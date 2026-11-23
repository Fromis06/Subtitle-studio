"""Persistent JSON-lines worker used by the Go desktop application.

Stdout is reserved for protocol messages. Diagnostics always go to stderr.
"""

import base64
import json
import math
import os
import sys
import threading
import traceback
import uuid

_write_lock = threading.Lock()
_job_lock = threading.Lock()
_current_job = None
_ocr_instance = None
_ocr_error = None
_ocr_ready = threading.Event()


def emit(message_type, **payload):
    message = {"type": message_type, **payload}
    with _write_lock:
        print(json.dumps(message, ensure_ascii=False), flush=True)


try:
    import cv2
    from algorithm import ProcessingCancelled, SubtitleExtractor
    from models.ocr_engine import SubtitleOCR
except Exception as import_exc:
    traceback.print_exc(file=sys.stderr)
    emit("fatal", message=f"Missing Python dependency: {import_exc}")
    raise SystemExit(1)


def validate_crop(crop):
    if not isinstance(crop, list) or len(crop) != 4:
        raise ValueError("crop must contain [x_min, x_max, y_min, y_max]")
    values = tuple(float(value) for value in crop)
    x_min, x_max, y_min, y_max = values
    if not (0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1):
        raise ValueError("crop coordinates must be normalized between 0 and 1")
    return values


def create_preview(video_path, ratio):
    if not os.path.isfile(video_path):
        raise FileNotFoundError("Video file does not exist")

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or math.isnan(fps) or fps <= 0:
            fps = 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / fps if frame_count > 0 else 0.0
        ratio = max(0.0, min(1.0, float(ratio)))
        timestamp = ratio * max(0.0, duration - (1.0 / fps))
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if (not ok or frame is None) and timestamp > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_count) - 2))
            ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("Cannot decode preview frame")

        height, width = frame.shape[:2]
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise ValueError("Cannot encode preview frame")

        return {
            "image": "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"),
            "width": width,
            "height": height,
            "duration": round(duration, 3),
            "timestamp": round(timestamp, 3),
        }
    finally:
        cap.release()


def run_job(command, ocr, cancel_event):
    global _current_job
    job_id = command.get("job_id") or str(uuid.uuid4())

    try:
        language = str(command.get("language") or "auto").strip().lower()
        preprocess_mode = str(command.get("preprocess_mode") or "natural").strip().lower()
        if language != getattr(ocr, "language", None):
            emit("loading", message=f"Preparing OCR model ({language.upper()})…")
            label = ocr.ensure_language(language)
            emit("ready", message=f"Ready · {label}")

        extractor = SubtitleExtractor(
            video_path=command["video"],
            crop_box=validate_crop(command["crop"]),
            step_sec=float(command.get("step", 1.0)),
            boundary_step=float(command.get("boundary_step", 0.1)),
            max_subtitle_duration=float(command.get("max_subtitle_duration", 10.0)),
            min_confidence=float(command.get("min_confidence", 0.40)),
            language=language,
            preprocess_mode=preprocess_mode,
            debug_timing=bool(command.get("debug_timing", False)),
            ocr=ocr,
            cancel_event=cancel_event,
        )

        emit("started", job_id=job_id)
        last_percent = 0

        def on_progress(percent, current_sub):
            nonlocal last_percent
            last_percent = max(last_percent, min(100, int(percent)))
            emit(
                "progress",
                job_id=job_id,
                percent=last_percent,
                current_sub=current_sub,
            )

        results = extractor.process(progress_callback=on_progress)
        emit(
            "done",
            job_id=job_id,
            percent=100,
            total_subtitles=len(results),
            data=results,
        )
    except ProcessingCancelled:
        emit("cancelled", job_id=job_id)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit("error", job_id=job_id, message=str(exc))
    finally:
        with _job_lock:
            _current_job = None


def _init_ocr_worker():
    global _ocr_instance, _ocr_error
    try:
        emit("loading", message="Initializing PP-OCRv6 engine…")
        _ocr_instance = SubtitleOCR(language="auto")
        device_label = getattr(_ocr_instance, "device_label", "ONNX")
        emit("ready", message=f"Ready · {device_label}")
    except Exception as exc:
        _ocr_error = str(exc)
        traceback.print_exc(file=sys.stderr)
        emit("fatal", message=str(exc))
    finally:
        _ocr_ready.set()


def main():
    global _current_job

    threading.Thread(target=_init_ocr_worker, daemon=True).start()

    for raw_line in sys.stdin:
        try:
            command = json.loads(raw_line)
            command_type = command.get("type")

            if command_type == "preview":
                with _job_lock:
                    busy = _current_job is not None
                if busy:
                    emit(
                        "preview_error",
                        request_id=command.get("request_id"),
                        message="Worker is busy processing a video",
                    )
                    continue
                try:
                    preview = create_preview(command["video"], command.get("ratio", 0.5))
                    emit("preview", request_id=command.get("request_id"), **preview)
                except Exception as exc:
                    emit(
                        "preview_error",
                        request_id=command.get("request_id"),
                        message=str(exc),
                    )

            elif command_type == "start":
                _ocr_ready.wait()
                if _ocr_instance is None:
                    emit(
                        "error",
                        job_id=command.get("job_id"),
                        message=_ocr_error or "OCR engine failed to initialize",
                    )
                    continue

                with _job_lock:
                    if _current_job is not None:
                        emit(
                            "error",
                            job_id=command.get("job_id"),
                            message="Another job is already running",
                        )
                        continue
                    cancel_event = threading.Event()
                    thread = threading.Thread(
                        target=run_job,
                        args=(command, _ocr_instance, cancel_event),
                        daemon=True,
                    )
                    _current_job = {
                        "job_id": command.get("job_id"),
                        "cancel": cancel_event,
                        "thread": thread,
                    }
                    thread.start()

            elif command_type == "cancel":
                with _job_lock:
                    current = _current_job
                    if current is not None and (
                        not command.get("job_id")
                        or command.get("job_id") == current["job_id"]
                    ):
                        current["cancel"].set()

            elif command_type == "shutdown":
                with _job_lock:
                    current = _current_job
                    if current is not None:
                        current["cancel"].set()
                return 0
            else:
                emit("protocol_error", message=f"Unknown command: {command_type}")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit("protocol_error", message=str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
