import concurrent.futures
import difflib
import sys
import time
import cv2
import numpy as np
from models.ocr_engine import OCR_BATCH_SIZE, SubtitleOCR
from math_engine import MathEngine


class ProcessingCancelled(Exception):
    """Raised when the active job is cancelled."""


class SubtitleExtractor:
    def __init__(
        self,
        video_path,
        crop_box,
        step_sec=1.0,
        boundary_step=0.1,
        max_subtitle_duration=10.0,
        min_confidence=0.40,
        language="en",
        preprocess_mode="clean",
        debug_timing=False,
        ocr=None,
        cancel_event=None,
    ):
        self.video_path = video_path
        self.crop_box = crop_box
        self.step_sec = step_sec
        self.boundary_step = boundary_step
        self.max_subtitle_duration = max_subtitle_duration
        self.min_confidence = min_confidence
        self.language = language
        self.preprocess_mode = preprocess_mode
        self.ocr = ocr or SubtitleOCR(language=language)
        self.cancel_event = cancel_event
        self.math = MathEngine()
        self.debug_timing = debug_timing
        self._timing = {"position": 0.0, "has_text": 0.0, "find_t_end": 0.0, "read_text": 0.0}
        self._timing_count = {"position": 0, "has_text": 0, "find_t_end": 0, "read_text": 0}

    def _check_cancelled(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise ProcessingCancelled("Job was cancelled")

    def _log_stage(self, stage, elapsed):
        self._timing[stage] += elapsed
        self._timing_count[stage] += 1

    def _print_timing_summary(self):
        total = sum(self._timing.values())
        print("=== Processing timing summary (seconds) ===", file=sys.stderr)
        for stage, secs in self._timing.items():
            n = self._timing_count[stage]
            avg_ms = (secs / n * 1000) if n else 0
            print(f"  {stage:12s}: {secs:8.2f}s total | {n:5d} calls | {avg_ms:7.2f}ms average", file=sys.stderr)
        print(f"  {'TOTAL':12s}: {total:8.2f}s", file=sys.stderr)

    def _crop(self, frame):
        h, w = frame.shape[:2]
        x_min, x_max, y_min, y_max = self.crop_box
        y1 = max(0, min(h - 1, int(h * y_min)))
        y2 = max(y1 + 1, min(h, int(h * y_max)))
        x1 = max(0, min(w - 1, int(w * x_min)))
        x2 = max(x1 + 1, min(w, int(w * x_max)))
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _select_representative_crop(samples: list[np.ndarray]) -> np.ndarray:
        """Build a clean representative crop avoiding fade-in edges and moving noise."""
        if len(samples) == 1:
            return samples[0]
        if len(samples) == 2:
            # Prefer the second stable frame (past initial fade-in)
            return samples[1]
        # Pick 3 evenly distributed stable samples (around 25%, 50%, 75%) and take pixel median
        idx1 = max(0, len(samples) // 4)
        idx2 = len(samples) // 2
        idx3 = min(len(samples) - 1, (3 * len(samples)) // 4)
        stack = np.stack([samples[idx1], samples[idx2], samples[idx3]], axis=0)
        return np.median(stack, axis=0).astype(np.uint8)

    def _find_t_end(self, cap, anchor_crop, start_time, fps):
        """Find the subtitle end using sequential frame grabs.

        Returns (t_end, last_decoded_time, is_valid_anchor, representative_crop).
        """
        t0 = time.perf_counter()
        step_frames = max(1, int(round(fps * self.boundary_step)))
        actual_step = step_frames / fps

        anchor_signature = self.math.compute_anchor_signature(anchor_crop)
        if anchor_signature[2] < 10:
            if self.debug_timing:
                self._log_stage("find_t_end", time.perf_counter() - t0)
            return start_time, start_time, False, anchor_crop

        curr_time = start_time
        last_stable_time = start_time
        max_time = start_time + self.max_subtitle_duration
        stable_samples = [anchor_crop.copy()]
        max_samples = 12

        while curr_time < max_time:
            self._check_cancelled()
            for _ in range(step_frames - 1):
                cap.grab()

            ret, frame = cap.read()
            if not ret:
                break

            curr_time += actual_step
            test_crop = self._crop(frame)

            if not self.math.is_subtitle_stable_precomputed(anchor_signature, test_crop):
                if self.debug_timing:
                    self._log_stage("find_t_end", time.perf_counter() - t0)
                t_end = max(start_time + actual_step, last_stable_time)
                rep_crop = self._select_representative_crop(stable_samples)
                return t_end, curr_time, True, rep_crop

            last_stable_time = curr_time
            if len(stable_samples) < max_samples:
                stable_samples.append(test_crop.copy())
            elif len(stable_samples) == max_samples:
                # Keep early, mid, and latest frames bounded in memory
                stable_samples[max_samples // 2 :] = stable_samples[max_samples // 2 + 1 :] + [test_crop.copy()]

        if self.debug_timing:
            self._log_stage("find_t_end", time.perf_counter() - t0)
        rep_crop = self._select_representative_crop(stable_samples)
        return max(start_time + actual_step, last_stable_time), curr_time, True, rep_crop

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.lower().split())

    @staticmethod
    def _is_meaningful_text(text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        return any(ch.isalnum() for ch in stripped)

    def _merge_consecutive_subtitles(self, subtitles):
        """Merge consecutive segments with identical or near-identical text separated by a tiny gap."""
        if not subtitles:
            return []

        cleaned = [sub for sub in subtitles if self._is_meaningful_text(sub["text"])]
        if not cleaned:
            return []

        merged = [dict(cleaned[0])]
        max_merge_gap = max(self.step_sec * 0.6, self.boundary_step * 3, 0.35)

        for current in cleaned[1:]:
            prev = merged[-1]
            gap = current["start"] - prev["end"]
            norm_prev = self._normalize_text(prev["text"])
            norm_curr = self._normalize_text(current["text"])

            if 0 <= gap <= max_merge_gap:
                similarity = difflib.SequenceMatcher(None, norm_prev, norm_curr).ratio()
                if similarity >= 0.84:
                    prev["end"] = max(prev["end"], current["end"])
                    # Prefer the higher-confidence / cleaner reading
                    if current.get("_conf", 0.0) > prev.get("_conf", 0.0) and len(current["text"]) >= len(prev["text"]) * 0.85:
                        prev["text"] = current["text"]
                        prev["_conf"] = current.get("_conf", 0.0)
                    continue

            merged.append(dict(current))

        for item in merged:
            item.pop("_conf", None)
        return merged

    def process(self, progress_callback=None):
        self._check_cancelled()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        total_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0

        def to_percent(seconds):
            if total_sec <= 0:
                return 0
            return min(100, max(0, (seconds / total_sec) * 100))

        curr_sec = 0.0
        last_decoded_sec = None
        results = []
        pending_ocr = []
        active_batch = None

        SEEK_THRESHOLD_SEC = 2.0

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ocr_executor:
            def collect_active_batch():
                nonlocal active_batch
                if active_batch is None:
                    return

                scored_texts = active_batch["future"].result()
                self._check_cancelled()
                if self.debug_timing:
                    self._log_stage("read_text", time.perf_counter() - active_batch["submitted_at"])

                for candidate, (text, conf) in zip(active_batch["candidates"], scored_texts):
                    if not self._is_meaningful_text(text):
                        continue
                    sub = {
                        "start": round(candidate["start"], 2),
                        "end": round(candidate["end"], 2),
                        "text": text,
                        "_conf": conf,
                    }
                    results.append(sub)
                    if progress_callback:
                        progress_callback(
                            to_percent(candidate["end"]),
                            {"start": sub["start"], "end": sub["end"], "text": sub["text"]},
                        )
                active_batch = None

            def submit_pending_batch():
                nonlocal active_batch, pending_ocr
                collect_active_batch()
                if not pending_ocr:
                    return
                candidates = pending_ocr
                pending_ocr = []
                active_batch = {
                    "candidates": candidates,
                    "submitted_at": time.perf_counter(),
                    "future": ocr_executor.submit(
                        self.ocr.read_text_batch_with_scores,
                        [candidate["crop"] for candidate in candidates],
                        self.min_confidence,
                        self.language,
                        self.preprocess_mode,
                    ),
                }

            while curr_sec < total_sec:
                self._check_cancelled()
                t_pos0 = time.perf_counter()
                if last_decoded_sec is None:
                    cap.set(cv2.CAP_PROP_POS_MSEC, curr_sec * 1000.0)
                else:
                    gap = curr_sec - last_decoded_sec
                    if 0 <= gap <= SEEK_THRESHOLD_SEC:
                        skip_frames = max(0, int(round(gap * fps)) - 1)
                        for _ in range(skip_frames):
                            cap.grab()
                    else:
                        cap.set(cv2.CAP_PROP_POS_MSEC, curr_sec * 1000.0)

                ret, frame = cap.read()
                if self.debug_timing:
                    self._log_stage("position", time.perf_counter() - t_pos0)
                if not ret:
                    break
                last_decoded_sec = curr_sec

                crop_img = self._crop(frame)

                t_ht0 = time.perf_counter()
                has_text = self.ocr.has_text(crop_img)
                if self.debug_timing:
                    self._log_stage("has_text", time.perf_counter() - t_ht0)

                if has_text:
                    t_start = curr_sec
                    t_end, decoded_sec, valid_anchor, rep_crop = self._find_t_end(
                        cap, crop_img, curr_sec, fps
                    )
                    last_decoded_sec = decoded_sec

                    if not valid_anchor:
                        curr_sec += self.step_sec
                        if progress_callback:
                            progress_callback(to_percent(curr_sec), None)
                        continue

                    pending_ocr.append(
                        {"start": t_start, "end": t_end, "crop": rep_crop}
                    )
                    if len(pending_ocr) >= OCR_BATCH_SIZE:
                        submit_pending_batch()

                    curr_sec = max(
                        decoded_sec + (1.0 / fps),
                        t_end + max(self.boundary_step, 1.0 / fps),
                    )
                    if progress_callback:
                        progress_callback(to_percent(curr_sec), None)
                    continue

                curr_sec += self.step_sec
                if progress_callback:
                    progress_callback(to_percent(curr_sec), None)

            submit_pending_batch()
            collect_active_batch()

        cap.release()
        results.sort(key=lambda sub: sub["start"])
        results = self._merge_consecutive_subtitles(results)
        if self.debug_timing:
            self._print_timing_summary()
        return results
