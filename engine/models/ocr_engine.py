import os
import threading
import cv2
import numpy as np
import torch

try:
    from rapidocr import RapidOCR, OCRVersion, LangRec, ModelType
except ImportError:
    RapidOCR = None
    OCRVersion = None
    LangRec = None
    ModelType = None

try:
    import easyocr
except ImportError:
    easyocr = None


GPU_DEVICE = "cuda:0"
OCR_BATCH_SIZE = 8

SUPPORTED_LANGUAGES = {"auto", "en", "ja", "zh", "vi", "ko"}


class SubtitleOCR:
    """Hybrid RapidOCR (PP-OCRv6/v5) + EasyOCR subtitle recognition engine."""

    def __init__(self, language: str = "auto"):
        self._lock = threading.Lock()
        self.language = language if language in SUPPORTED_LANGUAGES else "auto"
        self._rapid_engines: dict[str, object] = {}
        self._easyocr_readers: dict[str, object] = {}

        has_cuda = torch.cuda.is_available()
        require_cuda = os.environ.get("SUBTITLE_REQUIRE_CUDA", "").strip() == "1"
        if not has_cuda and require_cuda:
            raise RuntimeError(
                "CUDA is unavailable in PyTorch. Install a CUDA-enabled PyTorch build before running GPU OCR."
            )

        self.has_cuda = has_cuda
        self.easy_gpu = GPU_DEVICE if has_cuda else False

        if RapidOCR is None and easyocr is None:
            raise RuntimeError("Neither rapidocr nor easyocr is installed.")

        # Preload default PP-OCRv6 multilingual engine (handles EN/JA/ZH seamlessly)
        if RapidOCR is not None:
            self._get_rapid_engine("v6_multi")

        if RapidOCR is None or self.language == "vi":
            self._get_easyocr_reader("vi")
        elif self.language == "ko" and RapidOCR is not None:
            self._get_rapid_engine("v5_ko")

        self.device_label = self._describe_device(self.language)

    def _describe_device(self, language: str) -> str:
        if language == "vi":
            if self.has_cuda:
                try:
                    return f"PP-OCRv6 + {torch.cuda.get_device_name(0)}"
                except Exception:
                    return "PP-OCRv6 + EasyOCR (CUDA)"
            return "PP-OCRv6 + EasyOCR (CPU)"
        if language == "ko" and RapidOCR is not None:
            return "RapidOCR PP-OCRv5 Korean"
        if RapidOCR is not None:
            return "RapidOCR PP-OCRv6 (ONNX)"
        return "EasyOCR CUDA" if self.has_cuda else "EasyOCR CPU"

    def _get_rapid_engine(self, key: str):
        if RapidOCR is None:
            return None
        with self._lock:
            engine = self._rapid_engines.get(key)
            if engine is not None:
                return engine

            base_params = {
                "Global.log_level": "warning",
                "Global.use_cls": False,
                "Det.limit_type": "max",
                "Det.limit_side_len": 960,
                "Det.unclip_ratio": 1.6,
                "Det.box_thresh": 0.45,
            }
            if key == "v5_ko":
                base_params.update({
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.lang_type": LangRec.KOREAN,
                    "Rec.model_type": ModelType.MOBILE,
                })
            else:
                base_params.update({
                    "Rec.ocr_version": OCRVersion.PPOCRV6,
                    "Rec.model_type": ModelType.MEDIUM,
                })

            engine = RapidOCR(params=base_params)
            self._rapid_engines[key] = engine
            return engine

    def _get_easyocr_reader(self, language: str):
        if easyocr is None:
            raise RuntimeError("EasyOCR is required for this language but is not installed.")
        lang_map = {
            "vi": ["vi", "en"],
            "en": ["en"],
            "ja": ["ja", "en"],
            "zh": ["ch_sim", "en"],
            "ko": ["ko", "en"],
        }
        key = language if language in lang_map else "vi"
        with self._lock:
            reader = self._easyocr_readers.get(key)
            if reader is None:
                reader = easyocr.Reader(
                    lang_map[key],
                    gpu=self.easy_gpu,
                    verbose=False,
                )
                self._easyocr_readers[key] = reader
            return reader

    def ensure_language(self, language: str) -> str:
        lang = language if language in SUPPORTED_LANGUAGES else "auto"
        if lang == "vi" or RapidOCR is None:
            self._get_easyocr_reader("vi" if lang == "vi" else lang)
            if RapidOCR is not None:
                self._get_rapid_engine("v6_multi")
        elif lang == "ko":
            self._get_rapid_engine("v5_ko")
        else:
            self._get_rapid_engine("v6_multi")
        self.language = lang
        self.device_label = self._describe_device(lang)
        return self.device_label

    def has_text(self, crop_img: np.ndarray) -> bool:
        """Reject crops that clearly contain no text before invoking OCR."""
        if crop_img is None or crop_img.size == 0:
            return False

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        if height < 8 or width < 16:
            return False

        # Count stroke edges rather than flat bright blobs (shirts, sky, light panels)
        edges = cv2.Canny(gray, 85, 185)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))

        active_per_row = np.count_nonzero(edges, axis=1)
        row_threshold = max(8, min(24, int(width * 0.012)))
        active_rows = np.count_nonzero(active_per_row >= row_threshold)
        required_rows = max(3, min(8, int(height * 0.06)))
        return active_rows >= required_rows

    @staticmethod
    def _prepare_crop(crop_img: np.ndarray, preprocess_mode: str = "natural") -> np.ndarray:
        """Prepare subtitle crop cleanly with border padding and optional bright-text isolation."""
        if crop_img is None or crop_img.size == 0:
            return crop_img

        h, w = crop_img.shape[:2]
        prepared = crop_img
        pad_color = (0, 0, 0)

        if preprocess_mode == "clean":
            hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2]
            s = hsv[:, :, 1]
            bright = ((v > 175) & (s < 90)) | (
                (v > 175) & (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 42)
            )
            mask = bright.astype(np.uint8) * 255
            dark_halo = cv2.dilate(
                cv2.inRange(v, 0, 105),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            )
            outlined = cv2.bitwise_and(mask, dark_halo)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            keep = np.zeros_like(mask)
            for idx in range(1, num_labels):
                area = stats[idx, cv2.CC_STAT_AREA]
                bh = stats[idx, cv2.CC_STAT_HEIGHT]
                bw = stats[idx, cv2.CC_STAT_WIDTH]
                if area < 4 or bh > h * 0.7 or bw > w * 0.45:
                    continue
                comp = labels == idx
                if np.any(outlined[comp] > 0):
                    keep[comp] = 255

            if cv2.countNonZero(keep) >= 24:
                roi_mask = cv2.dilate(
                    keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                )
                soft_alpha = cv2.GaussianBlur(
                    roi_mask.astype(np.float32) / 255.0, (5, 5), 0
                )
                gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                blended = gray * soft_alpha + 24.0 * (1.0 - soft_alpha)
                out_gray = np.clip(blended, 0, 255).astype(np.uint8)
                prepared = cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)
                pad_color = (24, 24, 24)

        ph, pw = prepared.shape[:2]
        if ph < 96:
            scale = min(2.0, 128.0 / max(1, ph))
            prepared = cv2.resize(
                prepared,
                (int(pw * scale), int(ph * scale)),
                interpolation=cv2.INTER_LANCZOS4,
            )

        return cv2.copyMakeBorder(
            prepared, 24, 24, 32, 32, cv2.BORDER_CONSTANT, value=pad_color
        )

    @staticmethod
    def _merge_line_boxes(dt_boxes, img_shape) -> list[tuple[int, int, int, int]]:
        """Group DBNet word/chunk boxes into clean full-line boxes in reading order."""
        if dt_boxes is None or len(dt_boxes) == 0:
            return []

        h_img, w_img = img_shape[:2]
        rects = []
        for box in dt_boxes:
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            x1, y1, x2, y2 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
            if (x2 - x1) >= 3 and (y2 - y1) >= 6:
                rects.append([x1, y1, x2, y2])

        if not rects:
            return []

        rects.sort(key=lambda r: (r[1] + r[3]) / 2.0)
        rows = []
        for r in rects:
            cy = (r[1] + r[3]) / 2.0
            rh = max(1.0, r[3] - r[1])
            if not rows or abs(cy - rows[-1]["cy"]) > max(12.0, rh * 0.65):
                rows.append({"cy": cy, "rects": [r]})
            else:
                rows[-1]["rects"].append(r)

        line_boxes = []
        for row in rows:
            rs = sorted(row["rects"], key=lambda r: r[0])
            merged = [list(rs[0])]
            for r in rs[1:]:
                prev = merged[-1]
                rh = max(prev[3] - prev[1], r[3] - r[1])
                if r[0] - prev[2] <= rh * 1.4:
                    prev[0] = min(prev[0], r[0])
                    prev[1] = min(prev[1], r[1])
                    prev[2] = max(prev[2], r[2])
                    prev[3] = max(prev[3], r[3])
                else:
                    merged.append(list(r))

            for m in merged:
                x1 = max(0, int(m[0]) - 6)
                y1 = max(0, int(m[1]) - 3)
                x2 = min(w_img, int(m[2]) + 6)
                y2 = min(h_img, int(m[3]) + 3)
                if x2 > x1 + 4 and y2 > y1 + 4:
                    line_boxes.append((x1, y1, x2, y2))
        return line_boxes

    @staticmethod
    def _assemble_reading_order(results, min_confidence=0.40) -> tuple[str, float]:
        """Sort recognized text boxes from top to bottom and left to right."""
        candidates = []
        for box, text, score in results:
            cleaned = " ".join(str(text).strip().split())
            if not cleaned or float(score) < min_confidence:
                continue
            xs = [point[0] for point in box]
            ys = [point[1] for point in box]
            candidates.append(
                {
                    "text": cleaned,
                    "score": float(score),
                    "left": min(xs),
                    "center_y": (min(ys) + max(ys)) / 2,
                    "height": max(1, max(ys) - min(ys)),
                }
            )

        if not candidates:
            return "", 0.0

        candidates.sort(key=lambda item: item["center_y"])
        row_tolerance = max(
            12, float(np.median([item["height"] for item in candidates])) * 0.8
        )
        rows = []
        for item in candidates:
            if not rows or item["center_y"] - rows[-1]["center_y"] > row_tolerance:
                rows.append({"center_y": item["center_y"], "items": [item]})
            else:
                rows[-1]["items"].append(item)

        texts = []
        scores = []
        for row in rows:
            row["items"].sort(key=lambda item: item["left"])
            for item in row["items"]:
                texts.append(item["text"])
                scores.append(item["score"])
        return " ".join(texts).strip(), (sum(scores) / len(scores)) if scores else 0.0

    def _recognize_single(
        self, prepared: np.ndarray, min_confidence: float, language: str
    ) -> tuple[str, float]:
        if language == "vi":
            det_engine = self._get_rapid_engine("v6_multi")
            det_res = det_engine(prepared, use_det=True, use_cls=False, use_rec=False)
            boxes = getattr(det_res, "boxes", None)
            line_boxes = self._merge_line_boxes(boxes, prepared.shape)
            if not line_boxes:
                return "", 0.0
            reader = self._get_easyocr_reader("vi")
            texts = []
            scores = []
            for x1, y1, x2, y2 in line_boxes:
                res = reader.recognize(prepared[y1:y2, x1:x2])
                for _, text, score in res or []:
                    cleaned = " ".join(str(text).strip().split())
                    if cleaned and float(score) >= min_confidence:
                        texts.append(cleaned)
                        scores.append(float(score))
            if not texts:
                return "", 0.0
            return " ".join(texts), sum(scores) / len(scores)

        engine_key = "v5_ko" if language == "ko" else "v6_multi"
        engine = self._get_rapid_engine(engine_key)
        res = engine(prepared, use_det=True, use_cls=False, use_rec=True, text_score=min_confidence)
        boxes = getattr(res, "boxes", None)
        txts = getattr(res, "txts", None)
        scores = getattr(res, "scores", None)
        if boxes is None or len(boxes) == 0 or not txts or not scores:
            return "", 0.0
        return self._assemble_reading_order(zip(boxes, txts, scores), min_confidence)

    def read_text_batch_with_scores(
        self,
        crop_images: list[np.ndarray],
        min_confidence: float = 0.40,
        language: str | None = None,
        preprocess_mode: str = "natural",
    ) -> list[tuple[str, float]]:
        """Recognize a batch of subtitle crops, returning (text, confidence)."""
        if not crop_images:
            return []

        lang = language if language in SUPPORTED_LANGUAGES else self.language
        min_confidence = max(0.10, min(0.95, float(min_confidence)))

        try:
            if RapidOCR is not None:
                outputs = []
                for raw_img in crop_images:
                    prepared = self._prepare_crop(raw_img, preprocess_mode=preprocess_mode)
                    text, conf = self._recognize_single(prepared, min_confidence, lang)
                    if not text and preprocess_mode == "clean":
                        natural = self._prepare_crop(raw_img, preprocess_mode="natural")
                        text, conf = self._recognize_single(natural, min_confidence, lang)
                    outputs.append((text, conf))
                return outputs

            # Fallback if RapidOCR is unavailable
            prepared_images = [
                self._prepare_crop(image, preprocess_mode=preprocess_mode)
                for image in crop_images
            ]
            reader = self._get_easyocr_reader("vi" if lang == "auto" else lang)
            results_per_image = reader.readtext_batched(
                prepared_images,
                batch_size=min(OCR_BATCH_SIZE, len(prepared_images)),
                detail=1,
                paragraph=False,
                text_threshold=0.65,
                low_text=0.35,
                link_threshold=0.45,
                width_ths=0.8,
            )
            return [
                self._assemble_reading_order(result, min_confidence)
                for result in results_per_image
            ]
        except Exception as exc:
            import sys
            print(f"[ocr_engine] OCR failed: {exc}", file=sys.stderr)
            return [("", 0.0)] * len(crop_images)

    def read_text_batch(
        self,
        crop_images: list[np.ndarray],
        min_confidence: float = 0.40,
        language: str | None = None,
        preprocess_mode: str = "natural",
    ) -> list[str]:
        """Recognize a batch of crops."""
        return [
            text
            for text, _ in self.read_text_batch_with_scores(
                crop_images,
                min_confidence=min_confidence,
                language=language,
                preprocess_mode=preprocess_mode,
            )
        ]

    def read_text(
        self,
        crop_img: np.ndarray,
        min_confidence: float = 0.40,
        language: str | None = None,
        preprocess_mode: str = "natural",
    ) -> str:
        """Recognize a single crop."""
        return self.read_text_batch(
            [crop_img],
            min_confidence=min_confidence,
            language=language,
            preprocess_mode=preprocess_mode,
        )[0]
