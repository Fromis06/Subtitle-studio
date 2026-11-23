import cv2
import numpy as np


class MathEngine:
    @staticmethod
    def _focused_edges(gray: np.ndarray):
        """Extract edges, preferring edges around bright subtitle strokes when present."""
        edges = cv2.Canny(gray, 90, 180)
        bright = cv2.inRange(gray, 165, 255)
        bright_band = cv2.dilate(
            bright, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        )
        focused = cv2.bitwise_and(edges, bright_band)
        focused_count = cv2.countNonZero(focused)
        if focused_count >= 15:
            return focused, focused_count
        return edges, cv2.countNonZero(edges)

    @staticmethod
    def compute_anchor_signature(anchor_img: np.ndarray):
        """Build a text-focused grayscale and edge signature for an anchor frame."""
        g_anchor = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2GRAY)
        edges, edge_count = MathEngine._focused_edges(g_anchor)
        return g_anchor, edges, edge_count

    @staticmethod
    def is_subtitle_stable_precomputed(
        anchor_signature, test_img: np.ndarray, diff_threshold=26
    ) -> bool:
        """Compare a frame with a precomputed anchor signature symmetrically."""
        g_anchor, anchor_edges, anchor_edge_count = anchor_signature

        if anchor_edge_count < 10 or test_img is None or test_img.size == 0:
            return False

        g_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        if g_test.shape != g_anchor.shape:
            g_test = cv2.resize(g_test, (g_anchor.shape[1], g_anchor.shape[0]))

        diff = cv2.absdiff(g_anchor, g_test)
        _, static_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY_INV)

        stable_text = cv2.bitwise_and(anchor_edges, static_mask)
        stable_count = cv2.countNonZero(stable_text)
        if stable_count <= (anchor_edge_count * 0.45):
            return False

        # Reject if substantial new text edges appeared compared to the anchor frame
        test_edges, _ = MathEngine._focused_edges(g_test)
        changed_mask = cv2.bitwise_not(static_mask)
        new_edges = cv2.bitwise_and(test_edges, changed_mask)
        new_count = cv2.countNonZero(new_edges)
        if new_count > max(200, int(anchor_edge_count * 0.85)):
            return False

        return True

    @staticmethod
    def is_subtitle_stable(
        anchor_img: np.ndarray, test_img: np.ndarray, diff_threshold=26
    ) -> bool:
        """Compare two frames without a precomputed signature."""
        anchor_signature = MathEngine.compute_anchor_signature(anchor_img)
        return MathEngine.is_subtitle_stable_precomputed(
            anchor_signature, test_img, diff_threshold
        )
