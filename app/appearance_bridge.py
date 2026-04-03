"""
Optional stable IDs on top of YOLO/ByteTrack/BoT-SORT IDs using HSV histogram similarity.

Helps keep one timeline when the backend briefly drops an object and returns a new track id.
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np


def _crop_hist(bgr: np.ndarray) -> np.ndarray:
    if bgr is None or bgr.size < 100:
        return np.zeros(96, dtype=np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.astype(np.float32).flatten()


class AppearanceTrackBridge:
    """Maps backend track ids to stable ids using label + histogram correlation."""

    def __init__(self, max_gap_frames: int = 120, min_correlation: float = 0.72) -> None:
        self.max_gap = max_gap_frames
        self.min_corr = min_correlation
        self._next_stable = 1
        self._backend_to_stable: Dict[int, int] = {}
        self._bank: Dict[int, Dict[str, object]] = {}

    def reset(self) -> None:
        self._next_stable = 1
        self._backend_to_stable.clear()
        self._bank.clear()

    def stable_id(self, backend_tid: int, frame_idx: int, crop_bgr: np.ndarray, label: str) -> int:
        h = _crop_hist(crop_bgr)
        if backend_tid in self._backend_to_stable:
            sid = self._backend_to_stable[backend_tid]
            self._touch(sid, h, frame_idx, label)
            return sid

        best_sid: Optional[int] = None
        best_c = -1.0
        for sid, meta in self._bank.items():
            if str(meta["label"]).lower() != label.lower():
                continue
            last_fi = int(meta["last_fi"])
            if frame_idx - last_fi > self.max_gap:
                continue
            ref = meta["hist"]
            if isinstance(ref, np.ndarray) and ref.shape == h.shape:
                c = cv2.compareHist(ref, h, cv2.HISTCMP_CORREL)
                if c > best_c:
                    best_c = c
                    best_sid = sid

        if best_sid is not None and best_c >= self.min_corr:
            self._backend_to_stable[backend_tid] = best_sid
            self._touch(best_sid, h, frame_idx, label)
            return best_sid

        sid = self._next_stable
        self._next_stable += 1
        self._backend_to_stable[backend_tid] = sid
        self._bank[sid] = {"hist": h.copy(), "last_fi": frame_idx, "label": label}
        return sid

    def _touch(self, sid: int, h: np.ndarray, frame_idx: int, label: str) -> None:
        meta = self._bank[sid]
        old = meta["hist"]
        if isinstance(old, np.ndarray) and old.shape == h.shape:
            meta["hist"] = (0.88 * old + 0.12 * h).astype(np.float32)
        else:
            meta["hist"] = h.copy()
        meta["last_fi"] = frame_idx
        meta["label"] = label
