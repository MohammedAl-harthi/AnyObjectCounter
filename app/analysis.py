"""Time-series statistics for object counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class FrameRecord:
    frame_index: int
    time_sec: float
    count: int
    algorithm: str


class CountSeries:
    def __init__(self, window: int = 60) -> None:
        self.window = max(5, window)
        self.records: List[FrameRecord] = []

    def clear(self) -> None:
        self.records.clear()

    def append(self, frame_index: int, time_sec: float, count: int, algorithm: str) -> None:
        self.records.append(FrameRecord(frame_index, time_sec, count, algorithm))

    def upsert(self, frame_index: int, time_sec: float, count: int, algorithm: str) -> None:
        """Replace the last sample if it refers to the same frame (e.g. slider idle)."""
        rec = FrameRecord(frame_index, time_sec, count, algorithm)
        if self.records and self.records[-1].frame_index == frame_index:
            self.records[-1] = rec
        else:
            self.records.append(rec)

    def arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.records:
            return np.array([]), np.array([]), np.array([])
        fi = np.array([r.frame_index for r in self.records], dtype=np.float64)
        t = np.array([r.time_sec for r in self.records], dtype=np.float64)
        c = np.array([r.count for r in self.records], dtype=np.float64)
        return fi, t, c

    def rolling_mean_std(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        _, _, c = self.arrays()
        if c.size == 0:
            return None, None
        w = min(self.window, max(3, c.size))
        kernel = np.ones(w) / w
        pad = w // 2
        padded = np.pad(c, (pad, pad), mode="edge")
        mean = np.convolve(padded, kernel, mode="valid")[: c.size]
        mean_pad = np.pad(mean, (pad, pad), mode="edge")[: c.size + 2 * pad]
        # rolling std via conv of squares
        sq = np.pad(c**2, (pad, pad), mode="edge")
        mean_sq = np.convolve(sq, kernel, mode="valid")[: c.size]
        var = np.maximum(0.0, mean_sq - mean**2)
        std = np.sqrt(var)
        return mean, std

    def summary(self) -> dict:
        _, _, c = self.arrays()
        if c.size == 0:
            return {
                "n": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0,
                "max": 0,
                "trend_per_100_frames": 0.0,
            }
        slope = 0.0
        if c.size >= 2:
            x = np.arange(c.size, dtype=np.float64)
            slope = float(np.polyfit(x, c, 1)[0])
        return {
            "n": int(c.size),
            "mean": float(np.mean(c)),
            "std": float(np.std(c)),
            "min": int(np.min(c)),
            "max": int(np.max(c)),
            "trend_per_100_frames": float(slope * 100.0),
        }
