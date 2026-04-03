"""
Object counting: YOLO (Ultralytics), HOG people detector, and legacy OpenCV pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from app.yolo_runner import (
    TRACKER_YAML_REID,
    reset_yolo_trackers,
    run_yolo_detect,
    run_yolo_segment,
    run_yolo_track,
)


@dataclass
class DetectionResult:
    count: int
    boxes: List[Tuple[int, int, int, int]]  # x, y, w, h
    snippets: List[np.ndarray]
    edge_map: np.ndarray
    binary_mask: np.ndarray
    algorithm_name: str = ""
    labels: List[str] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    track_ids: List[int] = field(default_factory=list)


def filter_detection_by_classes(
    r: DetectionResult,
    frame_bgr: np.ndarray,
    enabled: Optional[Set[str]],
) -> DetectionResult:
    """If enabled is None or empty, show all. Otherwise keep only listed class names (case-insensitive)."""
    if not enabled:
        return r
    if not r.labels or len(r.labels) != len(r.boxes):
        return r
    en = {x.lower() for x in enabled}
    idx = [i for i, lb in enumerate(r.labels) if lb.lower() in en]
    if len(idx) == len(r.boxes):
        return r
    boxes = [r.boxes[i] for i in idx]
    labels = [r.labels[i] for i in idx]
    confs = [r.confidences[i] for i in idx] if r.confidences and len(r.confidences) == len(r.boxes) else []
    tids = [r.track_ids[i] for i in idx] if r.track_ids and len(r.track_ids) == len(r.boxes) else [-1] * len(boxes)
    snippets = _snippets_from_boxes(frame_bgr, boxes)
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for x, y, bw, bh in boxes:
        y1 = min(h, y + bh)
        x1 = min(w, x + bw)
        mask[y:y1, x:x1] = 255
    return DetectionResult(
        len(boxes),
        boxes,
        snippets,
        r.edge_map,
        mask,
        r.algorithm_name,
        labels=list(labels),
        confidences=list(confs),
        track_ids=list(tids),
    )


class AlgorithmKind(Enum):
    YOLO_V8 = auto()
    YOLO_V8_SEG = auto()
    HOG_PEOPLE = auto()
    EDGE_CONTOUR = auto()
    ADAPTIVE_THRESHOLD = auto()
    BACKGROUND_SUBTRACT = auto()
    HYBRID_EDGE_BLOB = auto()


def _nms_boxes(
    boxes: Sequence[Tuple[int, int, int, int]],
    scores: Sequence[float],
    iou_thresh: float = 0.35,
) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    x1, y1 = b[:, 0], b[:, 1]
    x2, y2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[rest] - inter + 1e-6
        iou = inter / union
        order = rest[iou < iou_thresh]
    return [tuple(int(round(v)) for v in boxes[k]) for k in keep]


def _boxes_from_contours(
    contours: Sequence,
    frame_shape: Tuple[int, int, int],
    min_area: float,
    max_area_ratio: float = 0.65,
) -> List[Tuple[int, int, int, int]]:
    h, w = frame_shape[:2]
    max_area = h * w * max_area_ratio
    boxes = []
    scores = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 4 or bh < 4:
            continue
        boxes.append((x, y, bw, bh))
        scores.append(area)
    return _nms_boxes(boxes, scores)


def _strong_edges(
    gray: np.ndarray,
    blur_ksize: int,
    canny_low: int,
    canny_high: int,
    morph_kernel: int,
    close_iters: int,
    dilate_iters: int,
) -> Tuple[np.ndarray, np.ndarray]:
    k = max(3, blur_ksize | 1)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_n = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, strong_grad = cv2.threshold(mag_n, int(canny_low * 0.9), 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    combined = cv2.bitwise_or(edges, strong_grad)
    mk = max(3, morph_kernel | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
    closed = cv2.dilate(closed, kernel, iterations=dilate_iters)
    return combined, closed


def _snippets_from_boxes(frame_bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]], pad: int = 4) -> List[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    out: List[np.ndarray] = []
    for x, y, bw, bh in boxes:
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)
        crop = frame_bgr[y0:y1, x0:x1].copy()
        if crop.size > 0:
            out.append(crop)
    return out


def draw_overlay(frame_bgr: np.ndarray, result: DetectionResult) -> np.ndarray:
    vis = frame_bgr.copy()
    for i, (x, y, bw, bh) in enumerate(result.boxes):
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 220, 0), 2)
        parts = [str(i + 1)]
        if i < len(result.labels) and result.labels[i]:
            parts.append(result.labels[i])
        if i < len(result.confidences):
            parts.append(f"{result.confidences[i]:.2f}")
        if result.track_ids and i < len(result.track_ids) and result.track_ids[i] >= 0:
            parts.append(f"id{result.track_ids[i]}")
        text = " ".join(parts)
        cv2.putText(
            vis,
            text,
            (x + 2, max(22, y + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        vis,
        f"count={result.count}  [{result.algorithm_name}]",
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (40, 40, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


class MultiAlgorithmDetector:
    def __init__(self) -> None:
        self._bg_sub: cv2.BackgroundSubtractor = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=32, detectShadows=False
        )
        self._warmup_left = 25
        self._yolo_cache: dict = {}
        self._hog: cv2.HOGDescriptor | None = None

    def reset_background(self) -> None:
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=32, detectShadows=False
        )
        self._warmup_left = 25

    def clear_yolo_cache(self) -> None:
        reset_yolo_trackers(self._yolo_cache)
        self._yolo_cache.clear()

    def reset_yolo_tracker_state(self) -> None:
        reset_yolo_trackers(self._yolo_cache)

    def _ensure_hog(self) -> cv2.HOGDescriptor:
        if self._hog is None:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        return self._hog

    def detect(
        self,
        frame_bgr: np.ndarray,
        kind: AlgorithmKind,
        *,
        min_area_ratio: float = 0.0008,
        blur_ksize: int = 5,
        canny_low: int = 48,
        canny_high: int = 140,
        morph_kernel: int = 5,
        close_iters: int = 2,
        dilate_iters: int = 1,
        adapt_block: int = 31,
        adapt_c: int = -2,
        yolo_model: str = "yolov8n.pt",
        yolo_conf: float = 0.35,
        yolo_iou: float = 0.5,
        yolo_imgsz: int = 640,
        yolo_device: str = "",
        hog_hit_threshold: float = 0.5,
        hog_scale: float = 1.05,
        yolo_track: bool = True,
        yolo_tracker_yaml: str = TRACKER_YAML_REID,
    ) -> DetectionResult:
        h, w = frame_bgr.shape[:2]
        min_area = float(h * w * min_area_ratio)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if kind == AlgorithmKind.YOLO_V8:
            d = str(yolo_device).strip()
            dev = None if not d else d
            if yolo_track:
                boxes, labels, confs, edge_vis, mask, name, tids = run_yolo_track(
                    frame_bgr,
                    model_name=yolo_model,
                    conf=yolo_conf,
                    iou=yolo_iou,
                    imgsz=yolo_imgsz,
                    device=dev,
                    cache=self._yolo_cache,
                    segment=False,
                    tracker=yolo_tracker_yaml,
                )
            else:
                boxes, labels, confs, edge_vis, mask, name, tids = run_yolo_detect(
                    frame_bgr,
                    model_name=yolo_model,
                    conf=yolo_conf,
                    iou=yolo_iou,
                    imgsz=yolo_imgsz,
                    device=dev,
                    cache=self._yolo_cache,
                )
            return DetectionResult(
                len(boxes),
                boxes,
                _snippets_from_boxes(frame_bgr, boxes),
                edge_vis,
                mask,
                name,
                labels=list(labels),
                confidences=list(confs),
                track_ids=list(tids),
            )

        if kind == AlgorithmKind.YOLO_V8_SEG:
            d = str(yolo_device).strip()
            dev = None if not d else d
            if yolo_track:
                boxes, labels, confs, edge_vis, mask, name, tids = run_yolo_track(
                    frame_bgr,
                    model_name=yolo_model,
                    conf=yolo_conf,
                    iou=yolo_iou,
                    imgsz=yolo_imgsz,
                    device=dev,
                    cache=self._yolo_cache,
                    segment=True,
                    tracker=yolo_tracker_yaml,
                )
            else:
                boxes, labels, confs, edge_vis, mask, name, tids = run_yolo_segment(
                    frame_bgr,
                    model_name=yolo_model,
                    conf=yolo_conf,
                    iou=yolo_iou,
                    imgsz=yolo_imgsz,
                    device=dev,
                    cache=self._yolo_cache,
                )
            return DetectionResult(
                len(boxes),
                boxes,
                _snippets_from_boxes(frame_bgr, boxes),
                edge_vis,
                mask,
                name,
                labels=list(labels),
                confidences=list(confs),
                track_ids=list(tids),
            )

        if kind == AlgorithmKind.HOG_PEOPLE:
            hog = self._ensure_hog()
            rects, weights = hog.detectMultiScale(
                frame_bgr,
                winStride=(8, 8),
                padding=(16, 16),
                scale=float(hog_scale),
                hitThreshold=float(hog_hit_threshold),
            )
            raw = [(int(x), int(y), int(bw), int(bh)) for (x, y, bw, bh) in rects]
            scores = [float(w[0]) for w in weights] if len(weights) else [1.0] * len(raw)
            if len(scores) != len(raw):
                scores = [1.0] * len(raw)
            boxes = _nms_boxes(raw, scores, iou_thresh=0.4)
            labels = ["person"] * len(boxes)
            confidences = [1.0] * len(boxes)
            edge_vis, binary = _strong_edges(
                gray, blur_ksize, canny_low, canny_high, morph_kernel, close_iters, dilate_iters
            )
            mask = np.zeros((h, w), np.uint8)
            for x, y, bw, bh in boxes:
                mask[y : y + bh, x : x + bw] = 255
            name = "HOG + SVM (OpenCV people)"
            return DetectionResult(
                len(boxes),
                boxes,
                _snippets_from_boxes(frame_bgr, boxes),
                edge_vis,
                mask,
                name,
                labels=labels,
                confidences=confidences,
                track_ids=[-1] * len(boxes),
            )

        if kind == AlgorithmKind.EDGE_CONTOUR:
            edge_vis, binary = _strong_edges(
                gray, blur_ksize, canny_low, canny_high, morph_kernel, close_iters, dilate_iters
            )
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = _boxes_from_contours(contours, frame_bgr.shape, min_area)
            name = "Legacy: edge + contour"
            return DetectionResult(
                len(boxes),
                boxes,
                _snippets_from_boxes(frame_bgr, boxes),
                edge_vis,
                binary,
                name,
                track_ids=[-1] * len(boxes),
            )

        if kind == AlgorithmKind.ADAPTIVE_THRESHOLD:
            edge_vis, edge_bin = _strong_edges(
                gray, blur_ksize, canny_low, canny_high, morph_kernel, 1, 0
            )
            ab = max(3, adapt_block | 1)
            at = cv2.adaptiveThreshold(
                cv2.GaussianBlur(gray, (blur_ksize | 1, blur_ksize | 1), 0),
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                ab,
                adapt_c,
            )
            binary = cv2.bitwise_and(at, cv2.dilate(edge_bin, np.ones((3, 3), np.uint8), iterations=1))
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = _boxes_from_contours(contours, frame_bgr.shape, min_area)
            name = "Legacy: adaptive threshold × edges"
            return DetectionResult(
                len(boxes),
                boxes,
                _snippets_from_boxes(frame_bgr, boxes),
                edge_vis,
                binary,
                name,
                track_ids=[-1] * len(boxes),
            )

        if kind == AlgorithmKind.BACKGROUND_SUBTRACT:
            fg = self._bg_sub.apply(frame_bgr)
            if self._warmup_left > 0:
                self._warmup_left -= 1
            _, edge_bin = _strong_edges(
                gray, blur_ksize, max(20, canny_low // 2), canny_high, morph_kernel, 1, 1
            )
            binary = cv2.bitwise_and(fg, edge_bin)
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=2,
            )
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = _boxes_from_contours(contours, frame_bgr.shape, min_area * 1.2)
            edge_vis, _ = _strong_edges(gray, blur_ksize, canny_low, canny_high, morph_kernel, close_iters, dilate_iters)
            name = "Legacy: MOG2 + edges"
            return DetectionResult(
                len(boxes),
                boxes,
                _snippets_from_boxes(frame_bgr, boxes),
                edge_vis,
                binary,
                name,
                track_ids=[-1] * len(boxes),
            )

        edge_vis, binary = _strong_edges(
            gray, blur_ksize, canny_low, canny_high, morph_kernel, close_iters + 1, dilate_iters + 1
        )
        filled = binary.copy()
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > min_area * 0.5:
                cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)
        dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
        if dist.max() < 1e-3:
            contours2, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = _boxes_from_contours(contours2, frame_bgr.shape, min_area)
        else:
            _, peaks = cv2.threshold(dist, 0.35 * dist.max(), 255, cv2.THRESH_BINARY)
            peaks = peaks.astype(np.uint8)
            n, markers = cv2.connectedComponents(peaks)
            boxes = []
            scores = []
            for lab in range(1, n):
                ys, xs = np.where(markers == lab)
                if xs.size < 10:
                    continue
                x, y, bw, bh = cv2.boundingRect(np.column_stack((xs, ys)))
                area = float(bw * bh)
                if area < min_area:
                    continue
                boxes.append((x, y, bw, bh))
                scores.append(float(dist[ys, xs].max()))
            boxes = _nms_boxes(boxes, scores, iou_thresh=0.4)
        name = "Legacy: hybrid peaks on edges"
        return DetectionResult(
            len(boxes),
            boxes,
            _snippets_from_boxes(frame_bgr, boxes),
            edge_vis,
            filled,
            name,
            track_ids=[-1] * len(boxes),
        )
