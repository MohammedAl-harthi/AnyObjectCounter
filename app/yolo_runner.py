"""Ultralytics YOLO inference with lazy model loading."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# BoT-SORT + ReID (appearance) for re-acquiring objects after short gaps; path is absolute for Ultralytics.
TRACKER_YAML_REID = str(Path(__file__).resolve().parent / "trackers" / "botsort_reid.yaml")
TRACKER_YAML_BYTE = "bytetrack.yaml"


def _empty_masks(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    gray = np.zeros((h, w), np.uint8)
    return gray, gray.copy()


def reset_yolo_trackers(cache: Dict[str, object]) -> None:
    """Reset ByteTrack/BoT-SORT state for a new clip.

    Never set ``predictor.trackers = None``: Ultralytics still has the attribute and
    ``on_predict_start(..., persist=True)`` will skip re-init, leaving trackers subscriptable
    as None and crashing in ``on_predict_postprocess_end``.
    """
    for model in cache.values():
        pred = getattr(model, "predictor", None)
        if pred is None:
            continue
        if not hasattr(pred, "trackers"):
            continue
        trackers = pred.trackers
        if trackers is None:
            try:
                delattr(pred, "trackers")
            except Exception:
                pass
            continue
        for tr in trackers:
            if tr is not None and hasattr(tr, "reset"):
                try:
                    tr.reset()
                except Exception:
                    pass


def _parse_boxes(
    r,
    h: int,
    w: int,
    *,
    with_track_ids: bool,
) -> Tuple[
    List[Tuple[int, int, int, int]],
    List[str],
    List[float],
    List[int],
    np.ndarray,
]:
    boxes_xywh: List[Tuple[int, int, int, int]] = []
    labels: List[str] = []
    confidences: List[float] = []
    track_ids: List[int] = []
    mask = np.zeros((h, w), np.uint8)

    if r.boxes is None or len(r.boxes) == 0:
        return boxes_xywh, labels, confidences, track_ids, mask

    xyxy_t = r.boxes.xyxy
    xyxy = xyxy_t.cpu().numpy() if hasattr(xyxy_t, "cpu") else np.asarray(xyxy_t)
    cfs_t = r.boxes.conf
    cfs = cfs_t.cpu().numpy() if hasattr(cfs_t, "cpu") else np.asarray(cfs_t)
    clss_t = r.boxes.cls
    clss = (clss_t.cpu().numpy() if hasattr(clss_t, "cpu") else np.asarray(clss_t)).astype(int)
    names = r.names if isinstance(r.names, dict) else {i: str(i) for i in range(1000)}

    id_arr: Optional[np.ndarray] = None
    if with_track_ids and getattr(r.boxes, "id", None) is not None:
        id_t = r.boxes.id
        id_arr = id_t.cpu().numpy() if hasattr(id_t, "cpu") else np.asarray(id_t)

    n = len(xyxy)
    for i in range(n):
        x1, y1, x2, y2 = xyxy[i]
        xi, yi = int(x1), int(y1)
        bw = int(max(1, x2 - x1))
        bh = int(max(1, y2 - y1))
        boxes_xywh.append((xi, yi, bw, bh))
        confidences.append(float(cfs[i]))
        cid = int(clss[i])
        labels.append(str(names.get(cid, cid)))
        if id_arr is not None and i < len(id_arr):
            track_ids.append(int(id_arr[i]))
        else:
            track_ids.append(-1)

        x0 = max(0, xi)
        y0 = max(0, yi)
        x1b = min(w, xi + bw)
        y1b = min(h, yi + bh)
        mask[y0:y1b, x0:x1b] = 255

    return boxes_xywh, labels, confidences, track_ids, mask


def run_yolo_detect(
    frame_bgr: np.ndarray,
    *,
    model_name: str,
    conf: float,
    iou: float,
    imgsz: int,
    device: Optional[str],
    cache: Dict[str, object],
) -> Tuple[
    List[Tuple[int, int, int, int]],
    List[str],
    List[float],
    np.ndarray,
    np.ndarray,
    str,
    List[int],
]:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        h, w = frame_bgr.shape[:2]
        e0, b0 = _empty_masks(h, w)
        return [], [], [], e0, b0, f"Ultralytics missing: {e} (pip install ultralytics)", []

    if model_name not in cache:
        cache[model_name] = YOLO(model_name)
    model = cache[model_name]

    h, w = frame_bgr.shape[:2]
    pred_kw: Dict[str, Union[str, int, float, bool, None]] = dict(
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        verbose=False,
    )
    if device:
        pred_kw["device"] = device
    results = model.predict(frame_bgr, **pred_kw)
    r = results[0]
    boxes, labels, confs, tids, mask = _parse_boxes(r, h, w, with_track_ids=False)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edge_vis = cv2.Canny(gray, 50, 150)
    algo = f"YOLO detect ({model_name})"
    return boxes, labels, confs, edge_vis, mask, algo, tids


def run_yolo_track(
    frame_bgr: np.ndarray,
    *,
    model_name: str,
    conf: float,
    iou: float,
    imgsz: int,
    device: Optional[str],
    cache: Dict[str, object],
    tracker: str = TRACKER_YAML_REID,
    segment: bool = False,
) -> Tuple[
    List[Tuple[int, int, int, int]],
    List[str],
    List[float],
    np.ndarray,
    np.ndarray,
    str,
    List[int],
]:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        h, w = frame_bgr.shape[:2]
        e0, b0 = _empty_masks(h, w)
        return [], [], [], e0, b0, f"Ultralytics missing: {e} (pip install ultralytics)", []

    if model_name not in cache:
        cache[model_name] = YOLO(model_name)
    model = cache[model_name]

    h, w = frame_bgr.shape[:2]
    pred_kw: Dict[str, Union[str, int, float, bool, None]] = dict(
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        verbose=False,
        persist=True,
        tracker=tracker,
    )
    if device:
        pred_kw["device"] = device

    results = model.track(frame_bgr, **pred_kw)
    r = results[0]
    boxes, labels, confs, tids, mask = _parse_boxes(r, h, w, with_track_ids=True)

    if segment and r.masks is not None and getattr(r.masks, "xy", None) is not None:
        mask.fill(0)
        for poly in r.masks.xy:
            if poly is None:
                continue
            arr = np.asarray(poly, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 3:
                continue
            pts = np.array([arr], dtype=np.int32)
            cv2.fillPoly(mask, pts, 255)
        if not mask.any() and boxes:
            for x, y, bw, bh in boxes:
                x0, y0 = max(0, x), max(0, y)
                x1b, y1b = min(w, x + bw), min(h, y + bh)
                mask[y0:y1b, x0:x1b] = 255

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edge_vis = cv2.Canny(gray, 40, 120)
    if mask.any():
        edge_vis = cv2.bitwise_or(edge_vis, cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)))

    algo = f"YOLO track{'+seg' if segment else ''} ({model_name}, {tracker})"
    return boxes, labels, confs, edge_vis, mask, algo, tids


def run_yolo_segment(
    frame_bgr: np.ndarray,
    *,
    model_name: str,
    conf: float,
    iou: float,
    imgsz: int,
    device: Optional[str],
    cache: Dict[str, object],
) -> Tuple[
    List[Tuple[int, int, int, int]],
    List[str],
    List[float],
    np.ndarray,
    np.ndarray,
    str,
    List[int],
]:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        h, w = frame_bgr.shape[:2]
        e0, b0 = _empty_masks(h, w)
        return [], [], [], e0, b0, f"Ultralytics missing: {e} (pip install ultralytics)", []

    if model_name not in cache:
        cache[model_name] = YOLO(model_name)
    model = cache[model_name]

    h, w = frame_bgr.shape[:2]
    pred_kw: Dict[str, Union[str, int, float, bool, None]] = dict(
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        verbose=False,
    )
    if device:
        pred_kw["device"] = device
    results = model.predict(frame_bgr, **pred_kw)
    r = results[0]
    boxes, labels, confs, tids, mask = _parse_boxes(r, h, w, with_track_ids=False)

    if r.masks is not None and getattr(r.masks, "xy", None) is not None:
        mask.fill(0)
        for poly in r.masks.xy:
            if poly is None:
                continue
            arr = np.asarray(poly, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 3:
                continue
            pts = np.array([arr], dtype=np.int32)
            cv2.fillPoly(mask, pts, 255)
    if not mask.any() and boxes:
        for x, y, bw, bh in boxes:
            x0, y0 = max(0, x), max(0, y)
            x1b, y1b = min(w, x + bw), min(h, y + bh)
            mask[y0:y1b, x0:x1b] = 255

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edge_vis = cv2.Canny(gray, 40, 120)
    if mask.any():
        edge_vis = cv2.bitwise_or(edge_vis, cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)))

    algo = f"YOLO segment ({model_name})"
    return boxes, labels, confs, edge_vis, mask, algo, tids
