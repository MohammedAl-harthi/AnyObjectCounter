from __future__ import annotations

import os
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.analysis import CountSeries
from app.appearance_bridge import AppearanceTrackBridge
from app.coco_names import COCO_CLASS_NAMES
from app.detectors import (
    AlgorithmKind,
    DetectionResult,
    MultiAlgorithmDetector,
    draw_overlay,
    filter_detection_by_classes,
)
from app.yolo_runner import TRACKER_YAML_BYTE, TRACKER_YAML_REID

ALGO_ITEMS: List[Tuple[AlgorithmKind, str]] = [
    (AlgorithmKind.YOLO_V8, "YOLOv8 — detect (recommended)"),
    (AlgorithmKind.YOLO_V8_SEG, "YOLOv8 — segment (masks)"),
    (AlgorithmKind.HOG_PEOPLE, "HOG — pedestrians (CPU)"),
    (AlgorithmKind.EDGE_CONTOUR, "Legacy: edge + contour"),
    (AlgorithmKind.ADAPTIVE_THRESHOLD, "Legacy: adaptive threshold × edges"),
    (AlgorithmKind.BACKGROUND_SUBTRACT, "Legacy: MOG2 + edges"),
    (AlgorithmKind.HYBRID_EDGE_BLOB, "Legacy: hybrid edge peaks"),
]

YOLO_DETECT_MODELS = ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt")
YOLO_SEG_MODELS = ("yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt")


def bgr_to_qpixmap_fit(bgr: np.ndarray, max_w: int, max_h: int) -> QPixmap:
    """Scale BGR image to fit inside max_w×max_h while preserving aspect ratio."""
    if bgr is None or bgr.size == 0 or max_w < 4 or max_h < 4:
        return QPixmap()
    h, w = bgr.shape[:2]
    if w < 1 or h < 1:
        return QPixmap()
    scale = min(max_w / w, max_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr, (nw, nh), interpolation=interp)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    hh, ww, ch = rgb.shape
    qimg = QImage(rgb.data, ww, hh, ch * ww, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def gray_to_qpixmap_fit(gray: np.ndarray, max_w: int, max_h: int) -> QPixmap:
    if gray is None or gray.size == 0 or max_w < 4 or max_h < 4:
        return QPixmap()
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if w < 1 or h < 1:
        return QPixmap()
    scale = min(max_w / w, max_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(gray, (nw, nh), interpolation=interp)
    hh, ww = resized.shape[:2]
    qimg = QImage(resized.data, ww, hh, ww, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())


class VideoFrameLabel(QLabel):
    """Shows a frame pixmap; maps clicks to original frame coordinates (centered pixmap)."""

    clicked_frame = pyqtSignal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self._fh = 0
        self._fw = 0

    def set_frame_shape(self, height: int, width: int) -> None:
        self._fh = int(height)
        self._fw = int(width)

    def mousePressEvent(self, event) -> None:
        if self._fw <= 0 or self._fh <= 0:
            super().mousePressEvent(event)
            return
        pm = self.pixmap()
        if pm is None or pm.isNull():
            super().mousePressEvent(event)
            return
        pw, ph = pm.width(), pm.height()
        lw, lh = self.width(), self.height()
        ox = max(0, (lw - pw) // 2)
        oy = max(0, (lh - ph) // 2)
        px = int(event.position().x())
        py = int(event.position().y())
        if px < ox or px >= ox + pw or py < oy or py >= oy + ph:
            super().mousePressEvent(event)
            return
        fx = int((px - ox) * self._fw / max(1, pw))
        fy = int((py - oy) * self._fh / max(1, ph))
        fx = max(0, min(self._fw - 1, fx))
        fy = max(0, min(self._fh - 1, fy))
        self.clicked_frame.emit(fx, fy)
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Any Object Counter — multi-algorithm dashboard")
        self.resize(1400, 900)

        pg.setConfigOptions(foreground="#c8cad8", background="#1a1a22")

        self._cap: Optional[cv2.VideoCapture] = None
        self._source_path: Optional[str] = None
        self._webcam = False
        self._frame_count = 0
        self._fps = 30.0
        self._paused = True
        self._current_index = 0
        self._detector = MultiAlgorithmDetector()
        self._series = CountSeries(window=90)
        self._last_result: Optional[DetectionResult] = None
        self._live_idx = 0
        self._pix_vis: Optional[np.ndarray] = None
        self._pix_edge: Optional[np.ndarray] = None
        self._last_snippets: List[np.ndarray] = []
        self._last_snippet_labels: List[str] = []
        self._video_has_frame = False
        self._last_yolo_model_pick = ""
        self._pick_boxes: List[Tuple[int, int, int, int]] = []
        self._pick_labels: List[str] = []
        self._track_buffers: dict[int, List[Tuple[int, np.ndarray]]] = {}
        self._track_label: dict[int, str] = {}
        self._max_track_frames = 4000
        self._track_play_timer = QTimer(self)
        self._track_play_timer.timeout.connect(self._on_track_play_tick)
        self._appearance_bridge = AppearanceTrackBridge(max_gap_frames=120, min_correlation=0.72)

        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._snip_resize_timer = QTimer(self)
        self._snip_resize_timer.setSingleShot(True)
        self._snip_resize_timer.timeout.connect(self._relayout_snippets)

    def _build_ui(self) -> None:
        self._video_label = VideoFrameLabel()
        self._video_label.setText("Open a video file or start webcam")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(200, 120)
        self._video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._video_label.setObjectName("PanelVideo")
        self._video_label.setToolTip(
            "Click an object to add its class to the filter (YOLO modes). "
            "Smallest box under the cursor wins."
        )
        self._video_label.clicked_frame.connect(self._on_video_frame_click)

        self._edge_label = QLabel("Structure / detection mask")
        self._edge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._edge_label.setMinimumHeight(72)
        self._edge_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._edge_label.setObjectName("PanelEdge")

        left_split = QSplitter(Qt.Orientation.Vertical)
        left_split.addWidget(self._video_label)
        left_split.addWidget(self._edge_label)
        left_split.setStretchFactor(0, 3)
        left_split.setStretchFactor(1, 1)
        left_split.setCollapsible(0, False)
        left_split.setCollapsible(1, False)

        self._snip_scroll = QScrollArea()
        self._snip_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._snip_container = QWidget()
        self._snip_grid = QGridLayout(self._snip_container)
        self._snip_grid.setSpacing(6)
        self._snip_grid.setContentsMargins(8, 8, 8, 8)
        self._snip_scroll.setWidget(self._snip_container)
        self._snip_scroll.setWidgetResizable(True)
        self._snip_scroll.viewport().installEventFilter(self)

        self._plot = pg.PlotWidget(title="Object count vs frame")
        self._plot.showGrid(x=True, y=True, alpha=0.18)
        self._plot.setLabel("bottom", "Frame index")
        self._plot.setLabel("left", "Count")
        self._curve = self._plot.plot(pen=pg.mkPen("#8fb4ff", width=2), name="count")
        self._mean_curve = self._plot.plot(pen=pg.mkPen("#7dd3a8", width=1, style=Qt.PenStyle.DashLine), name="rolling mean")
        self._plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._plot.setMinimumHeight(160)

        self._hist_plot = pg.PlotWidget(title="Count distribution (session)")
        self._hist_plot.setLabel("bottom", "Count (binned)")
        self._hist_plot.setLabel("left", "Frames")
        self._hist = pg.BarGraphItem(x=[], height=[], width=0.8, brush=pg.mkBrush("#9b7fd6"))
        self._hist_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._hist_plot.setMinimumHeight(140)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Frame", "Time (s)", "Count", "Δ"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._table.setMinimumHeight(120)

        dash = QWidget()
        dl = QVBoxLayout(dash)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.addWidget(self._plot, stretch=2)
        hrow = QHBoxLayout()
        hrow.addWidget(self._hist_plot, stretch=1)
        stats_box = QGroupBox("Session statistics")
        sl = QVBoxLayout(stats_box)
        self._stats_label = QLabel("No data yet.")
        self._stats_label.setWordWrap(True)
        sl.addWidget(self._stats_label)
        hrow.addWidget(stats_box, stretch=1)
        dl.addLayout(hrow, stretch=1)
        dl.addWidget(self._table, stretch=1)

        self._hist_plot.addItem(self._hist)
        self._style_analysis_plots()

        tabs = QTabWidget()
        tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        tabs.addTab(self._snip_scroll, "Object snippets")

        focus_tab = QWidget()
        fl = QVBoxLayout(focus_tab)
        fc = QGroupBox("Which object types to show (YOLO / HOG)")
        fcl = QVBoxLayout(fc)
        fcl.addWidget(
            QLabel(
                "No classes checked → show everything. "
                "Check one or more → only those types (e.g. person + car). "
                "Click the main video on a detection to check that class."
            )
        )
        self._class_search = QLineEdit()
        self._class_search.setPlaceholderText("Search class name…")
        self._class_search.textChanged.connect(self._filter_class_list)
        fcl.addWidget(self._class_search)
        crow = QHBoxLayout()
        b_all = QPushButton("Check all COCO classes")
        b_all.clicked.connect(self._class_check_all)
        crow.addWidget(b_all)
        b_clear = QPushButton("Uncheck all (show all types)")
        b_clear.clicked.connect(self._class_uncheck_all)
        crow.addWidget(b_clear)
        b_vis = QPushButton("Check types visible this frame")
        b_vis.clicked.connect(self._class_check_visible_frame)
        crow.addWidget(b_vis)
        fcl.addLayout(crow)
        self._class_list = QListWidget()
        self._class_list.setMinimumHeight(160)
        for name in sorted(COCO_CLASS_NAMES):
            it = QListWidgetItem(name)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Unchecked)
            self._class_list.addItem(it)
        self._class_list.itemChanged.connect(lambda _: self._on_class_filter_changed())
        fcl.addWidget(self._class_list)
        fl.addWidget(fc)

        tg = QGroupBox("Per-object crop timeline (YOLO tracking)")
        tgl = QVBoxLayout(tg)
        tgl.addWidget(
            QLabel(
                "With tracking on, each object stores cropped frames over time. "
                "BoT-SORT + ReID matches by appearance when boxes disappear briefly; "
                "the HSV option can merge timelines if IDs still reset. "
                "Select a track to scrub or export an MP4."
            )
        )
        trow = QHBoxLayout()
        self._track_list = QListWidget()
        self._track_list.setMinimumHeight(120)
        self._track_list.currentRowChanged.connect(self._on_track_row_changed)
        trow.addWidget(self._track_list, stretch=1)
        tprev = QVBoxLayout()
        self._track_preview = QLabel("Preview")
        self._track_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._track_preview.setMinimumSize(200, 200)
        self._track_preview.setObjectName("TrackPreview")
        self._track_frame_lbl = QLabel("—")
        self._track_slider = QSlider(Qt.Orientation.Horizontal)
        self._track_slider.setMinimum(0)
        self._track_slider.setMaximum(0)
        self._track_slider.valueChanged.connect(self._on_track_slider_changed)
        tprev.addWidget(self._track_preview)
        tprev.addWidget(self._track_frame_lbl)
        tprev.addWidget(self._track_slider)
        pb = QHBoxLayout()
        self._track_play_btn = QPushButton("Play")
        self._track_play_btn.setCheckable(True)
        self._track_play_btn.toggled.connect(self._on_track_play_toggled)
        pb.addWidget(self._track_play_btn)
        self._track_save_btn = QPushButton("Save track as MP4…")
        self._track_save_btn.clicked.connect(self._save_track_mp4)
        pb.addWidget(self._track_save_btn)
        self._track_clear_btn = QPushButton("Clear timelines")
        self._track_clear_btn.clicked.connect(self._clear_track_timelines)
        pb.addWidget(self._track_clear_btn)
        tprev.addLayout(pb)
        trow.addLayout(tprev, stretch=1)
        tgl.addLayout(trow)
        fl.addWidget(tg)
        fl.addStretch()

        tabs.addTab(focus_tab, "Classes & tracks")
        tabs.addTab(dash, "Analysis dashboard")

        ctrl = self._build_controls_widget()
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setWidget(ctrl)
        ctrl_scroll.setFrameShape(QFrame.Shape.NoFrame)
        ctrl_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        tabs.addTab(ctrl_scroll, "Detection settings")

        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.addWidget(left_split)
        main_split.addWidget(tabs)
        main_split.setStretchFactor(0, 55)
        main_split.setStretchFactor(1, 45)
        main_split.setCollapsible(0, False)
        main_split.setCollapsible(1, False)

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(6, 6, 6, 6)
        cl.setSpacing(6)
        cl.addWidget(main_split, stretch=1)

        bar = QHBoxLayout()
        self._play_btn = QPushButton("Play")
        self._play_btn.clicked.connect(self._toggle_play)
        bar.addWidget(self._play_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        bar.addWidget(self._slider, stretch=1)

        self._time_label = QLabel("—")
        bar.addWidget(self._time_label)

        self._reset_stats_btn = QPushButton("Clear session stats")
        self._reset_stats_btn.clicked.connect(self._reset_stats)
        bar.addWidget(self._reset_stats_btn)

        cl.addLayout(bar)

        self.setCentralWidget(central)

        m = self.menuBar().addMenu("File")
        open_a = QAction("Open video…", self)
        open_a.triggered.connect(self._open_video)
        m.addAction(open_a)
        cam_a = QAction("Use webcam", self)
        cam_a.triggered.connect(self._open_webcam)
        m.addAction(cam_a)

    def _style_analysis_plots(self) -> None:
        axis_pen = pg.mkPen("#8a90a8")
        bg = pg.mkColor("#1a1a22")
        for w in (self._plot, self._hist_plot):
            w.setBackground(bg)
            for name in ("left", "bottom"):
                ax = w.getAxis(name)
                ax.setPen(axis_pen)
                ax.setTextPen(axis_pen)
        self._hist_plot.showGrid(x=True, y=True, alpha=0.15)
        tl = getattr(self._plot.plotItem, "titleLabel", None)
        if tl is not None:
            tl.setText(getattr(tl, "text", " "), color="#c8cbdc", size="11pt")
        tl2 = getattr(self._hist_plot.plotItem, "titleLabel", None)
        if tl2 is not None:
            tl2.setText(getattr(tl2, "text", " "), color="#c8cbdc", size="11pt")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_media_scaling()

    def eventFilter(self, obj, event) -> bool:
        if obj is self._snip_scroll.viewport() and event.type() == QEvent.Type.Resize:
            self._snip_resize_timer.stop()
            self._snip_resize_timer.start(80)
        return super().eventFilter(obj, event)

    def _apply_media_scaling(self) -> None:
        if not self._video_has_frame:
            return
        vw = max(64, self._video_label.width())
        vh = max(64, self._video_label.height())
        if self._pix_vis is not None:
            self._video_label.setPixmap(bgr_to_qpixmap_fit(self._pix_vis, vw, vh))
        ew = max(48, self._edge_label.width())
        eh = max(48, self._edge_label.height())
        if self._pix_edge is not None:
            self._edge_label.setPixmap(gray_to_qpixmap_fit(self._pix_edge, ew, eh))

    def _snippet_layout_dims(self) -> tuple[int, int]:
        m = self._snip_grid.contentsMargins()
        w = self._snip_scroll.viewport().width() - m.left() - m.right() - 8
        if w < 100:
            w = max(100, self._snip_scroll.width() - 32)
        cols = max(1, min(8, w // 120))
        thumb = max(72, min(360, (w - (cols - 1) * self._snip_grid.horizontalSpacing()) // cols - 4))
        return cols, int(thumb)

    def _relayout_snippets(self) -> None:
        if self._last_snippets:
            lb = self._last_snippet_labels if self._last_snippet_labels else None
            self._update_snippets(self._last_snippets, lb)

    def _build_controls_widget(self) -> QWidget:
        w = QWidget()
        g = QVBoxLayout(w)

        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel("Algorithm:"))
        self._algo = QComboBox()
        for kind, title in ALGO_ITEMS:
            self._algo.addItem(title, kind)
        self._algo.currentIndexChanged.connect(self._on_algo_change)
        algo_row.addWidget(self._algo)
        g.addLayout(algo_row)

        self._panel_yolo = QGroupBox("YOLO (Ultralytics)")
        yl = QVBoxLayout(self._panel_yolo)
        ymod = QHBoxLayout()
        ymod.addWidget(QLabel("Weights:"))
        self._yolo_model = QComboBox()
        self._yolo_model.setMinimumWidth(200)
        self._yolo_model.currentIndexChanged.connect(self._on_yolo_model_changed)
        ymod.addWidget(self._yolo_model, stretch=1)
        yl.addLayout(ymod)

        self._yolo_conf = self._double_row_layout(yl, "Confidence ≥", 0.05, 0.99, 0.35, 2)
        self._yolo_iou = self._double_row_layout(yl, "NMS IoU", 0.1, 0.95, 0.5, 2)
        iz_row = QHBoxLayout()
        iz_row.addWidget(QLabel("Inference size:"))
        self._yolo_imgsz = QComboBox()
        for z in (320, 416, 512, 640, 960):
            self._yolo_imgsz.addItem(str(z), z)
        self._yolo_imgsz.setCurrentIndex(3)
        self._yolo_imgsz.currentIndexChanged.connect(lambda _: self._refresh_frame())
        iz_row.addWidget(self._yolo_imgsz)
        yl.addLayout(iz_row)
        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Device:"))
        self._yolo_device = QComboBox()
        self._yolo_device.addItem("Auto (Ultralytics default)", "")
        self._yolo_device.addItem("cpu", "cpu")
        self._yolo_device.addItem("cuda:0", "cuda:0")
        self._yolo_device.addItem("mps (Apple GPU)", "mps")
        self._yolo_device.currentIndexChanged.connect(lambda _: self._refresh_frame())
        dev_row.addWidget(self._yolo_device)
        yl.addLayout(dev_row)
        self._yolo_unload = QPushButton("Unload YOLO weights (free RAM)")
        self._yolo_unload.clicked.connect(self._on_yolo_unload)
        yl.addWidget(self._yolo_unload)
        self._yolo_track = QCheckBox("Tracking: persist IDs + crop timelines (recommended for video)")
        self._yolo_track.setChecked(True)
        self._yolo_track.toggled.connect(self._on_yolo_track_toggled)
        yl.addWidget(self._yolo_track)
        tr_row = QHBoxLayout()
        tr_row.addWidget(QLabel("Tracker backend:"))
        self._tracker_backend = QComboBox()
        self._tracker_backend.setMinimumWidth(280)
        self._tracker_backend.blockSignals(True)
        self._tracker_backend.addItem("BoT-SORT + ReID (appearance — best after occlusions)", "reid")
        self._tracker_backend.addItem("ByteTrack (faster, box overlap only)", "byte")
        self._tracker_backend.setCurrentIndex(0)
        self._tracker_backend.blockSignals(False)
        self._tracker_backend.currentIndexChanged.connect(self._on_tracker_backend_changed)
        tr_row.addWidget(self._tracker_backend, stretch=1)
        yl.addLayout(tr_row)
        self._appearance_merge = QCheckBox(
            "Extra: merge crop timelines by HSV similarity if the backend still gives a new ID"
        )
        self._appearance_merge.setChecked(True)
        self._appearance_merge.toggled.connect(lambda _: self._on_appearance_merge_toggled())
        yl.addWidget(self._appearance_merge)
        g.addWidget(self._panel_yolo)

        self._panel_hog = QGroupBox("HOG pedestrian detector")
        hl = QVBoxLayout(self._panel_hog)
        self._hog_hit = self._double_row_layout(hl, "Hit threshold", 0.0, 1.0, 0.5, 2)
        self._hog_scale = self._double_row_layout(hl, "Pyramid scale step", 1.01, 1.15, 1.05, 3)
        g.addWidget(self._panel_hog)

        self._panel_classical = QGroupBox("Legacy OpenCV parameters")
        cl = QVBoxLayout(self._panel_classical)
        self._canny_lo = self._spin_slider_row(cl, "Canny low", 10, 200, 48)
        self._canny_hi = self._spin_slider_row(cl, "Canny high", 20, 300, 140)
        self._min_area = self._double_row_in_layout(cl, "Min area (fraction of frame)", 0.0001, 0.05, 0.0008, 4)
        self._blur = self._spin_slider_row(cl, "Gaussian blur (odd)", 3, 15, 5)
        self._morph = self._spin_slider_row(cl, "Morph kernel", 3, 21, 5)
        self._close_i = self._spin_slider_row(cl, "Close iterations", 0, 6, 2)
        self._dilate_i = self._spin_slider_row(cl, "Dilate iterations", 0, 5, 1)
        self._reset_bg_btn = QPushButton("Reset background model (MOG2)")
        self._reset_bg_btn.clicked.connect(self._detector.reset_background)
        cl.addWidget(self._reset_bg_btn)
        g.addWidget(self._panel_classical)

        self._show_edges = QPushButton("Blend structure/mask into main view")
        self._show_edges.setCheckable(True)
        self._show_edges.toggled.connect(lambda _: self._refresh_frame())
        g.addWidget(self._show_edges)

        g.addStretch()

        self._sync_yolo_model_combo(self._algo.currentData())
        self._update_panels_visibility()
        return w

    def _double_row_layout(
        self, layout: QVBoxLayout, title: str, lo: float, hi: float, default: float, decimals: int
    ) -> QDoubleSpinBox:
        row = QHBoxLayout()
        row.addWidget(QLabel(title))
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setSingleStep(0.01 if decimals <= 2 else 0.001)
        sp.setValue(default)
        sp.valueChanged.connect(lambda _: self._refresh_frame())
        row.addWidget(sp)
        layout.addLayout(row)
        return sp

    def _double_row_in_layout(
        self, layout: QVBoxLayout, title: str, lo: float, hi: float, default: float, decimals: int
    ) -> QDoubleSpinBox:
        row = QHBoxLayout()
        row.addWidget(QLabel(title))
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setSingleStep(0.0002)
        sp.setValue(default)
        sp.valueChanged.connect(lambda _: self._refresh_frame())
        row.addWidget(sp)
        layout.addLayout(row)
        return sp

    def _on_yolo_model_changed(self) -> None:
        self._last_yolo_model_pick = self._yolo_model.currentText()
        self._refresh_frame()

    def _sync_yolo_model_combo(self, kind: AlgorithmKind) -> None:
        if kind not in (AlgorithmKind.YOLO_V8, AlgorithmKind.YOLO_V8_SEG):
            return
        self._yolo_model.blockSignals(True)
        self._yolo_model.clear()
        prev = self._last_yolo_model_pick or ""
        if kind == AlgorithmKind.YOLO_V8_SEG:
            models = YOLO_SEG_MODELS
            if prev in YOLO_SEG_MODELS:
                pick = prev
            elif prev in YOLO_DETECT_MODELS:
                i = YOLO_DETECT_MODELS.index(prev)
                pick = YOLO_SEG_MODELS[min(i, len(YOLO_SEG_MODELS) - 1)]
            else:
                pick = YOLO_SEG_MODELS[0]
        else:
            models = YOLO_DETECT_MODELS
            if prev in YOLO_DETECT_MODELS:
                pick = prev
            elif prev in YOLO_SEG_MODELS:
                i = YOLO_SEG_MODELS.index(prev)
                pick = YOLO_DETECT_MODELS[min(i, len(YOLO_DETECT_MODELS) - 1)]
            else:
                pick = YOLO_DETECT_MODELS[0]
        for m in models:
            self._yolo_model.addItem(m)
        idx = self._yolo_model.findText(pick)
        self._yolo_model.setCurrentIndex(max(0, idx))
        self._yolo_model.blockSignals(False)
        self._last_yolo_model_pick = self._yolo_model.currentText()

    def _update_panels_visibility(self) -> None:
        k = self._algo.currentData()
        yolo = k in (AlgorithmKind.YOLO_V8, AlgorithmKind.YOLO_V8_SEG)
        hog = k == AlgorithmKind.HOG_PEOPLE
        self._panel_yolo.setVisible(yolo)
        self._panel_hog.setVisible(hog)
        self._panel_classical.setVisible(not yolo and not hog)

    def _on_yolo_unload(self) -> None:
        self._detector.clear_yolo_cache()
        self._refresh_frame()

    def _resolved_tracker_yaml(self) -> str:
        key = self._tracker_backend.currentData()
        if key == "byte":
            return TRACKER_YAML_BYTE
        return TRACKER_YAML_REID

    def _on_tracker_backend_changed(self) -> None:
        self._appearance_bridge.reset()
        self._detector.clear_yolo_cache()
        self._clear_track_timelines()
        if self._webcam and self._cap is not None:
            self._read_and_show()
        else:
            self._refresh_frame()

    def _on_appearance_merge_toggled(self) -> None:
        self._appearance_bridge.reset()
        self._clear_track_timelines()
        if self._webcam and self._cap is not None:
            self._read_and_show()
        else:
            self._refresh_frame()

    def _on_yolo_track_toggled(self) -> None:
        self._detector.reset_yolo_tracker_state()
        self._appearance_bridge.reset()
        self._clear_track_timelines()
        if self._webcam and self._cap is not None:
            self._read_and_show()
        else:
            self._refresh_frame()

    def _spin_slider_row(
        self, layout: QVBoxLayout, title: str, lo: int, hi: int, default: int
    ) -> tuple[QSlider, QSpinBox]:
        row = QHBoxLayout()
        row.addWidget(QLabel(title))
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(default)
        sp = QSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(default)
        s.valueChanged.connect(sp.setValue)
        sp.valueChanged.connect(s.setValue)
        s.valueChanged.connect(lambda _: self._refresh_frame())
        row.addWidget(s)
        row.addWidget(sp)
        layout.addLayout(row)
        return s, sp

    def _on_algo_change(self) -> None:
        kind = self._algo.currentData()
        if kind == AlgorithmKind.BACKGROUND_SUBTRACT:
            self._detector.reset_background()
        if kind in (AlgorithmKind.YOLO_V8, AlgorithmKind.YOLO_V8_SEG):
            self._sync_yolo_model_combo(kind)
        self._update_panels_visibility()
        self._refresh_frame()

    def _toggle_play(self) -> None:
        if self._cap is None:
            return
        self._paused = not self._paused
        self._play_btn.setText("Pause" if not self._paused else "Play")
        if not self._paused:
            self._timer.start(int(1000 / max(1.0, self._fps)))
        else:
            self._timer.stop()

    def _on_tick(self) -> None:
        if self._cap is None or self._paused:
            return
        if self._webcam:
            self._read_and_show()
            return
        nxt = self._current_index + 1
        if nxt >= self._frame_count:
            self._paused = True
            self._play_btn.setText("Play")
            self._timer.stop()
            return
        self._slider.blockSignals(True)
        self._slider.setValue(nxt)
        self._slider.blockSignals(False)
        self._current_index = nxt
        self._read_and_show()

    def _on_slider(self, v: int) -> None:
        if self._cap is None or self._webcam:
            return
        self._current_index = v
        self._seek_and_show(v)

    def _seek_and_show(self, idx: int) -> None:
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        self._read_and_show()

    def _read_and_show(self) -> None:
        if self._cap is None:
            return
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return
        if self._webcam:
            self._live_idx += 1
        else:
            self._current_index = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if self._current_index < 0:
                self._current_index = 0
        self._process_frame(frame)

    def _params(self):
        kind: AlgorithmKind = self._algo.currentData()
        return dict(
            kind=kind,
            min_area_ratio=float(self._min_area.value()),
            blur_ksize=int(self._blur[1].value()),
            canny_low=int(self._canny_lo[1].value()),
            canny_high=int(self._canny_hi[1].value()),
            morph_kernel=int(self._morph[1].value()),
            close_iters=int(self._close_i[1].value()),
            dilate_iters=int(self._dilate_i[1].value()),
            yolo_model=self._yolo_model.currentText(),
            yolo_conf=float(self._yolo_conf.value()),
            yolo_iou=float(self._yolo_iou.value()),
            yolo_imgsz=int(self._yolo_imgsz.currentData()),
            yolo_device=str(self._yolo_device.currentData() or ""),
            hog_hit_threshold=float(self._hog_hit.value()),
            hog_scale=float(self._hog_scale.value()),
            yolo_track=self._yolo_track.isChecked(),
            yolo_tracker_yaml=self._resolved_tracker_yaml(),
        )

    def _process_frame(self, frame_bgr: np.ndarray) -> None:
        p = self._params()
        base = self._detector.detect(frame_bgr, p.pop("kind"), **p)
        self._pick_boxes = list(base.boxes)
        self._pick_labels = list(base.labels)
        en = self._checked_class_filter()
        res = filter_detection_by_classes(base, frame_bgr, en)
        self._last_result = res

        hh, ww = frame_bgr.shape[0], frame_bgr.shape[1]
        self._video_label.set_frame_shape(hh, ww)

        vis = draw_overlay(frame_bgr, res)
        if self._show_edges.isChecked():
            edge_bgr = cv2.cvtColor(res.edge_map, cv2.COLOR_GRAY2BGR)
            vis = cv2.addWeighted(vis, 0.55, edge_bgr, 0.45, 0)

        self._pix_vis = vis
        self._pix_edge = res.edge_map
        self._video_has_frame = True
        self._apply_media_scaling()

        fi = self._live_idx if self._webcam else self._current_index
        t_sec = fi / max(1e-6, self._fps)
        self._series.upsert(fi, t_sec, res.count, res.algorithm_name)
        self._update_snippets(res.snippets, res.labels)
        self._record_track_crops(res, fi)
        self._refresh_track_list()
        self._update_plots_and_table()
        self._update_time_label()

    def _checked_class_filter(self) -> Optional[Set[str]]:
        names: List[str] = []
        for i in range(self._class_list.count()):
            it = self._class_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                names.append(it.text())
        if not names:
            return None
        return {n.lower() for n in names}

    def _on_class_filter_changed(self) -> None:
        if self._cap is None:
            return
        if self._webcam:
            return
        self._refresh_frame()

    def _filter_class_list(self, text: str) -> None:
        t = text.lower().strip()
        for i in range(self._class_list.count()):
            it = self._class_list.item(i)
            it.setHidden(bool(t) and t not in it.text().lower())

    def _class_check_all(self) -> None:
        self._class_list.blockSignals(True)
        for i in range(self._class_list.count()):
            self._class_list.item(i).setCheckState(Qt.CheckState.Checked)
        self._class_list.blockSignals(False)
        self._on_class_filter_changed()

    def _class_uncheck_all(self) -> None:
        self._class_list.blockSignals(True)
        for i in range(self._class_list.count()):
            self._class_list.item(i).setCheckState(Qt.CheckState.Unchecked)
        self._class_list.blockSignals(False)
        self._on_class_filter_changed()

    def _class_check_visible_frame(self) -> None:
        if not self._pick_labels:
            return
        seen = {lb.lower() for lb in self._pick_labels}
        self._class_list.blockSignals(True)
        for i in range(self._class_list.count()):
            it = self._class_list.item(i)
            if it.text().lower() in seen:
                it.setCheckState(Qt.CheckState.Checked)
        self._class_list.blockSignals(False)
        self._on_class_filter_changed()

    def _on_video_frame_click(self, fx: int, fy: int) -> None:
        if not self._pick_boxes or not self._pick_labels:
            return
        best_lb: Optional[str] = None
        best_area = 10**18
        for i, (x, y, bw, bh) in enumerate(self._pick_boxes):
            if x <= fx < x + bw and y <= fy < y + bh:
                a = bw * bh
                if a < best_area:
                    best_area = a
                    best_lb = self._pick_labels[i] if i < len(self._pick_labels) else None
        if not best_lb:
            return
        target = best_lb.lower()
        self._class_list.blockSignals(True)
        found = False
        for i in range(self._class_list.count()):
            it = self._class_list.item(i)
            if it.text().lower() == target:
                it.setCheckState(Qt.CheckState.Checked)
                found = True
                break
        if not found:
            nw = QListWidgetItem(best_lb)
            nw.setFlags(nw.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            nw.setCheckState(Qt.CheckState.Checked)
            self._class_list.addItem(nw)
        self._class_list.blockSignals(False)
        self._on_class_filter_changed()

    def _record_track_crops(self, fr: DetectionResult, fi: int) -> None:
        k = self._algo.currentData()
        if k not in (AlgorithmKind.YOLO_V8, AlgorithmKind.YOLO_V8_SEG):
            return
        if not self._yolo_track.isChecked():
            return
        if not fr.track_ids or len(fr.track_ids) != len(fr.boxes):
            return
        use_bridge = self._appearance_merge.isChecked()
        for i, tid in enumerate(fr.track_ids):
            if tid < 0 or i >= len(fr.snippets):
                continue
            crop = fr.snippets[i].copy()
            lab = fr.labels[i] if i < len(fr.labels) else "?"
            key_id = (
                self._appearance_bridge.stable_id(tid, fi, crop, lab)
                if use_bridge
                else tid
            )
            if key_id not in self._track_buffers:
                self._track_buffers[key_id] = []
                self._track_label[key_id] = lab
            buf = self._track_buffers[key_id]
            if buf and buf[-1][0] == fi:
                buf[-1] = (fi, crop)
            else:
                buf.append((fi, crop))
            if len(buf) > self._max_track_frames:
                buf.pop(0)

    def _refresh_track_list(self) -> None:
        sel_tid: Optional[int] = None
        row = self._track_list.currentRow()
        if row >= 0:
            it = self._track_list.item(row)
            if it is not None:
                v = it.data(Qt.ItemDataRole.UserRole)
                if v is not None:
                    sel_tid = int(v)
        self._track_list.blockSignals(True)
        self._track_list.clear()
        for tid in sorted(self._track_buffers.keys()):
            n = len(self._track_buffers[tid])
            lab = self._track_label.get(tid, "?")
            item = QListWidgetItem(f"Track {tid} — {lab} ({n} crops)")
            item.setData(Qt.ItemDataRole.UserRole, tid)
            if self._appearance_merge.isChecked():
                item.setToolTip(
                    "Timeline key may combine multiple detector IDs when appearance matches within ~120 frames."
                )
            self._track_list.addItem(item)
        if sel_tid is not None:
            for r in range(self._track_list.count()):
                it2 = self._track_list.item(r)
                if it2 and int(it2.data(Qt.ItemDataRole.UserRole)) == sel_tid:
                    self._track_list.setCurrentRow(r)
                    break
        self._track_list.blockSignals(False)
        self._on_track_row_changed(self._track_list.currentRow())

    def _current_track_id(self) -> Optional[int]:
        row = self._track_list.currentRow()
        if row < 0:
            return None
        it = self._track_list.item(row)
        if it is None:
            return None
        v = it.data(Qt.ItemDataRole.UserRole)
        return int(v) if v is not None else None

    def _on_track_row_changed(self, row: int) -> None:
        tid = self._current_track_id()
        if tid is None:
            self._track_slider.setMaximum(0)
            self._track_preview.clear()
            self._track_frame_lbl.setText("—")
            return
        buf = self._track_buffers.get(tid, [])
        self._track_slider.blockSignals(True)
        self._track_slider.setMaximum(max(0, len(buf) - 1))
        self._track_slider.setValue(0)
        self._track_slider.blockSignals(False)
        self._on_track_slider_changed(0)

    def _on_track_slider_changed(self, v: int) -> None:
        tid = self._current_track_id()
        if tid is None:
            return
        buf = self._track_buffers.get(tid, [])
        if not buf or v < 0 or v >= len(buf):
            return
        fi, crop = buf[v]
        self._track_preview.setPixmap(bgr_to_qpixmap_fit(crop, 360, 360))
        self._track_frame_lbl.setText(f"Source frame #{fi}  •  crop {v + 1}/{len(buf)}")

    def _on_track_play_toggled(self, on: bool) -> None:
        if on:
            self._track_play_timer.start(50)
        else:
            self._track_play_timer.stop()

    def _on_track_play_tick(self) -> None:
        if not self._track_play_btn.isChecked():
            return
        m = self._track_slider.maximum()
        if m <= 0:
            return
        nxt = self._track_slider.value() + 1
        if nxt > m:
            nxt = 0
        self._track_slider.setValue(nxt)

    def _save_track_mp4(self) -> None:
        tid = self._current_track_id()
        if tid is None:
            QMessageBox.information(self, "Export", "Select a track in the list first.")
            return
        buf = self._track_buffers.get(tid, [])
        if not buf:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save track video",
            os.path.expanduser("~/track_%d.mp4" % tid),
            "MP4 video (*.mp4);;All (*.*)",
        )
        if not path:
            return
        h0, w0 = buf[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, max(1.0, self._fps), (w0, h0))
        if not out.isOpened():
            QMessageBox.warning(self, "Export", "Could not open video writer.")
            return
        for _, crop in buf:
            if crop.shape[0] != h0 or crop.shape[1] != w0:
                crop = cv2.resize(crop, (w0, h0))
            out.write(crop)
        out.release()
        QMessageBox.information(self, "Export", "Saved:\n%s" % path)

    def _clear_track_timelines(self) -> None:
        self._appearance_bridge.reset()
        self._track_buffers.clear()
        self._track_label.clear()
        self._track_play_timer.stop()
        self._track_play_btn.setChecked(False)
        self._track_list.clear()
        self._track_slider.setMaximum(0)
        self._track_preview.clear()
        self._track_frame_lbl.setText("—")

    def _refresh_frame(self) -> None:
        if self._cap is None:
            return
        if self._webcam:
            return
        self._seek_and_show(self._current_index)

    def _update_snippets(self, snippets: List[np.ndarray], labels: Optional[List[str]] = None) -> None:
        self._last_snippets = list(snippets)
        self._last_snippet_labels = list(labels) if labels else []
        while self._snip_grid.count():
            item = self._snip_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        cols, thumb = self._snippet_layout_dims()
        for c in range(cols):
            self._snip_grid.setColumnStretch(c, 1)
        for i, snip in enumerate(snippets[:24]):
            lab = QLabel()
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lab.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            lab.setPixmap(bgr_to_qpixmap_fit(snip, thumb, thumb))
            tip = f"Object {i + 1}"
            if labels and i < len(labels) and labels[i]:
                tip = f"{tip}: {labels[i]}"
            lab.setToolTip(tip)
            lab.setScaledContents(False)
            r, c = divmod(i, cols)
            self._snip_grid.addWidget(lab, r, c)

    def _update_plots_and_table(self) -> None:
        fi, _, c = self._series.arrays()
        if fi.size == 0:
            return
        self._curve.setData(fi, c)
        mean, _ = self._series.rolling_mean_std()
        if mean is not None:
            self._mean_curve.setData(fi, mean)
        summ = self._series.summary()
        self._stats_label.setText(
            f"Frames logged: {summ['n']}\n"
            f"Mean count: {summ['mean']:.2f}  σ: {summ['std']:.2f}\n"
            f"Min / Max: {summ['min']} / {summ['max']}\n"
            f"Linear trend (Δ per 100 frames): {summ['trend_per_100_frames']:+.3f}"
        )
        if c.size >= 2:
            nb = max(3, min(24, int(np.sqrt(c.size)) + 3))
            hist, edges = np.histogram(c, bins=nb)
            xc = (edges[:-1] + edges[1:]) / 2
            wbar = float(edges[1] - edges[0]) * 0.85 if len(edges) > 1 else 0.8
            self._hist.setOpts(x=xc, height=hist, width=wbar)

        self._table.setRowCount(min(200, int(c.size)))
        start = max(0, int(c.size) - 200)
        for row in range(self._table.rowCount()):
            idx = start + row
            rec = self._series.records[idx]
            prev = self._series.records[idx - 1].count if idx > 0 else rec.count
            delta = rec.count - prev
            self._table.setItem(row, 0, QTableWidgetItem(str(rec.frame_index)))
            self._table.setItem(row, 1, QTableWidgetItem(f"{rec.time_sec:.3f}"))
            self._table.setItem(row, 2, QTableWidgetItem(str(rec.count)))
            self._table.setItem(row, 3, QTableWidgetItem(f"{delta:+d}" if idx > 0 else "—"))

    def _update_time_label(self) -> None:
        if self._cap is None:
            self._time_label.setText("—")
            return
        fi = self._live_idx if self._webcam else self._current_index
        t = fi / max(1e-6, self._fps)
        if self._webcam:
            self._time_label.setText(
                f"live  •  frame #{fi}  •  t≈{t:.1f}s  •  "
                f"count: {self._last_result.count if self._last_result else 0}"
            )
        else:
            tot = self._frame_count / max(1e-6, self._fps)
            self._time_label.setText(
                f"frame {self._current_index + 1}/{self._frame_count}  •  "
                f"t={t:.2f}s / {tot:.2f}s  •  "
                f"count: {self._last_result.count if self._last_result else 0}"
            )

    def _reset_stats(self) -> None:
        self._series.clear()
        self._curve.setData([], [])
        self._mean_curve.setData([], [])
        self._hist.setOpts(x=[], height=[], width=0.8)
        self._table.setRowCount(0)
        self._stats_label.setText("No data yet.")
        self._refresh_frame()

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open video",
            os.path.expanduser("~"),
            "Video (*.mp4 *.avi *.mov *.mkv *.webm);;All (*.*)",
        )
        if not path:
            return
        self._open_capture(path, webcam=False)

    def _open_webcam(self) -> None:
        self._open_capture(0, webcam=True)

    def _open_capture(self, path, webcam: bool) -> None:
        self._timer.stop()
        self._paused = True
        self._play_btn.setText("Play")
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(0 if webcam else path)
        if not self._cap.isOpened():
            QMessageBox.warning(self, "Open failed", "Could not open video source.")
            self._cap = None
            return
        self._webcam = webcam
        self._source_path = None if webcam else str(path)
        self._live_idx = 0
        self._detector.reset_background()
        self._detector.reset_yolo_tracker_state()
        self._series.clear()
        self._clear_track_timelines()

        if webcam:
            self._frame_count = 10**9
            self._fps = 30.0
            self._slider.setMaximum(0)
            self._slider.setEnabled(False)
            self._paused = False
            self._play_btn.setText("Pause")
            self._timer.start(33)
        else:
            self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            self._fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
            self._slider.setEnabled(True)
            self._slider.setMaximum(max(0, self._frame_count - 1))
            self._slider.blockSignals(True)
            self._slider.setValue(0)
            self._slider.blockSignals(False)
            self._current_index = 0
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._read_and_show()

    def closeEvent(self, event) -> None:
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
        super().closeEvent(event)
