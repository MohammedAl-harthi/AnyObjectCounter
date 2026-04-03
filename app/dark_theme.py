"""Elegant dark palette and Qt Fusion + QSS for the whole app."""

from __future__ import annotations

from PyQt6.QtWidgets import QApplication

# Refined charcoal / slate with cool accent — easy on the eyes, not pure black.
ELEGANT_DARK_STYLESHEET = """
/* ---- Base ---- */
QWidget {
    background-color: #1c1c24;
    color: #e2e4ec;
    font-size: 13px;
    selection-background-color: #3d4a6e;
    selection-color: #f0f2ff;
}

QMainWindow::separator {
    background: #2a2a34;
    width: 4px;
    height: 4px;
}

/* ---- Menu ---- */
QMenuBar {
    background-color: #16161c;
    color: #d8dae6;
    border-bottom: 1px solid #32323e;
    padding: 2px 4px;
}
QMenuBar::item:selected {
    background-color: #2f3a55;
    border-radius: 4px;
}
QMenu {
    background-color: #24242e;
    color: #e2e4ec;
    border: 1px solid #3a3a48;
    padding: 6px;
}
QMenu::item:selected {
    background-color: #3d4f7a;
    border-radius: 4px;
}

/* ---- Tabs ---- */
QTabWidget::pane {
    border: 1px solid #353542;
    border-radius: 8px;
    top: -1px;
    background-color: #22222a;
}
QTabBar::tab {
    background-color: #1a1a22;
    color: #9898a8;
    border: 1px solid #353542;
    border-bottom: none;
    padding: 10px 18px;
    margin-right: 2px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    min-width: 72px;
}
QTabBar::tab:selected {
    background-color: #262630;
    color: #c8cbdc;
    border-bottom: 2px solid #6b8cce;
    font-weight: 600;
}
QTabBar::tab:hover:!selected {
    background-color: #2a2a34;
    color: #b8bbd0;
}

/* ---- Buttons ---- */
QPushButton {
    background-color: #2f3342;
    color: #e8eaf4;
    border: 1px solid #45455a;
    border-radius: 6px;
    padding: 8px 16px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #3a4054;
    border-color: #5a6b8c;
}
QPushButton:pressed {
    background-color: #252830;
}
QPushButton:checked {
    background-color: #3d4f7a;
    border-color: #6b8cce;
}
QPushButton:disabled {
    color: #5c5c6a;
    background-color: #222228;
    border-color: #333340;
}

/* ---- Inputs ---- */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #16161c;
    color: #e8eaf4;
    border: 1px solid #3a3a48;
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 22px;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border-color: #5a6b8c;
}
QComboBox::drop-down {
    border: none;
    width: 28px;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
    background: #2a2a34;
}
QComboBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #9a9dad;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #24242e;
    color: #e2e4ec;
    border: 1px solid #45455a;
    selection-background-color: #3d4f7a;
    outline: none;
}

/* ---- Sliders ---- */
QSlider::groove:horizontal {
    height: 6px;
    background: #2a2a34;
    border-radius: 3px;
    border: 1px solid #353542;
}
QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    margin: -6px 0;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #7a9fd4, stop:1 #5a7ab8);
    border: 1px solid #8aacdf;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #92b4e8, stop:1 #6b8cce);
}
QSlider::sub-page:horizontal {
    background: #3d4f6e;
    border-radius: 3px;
}

/* ---- Group box ---- */
QGroupBox {
    font-weight: 600;
    color: #c8cbdc;
    border: 1px solid #353542;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 12px;
    background-color: #22222a;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: #9db0d4;
}

/* ---- List / table ---- */
QListWidget, QTableWidget {
    background-color: #16161c;
    alternate-background-color: #1c1c24;
    color: #e2e4ec;
    border: 1px solid #353542;
    border-radius: 6px;
    gridline-color: #2f2f3a;
    outline: none;
}
QListWidget::item {
    padding: 6px;
    border-radius: 4px;
}
QListWidget::item:selected {
    background-color: #3d4f7a;
    color: #f0f2ff;
}
QTableWidget::item:selected {
    background-color: #3d4f7a;
}
QHeaderView::section {
    background-color: #24242e;
    color: #b8bbd0;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #45455a;
    border-right: 1px solid #2a2a34;
    font-weight: 600;
}

/* ---- Scroll areas ---- */
QScrollArea {
    border: none;
    background-color: transparent;
}
QAbstractScrollArea {
    background-color: #1c1c24;
}

/* ---- Scrollbars ---- */
QScrollBar:vertical {
    background: #1a1a22;
    width: 12px;
    margin: 0;
    border-radius: 6px;
}
QScrollBar::handle:vertical {
    background: #45455a;
    min-height: 32px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar::handle:vertical:hover {
    background: #5a5a6e;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: #1a1a22;
    height: 12px;
    margin: 0;
    border-radius: 6px;
}
QScrollBar::handle:horizontal {
    background: #45455a;
    min-width: 32px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar::handle:horizontal:hover {
    background: #5a5a6e;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* ---- Check box ---- */
QCheckBox {
    color: #d8dae6;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #45455a;
    background-color: #16161c;
}
QCheckBox::indicator:checked {
    background-color: #4a6494;
    border-color: #6b8cce;
}
QCheckBox::indicator:hover {
    border-color: #6b8cce;
}

/* ---- Splitter ---- */
QSplitter::handle {
    background-color: #2a2a34;
}
QSplitter::handle:hover {
    background-color: #3d4f6e;
}

/* ---- Labels: media panels (object names) ---- */
QLabel#PanelVideo {
    background-color: #121218;
    color: #8a8a9a;
    border: 1px solid #2f2f3a;
    border-radius: 8px;
}
QLabel#PanelEdge {
    background-color: #0f0f14;
    color: #6a6a78;
    border: 1px solid #2a2a34;
    border-radius: 6px;
}
QLabel#TrackPreview {
    background-color: #121218;
    color: #6a6a78;
    border: 1px solid #353542;
    border-radius: 8px;
}

/* ---- Tooltips ---- */
QToolTip {
    background-color: #2a2f3d;
    color: #e8eaf4;
    border: 1px solid #45455a;
    border-radius: 6px;
    padding: 8px;
}

/* ---- Message box (best-effort) ---- */
QMessageBox {
    background-color: #1c1c24;
}
QMessageBox QLabel {
    color: #e2e4ec;
}
"""


def apply_elegant_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(ELEGANT_DARK_STYLESHEET)
