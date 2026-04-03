#!/usr/bin/env python3
"""Entry point for Any Object Counter (PyQt6)."""

import sys

from PyQt6.QtWidgets import QApplication

from app.dark_theme import apply_elegant_dark_theme
from app.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Any Object Counter")
    apply_elegant_dark_theme(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
