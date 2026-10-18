"""GUI 진입점."""

from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow
    from .theme import apply_palette, stylesheet

    app = QApplication(sys.argv)
    app.setApplicationName("LoRA Trainer")
    app.setStyle("Fusion")
    apply_palette(app)
    font = QFont()
    font.setFamilies(["Pretendard", "Malgun Gothic", "Apple SD Gothic Neo", "Segoe UI"])
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(stylesheet())
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
