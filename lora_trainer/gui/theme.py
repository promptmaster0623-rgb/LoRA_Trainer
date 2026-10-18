"""다크 테마 (QSS) 및 색상 토큰."""

from __future__ import annotations

COLORS = {
    "bg": "#0f1115",
    "panel": "#151821",
    "card": "#1b1f2a",
    "card_hover": "#212635",
    "border": "#2a3040",
    "text": "#e7e9f0",
    "muted": "#8a93a8",
    "accent": "#7c5cff",
    "accent_hover": "#9277ff",
    "accent_soft": "#2a2350",
    "success": "#34d399",
    "warning": "#fbbf24",
    "danger": "#f87171",
    "chart_raw": "#4b5470",
    "chart_line": "#a48bff",
}

FONT_STACK = '"Pretendard", "Malgun Gothic", "Apple SD Gothic Neo", "Segoe UI", sans-serif'


def apply_palette(app) -> None:
    """Fusion 스타일 기본 팔레트를 다크로 (스크롤 영역 등 QSS 가 닿지 않는 곳)."""
    from PySide6.QtGui import QColor, QPalette

    c = COLORS
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(c["bg"]))
    pal.setColor(QPalette.WindowText, QColor(c["text"]))
    pal.setColor(QPalette.Base, QColor(c["panel"]))
    pal.setColor(QPalette.AlternateBase, QColor(c["card"]))
    pal.setColor(QPalette.Text, QColor(c["text"]))
    pal.setColor(QPalette.Button, QColor(c["card"]))
    pal.setColor(QPalette.ButtonText, QColor(c["text"]))
    pal.setColor(QPalette.Highlight, QColor(c["accent"]))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipBase, QColor(c["card"]))
    pal.setColor(QPalette.ToolTipText, QColor(c["text"]))
    pal.setColor(QPalette.PlaceholderText, QColor("#5d667c"))
    app.setPalette(pal)


def stylesheet() -> str:
    c = COLORS
    return f"""
* {{
    font-family: {FONT_STACK};
    font-size: 13px;
    color: {c['text']};
}}
QMainWindow, QWidget#Root {{ background: {c['bg']}; }}
QWidget#Sidebar {{ background: {c['panel']}; border-right: 1px solid {c['border']}; }}
QLabel#AppTitle {{ font-size: 17px; font-weight: 700; padding: 4px 0; }}
QLabel#AppSub {{ color: {c['muted']}; font-size: 11px; }}
QLabel#PageTitle {{ font-size: 20px; font-weight: 700; }}
QLabel#PageDesc {{ color: {c['muted']}; }}
QLabel#Muted {{ color: {c['muted']}; }}
QLabel#Hint {{ color: {c['muted']}; font-size: 11px; }}
QLabel#StatValue {{ font-size: 20px; font-weight: 700; }}
QLabel#StatLabel {{ color: {c['muted']}; font-size: 11px; }}
QLabel#Badge {{
    background: {c['accent_soft']}; color: {c['accent_hover']};
    border-radius: 9px; padding: 2px 9px; font-size: 11px; font-weight: 600;
}}
QFrame#Card {{
    background: {c['card']}; border: 1px solid {c['border']}; border-radius: 12px;
}}
QLabel#CardTitle {{ font-size: 14px; font-weight: 700; }}

QPushButton#NavButton {{
    text-align: left; padding: 10px 14px; border: none; border-radius: 8px;
    background: transparent; color: {c['muted']}; font-weight: 600;
}}
QPushButton#NavButton:hover {{ background: {c['card']}; color: {c['text']}; }}
QPushButton#NavButton:checked {{ background: {c['accent_soft']}; color: {c['text']}; }}

QPushButton {{
    background: {c['card']}; border: 1px solid {c['border']}; border-radius: 8px;
    padding: 7px 14px; font-weight: 600;
}}
QPushButton:hover {{ background: {c['card_hover']}; border-color: #3a4256; }}
QPushButton:pressed {{ background: {c['border']}; }}
QPushButton:disabled {{ color: #565d70; background: {c['panel']}; }}
QPushButton#Primary {{ background: {c['accent']}; border: 1px solid {c['accent']}; color: white; }}
QPushButton#Primary:hover {{ background: {c['accent_hover']}; }}
QPushButton#Primary:disabled {{ background: #3b3466; color: #a9a3c9; border-color: #3b3466; }}
QPushButton#Danger {{ background: transparent; border: 1px solid #5b2f36; color: {c['danger']}; }}
QPushButton#Danger:hover {{ background: #2b1a1e; }}
QPushButton#Ghost {{ background: transparent; border: none; color: {c['muted']}; }}
QPushButton#Ghost:hover {{ color: {c['text']}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 8px;
    padding: 6px 9px; selection-background-color: {c['accent']};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {c['accent']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {c['card']}; border: 1px solid {c['border']}; selection-background-color: {c['accent_soft']};
}}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 16px; border: none; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid #3a4256; background: {c['panel']}; }}
QCheckBox::indicator:checked {{ background: {c['accent']}; border-color: {c['accent']}; }}

QProgressBar {{
    background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 7px; height: 14px;
    text-align: center; font-size: 11px;
}}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 6px; }}

QListWidget, QTableWidget, QTreeWidget {{
    background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 10px; outline: 0;
}}
QListWidget::item {{ border-radius: 8px; padding: 4px; }}
QListWidget::item:selected {{ background: {c['accent_soft']}; }}
QHeaderView::section {{
    background: {c['card']}; color: {c['muted']}; border: none; border-bottom: 1px solid {c['border']};
    padding: 6px; font-weight: 600;
}}
QTableWidget {{ gridline-color: {c['border']}; }}
QTableWidget::item:selected {{ background: {c['accent_soft']}; }}

QScrollArea {{ border: none; background: {c['bg']}; }}
QWidget#PageContent, QWidget#PageHeader {{ background: {c['bg']}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #2f3546; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #3d4459; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #2f3546; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QSplitter::handle {{ background: {c['border']}; }}
QToolTip {{ background: {c['card']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 6px; }}
QStatusBar {{ background: {c['panel']}; color: {c['muted']}; border-top: 1px solid {c['border']}; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: transparent; color: {c['muted']}; padding: 8px 14px; border-bottom: 2px solid transparent; font-weight: 600;
}}
QTabBar::tab:selected {{ color: {c['text']}; border-bottom-color: {c['accent']}; }}
QMenu {{ background: {c['card']}; border: 1px solid {c['border']}; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 6px; }}
QMenu::item:selected {{ background: {c['accent_soft']}; }}
"""
