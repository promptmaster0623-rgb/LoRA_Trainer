"""공용 위젯: 카드, 경로 선택기, 폼 행, 통계 카드, 페이지 틀."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class Card(QFrame):
    def __init__(self, title: str = "", subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(18, 16, 18, 18)
        self.body.setSpacing(12)
        if title:
            head = QHBoxLayout()
            t = QLabel(title)
            t.setObjectName("CardTitle")
            head.addWidget(t)
            head.addStretch(1)
            self.head = head
            self.body.addLayout(head)
            if subtitle:
                s = QLabel(subtitle)
                s.setObjectName("Hint")
                s.setWordWrap(True)
                self.body.addWidget(s)

    def add(self, w):
        if isinstance(w, QWidget):
            self.body.addWidget(w)
        else:
            self.body.addLayout(w)
        return w


class FormGrid(QGridLayout):
    """라벨 | 위젯 | 도움말 3열 폼."""

    def __init__(self):
        super().__init__()
        self.setHorizontalSpacing(14)
        self.setVerticalSpacing(10)
        self.setColumnStretch(1, 1)
        self._row = 0

    def row(self, label: str, widget, hint: str = "") -> None:
        lab = QLabel(label)
        lab.setMinimumWidth(130)
        self.addWidget(lab, self._row, 0, Qt.AlignVCenter)
        if isinstance(widget, QWidget):
            self.addWidget(widget, self._row, 1)
        else:
            self.addLayout(widget, self._row, 1)
        if hint:
            h = QLabel(hint)
            h.setObjectName("Hint")
            h.setWordWrap(True)
            h.setMaximumWidth(330)
            self.addWidget(h, self._row, 2)
        self._row += 1

    def span(self, widget) -> None:
        self.addWidget(widget, self._row, 0, 1, 3)
        self._row += 1


class PathEdit(QWidget):
    changed = Signal(str)

    def __init__(self, mode: str = "file", placeholder: str = "", filter: str = "", parent=None):
        super().__init__(parent)
        self.mode = mode
        self.filter = filter
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.textChanged.connect(self.changed.emit)
        btn = QPushButton("찾기")
        btn.setFixedWidth(64)
        btn.clicked.connect(self._browse)
        lay.addWidget(self.edit, 1)
        lay.addWidget(btn)

    def _browse(self):
        start = self.edit.text() or ""
        if self.mode == "dir":
            p = QFileDialog.getExistingDirectory(self, "폴더 선택", start)
        else:
            p, _ = QFileDialog.getOpenFileName(self, "파일 선택", start, self.filter or "모든 파일 (*.*)")
        if p:
            self.edit.setText(p)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, t: str) -> None:
        self.edit.setText(t or "")

    def setPlaceholderText(self, t: str) -> None:
        self.edit.setPlaceholderText(t)


def combo(items: Sequence[Tuple[str, str]]) -> QComboBox:
    """(값, 표시이름) 목록으로 콤보 생성. currentData() 로 값 획득."""
    c = QComboBox()
    for value, label in items:
        c.addItem(label, value)
    return c


def set_combo(c: QComboBox, value) -> None:
    i = c.findData(value)
    if i >= 0:
        c.setCurrentIndex(i)


def spin(lo: int, hi: int, step: int = 1, suffix: str = "") -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    if suffix:
        s.setSuffix(suffix)
    s.setAccelerated(True)
    return s


def dspin(lo: float, hi: float, step: float, decimals: int = 4) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    return s


class SciSpin(QLineEdit):
    """학습률처럼 과학적 표기(1e-4)가 편한 값."""

    def __init__(self, value: float = 1e-4):
        super().__init__()
        self.setValue(value)

    def value(self) -> float:
        try:
            return float(self.text())
        except ValueError:
            return 1e-4

    def setValue(self, v: float) -> None:
        self.setText(f"{v:g}")


class StatCard(QFrame):
    def __init__(self, label: str, value: str = "-", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(2)
        self.value = QLabel(value)
        self.value.setObjectName("StatValue")
        lab = QLabel(label)
        lab.setObjectName("StatLabel")
        lay.addWidget(lab)
        lay.addWidget(self.value)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set(self, v: str) -> None:
        self.value.setText(v)


class Page(QWidget):
    """스크롤 가능한 페이지 틀 (제목 + 설명 + 내용)."""

    def __init__(self, title: str, desc: str = "", scroll: bool = True, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        header = QWidget()
        header.setObjectName("PageHeader")
        hl = QVBoxLayout(header)
        hl.setContentsMargins(28, 22, 28, 8)
        hl.setSpacing(4)
        t = QLabel(title)
        t.setObjectName("PageTitle")
        hl.addWidget(t)
        if desc:
            d = QLabel(desc)
            d.setObjectName("PageDesc")
            d.setWordWrap(True)
            hl.addWidget(d)
        outer.addWidget(header)
        content = QWidget()
        content.setObjectName("PageContent")
        self.content = QVBoxLayout(content)
        self.content.setContentsMargins(28, 10, 28, 28)
        self.content.setSpacing(16)
        if scroll:
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setWidget(content)
            area.setFrameShape(QFrame.NoFrame)
            outer.addWidget(area, 1)
        else:
            outer.addWidget(content, 1)

    def add(self, w):
        if isinstance(w, QWidget):
            self.content.addWidget(w)
        else:
            self.content.addLayout(w)
        return w

    # 하위 클래스 구현
    def load(self, job) -> None:  # pragma: no cover
        pass

    def store(self, job) -> None:  # pragma: no cover
        pass


def hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet("color: #2a3040;")
    return f


def hint(text: str) -> QLabel:
    l = QLabel(text)
    l.setObjectName("Hint")
    l.setWordWrap(True)
    return l
