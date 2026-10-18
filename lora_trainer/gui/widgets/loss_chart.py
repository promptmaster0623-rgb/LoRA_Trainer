"""loss 그래프 (외부 라이브러리 없이 QPainter 로 직접 그림)."""

from __future__ import annotations

from typing import List, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from ..theme import COLORS


class LossChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.raw: List[Tuple[int, float]] = []
        self.avg: List[Tuple[int, float]] = []
        self.marks: List[int] = []     # 저장/샘플 지점
        self.val: List[Tuple[int, float]] = []   # 검증 loss
        self.events: List[int] = []    # 적응형 자동 조정 지점
        self.setMinimumHeight(220)
        self.setMouseTracking(True)
        self._hover_x = None

    def clear(self) -> None:
        self.raw, self.avg, self.marks = [], [], []
        self.val, self.events = [], []
        self.update()

    def add(self, step: int, loss: float, avg: float) -> None:
        self.raw.append((step, loss))
        self.avg.append((step, avg))
        if len(self.raw) > 20000:  # 너무 길면 솎아냄
            self.raw = self.raw[::2]
            self.avg = self.avg[::2]
        self.update()

    def add_val(self, step: int, loss: float) -> None:
        self.val.append((step, loss))
        self.update()

    def mark_event(self, step: int) -> None:
        self.events.append(step)
        self.update()

    def mark(self, step: int) -> None:
        self.marks.append(step)
        self.update()

    def mouseMoveEvent(self, e):
        self._hover_x = e.position().x()
        self.update()

    def leaveEvent(self, e):
        self._hover_x = None
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect()
        left, right, top, bottom = 52, 14, 14, 26
        plot = QRectF(left, top, r.width() - left - right, r.height() - top - bottom)
        muted = QColor(COLORS["muted"])
        grid = QColor(COLORS["border"])
        font = QFont()
        font.setPointSize(8)
        p.setFont(font)

        if len(self.raw) < 2:
            p.setPen(muted)
            p.drawText(plot, Qt.AlignCenter, "학습이 시작되면 loss 그래프가 표시됩니다")
            p.end()
            return

        xs = [s for s, _ in self.raw]
        ys = sorted(v for _, v in self.raw)
        lo = ys[int(len(ys) * 0.02)]
        hi = ys[min(len(ys) - 1, int(len(ys) * 0.98))]
        if hi - lo < 1e-6:
            hi = lo + 1e-3
        pad = (hi - lo) * 0.1
        lo, hi = max(0.0, lo - pad), hi + pad
        x0, x1 = xs[0], max(xs[-1], xs[0] + 1)
        if self.val:  # 검증 loss 도 보이도록 범위 확장
            vv = [v for s, v in self.val if s >= x0]
            if vv:
                lo, hi = max(0.0, min(lo, min(vv) * 0.95)), max(hi, max(vv) * 1.05)

        def mx(s):
            return plot.left() + (s - x0) / (x1 - x0) * plot.width()

        def my(v):
            v = min(max(v, lo), hi)
            return plot.bottom() - (v - lo) / (hi - lo) * plot.height()

        # 격자 + 축 라벨
        p.setPen(QPen(grid, 1))
        for i in range(5):
            y = plot.top() + plot.height() * i / 4
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            val = hi - (hi - lo) * i / 4
            p.setPen(muted)
            p.drawText(QRectF(0, y - 8, left - 8, 16), Qt.AlignRight | Qt.AlignVCenter, f"{val:.4f}")
            p.setPen(QPen(grid, 1))
        for i in range(5):
            s = x0 + (x1 - x0) * i / 4
            x = mx(s)
            p.setPen(muted)
            p.drawText(QRectF(x - 30, plot.bottom() + 6, 60, 16), Qt.AlignCenter, f"{int(s)}")

        # 저장/샘플 마커
        p.setPen(QPen(QColor(COLORS["accent_soft"]), 1, Qt.DashLine))
        for m in self.marks:
            if x0 <= m <= x1:
                p.drawLine(QPointF(mx(m), plot.top()), QPointF(mx(m), plot.bottom()))

        # 적응형 자동 조정 지점 (주황 점선)
        p.setPen(QPen(QColor("#f59e0b"), 1.5, Qt.DashLine))
        for m in self.events:
            if x0 <= m <= x1:
                p.drawLine(QPointF(mx(m), plot.top()), QPointF(mx(m), plot.bottom()))
        # 검증 loss (초록 선 + 점)
        vpts = [(s, v) for s, v in self.val if x0 <= s <= x1]
        if vpts:
            vp = QPainterPath()
            vp.moveTo(mx(vpts[0][0]), my(vpts[0][1]))
            for s, v in vpts[1:]:
                vp.lineTo(mx(s), my(v))
            p.setPen(QPen(QColor("#34d399"), 2))
            p.drawPath(vp)
            p.setBrush(QColor("#34d399"))
            for s, v in vpts:
                p.drawEllipse(QPointF(mx(s), my(v)), 2.5, 2.5)
            p.setBrush(Qt.NoBrush)

        # 원본 loss (흐리게)
        def path(points):
            pp = QPainterPath()
            step = max(1, len(points) // int(max(plot.width(), 1)))
            pts = points[::step]
            pp.moveTo(mx(pts[0][0]), my(pts[0][1]))
            for s, v in pts[1:]:
                pp.lineTo(mx(s), my(v))
            return pp

        p.setPen(QPen(QColor(COLORS["chart_raw"]), 1))
        p.drawPath(path(self.raw))
        p.setPen(QPen(QColor(COLORS["chart_line"]), 2))
        p.drawPath(path(self.avg))

        # 호버 정보
        if self._hover_x is not None and plot.left() <= self._hover_x <= plot.right():
            s = x0 + (self._hover_x - plot.left()) / plot.width() * (x1 - x0)
            i = min(range(len(self.avg)), key=lambda k: abs(self.avg[k][0] - s))
            step, av = self.avg[i]
            raw = self.raw[i][1]
            x = mx(step)
            p.setPen(QPen(muted, 1, Qt.DotLine))
            p.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            p.setBrush(QColor(COLORS["chart_line"]))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(x, my(av)), 4, 4)
            text = f"스텝 {step}   평균 {av:.4f}   현재 {raw:.4f}"
            box = QRectF(min(x + 10, plot.right() - 220), plot.top() + 4, 210, 22)
            p.setBrush(QColor(COLORS["card"]))
            p.setPen(QPen(grid, 1))
            p.drawRoundedRect(box, 6, 6)
            p.setPen(QColor(COLORS["text"]))
            p.drawText(box, Qt.AlignCenter, text)

        # 범례
        p.setPen(QColor(COLORS["chart_line"]))
        p.drawText(QRectF(plot.right() - 170, plot.top() - 12, 80, 12), Qt.AlignRight, "━ 평균(EMA)")
        p.setPen(QColor(COLORS["chart_raw"]))
        p.drawText(QRectF(plot.right() - 80, plot.top() - 12, 80, 12), Qt.AlignRight, "━ 스텝 loss")
        p.end()
