"""사이드바 하단 시스템 상태 (CPU / RAM / GPU / VRAM) 실시간 표시.

측정은 별도 스레드에서 1초마다 수행해 화면이 멈추지 않게 한다.
VRAM 은 장치 전체 사용량이라 학습 프로세스가 쓰는 메모리도 포함된다.
"""

from __future__ import annotations

from PySide6.QtCore import QMetaObject, QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QSizePolicy, QWidget

from ...core.sysinfo import Usage, UsageSampler
from ..theme import COLORS


class _Sampler(QObject):
    sampled = Signal(object)

    def __init__(self):
        super().__init__()
        self._sampler = None
        self._timer = None

    @Slot()
    def start(self):
        self._sampler = UsageSampler()
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._tick()

    @Slot()
    def _tick(self):
        try:
            self.sampled.emit(self._sampler.sample())
        except Exception:
            pass

    @Slot()
    def stop(self):
        if self._timer:
            self._timer.stop()
        if self._sampler:
            self._sampler.close()


class MiniBar(QWidget):
    """얇은 사용률 막대 (임계값에 따라 색 변경)."""

    def __init__(self, warn: float = 0.85, danger: float = 0.95):
        super().__init__()
        self.ratio = 0.0
        self.warn, self.danger = warn, danger
        self.enabled_ = True
        self.setFixedHeight(5)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_ratio(self, r: float, enabled: bool = True):
        self.ratio = max(0.0, min(1.0, r))
        self.enabled_ = enabled
        self.update()

    def color(self) -> str:
        if self.ratio >= self.danger:
            return COLORS["danger"]
        if self.ratio >= self.warn:
            return COLORS["warning"]
        return COLORS["accent"]

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(COLORS["border"]))
        r = self.rect()
        p.drawRoundedRect(r, 2.5, 2.5)
        if self.enabled_ and self.ratio > 0:
            w = max(3, int(r.width() * self.ratio))
            p.setBrush(QColor(self.color()))
            p.drawRoundedRect(r.x(), r.y(), w, r.height(), 2.5, 2.5)


class SysMonitor(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SysMon")
        self.setStyleSheet(
            f"QFrame#SysMon {{ background: {COLORS['bg']}; border: 1px solid {COLORS['border']}; border-radius: 10px; }}"
            f"QLabel {{ font-size: 11px; background: transparent; }}"
            f"QLabel#SysKey {{ color: {COLORS['muted']}; font-weight: 600; }}"
            f"QLabel#SysVal {{ color: {COLORS['text']}; }}"
        )
        grid = QGridLayout(self)
        grid.setContentsMargins(10, 8, 10, 9)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        self.rows = {}
        for i, (key, label) in enumerate((("cpu", "CPU"), ("ram", "RAM"), ("gpu", "GPU"), ("vram", "VRAM"))):
            k = QLabel(label)
            k.setObjectName("SysKey")
            v = QLabel("-")
            v.setObjectName("SysVal")
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            bar = MiniBar(0.85, 0.95) if key in ("ram", "vram") else MiniBar(0.9, 0.99)
            grid.addWidget(k, i * 2, 0)
            grid.addWidget(v, i * 2, 1)
            grid.addWidget(bar, i * 2 + 1, 0, 1, 2)
            if i < 3:
                grid.setRowMinimumHeight(i * 2 + 1, 9)
            self.rows[key] = (v, bar)
        self.setToolTip("실시간 시스템 사용량 (1초 간격)\nVRAM 90% 이상 주황, 97% 이상 빨강")

        self._thread = QThread(self)
        self._worker = _Sampler()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.start)
        self._worker.sampled.connect(self._update)
        self._thread.start()

    def _set(self, key: str, text: str, ratio: float, enabled: bool = True):
        v, bar = self.rows[key]
        v.setText(text)
        bar.set_ratio(ratio, enabled)

    @Slot(object)
    def _update(self, u: Usage):
        self._set("cpu", f"{u.cpu_pct:.0f}%", u.cpu_pct / 100)
        if u.ram_total_gb:
            self._set("ram", f"{u.ram_used_gb:.1f} / {u.ram_total_gb:.0f} GB", u.ram_used_gb / u.ram_total_gb)
        if u.gpu_ok:
            if u.gpu_pct is None:
                self._set("gpu", "측정 불가", 0, False)
                self.rows["gpu"][0].setToolTip("GPU 사용률은 nvidia-ml-py 설치 시 표시됩니다")
            else:
                temp = f" · {u.gpu_temp:.0f}°C" if u.gpu_temp is not None else ""
                self._set("gpu", f"{u.gpu_pct:.0f}%{temp}", u.gpu_pct / 100)
            r = u.vram_used_gb / u.vram_total_gb if u.vram_total_gb else 0
            vbar = self.rows["vram"][1]
            vbar.warn, vbar.danger = 0.90, 0.97
            self._set("vram", f"{u.vram_used_gb:.1f} / {u.vram_total_gb:.0f} GB", r)
        else:
            self._set("gpu", "없음", 0, False)
            self._set("vram", "-", 0, False)

    def shutdown(self):
        if self._thread.isRunning():
            QMetaObject.invokeMethod(self._worker, "stop", Qt.BlockingQueuedConnection)
            self._thread.quit()
            self._thread.wait(2000)
