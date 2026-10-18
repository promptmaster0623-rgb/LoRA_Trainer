"""실행 · 모니터 페이지."""

from __future__ import annotations

import os
import time

from PySide6.QtCore import QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..widgets.common import Card, Page, StatCard, hint
from ..widgets.loss_chart import LossChart

PHASES = ["준비", "데이터셋", "텍스트 캐시", "Latent 캐시", "모델", "학습", "완료"]


def fmt_time(sec: float) -> str:
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class RunPage(Page):
    start_clicked = Signal()
    stop_clicked = Signal()
    pause_toggled = Signal(bool)
    sample_clicked = Signal()
    save_clicked = Signal()
    lr_scale_requested = Signal(float)

    def __init__(self):
        super().__init__("실행 · 모니터", "학습을 시작하고 진행 상황을 확인합니다. 학습은 별도 프로세스에서 실행되어 창이 멈추지 않습니다.", scroll=True)
        ctrl = QHBoxLayout()
        self.start_btn = QPushButton("▶  학습 시작")
        self.start_btn.setObjectName("Primary")
        self.start_btn.setMinimumHeight(38)
        self.pause_btn = QPushButton("일시정지")
        self.pause_btn.setCheckable(True)
        self.sample_btn = QPushButton("지금 샘플")
        self.save_btn = QPushButton("지금 저장")
        self.stop_btn = QPushButton("중지")
        self.stop_btn.setObjectName("Danger")
        self.open_btn = QPushButton("출력 폴더 열기")
        self.open_btn.setObjectName("Ghost")
        self.start_btn.clicked.connect(self.start_clicked.emit)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        self.pause_btn.toggled.connect(self._pause)
        self.sample_btn.clicked.connect(self.sample_clicked.emit)
        self.save_btn.clicked.connect(self.save_clicked.emit)
        self.open_btn.clicked.connect(self._open_output)
        for b in (self.start_btn, self.pause_btn, self.sample_btn, self.save_btn, self.stop_btn):
            ctrl.addWidget(b)
        ctrl.addSpacing(14)
        ctrl.addWidget(QLabel("학습률 배율"))
        self.lr_scale = QDoubleSpinBox()
        self.lr_scale.setRange(0.05, 10.0)
        self.lr_scale.setSingleStep(0.1)
        self.lr_scale.setDecimals(2)
        self.lr_scale.setPrefix("×")
        self.lr_scale.setValue(1.0)
        self.lr_scale.setFixedWidth(88)
        self.lr_scale.setToolTip("학습 중 학습률에 곱할 배율. '적용' 을 누르면 다음 스텝부터 반영됩니다.\n"
                                 "loss 가 튀면 ×0.5, 너무 느리면 ×1.5 처럼 조절해 보세요.")
        self.lr_apply_btn = QPushButton("적용")
        self.lr_apply_btn.clicked.connect(lambda: self.lr_scale_requested.emit(self.lr_scale.value()))
        ctrl.addWidget(self.lr_scale)
        ctrl.addWidget(self.lr_apply_btn)
        ctrl.addStretch(1)
        ctrl.addWidget(self.open_btn)
        self.add(ctrl)

        # 단계 표시
        phase_row = QHBoxLayout()
        phase_row.setSpacing(6)
        self.phase_labels = {}
        for p in PHASES:
            l = QLabel(p)
            l.setAlignment(Qt.AlignCenter)
            l.setMinimumHeight(26)
            self.phase_labels[p] = l
            phase_row.addWidget(l, 1)
        self.add(phase_row)
        self.set_phase(None)

        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m 스텝")
        self.progress.setValue(0)
        self.add(self.progress)
        self.sub_progress = QProgressBar()
        self.sub_progress.setVisible(False)
        self.add(self.sub_progress)
        self.status = hint("대기 중")
        self.add(self.status)

        stats = QGridLayout()
        stats.setSpacing(12)
        self.s_step = StatCard("스텝")
        self.s_loss = StatCard("loss (평균)")
        self.s_speed = StatCard("속도")
        self.s_eta = StatCard("남은 시간")
        self.s_vram = StatCard("최대 VRAM")
        self.s_lr = StatCard("학습률")
        for i, w in enumerate((self.s_step, self.s_loss, self.s_speed, self.s_eta, self.s_vram, self.s_lr)):
            stats.addWidget(w, 0, i)
        self.add(stats)

        chart = Card("Loss")
        self.chart = LossChart()
        chart.add(self.chart)
        chart.add(hint("실선: loss · 초록 선: 검증 loss · 주황 점선: 적응형 자동 조정 · 흐린 점선: 저장/샘플"))
        self.add(chart)

        split = QSplitter(Qt.Horizontal)
        samples = Card("샘플")
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.IconMode)
        self.gallery.setIconSize(QSize(200, 200))
        self.gallery.setGridSize(QSize(214, 244))
        self.gallery.setResizeMode(QListWidget.Adjust)
        self.gallery.setMovement(QListWidget.Static)
        self.gallery.setWordWrap(True)
        self.gallery.setMinimumHeight(280)
        self.gallery.itemDoubleClicked.connect(
            lambda it: QDesktopServices.openUrl(QUrl.fromLocalFile(it.data(Qt.UserRole)))
        )
        samples.add(self.gallery)
        samples.add(hint("더블클릭하면 원본 이미지를 엽니다."))
        split.addWidget(samples)
        logs = Card("로그")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setStyleSheet("font-family: Consolas, 'D2Coding', monospace; font-size: 12px;")
        logs.add(self.log_view)
        split.addWidget(logs)
        split.setSizes([560, 480])
        split.setMinimumHeight(360)
        self.add(split)
        self.output_dir = ""
        self.set_running(False)

    # ------------------------------------------------------------------
    def _pause(self, on: bool):
        self.pause_btn.setText("재개" if on else "일시정지")
        self.pause_toggled.emit(on)

    def _open_output(self):
        if self.output_dir and os.path.isdir(self.output_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_dir))

    def set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        for b in (self.pause_btn, self.sample_btn, self.save_btn, self.stop_btn, self.lr_scale, self.lr_apply_btn):
            b.setEnabled(running)
        if not running:
            self.pause_btn.blockSignals(True)
            self.pause_btn.setChecked(False)
            self.pause_btn.setText("일시정지")
            self.pause_btn.blockSignals(False)

    def set_phase(self, phase):
        reached = True
        for p, l in self.phase_labels.items():
            if phase is None:
                style = "background:#151821; color:#8a93a8;"
            elif p == phase:
                style = "background:#7c5cff; color:white; font-weight:700;"
                reached = False
            elif reached:
                style = "background:#2a2350; color:#c9bcff;"
            else:
                style = "background:#151821; color:#8a93a8;"
            l.setStyleSheet(style + "border-radius:6px; padding:4px;")

    def reset(self, total: int, output_dir: str):
        self.output_dir = output_dir
        self.chart.clear()
        self.gallery.clear()
        self.log_view.clear()
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(0)
        for s in (self.s_step, self.s_loss, self.s_speed, self.s_eta, self.s_vram, self.s_lr):
            s.set("-")
        self.lr_scale.setValue(1.0)
        self.set_phase("준비")

    def append_log(self, text: str):
        self.log_view.appendPlainText(text)
        self.log_view.moveCursor(QTextCursor.End)

    # 이벤트 처리 -------------------------------------------------------
    def on_event(self, kind: str, d: dict):
        if kind == "status":
            phase = d.get("phase", "")
            if phase in self.phase_labels:
                self.set_phase(phase)
            self.status.setText(f"{phase} · {d.get('message', '')}")
            self.sub_progress.setVisible(False)
        elif kind == "cache":
            self.sub_progress.setVisible(True)
            self.sub_progress.setMaximum(max(1, d.get("total", 1)))
            self.sub_progress.setValue(d.get("done", 0))
            label = "텍스트" if d.get("kind") == "text" else "이미지"
            self.sub_progress.setFormat(f"{label} 캐싱 %v / %m")
        elif kind == "progress":
            step, total = d.get("step", 0), d.get("total", 1)
            self.progress.setMaximum(total)
            self.progress.setValue(step)
            self.s_step.set(f"{step:,} / {total:,}")
            self.s_loss.set(f"{d.get('loss_avg', 0):.4f}")
            sps = d.get("sec_per_step", 0)
            self.s_speed.set(f"{sps:.2f} s/it" if sps >= 1 else f"{1 / max(sps, 1e-6):.2f} it/s")
            self.s_eta.set(fmt_time(d.get("eta_sec", 0)))
            self.s_vram.set(f"{d.get('vram_gb', 0):.1f} GB")
            sc = d.get("lr_scale", 1.0) or 1.0
            self.s_lr.set(f"{d.get('lr', 0):.2e}" + (f"  ×{sc:.2f}" if abs(sc - 1.0) > 1e-6 else ""))
            self.chart.add(step, d.get("loss", 0), d.get("loss_avg", 0))
        elif kind == "sample":
            step = d.get("step", 0)
            self.chart.mark(step)
            for path, prompt in zip(d.get("paths", []), d.get("prompts", [])):
                pm = QPixmap(path)
                if pm.isNull():
                    continue
                item = QListWidgetItem(QIcon(pm.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)),
                                       f"스텝 {step}\n{prompt[:40]}")
                item.setData(Qt.UserRole, path)
                item.setToolTip(prompt)
                self.gallery.insertItem(0, item)
        elif kind == "adaptive":
            self.chart.mark_event(d.get("step", 0))
            self.append_log("⚙ 적응형: " + d.get("message", ""))
            self.status.setText("자동 조정 · " + d.get("message", ""))
        elif kind == "val":
            self.chart.add_val(d.get("step", 0), d.get("loss", 0.0))
        elif kind == "saved":
            self.append_log(f"✔ 저장됨: {d.get('path')}")
        elif kind == "error":
            self.status.setText("오류: " + d.get("message", ""))
            self.status.setStyleSheet("color:#f87171;")
        elif kind == "done":
            self.set_phase("완료")
            path = d.get("path", "")
            self.status.setStyleSheet("")
            self.status.setText(("중지됨 · " if d.get("interrupted") else "완료 · ") + (path or "저장된 파일 없음"))
