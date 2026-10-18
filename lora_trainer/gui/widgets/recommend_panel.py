"""시작하기 페이지의 "시스템 맞춤 추천" 패널: 사양 요약, 퀄리티/최적/속도 3가지 카드, 경고."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ...core.recommend import PURPOSES, QUANT_LABEL, Analysis, Recommendation, fmt_duration, fmt_speed
from ..theme import COLORS
from .common import Card, hint

ACCENT = {"quality": "#a48bff", "balanced": COLORS["accent"], "speed": COLORS["success"]}
ICON = {"quality": "◆", "balanced": "●", "speed": "▲"}


class OptionCard(QFrame):
    apply_clicked = Signal(str)

    def __init__(self, key: str):
        super().__init__()
        self.key = key
        self.setObjectName("OptCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(6)
        top = QHBoxLayout()
        self.title = QLabel()
        self.title.setStyleSheet(f"font-size: 15px; font-weight: 800; color: {ACCENT[key]};")
        self.badge = QLabel("추천")
        self.badge.setObjectName("Badge")
        self.badge.setVisible(key == "balanced")
        top.addWidget(self.title)
        top.addStretch(1)
        top.addWidget(self.badge)
        lay.addLayout(top)
        self.subtitle = QLabel()
        self.subtitle.setObjectName("Hint")
        lay.addWidget(self.subtitle)

        self.total = QLabel("-")
        self.total.setStyleSheet("font-size: 24px; font-weight: 800; margin-top: 4px;")
        lay.addWidget(self.total)
        self.total_sub = QLabel()
        self.total_sub.setObjectName("Hint")
        self.total_sub.setWordWrap(True)
        lay.addWidget(self.total_sub)
        self.speed = QLabel()
        self.speed.setStyleSheet("font-weight: 700;")
        lay.addWidget(self.speed)

        self.vram_bar = QProgressBar()
        self.vram_bar.setTextVisible(True)
        self.vram_bar.setFixedHeight(16)
        lay.addWidget(self.vram_bar)

        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.RichText)
        self.details.setStyleSheet("font-size: 12px; line-height: 140%;")
        lay.addWidget(self.details)
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        self.notes.setObjectName("Hint")
        lay.addWidget(self.notes)
        lay.addStretch(1)
        self.apply = QPushButton("이 설정 적용")
        if key == "balanced":
            self.apply.setObjectName("Primary")
        self.apply.clicked.connect(lambda: self.apply_clicked.emit(self.key))
        lay.addWidget(self.apply)
        self.set_selected(False)

    def set_selected(self, on: bool):
        border = ACCENT[self.key] if on else COLORS["border"]
        bg = "#1e1a33" if on else COLORS["card"]
        self.setStyleSheet(f"QFrame#OptCard {{ background: {bg}; border: {2 if on else 1}px solid {border}; border-radius: 12px; }}")
        self.apply.setText("✓ 적용됨" if on else "이 설정 적용")

    def show_rec(self, r: Recommendation, vram_total: float):
        s = r.settings
        self.title.setText(f"{ICON[r.key]}  {r.title}")
        self.subtitle.setText(r.subtitle)
        self.total.setText(f"약 {fmt_duration(r.total_sec)}")
        self.total_sub.setText(
            f"학습 {fmt_duration(r.train_sec)} · 준비(캐시·로딩) {fmt_duration(r.prep_sec)}"
            + (f" · 샘플 {fmt_duration(r.sample_sec)}" if r.sample_sec > 30 else "")
        )
        basis = f"실측 보정 {r.calibrated}회" if r.calibrated else "추정"
        self.speed.setText(f"{fmt_speed(r.sec_per_step)}   ·  {basis}")
        pct = int(min(100, r.vram_gb / vram_total * 100)) if vram_total else 0
        self.vram_bar.setMaximum(100)
        self.vram_bar.setValue(pct)
        self.vram_bar.setFormat(f"예상 VRAM {r.vram_gb:.1f} / {vram_total:.0f} GB" if vram_total else "GPU 없음")
        color = COLORS["danger"] if not r.fits or pct >= 97 else (COLORS["warning"] if pct >= 90 else ACCENT[r.key])
        self.vram_bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; border-radius: 6px; }}")
        res = "+".join(str(x) for x in s["resolutions"])
        swap = f" · 스왑 {s['block_swap']}" if s["block_swap"] else ""
        rows = [
            ("DiT", f"{QUANT_LABEL[s['transformer_quant']]}{swap}"),
            ("해상도", res),
            ("rank", f"{s['rank']}  ·  {s['optimizer']}"),
            ("스텝", f"{s['max_steps']:,}  ·  lr {s['learning_rate']:.1e}"),
            ("체크포인팅", "켬" if s["gradient_checkpointing"] else "끔 (빠름)"),
        ]
        self.details.setText("<table cellspacing='0' cellpadding='1'>" + "".join(
            f"<tr><td style='color:{COLORS['muted']}; padding-right:10px'>{k}</td><td><b>{v}</b></td></tr>" for k, v in rows
        ) + "</table>")
        self.notes.setText("\n".join("· " + n for n in r.notes))
        self.notes.setVisible(bool(r.notes))
        self.apply.setEnabled(True)


class RecommendPanel(Card):
    analyze_requested = Signal(str)       # purpose
    apply_requested = Signal(str)         # option key

    def __init__(self):
        super().__init__("시스템 맞춤 추천 설정",
                         "이 PC 의 VRAM · RAM · CPU · 가상메모리와 데이터셋을 분석해 가능한 범위 안에서 3가지 조합을 제안합니다.")
        head = QHBoxLayout()
        head.addWidget(QLabel("학습 목적"))
        self.purpose = QComboBox()
        for k, v in PURPOSES.items():
            self.purpose.addItem(v, k)
        self.purpose.currentIndexChanged.connect(lambda: self.analyze_requested.emit(self.current_purpose()))
        head.addWidget(self.purpose)
        head.addStretch(1)
        self.refresh = QPushButton("↻  다시 분석")
        self.refresh.setToolTip("데이터셋이나 모델을 바꾼 뒤 눌러 주세요")
        self.refresh.clicked.connect(lambda: self.analyze_requested.emit(self.current_purpose()))
        head.addWidget(self.refresh)
        self.add(head)

        self.spec_label = QLabel()
        self.spec_label.setWordWrap(True)
        self.spec_label.setTextFormat(Qt.RichText)
        self.add(self.spec_label)

        grid = QGridLayout()
        grid.setSpacing(12)
        self.cards = {}
        for i, key in enumerate(("quality", "balanced", "speed")):
            c = OptionCard(key)
            c.apply_clicked.connect(self.apply_requested.emit)
            self.cards[key] = c
            grid.addWidget(c, 0, i)
            grid.setColumnStretch(i, 1)
        self.add(grid)
        self.warn_label = QLabel()
        self.warn_label.setWordWrap(True)
        self.warn_label.setTextFormat(Qt.RichText)
        self.add(self.warn_label)
        self.add(hint("예상 시간은 GPU 성능표 기반 근사치입니다. 학습을 한 번 마치면 실측 속도로 자동 보정되어 점점 정확해집니다. "
                      "적용하면 정밀도·해상도·rank·옵티마이저·스텝·저장/샘플 간격이 바뀌고, 이름·데이터셋·모델 경로·프롬프트는 유지됩니다."))

    def current_purpose(self) -> str:
        return self.purpose.currentData()

    def set_purpose(self, p: str):
        i = self.purpose.findData(p)
        if i >= 0:
            self.purpose.blockSignals(True)
            self.purpose.setCurrentIndex(i)
            self.purpose.blockSignals(False)

    def set_busy(self, busy: bool):
        self.refresh.setEnabled(not busy)
        self.refresh.setText("분석 중…" if busy else "↻  다시 분석")

    def show_analysis(self, a: Analysis, applied: str = ""):
        sp, st = a.spec, a.stats
        m = COLORS["muted"]
        gpu = f"{sp.gpu_name} · <b>{sp.vram_gb:.0f}GB</b>" if sp.gpu_available else "<b>CUDA GPU 없음</b>"
        data = (f"<b>{st.images}</b>장 (캡션 {st.captioned}) · 짧은 변 중앙값 {st.median_short}px"
                if not st.empty else "<span style='color:%s'>폴더 없음</span>" % COLORS["warning"])
        self.spec_label.setText(
            f"<span style='color:{m}'>GPU</span> {gpu} &nbsp;&nbsp; "
            f"<span style='color:{m}'>RAM</span> <b>{sp.ram_gb:.0f}GB</b> &nbsp;&nbsp; "
            f"<span style='color:{m}'>CPU</span> {sp.cpu_cores}코어/{sp.cpu_threads}스레드 &nbsp;&nbsp; "
            f"<span style='color:{m}'>가상메모리</span> <b>{sp.swap_gb:.0f}GB</b><br>"
            f"<span style='color:{m}'>데이터셋</span> {data} &nbsp;&nbsp; "
            f"<span style='color:{m}'>사용 가능 VRAM</span> 약 {a.usable_vram:.1f}GB"
        )
        for r in a.options:
            self.cards[r.key].show_rec(r, sp.vram_gb if sp.gpu_available else 0)
        self.mark_applied(applied)
        if a.warnings:
            self.warn_label.setText("<br>".join(f"<span style='color:{COLORS['warning']}'>⚠</span> {w}" for w in a.warnings))
            self.warn_label.setVisible(True)
        else:
            self.warn_label.setVisible(False)

    def mark_applied(self, key: str):
        for k, c in self.cards.items():
            c.set_selected(k == key)
