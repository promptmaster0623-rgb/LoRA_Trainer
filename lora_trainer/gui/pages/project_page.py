"""시작 페이지: 프로젝트 이름, 베이스 모델 선택, 시스템 맞춤 추천 설정, 프리셋, 대기열 추가."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from ...core.device import auto_memory_plan, gpu_info
from ..app_settings import ROOT
from ..theme import COLORS
from ..widgets.common import Card, FormGrid, Page, hint
from ..widgets.recommend_panel import RecommendPanel

MODEL_CARDS = {
    "krea2": {
        "title": "Krea 2",
        "badge": "12.9B · 1K",
        "lines": [
            "단일 스트림 DiT 28블록 · Qwen3-VL-4B 텍스트 인코더",
            "Raw 로 학습 → Turbo(8스텝)에서 사용 (공식 권장)",
            "스타일·질감·캐릭터 LoRA 에 강함",
        ],
    },
    "moss_image": {
        "title": "MOSS-Image",
        "badge": "5B · 텍스트 렌더링",
        "lines": [
            "단일 스트림 DiT 42블록 · Qwen3.5-9B 텍스트 인코더",
            "Base 로 학습 → Base/SFT 에서 사용",
            "글자·레이아웃·포스터 계열 LoRA 에 강함",
        ],
    },
}


class ModelChoice(QFrame):
    def __init__(self, arch: str):
        super().__init__()
        self.arch = arch
        self.setObjectName("Card")
        self.setCursor(Qt.PointingHandCursor)
        info = MODEL_CARDS[arch]
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        top = QHBoxLayout()
        t = QLabel(info["title"])
        t.setStyleSheet("font-size: 18px; font-weight: 800;")
        b = QLabel(info["badge"])
        b.setObjectName("Badge")
        top.addWidget(t)
        top.addStretch(1)
        top.addWidget(b)
        lay.addLayout(top)
        for line in info["lines"]:
            l = QLabel("·  " + line)
            l.setObjectName("Muted")
            lay.addWidget(l)
        self.files = QLabel()
        self.files.setObjectName("Hint")
        lay.addWidget(self.files)
        self.button = QPushButton("선택")
        self.button.setCheckable(True)
        lay.addWidget(self.button)
        self.button.toggled.connect(self._style)
        self._style(False)

    def mousePressEvent(self, e):
        if not self.button.isChecked():
            self.button.click()

    def set_files_status(self, text: str):
        self.files.setText(text)

    def _style(self, on: bool):
        self.setStyleSheet(
            f"QFrame#Card {{ border: 2px solid {COLORS['accent']}; background: #1e1a33; }}" if on else ""
        )
        self.button.setText("선택됨" if on else "선택")
        self.button.setObjectName("Primary" if on else "")
        self.button.style().unpolish(self.button)
        self.button.style().polish(self.button)


class ProjectPage(Page):
    arch_changed = Signal(str)
    preset_requested = Signal(str)
    recommend_requested = Signal(str)      # purpose
    recommend_apply = Signal(str)          # quality | balanced | speed
    queue_add_requested = Signal()

    def __init__(self):
        super().__init__("시작하기", "LoRA 이름과 베이스 모델을 고른 뒤, 이 PC 에 맞춘 추천 설정을 적용하세요. "
                                     "나머지는 왼쪽 메뉴 순서대로 확인하면 됩니다.")
        card = Card("프로젝트")
        form = FormGrid()
        name_row = QHBoxLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("예: my_character_v1")
        self.queue_btn = QPushButton("+ 대기열에 추가")
        self.queue_btn.setToolTip("지금 설정 그대로, LoRA 이름과 데이터셋 폴더만 바꿔 대기열에 예약합니다")
        self.queue_btn.clicked.connect(self.queue_add_requested.emit)
        name_row.addWidget(self.name, 1)
        name_row.addWidget(self.queue_btn)
        form.row("LoRA 이름", name_row, "출력 파일 이름이 됩니다. 영문/숫자/밑줄 권장")
        preset_row = QHBoxLayout()
        self.preset = QComboBox()
        self._fill_presets()
        apply = QPushButton("프리셋 적용")
        apply.clicked.connect(lambda: self.preset_requested.emit(self.preset.currentData() or ""))
        preset_row.addWidget(self.preset, 1)
        preset_row.addWidget(apply)
        form.row("프리셋", preset_row, "목적·VRAM 별 고정 설정 (아래 '시스템 맞춤 추천' 이 이 PC 에 더 정확합니다)")
        card.add(form)
        self.add(card)

        models = QHBoxLayout()
        models.setSpacing(16)
        self.choices = {a: ModelChoice(a) for a in MODEL_CARDS}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for a, w in self.choices.items():
            group.addButton(w.button)
            w.button.clicked.connect(lambda _=False, arch=a: self.arch_changed.emit(arch))
            models.addWidget(w, 1)
        self.add(models)

        self.recommend = RecommendPanel()
        self.recommend.analyze_requested.connect(self.recommend_requested.emit)
        self.recommend.apply_requested.connect(self.recommend_apply.emit)
        self.add(self.recommend)

        self.gpu_card = Card("GPU · 자동 메모리 계획", "VRAM 에 맞춰 양자화/블록스왑을 자동으로 고릅니다. 모델 페이지에서 직접 바꿀 수 있습니다.")
        self.gpu_label = QLabel()
        self.plan_label = QLabel()
        self.plan_label.setWordWrap(True)
        self.gpu_card.add(self.gpu_label)
        self.gpu_card.add(self.plan_label)
        self.add(self.gpu_card)

        tips = Card("작업 순서")
        for i, s in enumerate([
            "데이터셋 — 이미지 폴더 추가, 캡션 확인/자동 생성, 트리거 단어 지정",
            "모델 — 선택한 모델용 파일 경로 (비우면 허깅페이스에서 자동 다운로드)",
            "학습 설정 — 위 추천 설정을 적용했다면 그대로 사용해도 됩니다",
            "샘플 — 학습 중 확인할 프롬프트",
            "실행 — 시작 후 loss 그래프와 샘플로 진행 확인 (여러 개는 대기열로 예약)",
        ], 1):
            tips.add(hint(f"{i}.  {s}"))
        self.add(tips)
        self.content.addStretch(1)
        self.refresh_gpu("krea2")

    def _fill_presets(self):
        self.preset.clear()
        self.preset.addItem("— 선택 —", "")
        d = os.path.join(ROOT, "presets")
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith((".yaml", ".yml")):
                    self.preset.addItem(os.path.splitext(f)[0], os.path.join(d, f))

    def set_arch(self, arch: str):
        """외부(모델 페이지 등)에서 바뀐 모델을 신호 없이 반영."""
        w = self.choices.get(arch)
        if w and not w.button.isChecked():
            w.button.setChecked(True)
        self.refresh_gpu(arch)

    def set_files_status(self, arch: str, text: str):
        if arch in self.choices:
            self.choices[arch].set_files_status(text)

    def refresh_gpu(self, arch: str):
        info = gpu_info()
        self.gpu_label.setText(f"감지된 GPU: <b>{info.describe()}</b>")
        plan = auto_memory_plan(arch, info.total_gb if info.available else 0.0)
        self.plan_label.setText(
            f"<span style='color:{COLORS['muted']}'>자동 ({MODEL_CARDS[arch]['title']}):</span> "
            f"DiT <b>{plan.transformer_quant}</b> · 텍스트 인코더 <b>{plan.text_encoder_quant}</b> · "
            f"블록스왑 <b>{plan.block_swap}</b> — {plan.note}"
        )

    def load(self, job):
        self.name.setText(job.project.name)
        self.set_arch(job.model.arch)

    def store(self, job):
        job.project.name = self.name.text().strip() or "my_lora"
        for a, w in self.choices.items():
            if w.button.isChecked():
                job.model.arch = a
