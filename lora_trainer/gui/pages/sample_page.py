"""샘플 페이지: 학습 중 미리보기 이미지 설정."""

from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit

from ..widgets.common import Card, FormGrid, Page, dspin, hint, spin


class SamplePage(Page):
    def __init__(self):
        super().__init__("샘플", "학습 중 주기적으로 이미지를 생성해 진행 상황을 확인합니다. 프롬프트에 [trigger] 를 쓰면 트리거 단어로 바뀝니다.")
        card = Card("샘플 프롬프트", "한 줄에 하나씩. 트리거가 있는 것/없는 것을 섞으면 LoRA 의 영향을 비교하기 좋습니다.")
        self.prompts = QPlainTextEdit()
        self.prompts.setMinimumHeight(150)
        self.prompts.setPlaceholderText("[trigger], portrait photo, soft window light\n[trigger] standing in a neon city at night")
        card.add(self.prompts)
        self.negative = QLineEdit()
        self.negative.setPlaceholderText("네거티브 프롬프트 (비워두면 빈 문자열)")
        card.add(self.negative)
        self.add(card)

        opt = Card("생성 옵션")
        f = FormGrid()
        self.enabled = QCheckBox("학습 중 샘플 생성")
        f.row("", self.enabled)
        self.every = spin(1, 100000, 50)
        self.at_start = QCheckBox("학습 시작 전에도 생성 (베이스 비교용)")
        ev = QHBoxLayout()
        ev.setSpacing(8)
        ev.addWidget(QLabel("매"))
        ev.addWidget(self.every)
        ev.addWidget(QLabel("스텝마다"))
        ev.addSpacing(12)
        ev.addWidget(self.at_start)
        ev.addStretch(1)
        f.row("주기", ev)
        self.w, self.h = spin(256, 2048, 64), spin(256, 2048, 64)
        wh = QHBoxLayout()
        wh.setSpacing(8)
        wh.addWidget(self.w)
        wh.addWidget(QLabel("×"))
        wh.addWidget(self.h)
        wh.addStretch(1)
        f.row("크기", wh, "학습 시간 절약을 위해 768 권장 (16의 배수로 맞춰짐)")
        self.steps = spin(1, 200)
        f.row("스텝", self.steps, "Krea 2 Raw 공식 52 · MOSS 30. 미리보기는 20 정도면 충분")
        self.guidance = dspin(-1, 30, 0.5, 2)
        self.guidance.setSpecialValueText("모델 기본값")
        f.row("CFG", self.guidance, "Krea 2 Raw 3.5 (공식 정의) · MOSS 3.0")
        self.seed = spin(0, 2_147_483_647)
        f.row("시드", self.seed, "프롬프트마다 시드+순번 사용 (스텝 간 비교 가능)")
        opt.add(f)
        opt.add(hint("샘플 생성 중에는 학습이 잠시 멈춥니다. 저VRAM 에서는 스텝/크기를 줄이세요."))
        self.add(opt)
        self.every.setFixedWidth(110)
        self.w.setFixedWidth(110)
        self.h.setFixedWidth(110)
        self.content.addStretch(1)

    def load(self, job):
        s = job.sample
        self.prompts.setPlainText("\n".join(s.prompts))
        self.negative.setText(s.negative_prompt)
        self.enabled.setChecked(s.enabled)
        self.every.setValue(s.every)
        self.at_start.setChecked(s.at_start)
        self.w.setValue(s.width)
        self.h.setValue(s.height)
        self.steps.setValue(s.steps)
        self.guidance.setValue(s.guidance)
        self.seed.setValue(s.seed)

    def store(self, job):
        s = job.sample
        s.prompts = [p.strip() for p in self.prompts.toPlainText().splitlines() if p.strip()]
        s.negative_prompt = self.negative.text().strip()
        s.enabled = self.enabled.isChecked()
        s.every = self.every.value()
        s.at_start = self.at_start.isChecked()
        s.width, s.height = self.w.value(), self.h.value()
        s.steps = self.steps.value()
        s.guidance = self.guidance.value()
        s.seed = self.seed.value()
