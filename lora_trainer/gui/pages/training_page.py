"""학습 설정 페이지: LoRA 구조, 최적화, 타임스텝, 저장."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

from ...train.optim import OPTIMIZER_GROUPS, OPTIMIZER_SPECS, SCHEDULERS, optimizer_available
from ..widgets.common import Card, FormGrid, Page, PathEdit, SciSpin, combo, dspin, hint, set_combo, spin

TARGETS = [
    ("attn_mlp", "어텐션 + MLP (권장 · 표현력 높음)"),
    ("attn", "어텐션만 (가볍고 안정적)"),
    ("all_linear", "모든 Linear (텍스트 융합부 포함)"),
    ("custom", "사용자 정규식"),
]


class TrainingPage(Page):
    def __init__(self):
        super().__init__("학습 설정", "처음에는 프리셋 값을 그대로 쓰고, 샘플을 보며 스텝과 학습률을 조정하는 것을 권장합니다.")

        lora = Card("LoRA 구조")
        f = FormGrid()
        self.rank, self.alpha = spin(1, 256), dspin(0.1, 512, 1, 1)
        ra = QHBoxLayout()
        ra.setSpacing(8)
        ra.addWidget(QLabel("rank"))
        ra.addWidget(self.rank)
        ra.addSpacing(12)
        ra.addWidget(QLabel("alpha"))
        ra.addWidget(self.alpha)
        ra.addStretch(1)
        f.row("크기", ra, "캐릭터 16~32, 스타일 16~64. alpha=rank 이면 배율 1")
        self.target = combo(TARGETS)
        f.row("적용 레이어", self.target)
        self.include = QLineEdit()
        self.include.setPlaceholderText(r"예: blocks\.(1[0-9]|2[0-7])\.attn\.\w+")
        f.row("포함 정규식", self.include, "적용 레이어 = 사용자 정규식일 때")
        self.exclude = QLineEdit()
        self.exclude.setPlaceholderText(r"예: blocks\.(0|1)\.")
        f.row("제외 정규식", self.exclude)
        self.dropout, self.rank_dropout = dspin(0, 0.5, 0.05, 2), dspin(0, 0.5, 0.05, 2)
        dr = QHBoxLayout()
        dr.setSpacing(8)
        dr.addWidget(QLabel("입력"))
        dr.addWidget(self.dropout)
        dr.addSpacing(12)
        dr.addWidget(QLabel("rank"))
        dr.addWidget(self.rank_dropout)
        dr.addStretch(1)
        f.row("드롭아웃", dr, "과적합 방지 (보통 0)")
        self.rs_lora = QCheckBox("rsLoRA 스케일 (alpha/√rank) — 높은 rank 에서 안정적")
        f.row("", self.rs_lora)
        self.init_weights = PathEdit("file", filter="safetensors (*.safetensors)")
        f.row("이어 학습할 LoRA", self.init_weights, "기존 LoRA 가중치에서 시작 (선택)")
        lora.add(f)
        self.add(lora)

        opt = Card("최적화")
        f = FormGrid()
        self.steps, self.batch, self.accum = spin(1, 1_000_000, 100), spin(1, 64), spin(1, 64)
        f.row("총 스텝", self.steps, "이미지 20~40장 기준 1500~3000 스텝")
        sb = QHBoxLayout()
        sb.setSpacing(8)
        sb.addWidget(QLabel("배치"))
        sb.addWidget(self.batch)
        sb.addSpacing(12)
        sb.addWidget(QLabel("그래디언트 누적"))
        sb.addWidget(self.accum)
        sb.addStretch(1)
        f.row("배치", sb, "12GB 이하는 배치 1 권장. 실효 배치 = 배치 × 누적")
        self.lr = SciSpin()
        self.optimizer = self._optimizer_combo()
        self.opt_apply = QPushButton("권장값 적용")
        self.opt_apply.setObjectName("Ghost")
        self.opt_apply.setToolTip("선택한 옵티마이저의 권장 학습률 · weight decay · 스케줄을 채웁니다")
        oc = QHBoxLayout()
        oc.setSpacing(8)
        oc.addWidget(self.optimizer, 1)
        oc.addWidget(self.opt_apply)
        f.row("옵티마이저", oc)
        self.opt_info = hint("")
        self.opt_info.setTextFormat(Qt.RichText)
        f.row("", self.opt_info)
        f.row("학습률", self.lr, "옵티마이저를 바꾸면 권장값으로 자동 조정됩니다. 캐릭터 1e-4 · 스타일 1.5e-4~2e-4 (AdamW 기준)")
        self.optimizer.currentIndexChanged.connect(self._optimizer_changed)
        self.opt_apply.clicked.connect(lambda: self._apply_optimizer_defaults(force=True))
        self.scheduler = combo(list(SCHEDULERS.items()))
        self.warmup = spin(0, 100000, 10)
        sw = QHBoxLayout()
        sw.setSpacing(8)
        sw.addWidget(self.scheduler, 1)
        sw.addWidget(QLabel("웜업"))
        sw.addWidget(self.warmup)
        f.row("학습률 스케줄", sw)
        self.wd, self.clip = dspin(0, 1, 0.01, 3), dspin(0, 100, 0.1, 2)
        wc = QHBoxLayout()
        wc.setSpacing(8)
        wc.addWidget(QLabel("weight decay"))
        wc.addWidget(self.wd)
        wc.addSpacing(12)
        wc.addWidget(QLabel("grad clip"))
        wc.addWidget(self.clip)
        wc.addStretch(1)
        f.row("정규화", wc)
        self.loss = combo([("mse", "MSE (기본)"), ("huber", "Huber (이상치에 강함)")])
        f.row("손실 함수", self.loss)
        self.seed = spin(0, 2_147_483_647)
        f.row("시드", self.seed)
        opt.add(f)
        self.add(opt)

        ts = Card("타임스텝 (노이즈 단계) 분포", "자동은 모델별 권장값 (Krea 2: 해상도 연동 시간이동, MOSS: shift 3)")
        f = FormGrid()
        self.ts_sampling = combo([("logit_normal", "로짓-정규 (중간 노이즈 집중, 권장)"), ("uniform", "균등")])
        f.row("샘플링", self.ts_sampling)
        self.logit_mean, self.logit_std = dspin(-3, 3, 0.1, 2), dspin(0.1, 3, 0.1, 2)
        lm = QHBoxLayout()
        lm.setSpacing(8)
        lm.addWidget(QLabel("평균"))
        lm.addWidget(self.logit_mean)
        lm.addSpacing(12)
        lm.addWidget(QLabel("표준편차"))
        lm.addWidget(self.logit_std)
        lm.addStretch(1)
        f.row("로짓-정규", lm)
        self.shift_mode = combo([("auto", "자동 (모델 권장)"), ("none", "사용 안 함"), ("fixed", "고정 shift")])
        self.shift = dspin(0.1, 20, 0.5, 2)
        sm = QHBoxLayout()
        sm.setSpacing(8)
        sm.addWidget(self.shift_mode, 1)
        sm.addWidget(QLabel("값"))
        sm.addWidget(self.shift)
        f.row("시간 이동", sm, "값이 클수록 큰 노이즈(구도·형태) 위주 학습")
        ts.add(f)
        self.add(ts)

        ad = Card("적응형 학습 (실시간 자동 조정)",
                  "학습 중 loss 를 실시간으로 감시해 학습률을 스스로 조정하고, 발산하면 직전 정상 지점으로 되돌립니다. "
                  "모든 조정은 실행 페이지의 그래프(주황 점선)와 로그에 사유와 함께 표시됩니다.")
        self.ad_enabled = QCheckBox("적응형 학습 사용")
        self.ad_enabled.setStyleSheet("font-weight:600;")
        ad.add(self.ad_enabled)
        self.ad_box = QWidget()
        f = FormGrid()
        f.setContentsMargins(0, 0, 0, 0)
        self.ad_box.setLayout(f)

        self.ad_plateau = QCheckBox("사용")
        self.ad_patience, self.ad_factor, self.ad_min = spin(50, 100000, 50, " 스텝"), dspin(0.1, 0.95, 0.05, 2), dspin(0.01, 1, 0.05, 2)
        r = QHBoxLayout()
        r.setSpacing(8)
        for w in (self.ad_plateau, QLabel("인내"), self.ad_patience, QLabel("감소 ×"), self.ad_factor, QLabel("하한 ×"), self.ad_min):
            r.addWidget(w)
        r.addStretch(1)
        f.row("정체 감지", r, "개선이 멈추면 학습률을 낮춥니다. 검증 평가가 켜져 있으면 검증 loss 기준")

        self.ad_div = QCheckBox("사용")
        self.ad_spike, self.ad_maxrec = dspin(1.5, 20, 0.5, 1), spin(1, 20, 1, " 회")
        r = QHBoxLayout()
        r.setSpacing(8)
        for w in (self.ad_div, QLabel("급등 기준 ×"), self.ad_spike, QLabel("최대"), self.ad_maxrec):
            r.addWidget(w)
        r.addStretch(1)
        f.row("발산 복구", r, "loss 급등 · NaN 이 이어지면 직전 정상 지점으로 되돌리고 학습률을 절반으로")

        self.ad_clip = QCheckBox("자동")
        self.ad_clip_k = dspin(1, 10, 0.5, 1)
        r = QHBoxLayout()
        r.setSpacing(8)
        for w in (self.ad_clip, QLabel("최근 중앙값 ×"), self.ad_clip_k):
            r.addWidget(w)
        r.addStretch(1)
        f.row("그래디언트 클리핑", r, "켜면 위 grad clip 값 대신 최근 그래디언트 크기로 자동 결정")

        self.ad_eval, self.ad_eval_n = spin(0, 100000, 50), spin(1, 64)
        r = QHBoxLayout()
        r.setSpacing(8)
        for w in (QLabel("매"), self.ad_eval, QLabel("스텝마다 · 이미지"), self.ad_eval_n, QLabel("장")):
            r.addWidget(w)
        r.addStretch(1)
        f.row("검증 평가", r, "0 = 끔. 고정 노이즈로 loss 를 재서 흔들림 없는 추세를 봅니다 (평가 1회 ≈ 학습 몇 스텝)")

        self.ad_holdout = spin(0, 1000, 1, " 장")
        f.row("검증 전용 이미지", self.ad_holdout, "학습에서 빼서 평가에만 쓸 이미지 수. 과적합을 판단하려면 1~2장 권장 (0 = 학습 이미지로 평가)")

        self.ad_es = QCheckBox("사용")
        self.ad_es_pat = spin(1, 100, 1, " 회")
        r = QHBoxLayout()
        r.setSpacing(8)
        for w in (self.ad_es, QLabel("연속 미개선"), self.ad_es_pat):
            r.addWidget(w)
        r.addStretch(1)
        f.row("조기 종료", r, "검증 loss 가 계속 나빠지면 멈추고, 가장 좋았던 시점의 LoRA 로 저장합니다")

        ad.add(self.ad_box)
        ad.add(hint("학습 중에는 실행 · 모니터 페이지의 '학습률 배율' 로 학습률을 즉시 바꿀 수 있습니다 (적응형 사용 여부와 무관)."))
        self.ad_enabled.toggled.connect(self._adaptive_toggled)
        self.ad_es.toggled.connect(self._adaptive_toggled)
        self.ad_eval.valueChanged.connect(self._adaptive_toggled)
        for w in (self.ad_patience, self.ad_factor, self.ad_min, self.ad_spike, self.ad_clip_k, self.ad_eval, self.ad_eval_n,
                  self.ad_holdout, self.ad_es_pat):
            w.setFixedWidth(96 if w is not self.ad_patience else 120)
        self.add(ad)

        save = Card("저장")
        f = FormGrid()
        self.save_every, self.keep_last = spin(0, 100000, 50), spin(0, 100)
        se = QHBoxLayout()
        se.setSpacing(8)
        se.addWidget(QLabel("매"))
        se.addWidget(self.save_every)
        se.addWidget(QLabel("스텝마다 · 최근"))
        se.addWidget(self.keep_last)
        se.addWidget(QLabel("개 유지 (0=전부)"))
        se.addStretch(1)
        f.row("중간 저장", se)
        self.save_dtype = combo([("bf16", "bf16"), ("fp16", "fp16"), ("fp32", "fp32")])
        self.save_format = combo([("kohya", "kohya (lora_unet_*, 범용)"), ("comfy", "ComfyUI (diffusion_model.*)")])
        sf = QHBoxLayout()
        sf.setSpacing(8)
        sf.addWidget(self.save_format, 1)
        sf.addWidget(self.save_dtype)
        f.row("파일 형식", sf, "두 형식 모두 ComfyUI LoraLoader 에서 바로 사용 가능")
        self.save_state = QCheckBox("이어학습용 상태(옵티마이저 포함) 저장")
        f.row("", self.save_state)
        self.resume = PathEdit("dir")
        f.row("이어학습 (state 폴더)", self.resume, "<출력>/<이름>/state 폴더 지정")
        self.comfy_dir = PathEdit("dir")
        f.row("ComfyUI loras 폴더", self.comfy_dir, "지정 시 저장할 때마다 자동 복사 (설정 페이지 기본값 사용 가능)")
        save.add(f)
        self.add(save)
        self.rank.setFixedWidth(110)
        self.alpha.setFixedWidth(110)
        self.batch.setFixedWidth(110)
        self.accum.setFixedWidth(110)
        self.warmup.setFixedWidth(110)
        self.wd.setFixedWidth(110)
        self.clip.setFixedWidth(110)
        self.logit_mean.setFixedWidth(110)
        self.logit_std.setFixedWidth(110)
        self.shift.setFixedWidth(110)
        self.save_every.setFixedWidth(110)
        self.keep_last.setFixedWidth(110)
        self.dropout.setFixedWidth(110)
        self.rank_dropout.setFixedWidth(110)
        self.content.addStretch(1)

    # ------------------------------------------------------------------
    # 옵티마이저 선택
    # ------------------------------------------------------------------
    def _optimizer_combo(self) -> QComboBox:
        """그룹 머리글(선택 불가) + 옵티마이저 목록."""
        c = QComboBox()
        c.setMaxVisibleItems(24)
        for group in OPTIMIZER_GROUPS:
            c.addItem(f"── {group} ──", None)
            c.model().item(c.count() - 1).setEnabled(False)
            for spec in OPTIMIZER_SPECS.values():
                if spec.group != group:
                    continue
                mark = "" if optimizer_available(spec.key) else "  (설치 필요)"
                c.addItem(f"   {spec.label}{mark}", spec.key)
                c.setItemData(c.count() - 1, spec.desc, Qt.ToolTipRole)
        set_combo(c, "adamw8bit")
        return c

    def _optimizer_changed(self, *_):
        self._update_opt_info()
        if not getattr(self, "_loading", False):
            self._apply_optimizer_defaults(force=False)

    def _update_opt_info(self):
        spec = OPTIMIZER_SPECS.get(self.optimizer.currentData())
        if spec is None:
            self.opt_info.setText("")
            return
        tags = [f"메모리 {spec.memory}", f"권장 학습률 {spec.lr:g}"]
        if spec.lr_free:
            tags.append("학습률 자동")
        if spec.schedule_free:
            tags.append("스케줄러 불필요")
        if spec.self_tuning:
            tags.append("학습률 스스로 조정 · 칸은 시작값 · 스케줄러 미사용")
        html = f"{spec.desc}<br><span style='color:#a5b4fc'>{' · '.join(tags)}</span>"
        if not optimizer_available(spec.key):
            html += (f"<br><span style='color:#fbbf24'>설치 필요: install.bat 재실행 또는 "
                     f"<b>pip install {spec.pip}</b></span>")
        self.opt_info.setText(html)

    def _apply_optimizer_defaults(self, force: bool):
        """force=False 이면 학습률 범위가 맞지 않을 때만 바꾼다 (사용자 값 존중)."""
        spec = OPTIMIZER_SPECS.get(self.optimizer.currentData())
        if spec is None:
            return
        lr = self.lr.value()
        mismatch = (spec.lr_free and lr < 0.1) or (not spec.lr_free and lr >= 0.1)
        if spec.key.startswith("lion") and lr > spec.lr * 2:
            mismatch = True
        if spec.self_tuning and lr > 1e-4:
            mismatch = True
        if force or mismatch:
            self.lr.setValue(spec.lr)
        if force:
            self.wd.setValue(spec.weight_decay)
            if spec.schedule_free:
                set_combo(self.scheduler, "constant")

    def _adaptive_toggled(self, *_):
        on = self.ad_enabled.isChecked()
        self.ad_box.setEnabled(on)
        self.ad_es.setEnabled(on and self.ad_eval.value() > 0)
        self.ad_es_pat.setEnabled(on and self.ad_eval.value() > 0 and self.ad_es.isChecked())
        self.ad_holdout.setEnabled(on and self.ad_eval.value() > 0)
        self.ad_eval_n.setEnabled(on and self.ad_eval.value() > 0)

    def load(self, job):
        n, t = job.network, job.train
        self._loading = True
        self.rank.setValue(n.rank)
        self.alpha.setValue(n.alpha)
        set_combo(self.target, n.target)
        self.include.setText(n.include_regex)
        self.exclude.setText(n.exclude_regex)
        self.dropout.setValue(n.dropout)
        self.rank_dropout.setValue(n.rank_dropout)
        self.rs_lora.setChecked(n.rs_lora)
        self.init_weights.setText(n.init_weights)
        self.steps.setValue(t.max_steps)
        self.batch.setValue(t.batch_size)
        self.accum.setValue(t.grad_accum)
        self.lr.setValue(t.learning_rate)
        set_combo(self.optimizer, t.optimizer)
        set_combo(self.scheduler, t.lr_scheduler)
        self.warmup.setValue(t.warmup_steps)
        self.wd.setValue(t.weight_decay)
        self.clip.setValue(t.max_grad_norm)
        set_combo(self.loss, t.loss)
        self.seed.setValue(t.seed)
        set_combo(self.ts_sampling, t.timestep.sampling)
        self.logit_mean.setValue(t.timestep.logit_mean)
        self.logit_std.setValue(t.timestep.logit_std)
        set_combo(self.shift_mode, t.timestep.shift_mode)
        self.shift.setValue(t.timestep.shift)
        self.save_every.setValue(t.save_every)
        self.keep_last.setValue(t.keep_last)
        set_combo(self.save_dtype, t.save_dtype)
        set_combo(self.save_format, t.save_format)
        self.save_state.setChecked(t.save_state)
        self.resume.setText(t.resume_state)
        self.comfy_dir.setText(t.comfy_lora_dir)
        a = t.adaptive
        self.ad_enabled.setChecked(a.enabled)
        self.ad_plateau.setChecked(a.plateau)
        self.ad_patience.setValue(a.plateau_patience)
        self.ad_factor.setValue(a.plateau_factor)
        self.ad_min.setValue(a.min_lr_ratio)
        self.ad_div.setChecked(a.divergence)
        self.ad_spike.setValue(a.spike_ratio)
        self.ad_maxrec.setValue(a.max_recover)
        self.ad_clip.setChecked(a.auto_clip)
        self.ad_clip_k.setValue(a.auto_clip_k)
        self.ad_eval.setValue(a.eval_every)
        self.ad_eval_n.setValue(a.eval_samples)
        self.ad_holdout.setValue(a.val_holdout)
        self.ad_es.setChecked(a.early_stop)
        self.ad_es_pat.setValue(a.early_stop_patience)
        self._adaptive_toggled()
        self._update_opt_info()
        self._loading = False

    def store(self, job):
        n, t = job.network, job.train
        n.rank = self.rank.value()
        n.alpha = self.alpha.value()
        n.target = self.target.currentData()
        n.include_regex = self.include.text().strip()
        n.exclude_regex = self.exclude.text().strip()
        n.dropout = self.dropout.value()
        n.rank_dropout = self.rank_dropout.value()
        n.rs_lora = self.rs_lora.isChecked()
        n.init_weights = self.init_weights.text()
        t.max_steps = self.steps.value()
        t.batch_size = self.batch.value()
        t.grad_accum = self.accum.value()
        t.learning_rate = self.lr.value()
        t.optimizer = self.optimizer.currentData()
        t.lr_scheduler = self.scheduler.currentData()
        t.warmup_steps = self.warmup.value()
        t.weight_decay = self.wd.value()
        t.max_grad_norm = self.clip.value()
        t.loss = self.loss.currentData()
        t.seed = self.seed.value()
        t.timestep.sampling = self.ts_sampling.currentData()
        t.timestep.logit_mean = self.logit_mean.value()
        t.timestep.logit_std = self.logit_std.value()
        t.timestep.shift_mode = self.shift_mode.currentData()
        t.timestep.shift = self.shift.value()
        t.save_every = self.save_every.value()
        t.keep_last = self.keep_last.value()
        t.save_dtype = self.save_dtype.currentData()
        t.save_format = self.save_format.currentData()
        t.save_state = self.save_state.isChecked()
        t.resume_state = self.resume.text()
        t.comfy_lora_dir = self.comfy_dir.text()
        a = t.adaptive
        a.enabled = self.ad_enabled.isChecked()
        a.plateau = self.ad_plateau.isChecked()
        a.plateau_patience = self.ad_patience.value()
        a.plateau_factor = self.ad_factor.value()
        a.min_lr_ratio = self.ad_min.value()
        a.divergence = self.ad_div.isChecked()
        a.spike_ratio = self.ad_spike.value()
        a.max_recover = self.ad_maxrec.value()
        a.auto_clip = self.ad_clip.isChecked()
        a.auto_clip_k = self.ad_clip_k.value()
        a.eval_every = self.ad_eval.value()
        a.eval_samples = self.ad_eval_n.value()
        a.val_holdout = self.ad_holdout.value()
        a.early_stop = self.ad_es.isChecked()
        a.early_stop_patience = self.ad_es_pat.value()
