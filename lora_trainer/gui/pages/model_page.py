"""모델 페이지: 베이스 모델 선택, 모델별 구성 파일(상태 확인 · 자동 찾기), 정밀도/양자화, 블록 스왑.

모델(Krea 2 / MOSS-Image)마다 필요한 파일이 달라서, 페이지 맨 위에서 "지금 어떤 모델용 파일을
설정하는지" 를 고르고 그 모델에 필요한 칸만 보여준다. 경로는 모델별로 따로 기억된다
(user_data/settings.json 의 model_paths). 시작하기 페이지의 모델 선택과 양방향으로 연동된다.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.device import auto_memory_plan, gpu_info
from ...core.model_files import (
    ARCH_TITLES,
    COMPONENTS,
    DEFAULT,
    FIELD_OF,
    HUB,
    MISMATCH,
    MISSING,
    OK,
    check_path,
    find_models,
)
from ...models import get_adapter_class
from ..theme import COLORS
from ..widgets.common import Card, FormGrid, Page, PathEdit, combo, hint, set_combo, spin

ST_FILTER = "safetensors (*.safetensors);;모든 파일 (*.*)"
STATE_COLOR = {
    OK: COLORS["success"], DEFAULT: COLORS["muted"], HUB: COLORS["muted"],
    MISSING: COLORS["danger"], MISMATCH: COLORS["warning"],
}


class ArchSwitch(QWidget):
    """Krea 2 | MOSS-Image 세그먼트 토글."""

    changed = Signal(str)

    def __init__(self):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = {}
        archs = list(ARCH_TITLES)
        for i, arch in enumerate(archs):
            b = QPushButton(ARCH_TITLES[arch])
            b.setCheckable(True)
            b.setMinimumHeight(38)
            b.setMinimumWidth(150)
            left = "8px" if i == 0 else "0px"
            right = "8px" if i == len(archs) - 1 else "0px"
            b.setStyleSheet(
                f"QPushButton {{ border-radius: 0; border-top-left-radius: {left}; border-bottom-left-radius: {left};"
                f" border-top-right-radius: {right}; border-bottom-right-radius: {right}; font-size: 14px; }}"
                f"QPushButton:checked {{ background: {COLORS['accent']}; border-color: {COLORS['accent']}; color: white; }}"
            )
            b.clicked.connect(lambda _=False, a=arch: self.changed.emit(a))
            self.group.addButton(b)
            self.buttons[arch] = b
            lay.addWidget(b)
        lay.addStretch(1)

    def set_arch(self, arch: str):
        b = self.buttons.get(arch)
        if b:
            b.blockSignals(True)
            b.setChecked(True)
            b.blockSignals(False)


class _FindSignals(QObject):
    done = Signal(dict)


class _FindJob(QRunnable):
    def __init__(self, root, arch, signals):
        super().__init__()
        self.root, self.arch, self.signals = root, arch, signals

    def run(self):
        try:
            res = find_models(self.root, self.arch)
        except Exception:
            res = {}
        self.signals.done.emit(res)


class FoundDialog(QDialog):
    """자동 찾기 결과에서 적용할 파일 선택."""

    def __init__(self, arch: str, found: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"자동 찾기 결과 — {ARCH_TITLES[arch]}")
        self.setMinimumWidth(720)
        lay = QVBoxLayout(self)
        lay.addWidget(hint("찾은 파일 중 사용할 것을 고르세요. 형식이 확인된 파일이 위에 표시됩니다."))
        form = FormGrid()
        self.combos = {}
        for c in COMPONENTS[arch]:
            cb = QComboBox()
            cb.addItem("— 변경 안 함 —", "")
            for p in found.get(c.key, []):
                cb.addItem(p, p)
            if cb.count() > 1:
                cb.setCurrentIndex(1)
            else:
                cb.setEnabled(False)
                cb.setItemText(0, "— 찾지 못함 —")
            self.combos[c.key] = cb
            form.row(c.label, cb)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("적용")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def selection(self) -> dict:
        return {k: cb.currentData() for k, cb in self.combos.items() if cb.currentData()}


class ModelPage(Page):
    arch_changed = Signal(str)

    def __init__(self, settings_getter=None):
        super().__init__(
            "모델",
            "학습할 베이스 모델을 고르고, 그 모델에 필요한 파일을 지정하세요. 비워두면 허깅페이스에서 기본 모델을 자동으로 내려받습니다.",
        )
        self.settings_getter = settings_getter
        self.arch = "krea2"
        self.pool = QThreadPool.globalInstance()
        self.find_signals = _FindSignals()
        self.find_signals.done.connect(self._found)

        # 1) 모델 선택 ------------------------------------------------------
        sel = Card("베이스 모델", "모델마다 필요한 파일이 다릅니다. 선택한 모델용 경로만 아래에 표시되며, 모델별로 따로 기억됩니다.")
        self.switch = ArchSwitch()
        self.switch.changed.connect(self._switch_clicked)
        sel.add(self.switch)
        self.arch_hint = hint("")
        sel.add(self.arch_hint)
        self.add(sel)

        # 2) 모델 파일 ------------------------------------------------------
        self.paths_card = Card("모델 파일")
        self.paths_card.add(hint("로컬 파일/폴더, 허깅페이스 저장소(org/repo), 하위폴더(org/repo::vae), "
                                 "저장소 내 파일(org/repo/file.safetensors) 형식을 지원합니다."))
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        self.edits = {
            "transformer": PathEdit("file", filter=ST_FILTER),
            "text_encoder": PathEdit("dir"),
            "tokenizer": PathEdit("dir"),
            "vae": PathEdit("file", filter=ST_FILTER),
        }
        self.row_widgets = {}
        row = 0
        for key in ("transformer", "text_encoder", "tokenizer", "vae"):
            lab = QLabel()
            lab.setMinimumWidth(130)
            lab.setStyleSheet("font-weight: 600;")
            expect = QLabel()
            expect.setObjectName("Hint")
            expect.setWordWrap(True)
            status = QLabel()
            status.setStyleSheet("font-size: 11px;")
            status.setWordWrap(True)
            grid.addWidget(lab, row, 0, Qt.AlignVCenter)
            grid.addWidget(self.edits[key], row, 1)
            grid.addWidget(expect, row + 1, 1)
            grid.addWidget(status, row + 2, 1)
            spacer = QWidget()
            spacer.setFixedHeight(8)
            grid.addWidget(spacer, row + 3, 0)
            self.row_widgets[key] = (lab, self.edits[key], expect, status, spacer)
            row += 4
            self.edits[key].changed.connect(lambda _t, k=key: self._schedule_check(k))
        self.paths_card.add(grid)
        btns = QHBoxLayout()
        self.find_btn = QPushButton("폴더에서 자동 찾기…")
        self.find_btn.setToolTip("ComfyUI models 폴더 등을 고르면 이 모델에 맞는 파일을 찾아 채웁니다")
        self.find_btn.clicked.connect(self._find)
        clear = QPushButton("모두 비우기 (기본값 자동 다운로드)")
        clear.setObjectName("Ghost")
        clear.clicked.connect(self._clear)
        btns.addWidget(self.find_btn)
        btns.addWidget(clear)
        btns.addStretch(1)
        self.summary = QLabel()
        btns.addWidget(self.summary)
        self.paths_card.add(btns)
        self.model_note = hint("")
        self.paths_card.add(self.model_note)
        self.add(self.paths_card)
        self._check_timers = {}

        # 3) 메모리 · 정밀도 ---------------------------------------------------
        mem = Card("메모리 · 정밀도", "자동(auto)은 감지된 VRAM 에 맞춰 결정됩니다. 12GB 이하에서는 자동 사용을 권장합니다.")
        form2 = FormGrid()
        self.dtype = combo([("bf16", "bf16 (RTX 30/40/50 권장)"), ("fp16", "fp16 (구형 GPU)")])
        self.t_quant = combo([
            ("auto", "자동"), ("none", "없음 (bf16)"), ("fp8", "FP8 (메모리 1/2, 품질 우수)"),
            ("nf4", "NF4 4bit (메모리 1/4, bitsandbytes)"),
        ])
        self.te_quant = combo([("auto", "자동"), ("none", "없음 (bf16)"), ("int8", "INT8"), ("nf4", "NF4 4bit")])
        self.swap = spin(-1, 41)
        self.swap.setSpecialValueText("자동")
        self.gc = QCheckBox("그래디언트 체크포인팅 (VRAM 크게 절약, 속도 약간 감소)")
        form2.row("연산 정밀도", self.dtype)
        form2.row("DiT 양자화", self.t_quant, "LoRA 는 항상 고정밀(fp32)로 학습되고 베이스만 압축됩니다")
        form2.row("텍스트 인코더 양자화", self.te_quant, "캐싱 단계에서만 사용 (학습 중엔 VRAM 미사용)")
        form2.row("블록 스왑", self.swap, "N개 블록을 CPU 메모리에 두고 필요할 때만 GPU 로 (bf16/FP8 전용, 속도 감소)")
        form2.row("", self.gc)
        mem.add(form2)
        self.plan = QLabel()
        self.plan.setWordWrap(True)
        mem.add(self.plan)
        self.add(mem)
        self.content.addStretch(1)
        for w in (self.t_quant, self.te_quant):
            w.currentIndexChanged.connect(self._update_plan)
        self.swap.valueChanged.connect(self._update_plan)

    # ------------------------------------------------------------------
    # 모델 전환
    # ------------------------------------------------------------------
    def _switch_clicked(self, arch: str):
        if arch != self.arch:
            self.arch_changed.emit(arch)

    def get_paths(self) -> dict:
        return {k: e.text() for k, e in self.edits.items()}

    def set_paths(self, paths: dict):
        for k, e in self.edits.items():
            e.setText((paths or {}).get(k, ""))
        self._check_all()

    def set_arch(self, arch: str, paths: dict = None):
        """arch 로 화면 전환. paths 를 주면 그 경로로 채운다."""
        self.arch = arch
        self.switch.set_arch(arch)
        title = ARCH_TITLES.get(arch, arch)
        cls = get_adapter_class(arch)
        d = cls.defaults
        comps = {c.key: c for c in COMPONENTS[arch]}
        for key, (lab, edit, expect, status, spacer) in self.row_widgets.items():
            c = comps.get(key)
            visible = c is not None
            for w in (lab, edit, expect, status, spacer):
                w.setVisible(visible)
            if not visible:
                continue
            lab.setText(c.label + ("  (선택)" if c.optional else ""))
            expect.setText(c.expect)
            default = d.get(key) or ""
            edit.setPlaceholderText(f"비우면 기본값: {default}" if default else "비우면 텍스트 인코더 폴더 사용")
        self.paths_card.findChild(QLabel, "CardTitle").setText(f"모델 파일 — {title}")
        self.arch_hint.setText(
            f"지금 <b style='color:{COLORS['accent_hover']}'>{title}</b> 용 파일을 설정하고 있습니다. "
            f"시작하기 페이지의 모델 선택과 연동됩니다."
        )
        if arch == "krea2":
            self.model_note.setText(
                "Krea 2 Raw 는 게이트 모델입니다: 허깅페이스에서 라이선스 동의 후 설정 페이지에 토큰을 입력하세요. "
                "Turbo 체크포인트로도 학습은 되지만 공식 권장은 Raw 학습 → Turbo 사용입니다."
            )
        else:
            self.model_note.setText(
                "MOSS-Image 는 Base 학습을 권장합니다. ComfyUI 레이아웃(Comfy-Org/MOSS-Image)과 공식 diffusers 폴더 모두 지원합니다."
            )
        self.swap.setMaximum(27 if arch == "krea2" else 41)
        if paths is not None:
            self.set_paths(paths)
        else:
            self._check_all()
        self._update_plan()

    # ------------------------------------------------------------------
    # 상태 확인
    # ------------------------------------------------------------------
    def _schedule_check(self, key: str):
        t = self._check_timers.get(key)
        if t is None:
            t = QTimer(self)
            t.setSingleShot(True)
            t.setInterval(350)
            t.timeout.connect(lambda k=key: self._check(k))
            self._check_timers[key] = t
        t.start()

    def _check(self, key: str):
        comps = {c.key for c in COMPONENTS[self.arch]}
        if key not in comps:
            return
        d = get_adapter_class(self.arch).defaults
        st = check_path(self.arch, key, self.edits[key].text(), d.get(key) or "")
        status = self.row_widgets[key][3]
        status.setText(f"{st.icon}  {st.message}")
        status.setStyleSheet(f"font-size: 11px; color: {STATE_COLOR.get(st.state, COLORS['text'])};")
        self._update_summary()

    def _check_all(self):
        for c in COMPONENTS[self.arch]:
            self._check(c.key)

    def _update_summary(self):
        bad = 0
        for c in COMPONENTS[self.arch]:
            txt = self.row_widgets[c.key][3].text()
            if txt.startswith(("✗", "⚠")):
                bad += 1
        if bad:
            self.summary.setText(f"<span style='color:{COLORS['warning']}'>확인 필요 {bad}개</span>")
        else:
            self.summary.setText(f"<span style='color:{COLORS['success']}'>준비됨</span>")

    def has_problems(self) -> list:
        """학습 시작 전 확인용: [(구성요소 이름, 메시지)]"""
        out = []
        d = get_adapter_class(self.arch).defaults
        for c in COMPONENTS[self.arch]:
            st = check_path(self.arch, c.key, self.edits[c.key].text(), d.get(c.key) or "")
            if st.state in (MISSING, MISMATCH):
                out.append((c.label, st.message))
        return out

    # ------------------------------------------------------------------
    # 자동 찾기 / 비우기
    # ------------------------------------------------------------------
    def _find(self):
        start = ""
        s = self.settings_getter() if self.settings_getter else None
        if s is not None:
            start = getattr(s, "models_dir", "") or ""
        root = QFileDialog.getExistingDirectory(self, "모델을 찾을 폴더 (예: ComfyUI/models)", start)
        if not root:
            return
        self.find_btn.setEnabled(False)
        self.find_btn.setText("찾는 중…")
        self.pool.start(_FindJob(root, self.arch, self.find_signals))

    def _found(self, res: dict):
        self.find_btn.setEnabled(True)
        self.find_btn.setText("폴더에서 자동 찾기…")
        if not any(res.values()):
            QMessageBox.information(self, "자동 찾기", f"{ARCH_TITLES[self.arch]} 용 파일을 찾지 못했습니다.")
            return
        dlg = FoundDialog(self.arch, res, self)
        if dlg.exec() == QDialog.Accepted:
            for k, p in dlg.selection().items():
                self.edits[k].setText(p)

    def _clear(self):
        for c in COMPONENTS[self.arch]:
            self.edits[c.key].setText("")

    # ------------------------------------------------------------------
    def _update_plan(self):
        info = gpu_info()
        plan = auto_memory_plan(self.arch, info.total_gb if info.available else 0.0)
        tq = plan.transformer_quant if self.t_quant.currentData() == "auto" else self.t_quant.currentData()
        teq = plan.text_encoder_quant if self.te_quant.currentData() == "auto" else self.te_quant.currentData()
        sw = plan.block_swap if self.swap.value() < 0 else self.swap.value()
        warn = ""
        if tq == "nf4" and sw > 0:
            warn = f"<br><span style='color:{COLORS['warning']}'>NF4 는 블록 스왑과 함께 쓸 수 없어 스왑이 꺼집니다.</span>"
        self.plan.setText(f"적용될 설정 → DiT <b>{tq}</b> · 텍스트 인코더 <b>{teq}</b> · 블록 스왑 <b>{sw}</b>{warn}")

    def load(self, job):
        m = job.model
        set_combo(self.dtype, m.dtype)
        set_combo(self.t_quant, m.transformer_quant)
        set_combo(self.te_quant, m.text_encoder_quant)
        self.swap.setValue(m.block_swap)
        self.gc.setChecked(m.gradient_checkpointing)
        self.set_arch(m.arch, {k: getattr(m, f) for k, f in FIELD_OF.items()})

    def store(self, job):
        m = job.model
        m.arch = self.arch
        for k, f in FIELD_OF.items():
            setattr(m, f, self.edits[k].text())
        m.dtype = self.dtype.currentData()
        m.transformer_quant = self.t_quant.currentData()
        m.text_encoder_quant = self.te_quant.currentData()
        m.block_swap = self.swap.value()
        m.gradient_checkpointing = self.gc.isChecked()
