"""설정 페이지: 허깅페이스 토큰, 폴더 경로, MOSS 캡션용 Ollama(Gemma 4) 연결, 실행 환경 점검."""

from __future__ import annotations

import platform
import sys

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton

from ...data import moss_caption as mc
from ..app_settings import AppSettings
from ..widgets.common import Card, FormGrid, Page, PathEdit, combo, dspin, hint, set_combo, spin
from ..widgets.ollama import OllamaModelPicker, describe

VARIANT_ITEMS = [
    ("long", "long · 상세 설명 (권장 — MOSS 사전학습 형식에 가장 가까움)"),
    ("medium", "medium · 한 문단"),
    ("short", "short · 한 문장"),
]


class SettingsPage(Page):
    def __init__(self, settings: AppSettings):
        super().__init__("설정", "이 PC 에만 저장되는 설정입니다 (user_data/settings.json, 깃에 올라가지 않음).")
        self.settings = settings
        card = Card("허깅페이스")
        f = FormGrid()
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.Password)
        self.token.setPlaceholderText("hf_...")
        show = QPushButton("보기")
        show.setCheckable(True)
        show.toggled.connect(lambda on: self.token.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password))
        row = QHBoxLayout()
        row.addWidget(self.token, 1)
        row.addWidget(show)
        f.row("액세스 토큰", row, "Krea 2 Raw 같은 게이트 모델 다운로드에 필요 (read 권한이면 충분)")
        card.add(f)
        self.add(card)

        paths = Card("폴더")
        f = FormGrid()
        self.models_dir = PathEdit("dir")
        self.output_dir = PathEdit("dir")
        self.comfy_dir = PathEdit("dir")
        f.row("모델 다운로드 폴더", self.models_dir)
        f.row("출력 폴더", self.output_dir, "LoRA, 샘플, 캐시가 <출력>/<LoRA 이름>/ 에 저장")
        f.row("ComfyUI loras 폴더", self.comfy_dir, "예: D:/ComfyUI/models/loras — 학습 설정에서 비워두면 이 값을 사용")
        paths.add(f)
        save = QPushButton("설정 저장")
        save.setObjectName("Primary")
        save.clicked.connect(self.save)
        paths.add(save)
        self.saved_label = hint("")
        paths.add(self.saved_label)
        self.add(paths)

        # MOSS-Image 캡션용 Ollama ------------------------------------------
        ol = Card("MOSS-Image 캡션 · Ollama (Gemma 4) 로컬 연결",
                  "MOSS-Image 공식 가이드의 리캡션 형식(long / medium / short + 워터마크 · 테두리)으로 캡션을 만듭니다. "
                  "데이터셋 페이지의 [자동 캡션 생성] 에서 MOSS-Image 일 때 기본으로 사용됩니다.")
        f = FormGrid()
        self.ollama_url = QLineEdit()
        self.ollama_url.setPlaceholderText(mc.DEFAULT_OLLAMA_URL)
        f.row("Ollama 주소", self.ollama_url, "기본 http://localhost:11434 — 다른 PC 의 Ollama 라면 그 PC 주소")
        self.ollama_model = OllamaModelPicker(lambda: self.ollama_url.text().strip() or mc.DEFAULT_OLLAMA_URL)
        f.row("Gemma 4 모델", self.ollama_model,
              "12GB 이하: gemma4:e4b 권장 · 24GB+: gemma4:26b / 31b. [↻ 목록] 으로 설치된 모델 선택 (Gemma 4 가 맨 위)")
        self.ollama_think = QCheckBox("thinking 켜기 (공식 가이드 권장 · 더 정확하지만 느림)")
        f.row("생각 모드", self.ollama_think)
        samp = QHBoxLayout()
        samp.setSpacing(8)
        self.ollama_temp = dspin(0.0, 2.0, 0.05, 2)
        self.ollama_top_p = dspin(0.05, 1.0, 0.05, 2)
        self.ollama_ctx = spin(4096, 131072, 1024)
        for lab, w in (("temperature", self.ollama_temp), ("top_p", self.ollama_top_p), ("컨텍스트", self.ollama_ctx)):
            samp.addWidget(QLabel(lab))
            w.setFixedWidth(100)
            samp.addWidget(w)
            samp.addSpacing(8)
        reset = QPushButton("공식 권장값")
        reset.setToolTip("temperature 1.0 · top_p 0.95 · thinking 켬")
        reset.clicked.connect(self._official_defaults)
        samp.addWidget(reset)
        samp.addStretch(1)
        f.row("샘플링", samp, "공식 가이드: temperature 1 · top_p 0.95 · 최대 16384 토큰")
        self.ollama_timeout = spin(30, 3600, 30, " 초")
        f.row("장당 최대 대기", self.ollama_timeout, "큰 모델 + thinking 은 1장에 1~3분 걸릴 수 있음")
        self.variant = combo(VARIANT_ITEMS)
        f.row("학습에 쓸 캡션", self.variant, "공식 JSON 은 <이미지>.moss.json 에 전부 저장되고, 이 변형이 .txt 로 학습에 쓰입니다")
        self.ollama_unload = QCheckBox("캡션이 끝나면 Ollama 모델을 VRAM 에서 내리기 (학습 VRAM 확보)")
        f.row("메모리", self.ollama_unload)
        ol.add(f)
        trow = QHBoxLayout()
        test = QPushButton("연결 테스트")
        test.clicked.connect(self._test_ollama)
        save2 = QPushButton("설정 저장")
        save2.setObjectName("Primary")
        save2.clicked.connect(self.save)
        trow.addWidget(test)
        trow.addStretch(1)
        trow.addWidget(save2)
        ol.add(trow)
        self.ollama_status = hint("")
        self.ollama_status.setWordWrap(True)
        ol.add(self.ollama_status)
        self.add(ol)

        env = Card("실행 환경")
        self.env_text = QPlainTextEdit()
        self.env_text.setReadOnly(True)
        self.env_text.setMinimumHeight(170)
        btn = QPushButton("환경 점검")
        btn.clicked.connect(self.check_env)
        env.add(btn)
        env.add(self.env_text)
        self.add(env)
        self.content.addStretch(1)
        self.load_settings()

    def load_settings(self):
        s = self.settings
        self.token.setText(s.hf_token)
        self.models_dir.setText(s.models_dir)
        self.output_dir.setText(s.output_dir)
        self.comfy_dir.setText(s.comfy_lora_dir)
        self.ollama_url.setText(s.ollama_url)
        self.ollama_model.setText(s.ollama_model)
        self.ollama_think.setChecked(s.ollama_think)
        self.ollama_temp.setValue(s.ollama_temperature)
        self.ollama_top_p.setValue(s.ollama_top_p)
        self.ollama_ctx.setValue(s.ollama_num_ctx)
        self.ollama_timeout.setValue(s.ollama_timeout)
        set_combo(self.variant, s.moss_caption_variant)
        self.ollama_unload.setChecked(s.ollama_unload_after)

    def _official_defaults(self):
        self.ollama_temp.setValue(mc.OFFICIAL_OPTIONS["temperature"])
        self.ollama_top_p.setValue(mc.OFFICIAL_OPTIONS["top_p"])
        self.ollama_think.setChecked(True)
        self.ollama_ctx.setValue(16384)

    def _test_ollama(self):
        self.ollama_status.setText("연결 확인 중…")
        model = self.ollama_model.text() or mc.DEFAULT_OLLAMA_MODEL
        self.ollama_model.refresh(lambda res: self.ollama_status.setText(describe(res, model)))

    def save(self):
        s = self.settings
        s.hf_token = self.token.text().strip()
        s.models_dir = self.models_dir.text()
        s.output_dir = self.output_dir.text()
        s.comfy_lora_dir = self.comfy_dir.text()
        s.ollama_url = self.ollama_url.text().strip() or mc.DEFAULT_OLLAMA_URL
        s.ollama_model = self.ollama_model.text() or mc.DEFAULT_OLLAMA_MODEL
        s.ollama_think = self.ollama_think.isChecked()
        s.ollama_temperature = self.ollama_temp.value()
        s.ollama_top_p = self.ollama_top_p.value()
        s.ollama_num_ctx = self.ollama_ctx.value()
        s.ollama_timeout = self.ollama_timeout.value()
        s.moss_caption_variant = self.variant.currentData()
        s.ollama_unload_after = self.ollama_unload.isChecked()
        s.save()
        self.saved_label.setText("저장되었습니다.")
        self.ollama_status.setText("저장되었습니다.")

    def check_env(self):
        lines = [f"Python {sys.version.split()[0]} · {platform.system()} {platform.release()}"]
        try:
            import torch

            lines.append(f"PyTorch {torch.__version__} · CUDA {torch.version.cuda} · 사용 가능: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                lines.append(f"GPU {p.name} · {p.total_memory / 1024 ** 3:.1f} GB · bf16 {torch.cuda.is_bf16_supported()}")
        except Exception as e:
            lines.append(f"PyTorch 오류: {e}")
        for mod in ("transformers", "diffusers", "safetensors", "bitsandbytes", "huggingface_hub", "PySide6", "prodigyopt",
                    "schedulefree", "prodigyplus", "dadaptation", "came_pytorch"):
            try:
                m = __import__(mod)
                lines.append(f"{mod} {getattr(m, '__version__', '?')}")
            except Exception:
                lines.append(f"{mod} — 설치 안 됨" + (" (선택 · 옵티마이저)" if mod in ("prodigyopt", "schedulefree", "prodigyplus", "dadaptation", "came_pytorch") else ""))
        try:
            import transformers

            ok = hasattr(transformers, "Qwen3VLForConditionalGeneration")
            lines.append(f"Qwen3-VL (Krea 2 텍스트 인코더) 지원: {'예' if ok else '아니오 — transformers 업데이트 필요'}")
            ok35 = hasattr(transformers, "Qwen3_5Model") or hasattr(transformers, "Qwen3_5ForConditionalGeneration")
            lines.append(f"Qwen3.5 (MOSS 텍스트 인코더) 지원: {'예' if ok35 else '아니오 — transformers 업데이트 필요'}")
        except Exception:
            pass
        self.env_text.setPlainText("\n".join(lines))
