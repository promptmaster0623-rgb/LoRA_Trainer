"""앱 전역 설정 (user_data/settings.json, 깃에 올라가지 않음)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
USER_DIR = os.path.join(ROOT, "user_data")
SETTINGS_PATH = os.path.join(USER_DIR, "settings.json")
QUEUE_DIR = os.path.join(USER_DIR, "queue")
SPEED_CALIB_PATH = os.path.join(USER_DIR, "speed_calibration.json")


@dataclass
class AppSettings:
    hf_token: str = ""
    models_dir: str = os.path.join(ROOT, "models")
    output_dir: str = os.path.join(ROOT, "output")
    comfy_lora_dir: str = ""
    last_config: str = ""
    recent: list = field(default_factory=list)
    # 모델별로 기억하는 구성 파일 경로 {arch: {transformer, text_encoder, tokenizer, vae}}
    model_paths: dict = field(default_factory=dict)
    # 추천 설정의 학습 목적 (character | style | general)
    purpose: str = "character"
    # MOSS-Image 캡션용 Ollama (Gemma 4) 로컬 연결
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e4b"
    ollama_think: bool = True              # 공식 가이드 권장: thinking 켬
    ollama_temperature: float = 1.0        # 공식 가이드 권장 1.0
    ollama_top_p: float = 0.95             # 공식 가이드 권장 0.95
    ollama_num_ctx: int = 16384
    ollama_timeout: int = 600              # 이미지 1장당 최대 대기(초)
    ollama_unload_after: bool = True       # 캡션 후 VRAM 에서 내림 (학습 VRAM 확보)
    moss_caption_variant: str = "long"     # 학습 .txt 에 쓸 변형: long | medium | short

    @classmethod
    def load(cls) -> "AppSettings":
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        except (OSError, json.JSONDecodeError, TypeError):
            return cls()

    def save(self) -> None:
        os.makedirs(USER_DIR, exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    def push_recent(self, path: str) -> None:
        path = os.path.abspath(path)
        self.recent = [p for p in self.recent if p != path]
        self.recent.insert(0, path)
        self.recent = self.recent[:10]
        self.last_config = path

    def remember_paths(self, arch: str, paths: dict) -> None:
        self.model_paths[arch] = {k: (v or "") for k, v in (paths or {}).items()}

    def paths_for(self, arch: str) -> dict:
        return dict(self.model_paths.get(arch, {}))
