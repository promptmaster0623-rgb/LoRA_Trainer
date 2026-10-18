"""베이스 모델 어댑터 인터페이스.

학습 엔진은 모델 종류와 무관하게 이 인터페이스만 사용한다.
Flow-matching 규약: t ∈ [0, 1], t=1 은 순수 노이즈,
  x_t = (1 - t) * x0 + t * noise,  모델 출력(속도) 목표 = noise - x0
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from PIL import Image

from ..core.config import ModelConfig, TimestepConfig

Log = Callable[[str], None]


class ModelAdapter(ABC):
    arch: str = ""
    display_name: str = ""
    latent_channels: int = 16
    spatial_divisor: int = 16            # 이미지 크기 = 이 값의 배수
    lora_presets: Dict[str, List[str]] = {}
    default_lora_target: str = "attn_mlp"
    default_guidance: float = 3.5
    default_steps: int = 28
    # 블록 스왑 대상 블록 리스트 이름
    block_attr: str = "blocks"
    # 기본 다운로드 경로
    defaults: Dict[str, str] = {}

    def __init__(self, cfg: ModelConfig, models_dir: str, hf_token: str, device: torch.device,
                 dtype: torch.dtype, log: Log = print):
        self.cfg = cfg
        self.models_dir = models_dir
        self.hf_token = hf_token or None
        self.device = device
        self.dtype = dtype
        self.log = log
        self.transformer: Optional[nn.Module] = None
        self.swapper = None

    # 경로 ---------------------------------------------------------------
    def path_or_default(self, key: str) -> str:
        value = getattr(self.cfg, f"{key}_path", "") or ""
        return value.strip() or self.defaults.get(key, "")

    # 텍스트 인코더 -------------------------------------------------------
    @abstractmethod
    def load_text_encoder(self, quant: str) -> None: ...

    @abstractmethod
    def encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        """하나의 프롬프트 -> CPU 텐서 dict (가변 길이 허용)."""

    @abstractmethod
    def unload_text_encoder(self) -> None: ...

    @abstractmethod
    def collate_text(self, items: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """배치 단위 패딩 + 마스크 생성 (device/dtype 로 이동)."""

    # VAE ------------------------------------------------------------------
    @abstractmethod
    def load_vae(self) -> None: ...

    @abstractmethod
    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        """(1, 3, H, W) [-1, 1] -> (C, h, w) 정규화된 latent (CPU)."""

    @abstractmethod
    def decode_latents(self, latents: torch.Tensor) -> List[Image.Image]: ...

    @abstractmethod
    def unload_vae(self) -> None: ...

    # 트랜스포머 ------------------------------------------------------------
    @abstractmethod
    def load_transformer(self, quant: str, block_swap: int) -> nn.Module: ...

    @abstractmethod
    def predict(self, latents: torch.Tensor, t: torch.Tensor, text: Dict[str, torch.Tensor],
                skip_layers: Sequence[int] = ()) -> torch.Tensor:
        """속도(noise - x0) 예측. latents (B, C, h, w), t (B,) ∈ [0,1]."""

    def unload_transformer(self) -> None:
        if self.swapper is not None:
            self.swapper.remove()
            self.swapper = None
        self.transformer = None

    # 타임스텝 --------------------------------------------------------------
    def shift_for(self, latent_hw: tuple, ts_cfg: TimestepConfig) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        """학습 타임스텝 변환 함수. None 이면 변환 없음."""
        mode = ts_cfg.shift_mode
        if mode == "none":
            return None
        if mode == "fixed":
            s = float(ts_cfg.shift)
            return lambda t: s * t / (1 + (s - 1) * t)
        return self.default_shift(latent_hw)

    def default_shift(self, latent_hw: tuple):
        return None

    # 샘플링 ----------------------------------------------------------------
    @abstractmethod
    @torch.no_grad()
    def sample(self, cond: Dict[str, torch.Tensor], uncond: Optional[Dict[str, torch.Tensor]],
               width: int, height: int, steps: int, guidance: float, seed: int) -> torch.Tensor:
        """latent 반환 (decode 는 별도)."""

    # 메타데이터 ------------------------------------------------------------
    def base_model_version(self) -> str:
        return self.arch


def exp_time_shift(mu: float):
    """exp(mu) 시간 이동: t' = e^mu / (e^mu + (1/t - 1))."""
    em = math.exp(mu)

    def f(t: torch.Tensor) -> torch.Tensor:
        t = t.clamp(1e-6, 1 - 1e-6)
        return em / (em + (1.0 / t - 1.0))

    return f


def pil_from_decoded(images: torch.Tensor) -> List[Image.Image]:
    """(B, 3, H, W) [-1,1] -> PIL 목록."""
    images = images.float().clamp(-1, 1)
    arr = ((images + 1.0) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(a) for a in arr]
