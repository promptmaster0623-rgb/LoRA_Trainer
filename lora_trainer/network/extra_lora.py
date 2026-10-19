"""고정(학습 안 함) LoRA 를 모델에 잠깐 덧붙이기 — 미리보기용 Turbo LoRA 등.

- 학습 중인 LoRA 위에 한 겹 더 감싸며 (forward 체인), detach() 로 원래대로 되돌린다.
- kohya(lora_unet_*), comfy(diffusion_model.*), peft/diffusers(transformer.*) 키를 모두 읽는다.
- diffusers 식 Krea 2 이름(transformer_blocks_N_attn_to_q …)은 Krea 2 고유 이름으로 바꿔 매칭
  (Fizgig krea2/driver.py _ALIASES 와 같은 표).
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file

from ..quant.layers import linear_like
from .lora import _parse_lora_key

KREA2_ALIASES = (
    (r"^transformer_blocks_(\d+)_", r"blocks_\1_"), (r"^text_fusion_", "txtfusion_"),
    (r"_attn_to_out_0$", "_attn_wo"), (r"_attn_to_gate$", "_attn_gate"), (r"_attn_to_q$", "_attn_wq"),
    (r"_attn_to_k$", "_attn_wk"), (r"_attn_to_v$", "_attn_wv"), (r"_ff_up$", "_mlp_up"),
    (r"_ff_gate$", "_mlp_gate"), (r"_ff_down$", "_mlp_down"), (r"^img_in$", "first"),
    (r"^final_layer_linear$", "last_linear"), (r"^txt_in_linear_1$", "txtmlp_1"),
    (r"^txt_in_linear_2$", "txtmlp_3"), (r"^time_embed_linear_1$", "tmlp_0"),
    (r"^time_embed_linear_2$", "tmlp_2"), (r"^time_mod_proj$", "tproj_1"),
)


def _alias(flat: str) -> str:
    out = flat
    for pat, new in KREA2_ALIASES:
        out = re.sub(pat, new, out)
    return out


class _Wrap:
    def __init__(self, module: nn.Module, down: torch.Tensor, up: torch.Tensor, scale: float):
        self.module = module
        self.down, self.up, self.scale = down, up, scale
        self.prev = module.forward
        module.forward = self.forward

    def forward(self, x):
        out = self.prev(x)
        lx = x.to(self.down.dtype)
        return out + ((lx @ self.down.t()) @ self.up.t() * self.scale).to(out.dtype)

    def restore(self):
        self.module.forward = self.prev


class StaticLoRA:
    """파일 하나를 읽어 두고 attach / detach 를 반복할 수 있는 고정 LoRA."""

    def __init__(self, path: str, dtype: torch.dtype = torch.bfloat16):
        self.path = path
        self.dtype = dtype
        self.entries: Dict[str, Dict[str, torch.Tensor]] = {}
        for k, v in load_file(path).items():
            name, kind = _parse_lora_key(k)
            if name is None:
                continue
            self.entries.setdefault(name.replace(".", "_"), {})[kind] = v
        self.entries = {k: e for k, e in self.entries.items() if "down" in e and "up" in e}
        self._wraps: List[_Wrap] = []

    def attach(self, model: nn.Module, strength: float = 1.0) -> Tuple[int, int]:
        """(적용된 모듈 수, 파일의 모듈 수)."""
        self.detach()
        lookup = {}
        for n, m in model.named_modules():
            if n and linear_like(m):
                lookup[n.replace(".", "_")] = m
        dev = next((p.device for p in model.parameters() if p.device.type != "meta"), torch.device("cpu"))
        for key, e in self.entries.items():
            mod = lookup.get(key) or lookup.get(_alias(key))
            if mod is None:
                continue
            down, up = e["down"], e["up"]
            rank = down.shape[0]
            alpha = float(e["alpha"]) if "alpha" in e else float(rank)
            self._wraps.append(_Wrap(mod, down.to(dev, self.dtype), up.to(dev, self.dtype),
                                     strength * alpha / rank))
        return len(self._wraps), len(self.entries)

    def detach(self) -> None:
        for w in reversed(self._wraps):
            w.restore()
        self._wraps = []

    @property
    def attached(self) -> bool:
        return bool(self._wraps)
