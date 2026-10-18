"""LoRA 네트워크 (kohya 방식: 원본 Linear 의 forward 를 감싸서 덧셈).

저장 포맷
  - kohya  : lora_unet_<모듈경로_밑줄>.lora_down.weight / .lora_up.weight / .alpha
  - comfy  : diffusion_model.<모듈경로>.lora_A.weight / .lora_B.weight / .alpha
두 포맷 모두 ComfyUI 기본 LoRA 로더에서 바로 불러올 수 있다.
"""

from __future__ import annotations

import json
import math
import re
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from ..quant.layers import linear_like, linear_shape

KOHYA_PREFIX = "lora_unet_"
COMFY_PREFIX = "diffusion_model."


class LoRAModule(nn.Module):
    def __init__(
        self,
        name: str,
        org_module: nn.Module,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        rs_lora: bool = False,
    ):
        super().__init__()
        self.lora_name = name
        in_f, out_f = linear_shape(org_module)
        self.in_features, self.out_features = in_f, out_f
        self.rank = rank
        self.alpha_value = float(alpha if alpha else rank)
        self.register_buffer("alpha", torch.tensor(self.alpha_value))
        self.scale = self.alpha_value / (math.sqrt(rank) if rs_lora else rank)
        self.lora_down = nn.Linear(in_f, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.multiplier = 1.0
        self.enabled = True
        self.org_module = [org_module]  # 리스트로 감싸 state_dict 에서 제외
        self.org_forward = None

    def apply_to(self) -> None:
        org = self.org_module[0]
        self.org_forward = org.forward
        org.forward = self.forward

    def restore(self) -> None:
        if self.org_forward is not None:
            self.org_module[0].forward = self.org_forward
            self.org_forward = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.org_forward(x)
        if not self.enabled or self.multiplier == 0.0:
            return out
        lx = x.to(self.lora_down.weight.dtype) if not torch.is_autocast_enabled() else x
        if self.training and self.dropout > 0:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)
        lx = self.lora_down(lx)
        scale = self.scale
        if self.training and self.rank_dropout > 0:
            mask = torch.rand(lx.shape[-1], device=lx.device) > self.rank_dropout
            lx = lx * mask.to(lx.dtype)
            scale = scale * (1.0 / (1.0 - self.rank_dropout))
        lx = self.lora_up(lx)
        return out + (lx * (scale * self.multiplier)).to(out.dtype)


class LoRANetwork(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        target_filter: Callable[[str, nn.Module], bool],
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        rs_lora: bool = False,
        exclude_regex: str = "",
    ):
        super().__init__()
        exclude = re.compile(exclude_regex) if exclude_regex else None
        self.loras = nn.ModuleList()
        self.names: List[str] = []
        for name, module in model.named_modules():
            if not name or not linear_like(module):
                continue
            if not target_filter(name, module):
                continue
            if exclude is not None and exclude.search(name):
                continue
            lora = LoRAModule(name, module, rank, alpha, dropout, rank_dropout, rs_lora)
            self.loras.append(lora)
            self.names.append(name)
        if not self.loras:
            raise RuntimeError("LoRA 대상 레이어를 찾지 못했습니다. 대상(target) 설정을 확인하세요.")
        self.rank = rank
        self.alpha = alpha

    # ------------------------------------------------------------------
    def apply(self) -> None:
        for l in self.loras:
            l.apply_to()

    def restore(self) -> None:
        for l in self.loras:
            l.restore()

    def set_multiplier(self, m: float) -> None:
        for l in self.loras:
            l.multiplier = m

    def set_enabled(self, enabled: bool) -> None:
        for l in self.loras:
            l.enabled = enabled

    def trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.trainable_params())

    # ------------------------------------------------------------------
    # 저장 / 로드
    # ------------------------------------------------------------------
    def lora_state_dict(self, fmt: str = "kohya", dtype: torch.dtype = torch.bfloat16) -> Dict[str, torch.Tensor]:
        sd: Dict[str, torch.Tensor] = {}
        for l in self.loras:
            down = l.lora_down.weight.detach().to("cpu", dtype).contiguous()
            up = l.lora_up.weight.detach().to("cpu", dtype).contiguous()
            alpha = torch.tensor(l.alpha_value, dtype=dtype)
            if fmt == "comfy":
                base = COMFY_PREFIX + l.lora_name
                sd[f"{base}.lora_A.weight"] = down
                sd[f"{base}.lora_B.weight"] = up
                sd[f"{base}.alpha"] = alpha
            else:
                base = KOHYA_PREFIX + l.lora_name.replace(".", "_")
                sd[f"{base}.lora_down.weight"] = down
                sd[f"{base}.lora_up.weight"] = up
                sd[f"{base}.alpha"] = alpha
        return sd

    def save(self, path: str, fmt: str, dtype: torch.dtype, metadata: Optional[Dict[str, str]] = None) -> None:
        sd = self.lora_state_dict(fmt, dtype)
        meta = {k: (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) for k, v in (metadata or {}).items()}
        save_file(sd, path, metadata=meta)

    def load_weights(self, path: str) -> Tuple[int, int]:
        """kohya / comfy / peft 형식 LoRA 를 읽어 현재 네트워크에 채운다. (로드 수, 전체 수)"""
        sd = load_file(path)
        by_name = {}
        for k, v in sd.items():
            name, kind = _parse_lora_key(k)
            if name is None:
                continue
            by_name.setdefault(name, {})[kind] = v
        loaded = 0
        for l in self.loras:
            entry = by_name.get(l.lora_name) or by_name.get(l.lora_name.replace(".", "_"))
            if not entry or "down" not in entry or "up" not in entry:
                continue
            down, up = entry["down"], entry["up"]
            if down.shape != l.lora_down.weight.shape or up.shape != l.lora_up.weight.shape:
                continue
            with torch.no_grad():
                l.lora_down.weight.copy_(down.to(l.lora_down.weight.dtype))
                l.lora_up.weight.copy_(up.to(l.lora_up.weight.dtype))
                if "alpha" in entry:
                    # 저장된 alpha 를 현재 스케일에 맞게 반영
                    ratio = float(entry["alpha"]) / l.alpha_value
                    l.lora_up.weight.mul_(ratio)
            loaded += 1
        return loaded, len(self.loras)


def _parse_lora_key(key: str):
    """키 -> (모듈이름, down|up|alpha). kohya 는 밑줄 이름을 반환."""
    for prefix in (COMFY_PREFIX, "transformer.", "base_model.model.", "model.diffusion_model."):
        if key.startswith(prefix):
            body = key[len(prefix):]
            for suffix, kind in ((".lora_A.weight", "down"), (".lora_B.weight", "up"),
                                 (".lora_down.weight", "down"), (".lora_up.weight", "up"), (".alpha", "alpha")):
                if body.endswith(suffix):
                    return body[: -len(suffix)], kind
    if key.startswith(KOHYA_PREFIX):
        body = key[len(KOHYA_PREFIX):]
        for suffix, kind in ((".lora_down.weight", "down"), (".lora_up.weight", "up"), (".alpha", "alpha")):
            if body.endswith(suffix):
                return body[: -len(suffix)], kind
    return None, None


def build_target_filter(presets: Dict[str, Iterable[str]], target: str, include_regex: str = "") -> Callable[[str, nn.Module], bool]:
    """프리셋(정규식 목록) 또는 사용자 정규식으로 대상 판별 함수 생성."""
    if target == "custom":
        if not include_regex:
            raise ValueError("custom 대상은 포함 정규식이 필요합니다.")
        pats = [re.compile(include_regex)]
    else:
        if target not in presets:
            raise ValueError(f"알 수 없는 LoRA 대상 프리셋: {target}")
        pats = [re.compile(p) for p in presets[target]]

    def _filter(name: str, module: nn.Module) -> bool:
        return any(p.fullmatch(name) for p in pats)

    return _filter
