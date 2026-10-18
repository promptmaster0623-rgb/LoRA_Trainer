"""저정밀도(양자화) Linear 레이어.

- FP8Linear : 가중치를 float8_e4m3fn + 행(row)별 스케일로 저장, 연산 시 bf16 으로 복원.
              GPU 세대와 무관하게 동작하며 메모리를 bf16 대비 절반으로 줄인다.
- make_nf4_linear : bitsandbytes NF4(4bit) Linear 생성 (메모리 약 1/4).

LoRA 는 이 레이어들 위에 덧붙여지며(베이스는 고정), LoRA 가중치만 학습된다.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
FP8_MAX = 448.0


def fp8_supported() -> bool:
    return FP8_DTYPE is not None


class FP8Linear(nn.Module):
    """float8 가중치 저장 Linear. 학습 불가(동결) 베이스 레이어 전용."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=FP8_DTYPE, device=device),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.ones(out_features, dtype=torch.float32, device=device), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.compute_dtype = torch.bfloat16

    @classmethod
    def from_weight(
        cls,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        compute_dtype: torch.dtype,
        device,
    ) -> "FP8Linear":
        out_f, in_f = weight.shape
        layer = cls.__new__(cls)
        nn.Module.__init__(layer)
        layer.in_features, layer.out_features = in_f, out_f
        w = weight.to(device=device, dtype=torch.float32)
        amax = w.abs().amax(dim=1).clamp(min=1e-12)
        scale = amax / FP8_MAX
        q = (w / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        layer.weight = nn.Parameter(q, requires_grad=False)
        layer.weight_scale = nn.Parameter(scale, requires_grad=False)
        if bias is not None:
            layer.bias = nn.Parameter(bias.to(device=device, dtype=compute_dtype), requires_grad=False)
        else:
            layer.register_parameter("bias", None)
        layer.compute_dtype = compute_dtype
        return layer

    def dequantized_weight(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        dtype = dtype or self.compute_dtype
        return self.weight.to(dtype) * self.weight_scale.to(dtype)[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else self.compute_dtype
        w = self.dequantized_weight(dtype)
        b = self.bias.to(dtype) if self.bias is not None else None
        return F.linear(x.to(dtype), w, b)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, fp8"


def make_nf4_linear(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    compute_dtype: torch.dtype,
    device,
) -> nn.Module:
    """bitsandbytes NF4 Linear 를 만들어 device(GPU)로 올린다 (올릴 때 양자화됨)."""
    try:
        import bitsandbytes as bnb
    except ImportError as e:  # pragma: no cover - 환경 의존
        raise RuntimeError(
            "NF4 양자화에는 bitsandbytes 가 필요합니다. 'pip install bitsandbytes' 로 설치하세요."
        ) from e
    out_f, in_f = weight.shape
    layer = bnb.nn.Linear4bit(
        in_f,
        out_f,
        bias=bias is not None,
        compute_dtype=compute_dtype,
        quant_type="nf4",
        device="meta",
    )
    layer.weight = bnb.nn.Params4bit(
        weight.to("cpu", dtype=compute_dtype).contiguous(),
        requires_grad=False,
        quant_type="nf4",
        compress_statistics=True,
    )
    if bias is not None:
        layer.bias = nn.Parameter(bias.to("cpu", dtype=compute_dtype), requires_grad=False)
    return layer.to(device)


def is_quantized_linear(module: nn.Module) -> bool:
    if isinstance(module, FP8Linear):
        return True
    name = type(module).__name__
    return name in ("Linear4bit", "Linear8bitLt")


def linear_like(module: nn.Module) -> bool:
    """LoRA 를 붙일 수 있는 Linear 계열인지."""
    return isinstance(module, nn.Linear) or is_quantized_linear(module)


def linear_shape(module: nn.Module):
    if isinstance(module, FP8Linear):
        return module.in_features, module.out_features
    return module.in_features, module.out_features
