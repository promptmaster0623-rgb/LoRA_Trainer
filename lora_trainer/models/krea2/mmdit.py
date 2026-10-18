# Copyright 2026 Krea AI, Inc. (원본: https://github.com/krea-ai/krea-2 , Apache License 2.0)
# 이 파일은 공식 mmdit.py 를 학습용으로 수정한 것이다. 변경 사항:
#   - torch.compile / CUDNN 강제 백엔드 제거 (Windows·구형 GPU 호환)
#   - 어텐션 마스크를 key-padding 형태(B,1,1,L)로 변경 (유효 토큰 결과 동일, 전부 마스크된 행의 NaN 방지)
#   - 256 배수 패딩 제거 (컴파일 커널 안정화용이었음, 수치 결과 동일)
#   - 그래디언트 체크포인팅(재진입 방식) 지원
# 모듈/파라미터 이름은 공식 체크포인트(=ComfyUI)와 동일하게 유지한다.
"""Krea 2 SingleStreamDiT (12.9B, 28 블록, 폭 6144)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def rope(pos: Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float64), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape).to(xq.dtype), xk_.reshape(*xk.shape).to(xk.dtype)


def attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None, gqa: bool = False) -> Tensor:
    if gqa and q.shape[1] != k.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return rearrange(x, "B H L D -> B L (H D)")


def key_mask(mask: Tensor | None) -> Tensor | None:
    """(B, L) bool -> (B, 1, 1, L) key-padding 마스크. 전부 유효하면 None (flash 경로)."""
    if mask is None:
        return None
    mask = mask.bool()
    if bool(mask.all()):
        return None
    return mask[:, None, None, :]


def temb(t: Tensor, dim: int, period: float = 1e4, tfactor: float = 1e3, device=None, dtype=None) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(period) * torch.arange(half, dtype=torch.float32, device=device) / half)
    args = (t.float() * tfactor)[:, None, None] * freqs
    sin, cos = torch.sin(args), torch.cos(args)
    return torch.cat((cos, sin), dim=-1).to(dtype=dtype)


@dataclass
class SingleMMDiTConfig:
    features: int = 6144
    tdim: int = 256
    txtdim: int = 2560
    heads: int = 48
    multiplier: int = 4
    layers: int = 28
    patch: int = 2
    channels: int = 16
    bias: bool = False
    theta: float = 1e3
    kvheads: int | None = 12
    txtlayers: int = 12
    txtheads: int = 20
    txtkvheads: int = 20


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim))
        self.multiplier = 2

    def forward(self, vec: Tensor):
        out = vec + rearrange(self.lin, "two d -> 1 two d")
        scale, shift = out.chunk(self.multiplier, dim=1)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, vec: Tensor):
        out = vec + self.lin
        return out.chunk(6, dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, axdims: list[int], theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims
        self.theta = theta
        self.ntk = ntk

    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat(
            [rope(pos[..., i], d, self.theta, self.ntk) for i, d in enumerate(self.axdims)],
            dim=-3,
        )


class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-05, device=None):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(features, device=device, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        t, dtype = x.float(), x.dtype
        t = F.rms_norm(t, (self.features,), eps=self.eps, weight=(self.scale.float() + 1.0))
        return t.to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        return self.qnorm(q), self.knorm(k), v


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads
        self.wq = nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.gqa = self.heads != self.kvheads
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(self, qkv: Tensor, freqs: Tensor | None = None, mask: Tensor | None = None) -> Tensor:
        q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)
        q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
        q, k, v = self.qknorm(q, k, v)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)
        return self.wo(attention(q, k, v, mask=mask, gqa=self.gqa) * torch.sigmoid(gate))


class LastLayer(nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        return self.linear(x)


class TextFusionBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: int = None):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.prenorm(x), mask=mask)
        return x + self.mlp(self.postnorm(x))


class TextFusionTransformer(nn.Module):
    def __init__(self, num_txt_layers: int, txt_dim: int, heads: int, multiplier: int, bias: bool = False, kvheads: int = None):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        )
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None)
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: int = None):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, vec: Tensor, freqs: Tensor, mask: Tensor | None = None) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs, mask)
        return x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)


class SingleStreamDiT(nn.Module):
    def __init__(self, config: SingleMMDiTConfig):
        super().__init__()
        self.config = config
        headdim = config.features // config.heads
        axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
        assert sum(axes) == headdim and all(a % 2 == 0 for a in axes), axes
        self.posemb = PositionalEncoding(config.features, axes, theta=config.theta, ntk=1.0)
        self.first = nn.Linear(config.channels * config.patch ** 2, config.features, bias=True)
        self.blocks = nn.ModuleList(
            [
                SingleStreamBlock(config.features, config.heads, config.multiplier, config.bias, config.kvheads)
                for _ in range(config.layers)
            ]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(config.tdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.txtfusion = TextFusionTransformer(
            config.txtlayers, config.txtdim, config.txtheads, config.multiplier, config.bias, config.txtkvheads
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(config.txtdim),
            nn.Linear(config.txtdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.last = LastLayer(config.features, config.patch, config.channels)
        self.tproj = nn.Sequential(nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features * 6))
        self.gradient_checkpointing = False

    def forward(self, img: Tensor, context: Tensor, t: Tensor, pos: Tensor, mask: Tensor | None = None) -> Tensor:
        """img: (B, N, C*p*p) 패치 토큰, context: (B, Lt, 12, 2560), t: (B,) [0,1] (1 = 노이즈),
        pos: (B, Lt+N, 3), mask: (B, Lt+N) bool."""
        img = self.first(img)
        t = self.tmlp(temb(t, self.config.tdim, device=img.device, dtype=img.dtype))
        tvec = self.tproj(t)

        txtlen = context.shape[1]
        txtmask = key_mask(mask[:, :txtlen]) if mask is not None else None
        context = self.txtfusion(context, mask=txtmask)
        context = self.txtmlp(context)

        imglen = img.shape[1]
        combined = torch.cat((context, img), dim=1)
        attn_mask = key_mask(mask)
        freqs = self.posemb(pos)

        use_ckpt = self.gradient_checkpointing and torch.is_grad_enabled() and self.training
        if use_ckpt and not combined.requires_grad:
            combined.requires_grad_(True)  # 재진입 체크포인트에서 LoRA grad 가 흐르도록
        for block in self.blocks:
            if use_ckpt:
                combined = checkpoint(block, combined, tvec, freqs, attn_mask, use_reentrant=True)
            else:
                combined = block(combined, tvec, freqs, attn_mask)

        final = self.last(combined, t)
        return final[:, txtlen: txtlen + imglen, :]
