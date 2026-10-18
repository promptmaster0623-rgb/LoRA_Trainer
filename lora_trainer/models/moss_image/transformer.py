# 원본: ostris/ai-toolkit extensions_built_in/diffusion_models/moss_image/src/transformer.py (MIT License,
# Copyright (c) 2024 Ostris, LLC). OpenMOSS 의 MOSSImageTransformer(pipeline_moss_image.py, Apache-2.0) 포팅본을
# 학습용으로 단순화 (참조이미지/KV 캐시 기능 제거, 재진입 체크포인트 추가).
# 키 이름은 Comfy-Org/MOSS-Image 레이아웃(q/k/v 결합 = proj.qkv)이며 ComfyUI LoRA 키와 일치한다.
"""MOSS-Image DiT (5B, 42 블록, 폭 3072, latent 128ch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class MossImageParams:
    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096
    freq_dim: int = 256
    latent_channels: int = 128
    rope_base: float = 10000.0


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = F.rms_norm(x.float(), (x.shape[-1],), eps=1e-6)
        return y.to(x.dtype) * self.weight.to(x.dtype)


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads, mask):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    out = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v, attn_mask=mask)
    return out.transpose(1, 2).flatten(2)


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope, mask):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads, mask))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict(
            {
                "qkv": nn.Linear(dim, 3 * dim, bias=False),
                "o": nn.Linear(dim, dim, bias=False),
                "ffn_up_gate": nn.Linear(dim, 2 * ffn_dim, bias=False),
                "ffn_down": nn.Linear(ffn_dim, dim, bias=False),
            }
        )

    def forward(self, x, rope, modulation, mask):
        # modulation (B, 6, D): [shift_a, scale_a, gate_a, shift_f, scale_f, gate_f]
        m = modulation.unbind(1)
        y = self.norm1(x) * (1 + m[1][:, None]) + m[0][:, None]
        q, k, v = self.proj["qkv"](y).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        attn = _attention(q, k, v, rope, self.heads, mask)
        x = x + self.proj["o"](attn) * m[2][:, None]
        y = self.norm2(x) * (1 + m[4][:, None]) + m[3][:, None]
        hidden, ffn_gate = self.proj["ffn_up_gate"](y).chunk(2, -1)
        return x + self.proj["ffn_down"](_situ(hidden, ffn_gate)) * m[5][:, None]


def _rope_bands(device, head_dim: int, base: float):
    num_bands = head_dim // 8
    exponent = torch.arange(num_bands, dtype=torch.float32) / num_bands
    return (1 / (base ** exponent)).to(device)


class MossImageTransformer(nn.Module):
    def __init__(self, params: Optional[MossImageParams] = None):
        super().__init__()
        params = params or MossImageParams()
        self.params = params
        dim, ffn_dim, heads = params.dim, params.ffn_dim, params.num_heads
        self.video_embedder = nn.Linear(params.latent_channels, dim)
        self.text_embedder = _TextProjection(params.text_dim, dim, ffn_dim, heads)
        self.time_embedding = nn.Sequential(nn.Linear(params.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([_Block(dim, ffn_dim, heads) for _ in range(params.num_layers)])
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, params.latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)
        self.gradient_checkpointing = False
        self._bands = {}

    def _rope_from_coords(self, coords):
        device = coords.device
        bands = self._bands.get(device)
        if bands is None:
            bands = _rope_bands(device, self.params.dim // self.params.num_heads, self.params.rope_base)
            self._bands[device] = bands
        phase = coords.unsqueeze(-1) * bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)

    def _rope(self, lengths, height, width, text_len, device):
        """이미지 토큰 좌표 (0, 텍스트길이, 중심기준 h, w), 텍스트 i 번째 토큰 (0, i, 0, 0)."""
        batch = lengths.shape[0]
        hh, ww = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        image = torch.zeros(height * width, 4, device=device)
        image[:, 2] = (hh - (height - 1) / 2).flatten()
        image[:, 3] = (ww - (width - 1) / 2).flatten()
        image = image.unsqueeze(0).repeat(batch, 1, 1)
        image[..., 1] = lengths.float()[:, None]
        text = torch.zeros(batch, text_len, 4, device=device)
        text[..., 1] = torch.arange(text_len, device=device, dtype=torch.float32)
        return self._rope_from_coords(torch.cat((image, text), 1))

    def _time_modulation(self, times, dtype):
        half = self.params.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(dtype)).float()
        mod = self.shared_block_modulation(time.to(dtype)).view(times.shape[0], 6, -1)
        return mod, time

    def forward(
        self,
        hidden_states: torch.Tensor,           # (B, 128, H, W)
        timestep: torch.Tensor,                # 0..1000, 1000 = 노이즈
        encoder_hidden_states: torch.Tensor,   # (B, S, 4096)
        encoder_attention_mask: Optional[torch.Tensor] = None,  # (B, S) bool, 앞쪽이 유효 토큰
        skip_layers: Optional[Iterable[int]] = None,
    ):
        skip = frozenset(skip_layers or ())
        batch, _, height, width = hidden_states.shape
        num_image = height * width
        device = hidden_states.device
        dtype = self.final_norm_video.weight.dtype

        context = encoder_hidden_states
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(context.shape[:2], dtype=torch.bool, device=device)
        mask = mask.to(device=device, dtype=torch.bool)
        lengths = mask.sum(1)
        text_len = int(lengths.max())
        padded = bool((lengths < text_len).any())
        context = context[:, :text_len]
        valid = torch.arange(text_len, device=device)[None] < lengths[:, None]

        times = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1).expand(batch)
        mod, time = self._time_modulation(times, dtype)
        head = self.video_head_modulation(time.to(dtype)).view(batch, 2, -1)
        rope = self._rope(lengths, height, width, text_len, device).to(dtype)

        text_mask = valid[:, None, None, :] if padded else None
        joint_mask = None
        if padded:
            live = torch.cat((torch.ones(batch, num_image, dtype=torch.bool, device=device), valid), 1)
            joint_mask = live[:, None, None, :]

        video = hidden_states.flatten(2).transpose(1, 2).to(dtype)
        video = self.video_embedder(video.contiguous())
        text = self.text_embedder.proj(context.to(dtype))
        text_rope = rope[:, num_image:]

        use_ckpt = self.gradient_checkpointing and torch.is_grad_enabled() and self.training
        for block in self.text_embedder.blocks:
            if use_ckpt:
                if not text.requires_grad:
                    text.requires_grad_(True)
                text = checkpoint(block, text, text_rope, text_mask, use_reentrant=True)
            else:
                text = block(text, text_rope, text_mask)

        x = torch.cat((video, text), 1)
        if use_ckpt and not x.requires_grad:
            x.requires_grad_(True)
        for index, layer in enumerate(self.layers):
            if index in skip:
                continue
            if use_ckpt:
                x = checkpoint(layer, x, rope, mod, joint_mask, use_reentrant=True)
            else:
                x = layer(x, rope, mod, joint_mask)

        x = self.final_norm_video(x[:, :num_image])
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        x = self.final_linear_video(x)
        return x.transpose(1, 2).reshape(batch, -1, height, width)


def moss_sources(model_keys, ckpt, strip_prefixes=("model.diffusion_model.", "diffusion_model.")):
    """모델 키 -> 체크포인트 키. 분리된 proj.q/k/v (공식 diffusers 샤드)는 qkv 로 결합."""
    available = {}
    for k in ckpt.keys():
        kk = k
        for p in strip_prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
                break
        available[kk] = k
    out = {}
    for key in model_keys:
        if key in available:
            out[key] = [available[key]]
        elif key.endswith(".proj.qkv.weight"):
            base = key[: -len("qkv.weight")]
            parts = [f"{base}{p}.weight" for p in ("q", "k", "v")]
            if all(p in available for p in parts):
                out[key] = [available[p] for p in parts]
    return out
