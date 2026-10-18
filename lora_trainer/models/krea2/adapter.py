"""Krea 2 어댑터.

구성
  - DiT       : SingleStreamDiT 12.9B (공식 krea-2 코드, 키 이름 = 공식/ComfyUI)
  - 텍스트    : Qwen3-VL-4B-Instruct, 12개 레이어 hidden state 스택 (공식 템플릿)
  - VAE       : Qwen-Image VAE (f8, 16ch, latents_mean/std 정규화)
권장: Raw 체크포인트로 학습 → Turbo 에서 사용 (공식 권장, LoRA 호환)
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from einops import rearrange, repeat
from PIL import Image

from ...core.device import free_memory
from ...quant.loader import LazyCheckpoint, identity_sources, load_streaming, model_tensor_keys
from ...quant.offload import BlockSwapper
from ...utils.hub import find_safetensors, resolve_path
from ..base import ModelAdapter, exp_time_shift, pil_from_decoded
from ..hf_loading import drop_vision_tower, load_hf_model
from .mmdit import SingleMMDiTConfig, SingleStreamDiT

SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
PROMPT_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PREFIX_TOKENS = 34
MAX_TEXT_TOKENS = 512

# 해상도 의존 시간 이동 (공식 sampling.py): 256px -> mu 0.5, 1280px -> mu 1.15
MU_X1, MU_X2 = (256 // 16) ** 2, (1280 // 16) ** 2
MU_Y1, MU_Y2 = 0.5, 1.15


def mu_for_seq_len(seq_len: int) -> float:
    slope = (MU_Y2 - MU_Y1) / (MU_X2 - MU_X1)
    return slope * seq_len + (MU_Y1 - slope * MU_X1)


def prepare_tokens(img: torch.Tensor, txtlen: int, patch: int, txtmask: torch.Tensor):
    """latent 를 패치 토큰으로, 텍스트+이미지 위치/마스크 생성 (공식 prepare)."""
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    tokens = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    mask = torch.cat((txtmask.to(img.device).bool(), imgmask), dim=1)
    pos = torch.cat((txtpos, imgpos), dim=1)
    return tokens, pos, mask


class Krea2Adapter(ModelAdapter):
    arch = "krea2"
    display_name = "Krea 2 (12.9B)"
    latent_channels = 16
    spatial_divisor = 16
    block_attr = "blocks"
    default_guidance = 3.5
    default_steps = 28
    defaults = {
        "transformer": "krea/Krea-2-Raw/raw.safetensors",
        "text_encoder": "Qwen/Qwen3-VL-4B-Instruct",
        "tokenizer": "",
        "vae": "Qwen/Qwen-Image::vae",
    }
    lora_presets = {
        "attn": [r"blocks\.\d+\.attn\.(wq|wk|wv|wo|gate)"],
        "attn_mlp": [
            r"blocks\.\d+\.attn\.(wq|wk|wv|wo|gate)",
            r"blocks\.\d+\.mlp\.(gate|up|down)",
        ],
        "all_linear": [
            r"blocks\.\d+\.(attn|mlp)\.\w+",
            r"txtfusion\.(layerwise|refiner)_blocks\.\d+\.(attn|mlp)\.\w+",
        ],
    }
    quant_exclude = ["first", "tmlp.*", "tproj.*", "txtmlp.*", "txtfusion.projector", "last.*"]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.config = SingleMMDiTConfig()
        self.text_model = None
        self.tokenizer = None
        self.vae = None
        self._latent_mean = None
        self._latent_std = None

    # ------------------------------------------------------------------
    # 텍스트 인코더
    # ------------------------------------------------------------------
    def load_text_encoder(self, quant: str) -> None:
        from transformers import AutoTokenizer

        try:
            from transformers import Qwen3VLForConditionalGeneration as TECls
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Qwen3-VL 지원 transformers(>=4.57) 가 필요합니다.") from e

        path = resolve_path(self.path_or_default("text_encoder"), self.models_dir, self.hf_token,
                            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"],
                            log=self.log)
        tok_path = self.cfg.tokenizer_path.strip() or path
        tok_path = resolve_path(tok_path, self.models_dir, self.hf_token, log=self.log)
        self.log(f"Qwen3-VL 텍스트 인코더 로딩 ({quant}) : {path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.text_model = load_hf_model(TECls, path, self.dtype, self.device, quant)
        if drop_vision_tower(self.text_model):
            free_memory()
        n_prefix = len(self.tokenizer(PROMPT_PREFIX)["input_ids"])
        if n_prefix != PREFIX_TOKENS:
            self.log(f"[경고] 프롬프트 접두 토큰 수가 {n_prefix} 입니다 (공식 34). 토크나이저를 확인하세요.")

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        tok = self.tokenizer
        dev = next(self.text_model.parameters()).device
        suffix = tok([PROMPT_SUFFIX], return_tensors="pt")
        inputs = tok([PROMPT_PREFIX + prompt], truncation=True, max_length=MAX_TEXT_TOKENS + PREFIX_TOKENS,
                     return_tensors="pt")
        input_ids = torch.cat([inputs["input_ids"], suffix["input_ids"]], dim=1).to(dev)
        mask = torch.cat([inputs["attention_mask"], suffix["attention_mask"]], dim=1).bool().to(dev)
        out = self.text_model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
        hiddens = torch.stack([out.hidden_states[i] for i in SELECT_LAYERS], dim=2)  # (1, L, 12, 2560)
        feats = hiddens[0, PREFIX_TOKENS:]
        return {"feats": feats.to("cpu", torch.bfloat16).contiguous()}

    def unload_text_encoder(self) -> None:
        self.text_model = None
        free_memory()

    def collate_text(self, items: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        lengths = [it["feats"].shape[0] for it in items]
        lmax = max(lengths)
        n, d = items[0]["feats"].shape[1:]
        feats = torch.zeros(len(items), lmax, n, d, dtype=self.dtype)
        mask = torch.zeros(len(items), lmax, dtype=torch.bool)
        for i, it in enumerate(items):
            feats[i, : lengths[i]] = it["feats"].to(self.dtype)
            mask[i, : lengths[i]] = True
        return {"feats": feats.to(self.device), "mask": mask.to(self.device)}

    # ------------------------------------------------------------------
    # VAE
    # ------------------------------------------------------------------
    def load_vae(self) -> None:
        from diffusers import AutoencoderKLQwenImage

        spec = self.path_or_default("vae")
        path = resolve_path(spec, self.models_dir, self.hf_token, log=self.log)
        self.log(f"Qwen-Image VAE 로딩 : {path}")
        if os.path.isfile(path):
            self.vae = AutoencoderKLQwenImage.from_single_file(path, torch_dtype=torch.float32)
        else:
            self.vae = AutoencoderKLQwenImage.from_pretrained(path, torch_dtype=torch.float32)
        self.vae.to(self.device).eval().requires_grad_(False)
        cfg = self.vae.config
        self._latent_mean = torch.tensor(cfg.latents_mean).view(1, -1, 1, 1, 1)
        self._latent_std = torch.tensor(cfg.latents_std).view(1, -1, 1, 1, 1)

    def _tiling(self, on: bool) -> None:
        """인코딩(캐시)은 정확도를 위해 타일 없이, 디코딩(샘플)은 VRAM 절약을 위해 타일로."""
        try:
            self.vae.enable_tiling() if on else self.vae.disable_tiling()
        except Exception:
            pass

    @torch.no_grad()
    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        self._tiling(False)
        x = pixels.to(self.device, torch.float32)[:, :, None]  # (B,3,1,H,W)
        z = self.vae.encode(x).latent_dist.mode()
        z = (z - self._latent_mean.to(z)) / self._latent_std.to(z)
        return z[0, :, 0].to("cpu", torch.bfloat16).contiguous()

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> List[Image.Image]:
        self._tiling(self.device.type == "cuda")
        z = latents.to(self.device, torch.float32)[:, :, None]
        z = z * self._latent_std.to(z) + self._latent_mean.to(z)
        img = self.vae.decode(z).sample[:, :, 0]
        return pil_from_decoded(img)

    def unload_vae(self) -> None:
        self.vae = None
        free_memory()

    # ------------------------------------------------------------------
    # 트랜스포머
    # ------------------------------------------------------------------
    def load_transformer(self, quant: str, block_swap: int) -> nn.Module:
        path = resolve_path(self.path_or_default("transformer"), self.models_dir, self.hf_token, log=self.log)
        files = find_safetensors(path)
        if not files:
            raise FileNotFoundError(f"트랜스포머 safetensors 를 찾지 못했습니다: {path}")
        self.is_turbo = "turbo" in os.path.basename(files[0]).lower()
        if self.is_turbo:
            self.log("[알림] Turbo 체크포인트입니다. 공식 권장은 Raw 로 학습 후 Turbo 에서 사용하는 것입니다.")
        self.log(f"Krea 2 DiT 로딩 (quant={quant}, block_swap={block_swap}) : {files[0]}")
        with torch.device("meta"):
            model = SingleStreamDiT(self.config)
        ckpt = LazyCheckpoint(files)
        sources = identity_sources(model_tensor_keys(model), ckpt,
                                   strip_prefixes=("model.diffusion_model.", "diffusion_model."))
        n_blocks = self.config.layers
        swap = max(0, min(block_swap, n_blocks - 1))
        cpu_prefixes = [f"blocks.{i}." for i in range(n_blocks - swap, n_blocks)]
        load_streaming(model, ckpt, sources, self.device, self.dtype, quant, self.quant_exclude,
                       cpu_prefixes=cpu_prefixes, keep_fp32=["*.scale"], log=self.log)
        ckpt.close()
        if swap:
            self.swapper = BlockSwapper([model.blocks[i] for i in range(n_blocks - swap, n_blocks)], self.device)
            self.log(f"블록 스왑: {swap}개 블록을 CPU 메모리에 둡니다.")
        model.gradient_checkpointing = bool(self.cfg.gradient_checkpointing or swap)
        self.transformer = model
        free_memory()
        return model

    def predict(self, latents, t, text, skip_layers: Sequence[int] = ()):
        context, txtmask = text["feats"], text["mask"]
        if context.shape[0] != latents.shape[0]:
            context = context.expand(latents.shape[0], *context.shape[1:])
            txtmask = txtmask.expand(latents.shape[0], -1)
        tokens, pos, mask = prepare_tokens(latents, context.shape[1], self.config.patch, txtmask)
        out = self.transformer(img=tokens.to(self.dtype), context=context.to(self.dtype),
                               t=t.to(self.dtype), pos=pos, mask=mask)
        b, c, h, w = latents.shape
        p = self.config.patch
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h // p, w=w // p, ph=p, pw=p)

    def default_shift(self, latent_hw):
        h, w = latent_hw
        seq = (h // self.config.patch) * (w // self.config.patch)
        return exp_time_shift(mu_for_seq_len(seq))

    @torch.no_grad()
    def sample(self, cond, uncond, width, height, steps, guidance, seed):
        h, w = height // 8, width // 8
        gen = torch.Generator("cpu").manual_seed(seed)
        x = torch.randn(1, self.latent_channels, h, w, generator=gen).to(self.device, self.dtype)
        seq = (h // 2) * (w // 2)
        mu = 1.15 if getattr(self, "is_turbo", False) else mu_for_seq_len(seq)
        ts = torch.linspace(1, 0, steps + 1)
        em = math.exp(mu)
        ts = [em / (em + (1.0 / float(v) - 1.0)) if v > 0 else 0.0 for v in ts]
        use_cfg = guidance > 0 and uncond is not None
        for tc, tp in zip(ts[:-1], ts[1:]):
            t = torch.full((1,), tc, device=self.device, dtype=torch.float32)
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
                v = self.predict(x, t, cond).float()
                if use_cfg:
                    vu = self.predict(x, t, uncond).float()
                    v = v + guidance * (v - vu)  # 공식 정의: cond + g * (cond - uncond)
            x = (x.float() + (tp - tc) * v).to(self.dtype)
        return x
