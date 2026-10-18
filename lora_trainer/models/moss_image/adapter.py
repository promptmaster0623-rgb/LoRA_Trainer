"""MOSS-Image 어댑터.

구성
  - DiT     : MOSS-Image 5B (Comfy-Org 레이아웃, qkv 결합. 공식 diffusers 샤드도 로딩 시 결합)
  - 텍스트  : Qwen3.5-9B 의 최종 hidden state (채팅 템플릿, thinking on, 프롬프트별 패딩 없이 인코딩)
  - VAE     : FLUX.2 VAE (2x2 패킹 128ch, BatchNorm 정규화)
권장: Base 로 학습 (SFT 에서도 동작). 샘플링: shift 5, CFG 3, 네거티브 분기에서 22번 블록 스킵.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file

from ...core.device import free_memory
from ...quant.loader import LazyCheckpoint, load_streaming, model_tensor_keys
from ...quant.offload import BlockSwapper
from ...utils.hub import find_safetensors, resolve_path
from ..base import ModelAdapter, pil_from_decoded
from ..hf_loading import drop_vision_tower, load_hf_model
from .transformer import MossImageParams, MossImageTransformer, moss_sources
from .vae_flux2 import load_flux2_vae

MAX_TEXT_TOKENS = 4096
SAMPLE_SHIFT = 5.0
UNCOND_SKIP_LAYERS = (22,)
NUM_TRAIN_TIMESTEPS = 1000


def moss_sigmas(num_steps: int, shift: float = SAMPLE_SHIFT, total: int = NUM_TRAIN_TIMESTEPS) -> torch.Tensor:
    """공식 스케줄러 sigma (shift 격자를 학습 격자에 스냅, 끝에 0 추가)."""

    def grid(count, s):
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(num_steps, shift)
    training = grid(total, 1.0).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


class MossImageAdapter(ModelAdapter):
    arch = "moss_image"
    display_name = "MOSS-Image (5B)"
    latent_channels = 128
    spatial_divisor = 16
    block_attr = "layers"
    default_guidance = 3.0
    default_steps = 30
    defaults = {
        "transformer": "Comfy-Org/MOSS-Image/diffusion_models/moss_image_base_bf16.safetensors",
        "text_encoder": "Qwen/Qwen3.5-9B",
        "tokenizer": "OpenMOSS-Team/MOSS-Image-Base::tokenizer",
        "vae": "Comfy-Org/MOSS-Image/vae/flux2-vae.safetensors",
    }
    lora_presets = {
        "attn": [r"layers\.\d+\.proj\.(qkv|o)"],
        "attn_mlp": [r"layers\.\d+\.proj\.(qkv|o|ffn_up_gate|ffn_down)"],
        "all_linear": [
            r"layers\.\d+\.proj\.(qkv|o|ffn_up_gate|ffn_down)",
            r"text_embedder\.blocks\.\d+\.(qkv|proj|ffn_gate|ffn_up|ffn_down)",
        ],
    }
    quant_exclude = [
        "video_embedder", "text_embedder*", "time_embedding*",
        "shared_block_modulation", "video_head_modulation", "final_linear_video",
    ]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.params = MossImageParams()
        self.text_model = None
        self.tokenizer = None
        self.vae = None

    # ------------------------------------------------------------------
    # 텍스트 인코더
    # ------------------------------------------------------------------
    def load_text_encoder(self, quant: str) -> None:
        from transformers import AutoModel, AutoTokenizer

        te_path = resolve_path(self.path_or_default("text_encoder"), self.models_dir, self.hf_token,
                               allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"], log=self.log)
        tok_path = resolve_path(self.path_or_default("tokenizer"), self.models_dir, self.hf_token, log=self.log)
        self.log(f"Qwen3.5 텍스트 인코더 로딩 ({quant}) : {te_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        except Exception:
            from transformers import Qwen2Tokenizer

            self.tokenizer = Qwen2Tokenizer.from_pretrained(tok_path)
        if getattr(self.tokenizer, "chat_template", None) is None:
            raise RuntimeError(f"토크나이저에 채팅 템플릿이 없습니다: {tok_path}")
        self.text_model = load_hf_model(AutoModel, te_path, self.dtype, self.device, quant)
        if drop_vision_tower(self.text_model):
            free_memory()

    def _chat(self, content: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        dev = next(self.text_model.parameters()).device
        ids = self.tokenizer(self._chat(prompt), return_tensors="pt", truncation=True,
                             max_length=MAX_TEXT_TOKENS)["input_ids"].to(dev)
        out = self.text_model(input_ids=ids, use_cache=False)
        states = out.last_hidden_state[0]
        return {"embeds": states.to("cpu", torch.bfloat16).contiguous()}

    def unload_text_encoder(self) -> None:
        self.text_model = None
        free_memory()

    def collate_text(self, items: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        lengths = [it["embeds"].shape[0] for it in items]
        lmax = max(lengths)
        d = items[0]["embeds"].shape[1]
        emb = torch.zeros(len(items), lmax, d, dtype=self.dtype)
        mask = torch.zeros(len(items), lmax, dtype=torch.bool)
        for i, it in enumerate(items):
            emb[i, : lengths[i]] = it["embeds"].to(self.dtype)
            mask[i, : lengths[i]] = True
        return {"embeds": emb.to(self.device), "mask": mask.to(self.device)}

    # ------------------------------------------------------------------
    # VAE
    # ------------------------------------------------------------------
    def load_vae(self) -> None:
        path = resolve_path(self.path_or_default("vae"), self.models_dir, self.hf_token, log=self.log)
        files = find_safetensors(path)
        if not files:
            raise FileNotFoundError(f"VAE safetensors 를 찾지 못했습니다: {path}")
        self.log(f"FLUX.2 VAE 로딩 : {files[0]}")
        sd = {}
        for f in files:
            sd.update(load_file(f))
        self.vae = load_flux2_vae(sd, torch.float32).to(self.device)

    @torch.no_grad()
    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(pixels.to(self.device, torch.float32))
        return z[0].to("cpu", torch.bfloat16).contiguous()

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> List[Image.Image]:
        img = self.vae.decode(latents.to(self.device, torch.float32))
        return pil_from_decoded(img)

    def unload_vae(self) -> None:
        self.vae = None
        free_memory()

    # ------------------------------------------------------------------
    # 트랜스포머
    # ------------------------------------------------------------------
    def load_transformer(self, quant: str, block_swap: int) -> nn.Module:
        path = resolve_path(self.path_or_default("transformer"), self.models_dir, self.hf_token, log=self.log)
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, "transformer")):
            path = os.path.join(path, "transformer")
        files = find_safetensors(path)
        if not files:
            raise FileNotFoundError(f"트랜스포머 safetensors 를 찾지 못했습니다: {path}")
        self.log(f"MOSS-Image DiT 로딩 (quant={quant}, block_swap={block_swap}) : {path}")
        with torch.device("meta"):
            model = MossImageTransformer(self.params)
        ckpt = LazyCheckpoint(files)
        sources = moss_sources(model_tensor_keys(model), ckpt)
        n = self.params.num_layers
        swap = max(0, min(block_swap, n - 1))
        cpu_prefixes = [f"layers.{i}." for i in range(n - swap, n)]
        load_streaming(model, ckpt, sources, self.device, self.dtype, quant, self.quant_exclude,
                       cpu_prefixes=cpu_prefixes, log=self.log)
        ckpt.close()
        if swap:
            self.swapper = BlockSwapper([model.layers[i] for i in range(n - swap, n)], self.device)
            self.log(f"블록 스왑: {swap}개 블록을 CPU 메모리에 둡니다.")
        model.gradient_checkpointing = bool(self.cfg.gradient_checkpointing or swap)
        self.transformer = model
        free_memory()
        return model

    def predict(self, latents, t, text, skip_layers: Sequence[int] = ()):
        emb, mask = text["embeds"], text["mask"]
        if emb.shape[0] != latents.shape[0]:
            emb = emb.expand(latents.shape[0], *emb.shape[1:])
            mask = mask.expand(latents.shape[0], -1)
        timestep = t.float() * NUM_TRAIN_TIMESTEPS
        return self.transformer(latents.to(self.dtype), timestep, emb, mask, skip_layers=skip_layers)

    def default_shift(self, latent_hw):
        s = 3.0
        return lambda t: s * t / (1 + (s - 1) * t)

    @torch.no_grad()
    def sample(self, cond, uncond, width, height, steps, guidance, seed):
        gen = torch.Generator("cpu").manual_seed(seed)
        x = torch.randn(1, self.latent_channels, height // 16, width // 16, generator=gen,
                        dtype=torch.float32).to(self.device)
        sigmas = moss_sigmas(steps).to(self.device)
        use_cfg = uncond is not None and guidance != 1.0 and guidance > 0
        for i in range(len(sigmas) - 1):
            s, sn = sigmas[i], sigmas[i + 1]
            t = s.reshape(1)
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
                v = self.predict(x, t, cond).float()
                if use_cfg:
                    vu = self.predict(x, t, uncond, skip_layers=UNCOND_SKIP_LAYERS).float()
                    v = vu + guidance * (v - vu)
            x = x + (sn - s) * v
        return x
