"""Krea 2 용 Qwen-Image VAE 로더 (단일 safetensors 파일 지원).

diffusers 의 AutoencoderKLQwenImage 는 from_single_file 을 지원하지 않는다
("FromOriginalModelMixin is currently only compatible with ..." 오류).
그래서 config 로 모델을 직접 만들고 state dict 를 넣는다.

지원 형식
  - diffusers 폴더      : <폴더>/config.json + diffusion_pytorch_model.safetensors
  - diffusers 단일 파일 : diffusion_pytorch_model.safetensors (config.json 이 옆에 없어도 됨)
  - ComfyUI 단일 파일   : Comfy-Org/Krea-2 vae/qwen_image_vae.safetensors (Wan 형식 키 → 자동 변환)

ComfyUI → diffusers 키 변환표는 musubi-tuner qwen_image_utils.convert_comfyui_state_dict
(Apache-2.0, Fizgig src/fizgig/krea2/vae_loader.py 경유) 에서 가져왔다.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import torch

# Qwen/Qwen-Image vae/config.json (Fizgig 와 동일 값). 옆에 config.json 이 없을 때 사용.
QWEN_IMAGE_VAE_CONFIG = {
    "attn_scales": [],
    "base_dim": 96,
    "dim_mult": [1, 2, 4, 4],
    "dropout": 0.0,
    "latents_mean": [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                     0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921],
    "latents_std": [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916],
    "num_res_blocks": 2,
    "temperal_downsample": [False, True, True],
    "z_dim": 16,
}

# ComfyUI(Wan 형식) 모듈 이름 -> diffusers 모듈 이름 (가중치 접미사 .weight/.bias/.gamma 제외)
COMFY_TO_DIFFUSERS = {
    "conv1": "quant_conv",
    "conv2": "post_quant_conv",
    "decoder.conv1": "decoder.conv_in",
    "decoder.head.0": "decoder.norm_out",
    "decoder.head.2": "decoder.conv_out",
    "decoder.middle.0.residual.0": "decoder.mid_block.resnets.0.norm1",
    "decoder.middle.0.residual.2": "decoder.mid_block.resnets.0.conv1",
    "decoder.middle.0.residual.3": "decoder.mid_block.resnets.0.norm2",
    "decoder.middle.0.residual.6": "decoder.mid_block.resnets.0.conv2",
    "decoder.middle.1.norm": "decoder.mid_block.attentions.0.norm",
    "decoder.middle.1.proj": "decoder.mid_block.attentions.0.proj",
    "decoder.middle.1.to_qkv": "decoder.mid_block.attentions.0.to_qkv",
    "decoder.middle.2.residual.0": "decoder.mid_block.resnets.1.norm1",
    "decoder.middle.2.residual.2": "decoder.mid_block.resnets.1.conv1",
    "decoder.middle.2.residual.3": "decoder.mid_block.resnets.1.norm2",
    "decoder.middle.2.residual.6": "decoder.mid_block.resnets.1.conv2",
    "decoder.upsamples.0.residual.0": "decoder.up_blocks.0.resnets.0.norm1",
    "decoder.upsamples.0.residual.2": "decoder.up_blocks.0.resnets.0.conv1",
    "decoder.upsamples.0.residual.3": "decoder.up_blocks.0.resnets.0.norm2",
    "decoder.upsamples.0.residual.6": "decoder.up_blocks.0.resnets.0.conv2",
    "decoder.upsamples.1.residual.0": "decoder.up_blocks.0.resnets.1.norm1",
    "decoder.upsamples.1.residual.2": "decoder.up_blocks.0.resnets.1.conv1",
    "decoder.upsamples.1.residual.3": "decoder.up_blocks.0.resnets.1.norm2",
    "decoder.upsamples.1.residual.6": "decoder.up_blocks.0.resnets.1.conv2",
    "decoder.upsamples.10.residual.0": "decoder.up_blocks.2.resnets.2.norm1",
    "decoder.upsamples.10.residual.2": "decoder.up_blocks.2.resnets.2.conv1",
    "decoder.upsamples.10.residual.3": "decoder.up_blocks.2.resnets.2.norm2",
    "decoder.upsamples.10.residual.6": "decoder.up_blocks.2.resnets.2.conv2",
    "decoder.upsamples.11.resample.1": "decoder.up_blocks.2.upsamplers.0.resample.1",
    "decoder.upsamples.12.residual.0": "decoder.up_blocks.3.resnets.0.norm1",
    "decoder.upsamples.12.residual.2": "decoder.up_blocks.3.resnets.0.conv1",
    "decoder.upsamples.12.residual.3": "decoder.up_blocks.3.resnets.0.norm2",
    "decoder.upsamples.12.residual.6": "decoder.up_blocks.3.resnets.0.conv2",
    "decoder.upsamples.13.residual.0": "decoder.up_blocks.3.resnets.1.norm1",
    "decoder.upsamples.13.residual.2": "decoder.up_blocks.3.resnets.1.conv1",
    "decoder.upsamples.13.residual.3": "decoder.up_blocks.3.resnets.1.norm2",
    "decoder.upsamples.13.residual.6": "decoder.up_blocks.3.resnets.1.conv2",
    "decoder.upsamples.14.residual.0": "decoder.up_blocks.3.resnets.2.norm1",
    "decoder.upsamples.14.residual.2": "decoder.up_blocks.3.resnets.2.conv1",
    "decoder.upsamples.14.residual.3": "decoder.up_blocks.3.resnets.2.norm2",
    "decoder.upsamples.14.residual.6": "decoder.up_blocks.3.resnets.2.conv2",
    "decoder.upsamples.2.residual.0": "decoder.up_blocks.0.resnets.2.norm1",
    "decoder.upsamples.2.residual.2": "decoder.up_blocks.0.resnets.2.conv1",
    "decoder.upsamples.2.residual.3": "decoder.up_blocks.0.resnets.2.norm2",
    "decoder.upsamples.2.residual.6": "decoder.up_blocks.0.resnets.2.conv2",
    "decoder.upsamples.3.resample.1": "decoder.up_blocks.0.upsamplers.0.resample.1",
    "decoder.upsamples.3.time_conv": "decoder.up_blocks.0.upsamplers.0.time_conv",
    "decoder.upsamples.4.residual.0": "decoder.up_blocks.1.resnets.0.norm1",
    "decoder.upsamples.4.residual.2": "decoder.up_blocks.1.resnets.0.conv1",
    "decoder.upsamples.4.residual.3": "decoder.up_blocks.1.resnets.0.norm2",
    "decoder.upsamples.4.residual.6": "decoder.up_blocks.1.resnets.0.conv2",
    "decoder.upsamples.4.shortcut": "decoder.up_blocks.1.resnets.0.conv_shortcut",
    "decoder.upsamples.5.residual.0": "decoder.up_blocks.1.resnets.1.norm1",
    "decoder.upsamples.5.residual.2": "decoder.up_blocks.1.resnets.1.conv1",
    "decoder.upsamples.5.residual.3": "decoder.up_blocks.1.resnets.1.norm2",
    "decoder.upsamples.5.residual.6": "decoder.up_blocks.1.resnets.1.conv2",
    "decoder.upsamples.6.residual.0": "decoder.up_blocks.1.resnets.2.norm1",
    "decoder.upsamples.6.residual.2": "decoder.up_blocks.1.resnets.2.conv1",
    "decoder.upsamples.6.residual.3": "decoder.up_blocks.1.resnets.2.norm2",
    "decoder.upsamples.6.residual.6": "decoder.up_blocks.1.resnets.2.conv2",
    "decoder.upsamples.7.resample.1": "decoder.up_blocks.1.upsamplers.0.resample.1",
    "decoder.upsamples.7.time_conv": "decoder.up_blocks.1.upsamplers.0.time_conv",
    "decoder.upsamples.8.residual.0": "decoder.up_blocks.2.resnets.0.norm1",
    "decoder.upsamples.8.residual.2": "decoder.up_blocks.2.resnets.0.conv1",
    "decoder.upsamples.8.residual.3": "decoder.up_blocks.2.resnets.0.norm2",
    "decoder.upsamples.8.residual.6": "decoder.up_blocks.2.resnets.0.conv2",
    "decoder.upsamples.9.residual.0": "decoder.up_blocks.2.resnets.1.norm1",
    "decoder.upsamples.9.residual.2": "decoder.up_blocks.2.resnets.1.conv1",
    "decoder.upsamples.9.residual.3": "decoder.up_blocks.2.resnets.1.norm2",
    "decoder.upsamples.9.residual.6": "decoder.up_blocks.2.resnets.1.conv2",
    "encoder.conv1": "encoder.conv_in",
    "encoder.downsamples.0.residual.0": "encoder.down_blocks.0.norm1",
    "encoder.downsamples.0.residual.2": "encoder.down_blocks.0.conv1",
    "encoder.downsamples.0.residual.3": "encoder.down_blocks.0.norm2",
    "encoder.downsamples.0.residual.6": "encoder.down_blocks.0.conv2",
    "encoder.downsamples.1.residual.0": "encoder.down_blocks.1.norm1",
    "encoder.downsamples.1.residual.2": "encoder.down_blocks.1.conv1",
    "encoder.downsamples.1.residual.3": "encoder.down_blocks.1.norm2",
    "encoder.downsamples.1.residual.6": "encoder.down_blocks.1.conv2",
    "encoder.downsamples.10.residual.0": "encoder.down_blocks.10.norm1",
    "encoder.downsamples.10.residual.2": "encoder.down_blocks.10.conv1",
    "encoder.downsamples.10.residual.3": "encoder.down_blocks.10.norm2",
    "encoder.downsamples.10.residual.6": "encoder.down_blocks.10.conv2",
    "encoder.downsamples.2.resample.1": "encoder.down_blocks.2.resample.1",
    "encoder.downsamples.3.residual.0": "encoder.down_blocks.3.norm1",
    "encoder.downsamples.3.residual.2": "encoder.down_blocks.3.conv1",
    "encoder.downsamples.3.residual.3": "encoder.down_blocks.3.norm2",
    "encoder.downsamples.3.residual.6": "encoder.down_blocks.3.conv2",
    "encoder.downsamples.3.shortcut": "encoder.down_blocks.3.conv_shortcut",
    "encoder.downsamples.4.residual.0": "encoder.down_blocks.4.norm1",
    "encoder.downsamples.4.residual.2": "encoder.down_blocks.4.conv1",
    "encoder.downsamples.4.residual.3": "encoder.down_blocks.4.norm2",
    "encoder.downsamples.4.residual.6": "encoder.down_blocks.4.conv2",
    "encoder.downsamples.5.resample.1": "encoder.down_blocks.5.resample.1",
    "encoder.downsamples.5.time_conv": "encoder.down_blocks.5.time_conv",
    "encoder.downsamples.6.residual.0": "encoder.down_blocks.6.norm1",
    "encoder.downsamples.6.residual.2": "encoder.down_blocks.6.conv1",
    "encoder.downsamples.6.residual.3": "encoder.down_blocks.6.norm2",
    "encoder.downsamples.6.residual.6": "encoder.down_blocks.6.conv2",
    "encoder.downsamples.6.shortcut": "encoder.down_blocks.6.conv_shortcut",
    "encoder.downsamples.7.residual.0": "encoder.down_blocks.7.norm1",
    "encoder.downsamples.7.residual.2": "encoder.down_blocks.7.conv1",
    "encoder.downsamples.7.residual.3": "encoder.down_blocks.7.norm2",
    "encoder.downsamples.7.residual.6": "encoder.down_blocks.7.conv2",
    "encoder.downsamples.8.resample.1": "encoder.down_blocks.8.resample.1",
    "encoder.downsamples.8.time_conv": "encoder.down_blocks.8.time_conv",
    "encoder.downsamples.9.residual.0": "encoder.down_blocks.9.norm1",
    "encoder.downsamples.9.residual.2": "encoder.down_blocks.9.conv1",
    "encoder.downsamples.9.residual.3": "encoder.down_blocks.9.norm2",
    "encoder.downsamples.9.residual.6": "encoder.down_blocks.9.conv2",
    "encoder.head.0": "encoder.norm_out",
    "encoder.head.2": "encoder.conv_out",
    "encoder.middle.0.residual.0": "encoder.mid_block.resnets.0.norm1",
    "encoder.middle.0.residual.2": "encoder.mid_block.resnets.0.conv1",
    "encoder.middle.0.residual.3": "encoder.mid_block.resnets.0.norm2",
    "encoder.middle.0.residual.6": "encoder.mid_block.resnets.0.conv2",
    "encoder.middle.1.norm": "encoder.mid_block.attentions.0.norm",
    "encoder.middle.1.proj": "encoder.mid_block.attentions.0.proj",
    "encoder.middle.1.to_qkv": "encoder.mid_block.attentions.0.to_qkv",
    "encoder.middle.2.residual.0": "encoder.mid_block.resnets.1.norm1",
    "encoder.middle.2.residual.2": "encoder.mid_block.resnets.1.conv1",
    "encoder.middle.2.residual.3": "encoder.mid_block.resnets.1.norm2",
    "encoder.middle.2.residual.6": "encoder.mid_block.resnets.1.conv2",
}

_STRIP_PREFIXES = ("first_stage_model.", "vae.", "model.")


def is_comfy_vae(sd: Dict[str, torch.Tensor]) -> bool:
    """ComfyUI/Wan 형식이면 True (conv1 = quant_conv 이름)."""
    return "conv1.bias" in sd or "encoder.conv1.weight" in sd


def convert_comfy_vae(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """ComfyUI(Wan) 키를 diffusers AutoencoderKLQwenImage 키로 바꾼다."""
    out = {}
    for key, value in sd.items():
        module, _, suffix = key.rpartition(".")
        new = COMFY_TO_DIFFUSERS.get(module)
        out[f"{new}.{suffix}" if new else key] = value
    return out


def _strip(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    for p in _STRIP_PREFIXES:
        if sd and all(k.startswith(p) for k in sd):
            return {k[len(p):]: v for k, v in sd.items()}
    return sd


def _read_config(path: str) -> dict:
    """파일 옆(또는 폴더 안)의 config.json, 없으면 내장값."""
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    cfg_path = os.path.join(folder, "config.json")
    cfg = dict(QWEN_IMAGE_VAE_CONFIG)
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("_class_name", "AutoencoderKLQwenImage") == "AutoencoderKLQwenImage":
                cfg.update({k: v for k, v in data.items() if not k.startswith("_") and k in cfg})
        except (OSError, ValueError):
            pass
    return cfg


def _weight_file(path: str) -> str:
    if os.path.isfile(path):
        return path
    for name in ("diffusion_pytorch_model.safetensors", "qwen_image_vae.safetensors"):
        p = os.path.join(path, name)
        if os.path.isfile(p):
            return p
    files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"VAE safetensors 를 찾지 못했습니다: {path}")
    return os.path.join(path, files[0])


def load_qwen_image_vae(path: str, dtype: torch.dtype = torch.float32, log=None):
    """경로(폴더 또는 단일 파일) -> diffusers AutoencoderKLQwenImage (eval, grad 끔)."""
    from diffusers import AutoencoderKLQwenImage
    from safetensors.torch import load_file

    wfile = _weight_file(path)
    sd = _strip(load_file(wfile))
    comfy = is_comfy_vae(sd)
    if comfy:
        sd = convert_comfy_vae(sd)
    cfg = _read_config(path)
    vae = AutoencoderKLQwenImage(**cfg)
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(
            f"Qwen-Image VAE 가 아니거나 형식을 알 수 없는 파일입니다: {wfile}\n"
            f"누락된 가중치 {len(missing)}개 (예: {missing[:3]}). "
            "Krea 2 는 Qwen-Image VAE (qwen_image_vae.safetensors 또는 Qwen/Qwen-Image::vae) 가 필요합니다."
        )
    if log:
        kind = "ComfyUI 형식 → 자동 변환" if comfy else "diffusers 형식"
        log(f"  VAE 형식: {kind}" + (f", 무시된 키 {len(unexpected)}개" if unexpected else ""))
    return vae.to(dtype).eval().requires_grad_(False)
