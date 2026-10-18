"""베이스 모델 어댑터 레지스트리."""

from __future__ import annotations

ARCHS = {
    "krea2": "Krea 2 (12.9B · Qwen3-VL-4B · Qwen-Image VAE)",
    "moss_image": "MOSS-Image (5B · Qwen3.5-9B · FLUX.2 VAE)",
}


def get_adapter_class(arch: str):
    if arch == "krea2":
        from .krea2.adapter import Krea2Adapter

        return Krea2Adapter
    if arch == "moss_image":
        from .moss_image.adapter import MossImageAdapter

        return MossImageAdapter
    raise ValueError(f"지원하지 않는 모델: {arch}")
