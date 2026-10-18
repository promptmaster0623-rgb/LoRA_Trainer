"""transformers 모델 로딩 공통 함수 (bitsandbytes 양자화 포함)."""

from __future__ import annotations

from typing import Optional

import torch


def quantization_config(quant: str, compute_dtype: torch.dtype):
    if quant in (None, "", "none", "auto"):
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:  # pragma: no cover
        raise RuntimeError("transformers 버전이 너무 낮습니다.")
    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(f"알 수 없는 텍스트 인코더 양자화: {quant}")


def load_hf_model(cls, path: str, dtype: torch.dtype, device: torch.device, quant: str,
                  subfolder: Optional[str] = None, token: Optional[str] = None, **kwargs):
    qcfg = quantization_config(quant, dtype)
    load_kwargs = dict(token=token, **kwargs)
    if subfolder:
        load_kwargs["subfolder"] = subfolder
    if qcfg is not None:
        if device.type != "cuda":
            raise RuntimeError("int8/nf4 텍스트 인코더 양자화는 CUDA GPU 가 필요합니다.")
        load_kwargs["quantization_config"] = qcfg
        load_kwargs["device_map"] = {"": device.index or 0}
    # transformers 5.x 는 dtype=, 4.x 는 torch_dtype=
    try:
        model = cls.from_pretrained(path, dtype=dtype, **load_kwargs)
    except TypeError:
        model = cls.from_pretrained(path, torch_dtype=dtype, **load_kwargs)
    if qcfg is None:
        model = model.to(device)
    model.eval().requires_grad_(False)
    return model


def drop_vision_tower(model) -> bool:
    """텍스트만 인코딩할 때 비전 타워를 제거해 VRAM 절약. 성공 여부 반환."""
    for owner in (getattr(model, "model", None), model):
        if owner is not None and hasattr(owner, "visual") and owner.visual is not None:
            try:
                owner.visual = None
                return True
            except Exception:
                return False
    return False
