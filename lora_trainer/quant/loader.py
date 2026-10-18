"""safetensors 스트리밍 로더.

거대한 DiT(Krea 2 = 26GB bf16)를 시스템 RAM 에 통째로 올리지 않도록
meta 디바이스에서 모델 뼈대를 만든 뒤, 텐서를 하나씩 읽어
  (1) 필요 시 dtype 변환 / ComfyUI scaled-fp8 역양자화
  (2) 대상 Linear 는 FP8 / NF4 로 양자화
  (3) 지정된 디바이스(GPU 또는 블록스왑용 CPU pinned 메모리)로 배치
하는 방식으로 로드한다.
"""

from __future__ import annotations

import fnmatch
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
from safetensors import safe_open

from .layers import FP8Linear, make_nf4_linear

SourceMap = Dict[str, List[str]]


class LazyCheckpoint:
    """여러 safetensors 파일을 하나의 키 공간처럼 지연 접근."""

    def __init__(self, files: Sequence[str]):
        self.files = list(files)
        self._handles = [safe_open(f, framework="pt", device="cpu") for f in self.files]
        self._where: Dict[str, int] = {}
        for i, h in enumerate(self._handles):
            for k in h.keys():
                self._where[k] = i

    def keys(self) -> List[str]:
        return list(self._where.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._where

    def get(self, key: str) -> torch.Tensor:
        t = self._handles[self._where[key]].get_tensor(key)
        # ComfyUI "scaled fp8" 파일: <name>.weight(fp8) * <name>.scale_weight
        if t.dtype in _fp8_dtypes() and key.endswith(".weight"):
            scale_key = key[: -len("weight")] + "scale_weight"
            t = t.to(torch.float32)
            if scale_key in self._where:
                s = self._handles[self._where[scale_key]].get_tensor(scale_key).to(torch.float32)
                t = t * s if s.numel() == 1 else t * s.reshape(-1, *([1] * (t.dim() - 1)))
        return t

    def close(self) -> None:
        self._handles = []


def _fp8_dtypes():
    out = []
    for name in ("float8_e4m3fn", "float8_e5m2"):
        if hasattr(torch, name):
            out.append(getattr(torch, name))
    return tuple(out)


def _match_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _set_tensor(module: nn.Module, name: str, tensor: torch.Tensor, is_buffer: bool) -> None:
    if is_buffer:
        module._buffers[name] = tensor
    else:
        module._parameters[name] = nn.Parameter(tensor, requires_grad=False)


def load_streaming(
    model: nn.Module,
    ckpt: LazyCheckpoint,
    sources: SourceMap,
    device: torch.device,
    dtype: torch.dtype,
    quant: str = "none",
    quant_exclude: Sequence[str] = (),
    cpu_prefixes: Sequence[str] = (),
    keep_fp32: Sequence[str] = (),
    log: Optional[Callable[[str], None]] = None,
) -> nn.Module:
    """meta 위에 만들어진 model 에 가중치를 채운다.

    sources  : 모델 키 -> 체크포인트 키 목록 (2개 이상이면 dim0 으로 이어붙임, 예: q/k/v 결합)
    quant    : none | fp8 | nf4 (nn.Linear 만, quant_exclude 패턴 제외)
    cpu_prefixes : 이 접두사로 시작하는 파라미터는 CPU(pinned)에 둔다 (블록 스왑용)
    keep_fp32    : fp32 로 유지할 파라미터 패턴
    """
    linear_names = {
        name for name, m in model.named_modules() if type(m) is nn.Linear
    }
    handled = set()
    total = len(linear_names)
    done_linear = 0

    def on_cpu(name: str) -> bool:
        return any(name.startswith(p) for p in cpu_prefixes)

    def fetch(key: str) -> torch.Tensor:
        if key not in sources:
            raise KeyError(f"체크포인트에 '{key}' 가 없습니다.")
        parts = [ckpt.get(k) for k in sources[key]]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    # 1) Linear 레이어: 양자화 대상이면 통째로 교체
    for mod_name in sorted(linear_names):
        parent_name, _, child = mod_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        lin: nn.Linear = getattr(parent, child)
        w_key, b_key = f"{mod_name}.weight", f"{mod_name}.bias"
        weight = fetch(w_key)
        bias = fetch(b_key) if lin.bias is not None else None
        target_dev = torch.device("cpu") if on_cpu(mod_name + ".") else device
        do_quant = quant in ("fp8", "nf4") and not _match_any(mod_name, quant_exclude)
        if do_quant and quant == "fp8":
            new = FP8Linear.from_weight(weight, bias, dtype, target_dev)
        elif do_quant and quant == "nf4":
            if target_dev.type != "cuda":
                raise RuntimeError("NF4 양자화는 CUDA GPU 와 블록스왑 0 에서만 사용할 수 있습니다.")
            new = make_nf4_linear(weight, bias, dtype, target_dev)
        else:
            new = nn.Linear(lin.in_features, lin.out_features, bias=lin.bias is not None, device="meta")
            _set_tensor(new, "weight", weight.to(target_dev, dtype), False)
            if bias is not None:
                _set_tensor(new, "bias", bias.to(target_dev, dtype), False)
        setattr(parent, child, new)
        handled.update({w_key, b_key})
        del weight, bias
        done_linear += 1
        if log and total and done_linear % max(1, total // 10) == 0:
            log(f"  가중치 로딩 {done_linear}/{total}")

    # 2) 나머지 파라미터/버퍼
    for mod_name, module in model.named_modules():
        for pname, p in list(module._parameters.items()):
            full = f"{mod_name}.{pname}" if mod_name else pname
            if full in handled or p is None:
                continue
            if not p.is_meta:
                continue
            t = fetch(full)
            tdtype = torch.float32 if _match_any(full, keep_fp32) else dtype
            target_dev = torch.device("cpu") if on_cpu(full) else device
            _set_tensor(module, pname, t.to(target_dev, tdtype), False)
        for bname, b in list(module._buffers.items()):
            full = f"{mod_name}.{bname}" if mod_name else bname
            if b is None or not b.is_meta:
                continue
            if full in sources:
                _set_tensor(module, bname, fetch(full).to(device), True)
            else:
                raise KeyError(f"버퍼 '{full}' 값을 찾을 수 없습니다.")

    leftover = [n for n, p in model.named_parameters() if p.is_meta]
    if leftover:
        raise RuntimeError(f"로딩되지 않은 파라미터가 있습니다: {leftover[:5]} ...")
    model.requires_grad_(False)
    return model


def identity_sources(model_keys: Iterable[str], ckpt: LazyCheckpoint, strip_prefixes=()) -> SourceMap:
    """모델 키와 체크포인트 키가 (접두사 제외) 같은 경우의 기본 매핑."""
    available = set(ckpt.keys())
    stripped = {}
    for k in available:
        kk = k
        for p in strip_prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
                break
        stripped[kk] = k
    out: SourceMap = {}
    for k in model_keys:
        if k in stripped:
            out[k] = [stripped[k]]
    return out


def model_tensor_keys(model: nn.Module) -> List[str]:
    keys = [n for n, _ in model.named_parameters()]
    keys += [n for n, b in model.named_buffers() if b is not None]
    return keys
