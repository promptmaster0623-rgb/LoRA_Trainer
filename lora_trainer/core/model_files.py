"""베이스 모델별 구성 파일 정의, 경로 상태 확인, 폴더 자동 찾기.

모델 페이지에서 "지금 어떤 모델용 파일을 넣어야 하는지" 를 분명히 보여주기 위해
Krea 2 / MOSS-Image 각각의 구성요소(DiT, 텍스트 인코더, 토크나이저, VAE)를 정의하고,
지정된 파일이 실제로 그 모델용인지 safetensors 헤더 / config.json 으로 가볍게 확인한다.
(가중치는 읽지 않으므로 수십 GB 파일도 즉시 확인됨)
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

ARCH_TITLES = {"krea2": "Krea 2", "moss_image": "MOSS-Image"}

# 구성요소 키 → ModelConfig 필드
FIELD_OF = {
    "transformer": "transformer_path",
    "text_encoder": "text_encoder_path",
    "tokenizer": "tokenizer_path",
    "vae": "vae_path",
}


@dataclass(frozen=True)
class Component:
    key: str            # transformer | text_encoder | tokenizer | vae
    label: str          # 화면 표시 이름
    expect: str         # 무엇을 넣어야 하는지 (사람용 설명)
    mode: str           # file | dir | any
    optional: bool = False


COMPONENTS: Dict[str, List[Component]] = {
    "krea2": [
        Component("transformer", "DiT (트랜스포머)", "Krea 2 Raw 체크포인트 (raw.safetensors, 약 26GB)", "file"),
        Component("text_encoder", "텍스트 인코더", "Qwen3-VL-4B-Instruct 폴더 (config.json 포함)", "dir"),
        Component("vae", "VAE", "Qwen-Image VAE (Qwen/Qwen-Image 의 vae 폴더)", "any"),
    ],
    "moss_image": [
        Component("transformer", "DiT (트랜스포머)", "MOSS-Image Base (moss_image_base_bf16.safetensors 또는 diffusers transformer 폴더)", "any"),
        Component("text_encoder", "텍스트 인코더", "Qwen3.5-9B 폴더 (config.json 포함)", "dir"),
        Component("tokenizer", "토크나이저", "MOSS-Image-Base 의 tokenizer 폴더 (비우면 기본값 자동 다운로드)", "dir", optional=True),
        Component("vae", "VAE", "FLUX.2 VAE (flux2-vae.safetensors)", "any"),
    ],
}

# 상태 코드
OK, DEFAULT, HUB, MISSING, MISMATCH, UNKNOWN = "ok", "default", "hub", "missing", "mismatch", "unknown"


@dataclass
class PathStatus:
    state: str
    message: str

    @property
    def icon(self) -> str:
        return {OK: "✓", DEFAULT: "⬇", HUB: "⬇", MISSING: "✗", MISMATCH: "⚠", UNKNOWN: "•"}[self.state]


# ---------------------------------------------------------------------------
# 파일 식별 (헤더만 읽음)
# ---------------------------------------------------------------------------
def read_safetensors_keys(path: str, limit: int = 4000) -> List[str]:
    """safetensors 헤더의 텐서 이름 목록 (가중치는 읽지 않음)."""
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            return []
        n = struct.unpack("<Q", head)[0]
        if n <= 0 or n > 200 * 1024 * 1024:
            return []
        meta = json.loads(f.read(n).decode("utf-8", errors="ignore"))
    return [k for k in meta.keys() if k != "__metadata__"][:limit]


def classify_keys(keys: List[str]) -> Optional[str]:
    """텐서 이름으로 파일 종류 추정.

    반환: krea2_dit | moss_dit | flux2_vae | qwen_vae | vae | None
    """
    if not keys:
        return None
    joined = "\n".join(keys)
    if ".attn.wq." in joined or "txtfusion" in joined or "layerwise_blocks" in joined:
        return "krea2_dit"
    if re.search(r"(^|\.)blocks\.\d+\..*qkv", joined) and ("text_refiner" in joined or "proj.qkv" in joined or "refiner" in joined):
        return "moss_dit"
    is_vae = any(k.startswith(("encoder.", "decoder.", "vae.")) or "quant_conv" in k for k in keys[:400])
    if is_vae:
        if any(k.startswith("bn.") or ".bn." in k for k in keys):
            return "flux2_vae"
        if "time_conv" in joined or "downsamples" in joined or "residual" in joined:
            return "qwen_vae"
        return "vae"
    if re.search(r"(^|\.)blocks\.\d+\..*qkv", joined):
        return "moss_dit"
    return None


def _dir_kind(path: str) -> Optional[str]:
    """폴더 종류 추정: qwen3_vl | qwen3_5 | qwen_vae | moss_dit | tokenizer | None"""
    cfg = os.path.join(path, "config.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        mt = str(data.get("model_type", "")).lower()
        cls = str(data.get("_class_name", ""))
        if mt.startswith("qwen3_vl"):
            return "qwen3_vl"
        if mt.startswith("qwen3_5") or mt.startswith("qwen3.5"):
            return "qwen3_5"
        if "QwenImage" in cls and "Autoencoder" in cls:
            return "qwen_vae"
        if "Autoencoder" in cls:
            return "vae"
        if "MOSS" in cls.upper() or "Moss" in cls:
            return "moss_dit"
    if os.path.isfile(os.path.join(path, "tokenizer.json")) or os.path.isfile(os.path.join(path, "tokenizer_config.json")):
        return "tokenizer"
    for sub in ("vae", "transformer", "tokenizer"):
        if os.path.isdir(os.path.join(path, sub)):
            return f"repo_with_{sub}"
    return None


EXPECTED = {
    ("krea2", "transformer"): {"krea2_dit"},
    ("krea2", "text_encoder"): {"qwen3_vl"},
    ("krea2", "tokenizer"): {"qwen3_vl", "tokenizer"},
    ("krea2", "vae"): {"qwen_vae", "vae", "repo_with_vae"},
    ("moss_image", "transformer"): {"moss_dit", "repo_with_transformer"},
    ("moss_image", "text_encoder"): {"qwen3_5"},
    ("moss_image", "tokenizer"): {"tokenizer", "qwen3_5", "repo_with_tokenizer"},
    ("moss_image", "vae"): {"flux2_vae", "vae"},
}

KIND_TO_TEXT = {
    "krea2_dit": "Krea 2 DiT",
    "moss_dit": "MOSS-Image DiT",
    "flux2_vae": "FLUX.2 VAE (MOSS-Image용)",
    "qwen_vae": "Qwen-Image VAE (Krea 2용)",
    "qwen3_vl": "Qwen3-VL (Krea 2 텍스트 인코더)",
    "qwen3_5": "Qwen3.5 (MOSS-Image 텍스트 인코더)",
}

# 다른 모델용임이 확실한 조합 → 불일치 경고
WRONG_FAMILY = {
    "krea2": {"moss_dit", "flux2_vae", "qwen3_5"},
    "moss_image": {"krea2_dit", "qwen_vae", "qwen3_vl"},
}


def identify(path: str) -> Optional[str]:
    try:
        if os.path.isdir(path):
            return _dir_kind(path)
        if path.lower().endswith(".safetensors"):
            return classify_keys(read_safetensors_keys(path))
    except Exception:
        return None
    return None


def _looks_like_hub(p: str) -> bool:
    if os.path.isabs(p) or p.startswith((".", "~")) or re.match(r"^[A-Za-z]:", p):
        return False
    return bool(re.match(r"^[\w.\-]+/[\w.\-]+(::[\w.\-/]+|/[\w.\-/]+)?$", p))


def check_path(arch: str, key: str, path: str, default: str = "") -> PathStatus:
    path = (path or "").strip()
    if not path:
        if key == "tokenizer" and arch == "krea2":
            return PathStatus(DEFAULT, "텍스트 인코더 폴더의 토크나이저 사용")
        return PathStatus(DEFAULT, "비어 있음 → 처음 학습할 때 허깅페이스 기본값을 자동으로 내려받습니다")
    if not os.path.exists(path):
        if _looks_like_hub(path):
            return PathStatus(HUB, "허깅페이스에서 자동 다운로드")
        return PathStatus(MISSING, "파일/폴더를 찾을 수 없습니다")
    kind = identify(path)
    title = ARCH_TITLES.get(arch, arch)
    if kind and kind in WRONG_FAMILY.get(arch, set()):
        return PathStatus(MISMATCH, f"{KIND_TO_TEXT.get(kind, kind)} 파일입니다 — {title}용이 아닙니다")
    if kind and kind in EXPECTED.get((arch, key), set()):
        return PathStatus(OK, f"확인됨 · {KIND_TO_TEXT.get(kind, '형식 일치')}")
    if os.path.isfile(path) and not path.lower().endswith(".safetensors"):
        return PathStatus(UNKNOWN, "safetensors 파일이 아닙니다")
    return PathStatus(UNKNOWN, "있음 · 형식 자동 확인 불가 (학습 시 확인)")


# ---------------------------------------------------------------------------
# 폴더 자동 찾기
# ---------------------------------------------------------------------------
NAME_HINTS = {
    ("krea2", "transformer"): [r"krea.*raw", r"^raw\.safetensors$", r"krea[-_ ]?2"],
    ("krea2", "vae"): [r"qwen[-_]?image.*vae", r"^vae$"],
    ("moss_image", "transformer"): [r"moss[-_]?image.*base", r"moss[-_]?image"],
    ("moss_image", "vae"): [r"flux2.*vae", r"flux\.?2.*ae"],
}


def find_models(root: str, arch: str, max_depth: int = 5, max_entries: int = 20000) -> Dict[str, List[str]]:
    """root 아래에서 arch 구성요소 후보를 찾는다. {key: [경로, ...]} (확실한 것 먼저)"""
    found: Dict[str, List[Tuple[int, str]]] = {c.key: [] for c in COMPONENTS[arch]}
    if not os.path.isdir(root):
        return {k: [] for k in found}
    base_depth = root.rstrip("\\/").count(os.sep)
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.count(os.sep) - base_depth
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__")) and d not in ("venv", "node_modules", "output")]
        if depth >= max_depth:
            dirnames[:] = []
        seen += len(filenames) + len(dirnames)
        if seen > max_entries:
            break
        kind = _dir_kind(dirpath) if "config.json" in filenames or "tokenizer.json" in filenames else None
        if kind:
            for c in COMPONENTS[arch]:
                if c.mode != "file" and kind in EXPECTED.get((arch, c.key), set()) and not kind.startswith("repo_"):
                    found[c.key].append((0, dirpath))
        for name in filenames:
            if not name.lower().endswith(".safetensors"):
                continue
            full = os.path.join(dirpath, name)
            low = name.lower()
            for c in COMPONENTS[arch]:
                if c.mode == "dir":
                    continue
                hints = NAME_HINTS.get((arch, c.key), [])
                if not any(re.search(h, low) for h in hints):
                    continue
                k = identify(full)
                if k in EXPECTED.get((arch, c.key), set()):
                    found[c.key].append((0, full))
                elif k is None or k == "vae":
                    found[c.key].append((1, full))
    out = {}
    for key, items in found.items():
        uniq = []
        for _, p in sorted(items, key=lambda x: (x[0], len(x[1]))):
            if p not in uniq:
                uniq.append(p)
        out[key] = uniq
    return out
