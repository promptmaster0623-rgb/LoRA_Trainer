"""데이터셋 스캔, 캡션 처리, 종횡비 버킷.

폴더 구조 예시
  dataset/
    img001.png
    img001.txt      <- 캡션 (같은 이름, 확장자만 다름)
    img002.jpg
    ...
캡션에 [trigger] 가 있으면 트리거 단어로 치환, 없으면 앞에 "트리거, " 를 붙인다.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageOps

from ..core.config import DatasetConfig, DatasetFolder

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
TRIGGER_TOKEN = "[trigger]"


@dataclass
class ImageItem:
    path: str
    caption: str
    width: int
    height: int
    repeats: int = 1


@dataclass
class Sample:
    """학습 단위 (해상도/반전 조합별 하나)."""

    item: ImageItem
    bucket: Tuple[int, int]      # (W, H)
    flip: bool
    caption: str


def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    out = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(IMAGE_EXTS):
            out.append(os.path.join(folder, name))
    return out


def caption_path(image_path: str, ext: str = ".txt") -> str:
    return os.path.splitext(image_path)[0] + ext


def read_caption(image_path: str, ext: str = ".txt", default: str = "") -> str:
    p = caption_path(image_path, ext)
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    return default


def write_caption(image_path: str, text: str, ext: str = ".txt") -> None:
    with open(caption_path(image_path, ext), "w", encoding="utf-8") as f:
        f.write(text.strip())


def apply_trigger(caption: str, trigger: str) -> str:
    caption = caption.strip()
    trigger = trigger.strip()
    if TRIGGER_TOKEN in caption:
        return caption.replace(TRIGGER_TOKEN, trigger).strip(" ,")
    if trigger and trigger.lower() not in caption.lower():
        return f"{trigger}, {caption}" if caption else trigger
    return caption


def image_size(path: str) -> Tuple[int, int]:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.size


# ---------------------------------------------------------------------------
# 버킷
# ---------------------------------------------------------------------------
def make_buckets(resolution: int, step: int, vmin: int, vmax: int, divisor: int = 16) -> List[Tuple[int, int]]:
    """면적 ≈ resolution² 인 (W, H) 후보 목록."""
    step = max(divisor, (step // divisor) * divisor)
    area = resolution * resolution
    buckets = set()
    w = vmin
    while w <= vmax:
        h = int(area / w // step) * step
        if vmin <= h <= vmax:
            buckets.add((w, h))
        w += step
    buckets.add((resolution, resolution))
    return sorted(buckets)


def pick_bucket(width: int, height: int, buckets: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    aspect = math.log(width / height)
    return min(buckets, key=lambda b: abs(math.log(b[0] / b[1]) - aspect))


def load_and_fit(path: str, bucket: Tuple[int, int], flip: bool = False) -> Image.Image:
    """비율 유지 확대/축소 후 중앙 크롭으로 버킷 크기에 맞춤."""
    bw, bh = bucket
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
        scale = max(bw / im.width, bh / im.height)
        nw, nh = max(bw, round(im.width * scale)), max(bh, round(im.height * scale))
        im = im.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - bw) // 2, (nh - bh) // 2
        im = im.crop((left, top, left + bw, top + bh))
        if flip:
            im = ImageOps.mirror(im)
        return im


# ---------------------------------------------------------------------------
# 스캔
# ---------------------------------------------------------------------------
def scan_dataset(cfg: DatasetConfig, divisor: int = 16) -> Tuple[List[ImageItem], List[Sample]]:
    items: List[ImageItem] = []
    for folder in cfg.folders:
        trigger = folder.trigger_word or cfg.trigger_word
        for path in list_images(folder.path):
            try:
                w, h = image_size(path)
            except Exception:
                continue
            raw = read_caption(path, folder.caption_ext or ".txt", cfg.default_caption)
            items.append(ImageItem(path, apply_trigger(raw, trigger), w, h, max(1, folder.repeats)))

    samples: List[Sample] = []
    bucket_sets: Dict[int, List[Tuple[int, int]]] = {
        r: make_buckets(r, cfg.bucket_step, min(cfg.bucket_min, r), max(cfg.bucket_max, r), divisor)
        for r in cfg.resolutions
    }
    for it in items:
        for r, buckets in bucket_sets.items():
            b = pick_bucket(it.width, it.height, buckets)
            samples.append(Sample(it, b, False, it.caption))
            if cfg.flip_horizontal:
                samples.append(Sample(it, b, True, it.caption))
    return items, samples


def dataset_summary(items: List[ImageItem], samples: List[Sample]) -> str:
    buckets: Dict[Tuple[int, int], int] = {}
    for s in samples:
        buckets[s.bucket] = buckets.get(s.bucket, 0) + s.item.repeats
    no_cap = sum(1 for i in items if not i.caption.strip())
    lines = [f"이미지 {len(items)}장, 학습 샘플 {sum(s.item.repeats for s in samples)}개 (반복 포함)"]
    if no_cap:
        lines.append(f"캡션 없는 이미지 {no_cap}장")
    lines.append("버킷: " + ", ".join(f"{w}x{h}({n})" for (w, h), n in sorted(buckets.items())))
    return "\n".join(lines)
