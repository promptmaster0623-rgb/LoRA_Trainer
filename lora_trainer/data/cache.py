"""텍스트 임베딩 / latent 디스크 캐시 및 배치 구성.

학습 전에 텍스트 인코더와 VAE 를 차례로 한 번씩만 GPU 에 올려 모든 결과를 디스크에 저장한다.
이후 학습 단계에서는 DiT 만 VRAM 에 상주하므로 12GB 이하 GPU 에서도 학습이 가능하다.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .dataset import Sample, load_and_fit

Progress = Optional[Callable[[int, int], None]]


def _hash(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:24]


class FeatureCache:
    def __init__(self, root: str, arch: str):
        self.arch = arch
        self.text_dir = os.path.join(root, arch, "text")
        self.latent_dir = os.path.join(root, arch, "latent")
        os.makedirs(self.text_dir, exist_ok=True)
        os.makedirs(self.latent_dir, exist_ok=True)

    # 텍스트 ------------------------------------------------------------
    def text_key(self, prompt: str, te_tag: str = "") -> str:
        return _hash(self.arch, te_tag, prompt)

    def text_path(self, key: str) -> str:
        return os.path.join(self.text_dir, key + ".safetensors")

    def has_text(self, key: str) -> bool:
        return os.path.isfile(self.text_path(key))

    def save_text(self, key: str, tensors: Dict[str, torch.Tensor]) -> None:
        save_file({k: v.contiguous() for k, v in tensors.items()}, self.text_path(key))

    def load_text(self, key: str) -> Dict[str, torch.Tensor]:
        return load_file(self.text_path(key))

    # latent ------------------------------------------------------------
    def latent_key(self, sample: Sample, vae_tag: str = "") -> str:
        st = os.stat(sample.item.path)
        return _hash(self.arch, vae_tag, os.path.abspath(sample.item.path), st.st_mtime_ns, st.st_size,
                     sample.bucket, sample.flip)

    def latent_path(self, key: str) -> str:
        return os.path.join(self.latent_dir, key + ".safetensors")

    def has_latent(self, key: str) -> bool:
        return os.path.isfile(self.latent_path(key))

    def save_latent(self, key: str, latent: torch.Tensor) -> None:
        save_file({"latent": latent.contiguous()}, self.latent_path(key))

    def load_latent(self, key: str) -> torch.Tensor:
        return load_file(self.latent_path(key))["latent"]


def pil_to_tensor(img) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def build_text_cache(adapter, cache: FeatureCache, prompts: Sequence[str], te_tag: str,
                     progress: Progress = None, should_stop: Callable[[], bool] = lambda: False) -> Dict[str, str]:
    """프롬프트 -> 캐시 키. 없는 것만 인코딩."""
    mapping: Dict[str, str] = {}
    todo = []
    for p in dict.fromkeys(prompts):
        key = cache.text_key(p, te_tag)
        mapping[p] = key
        if not cache.has_text(key):
            todo.append((p, key))
    if todo:
        for i, (p, key) in enumerate(todo):
            if should_stop():
                raise InterruptedError("사용자 중지")
            cache.save_text(key, adapter.encode_prompt(p))
            if progress:
                progress(i + 1, len(todo))
    return mapping


def build_latent_cache(adapter, cache: FeatureCache, samples: Sequence[Sample], vae_tag: str,
                       progress: Progress = None, should_stop: Callable[[], bool] = lambda: False) -> List[str]:
    keys = [cache.latent_key(s, vae_tag) for s in samples]
    todo = [(s, k) for s, k in zip(samples, keys) if not cache.has_latent(k)]
    for i, (s, k) in enumerate(todo):
        if should_stop():
            raise InterruptedError("사용자 중지")
        img = load_and_fit(s.item.path, s.bucket, s.flip)
        cache.save_latent(k, adapter.encode_image(pil_to_tensor(img)))
        if progress:
            progress(i + 1, len(todo))
    return keys


class BucketBatcher:
    """같은 버킷끼리 묶어 배치를 만들고, 에폭마다 섞는다 (repeats 반영)."""

    def __init__(self, entries: List[Tuple[str, str, Tuple[int, int], int]], batch_size: int, seed: int = 0):
        # entries: (latent_key, text_key, bucket, repeats)
        self.entries = entries
        self.batch_size = batch_size
        self.rng = random.Random(seed)

    def epoch(self) -> List[List[Tuple[str, str]]]:
        by_bucket: Dict[Tuple[int, int], List[Tuple[str, str]]] = {}
        for lat, txt, bucket, rep in self.entries:
            by_bucket.setdefault(bucket, []).extend([(lat, txt)] * rep)
        batches = []
        for items in by_bucket.values():
            self.rng.shuffle(items)
            for i in range(0, len(items), self.batch_size):
                batches.append(items[i: i + self.batch_size])
        self.rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[List[Tuple[str, str]]]:
        while True:
            for b in self.epoch():
                yield b

    def steps_per_epoch(self) -> int:
        counts: Dict[Tuple[int, int], int] = {}
        for _lat, _txt, bucket, rep in self.entries:
            counts[bucket] = counts.get(bucket, 0) + rep
        return sum((n + self.batch_size - 1) // self.batch_size for n in counts.values())
