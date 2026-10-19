"""LoRA 가중치 지수이동평균 (EMA).

Fizgig 기본값(0.98)과 같은 방식: 학습은 원래 가중치로 진행하고,
체크포인트 저장 · 샘플 생성 때만 최근 스텝들의 평균 가중치를 사용한다.
배치 1 · 무작위 타임스텝 학습의 스텝별 흔들림이 줄어 후반 에폭의 닮음(likeness)이 안정된다.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List

import torch
import torch.nn as nn


class LoRAEma:
    def __init__(self, network: nn.Module, decay: float = 0.98, warmup: bool = True):
        self.decay = float(decay)
        self.warmup = warmup
        self.params: List[nn.Parameter] = [p for p in network.parameters() if p.requires_grad]
        self.shadow: List[torch.Tensor] = [p.detach().float().clone() for p in self.params]
        self.updates = 0

    @property
    def enabled(self) -> bool:
        return 0.0 < self.decay < 1.0

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        # 초반에는 평균 창을 짧게 (0 에서 시작하는 lora_up 이 평균에 묻히지 않도록)
        return min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))

    @torch.no_grad()
    def update(self) -> None:
        if not self.enabled:
            return
        d = self.current_decay()
        for s, p in zip(self.shadow, self.params):
            s.mul_(d).add_(p.detach().float(), alpha=1.0 - d)
        self.updates += 1

    @torch.no_grad()
    def reset(self) -> None:
        """현재 가중치로 평균을 다시 시작 (발산 복구 직후 등)."""
        for s, p in zip(self.shadow, self.params):
            s.copy_(p.detach().float())

    @contextmanager
    def swap(self):
        """블록 안에서는 네트워크가 EMA 가중치를 사용, 끝나면 원래 가중치로 복원."""
        if not self.enabled or self.updates == 0:
            yield
            return
        backup = [p.detach().clone() for p in self.params]
        try:
            with torch.no_grad():
                for p, s in zip(self.params, self.shadow):
                    p.copy_(s.to(p.dtype))
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(self.params, backup):
                    p.copy_(b)

    def state_dict(self) -> Dict:
        return {"decay": self.decay, "updates": self.updates, "shadow": [s.cpu() for s in self.shadow]}

    def load_state_dict(self, state: Dict) -> bool:
        shadow = state.get("shadow") or []
        if len(shadow) != len(self.shadow) or any(a.shape != b.shape for a, b in zip(shadow, self.shadow)):
            return False
        for s, v in zip(self.shadow, shadow):
            s.copy_(v.to(s.device, s.dtype))
        self.updates = int(state.get("updates", 0))
        return True
