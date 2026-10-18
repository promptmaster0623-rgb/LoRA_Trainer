"""블록 스왑 (Block Swap).

VRAM 이 부족할 때 트랜스포머 블록 일부의 (동결된) 베이스 가중치를 CPU pinned 메모리에
두고, 해당 블록이 실행되기 직전에 GPU 로 복사, 끝나면 GPU 사본을 버린다.

규칙
  - 그래디언트 체크포인팅(재진입 방식)과 함께 사용해야 한다.
    * 1차 순전파(no_grad) : 블록 실행 후 즉시 GPU 사본 해제
    * 역전파 중 재계산      : 블록 입력의 grad 가 계산되는 순간(=블록 역전파 완료) 해제
  - LoRA 파라미터는 별도 모듈에 있으므로 스왑 대상이 아니다.
  - NF4(bitsandbytes) 가중치는 스왑할 수 없다.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class _BlockState:
    def __init__(self, block: nn.Module, device: torch.device):
        self.block = block
        self.device = device
        self.cpu: Dict[int, torch.Tensor] = {}
        self.params: List[nn.Parameter] = []
        self.resident = False
        for p in block.parameters():
            if p.requires_grad:
                continue
            cpu = p.data
            if cpu.device.type != "cpu":
                cpu = cpu.to("cpu")
            try:
                cpu = cpu.pin_memory()
            except RuntimeError:
                pass
            p.data = cpu
            self.cpu[id(p)] = cpu
            self.params.append(p)

    def load(self) -> None:
        if self.resident:
            return
        for p in self.params:
            p.data = self.cpu[id(p)].to(self.device, non_blocking=True)
        self.resident = True

    def offload(self) -> None:
        if not self.resident:
            return
        for p in self.params:
            p.data = self.cpu[id(p)]
        self.resident = False


class BlockSwapper:
    def __init__(self, blocks: List[nn.Module], device: torch.device):
        self.states = [_BlockState(b, device) for b in blocks]
        self._handles = []
        for st in self.states:
            self._handles.append(st.block.register_forward_pre_hook(self._make_pre(st), with_kwargs=True))
            self._handles.append(st.block.register_forward_hook(self._make_post(st), with_kwargs=True))

    @staticmethod
    def _make_pre(st: _BlockState):
        def hook(module, args, kwargs):
            st.load()
            if torch.is_grad_enabled():
                # 역전파 재계산 단계: 입력 grad 가 계산되면 이 블록의 역전파가 끝난 것
                x = args[0] if args else next(iter(kwargs.values()), None)
                if isinstance(x, torch.Tensor) and x.requires_grad:
                    x.register_hook(lambda g, _st=st: (_st.offload(), g)[1])
            return None

        return hook

    @staticmethod
    def _make_post(st: _BlockState):
        def hook(module, args, kwargs, output):
            if not torch.is_grad_enabled():
                st.offload()
            return None

        return hook

    def offload_all(self) -> None:
        for st in self.states:
            st.offload()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    @property
    def count(self) -> int:
        return len(self.states)
