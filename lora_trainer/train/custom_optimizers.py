"""내장 옵티마이저 (추가 설치 불필요): Lion, Automagic.

Automagic 은 AI-Toolkit 의 Automagic 방식을 참고해 새로 작성했다.
  - 2차 모멘트는 Adafactor 처럼 행·열로 분해해 저장 (메모리 최소)
  - 파라미터 원소마다 학습률(lr_mask)을 따로 두고,
    직전 스텝과 업데이트 방향(부호)이 같으면 조금 올리고 다르면 조금 내린다
  - 스케줄러 / 적응형 학습 배율은 lr_mask 전체에 곱해진다 (group["lr"] / 시작 lr)
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """Lion (Chen et al. 2023): update = sign(β1·m + (1-β1)·g)."""

    def __init__(self, params: Iterable, lr: float = 3e-5, betas: Tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0):
        if lr <= 0:
            raise ValueError("학습률은 0 보다 커야 합니다.")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), wd = group["lr"], group["betas"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                m = state["exp_avg"]
                update = (m * b1 + g * (1 - b1)).sign_()
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update.to(p.dtype), alpha=-lr)
                m.mul_(b2).add_(g, alpha=1 - b2)
        return loss


class Automagic(Optimizer):
    """원소별 학습률 자동 조정 옵티마이저 (Adafactor 기반)."""

    def __init__(self, params: Iterable, lr: float = 1e-4, min_lr: float = 1e-7, max_lr: float = 1e-3,
                 lr_bump: float = 1e-6, eps: Tuple[float, float] = (1e-30, 1e-3), clip_threshold: float = 1.0,
                 decay_rate: float = -0.8, weight_decay: float = 0.0):
        if lr <= 0:
            raise ValueError("학습률은 0 보다 커야 합니다.")
        defaults = dict(lr=lr, start_lr=lr, min_lr=min_lr, max_lr=max(max_lr, lr), lr_bump=lr_bump, eps=eps,
                        clip_threshold=clip_threshold, decay_rate=decay_rate, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        return t.norm(2) / math.sqrt(t.numel())

    @staticmethod
    def _approx_sq_grad(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        r = (row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c = col.unsqueeze(-2).rsqrt()
        return r * c

    def _scale(self, group) -> float:
        """스케줄러가 바꾼 group['lr'] 의 비율 (시작값 대비)."""
        start = group.get("initial_lr", group["start_lr"]) or group["start_lr"]
        return float(group["lr"]) / float(start) if start else 1.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            eps1, _eps2 = group["eps"]
            scale = self._scale(group)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                factored = g.dim() >= 2
                if not state:
                    state["step"] = 0
                    state["lr_mask"] = torch.full_like(p, group["start_lr"], dtype=torch.float32)
                    state["last_polarity"] = torch.zeros_like(p, dtype=torch.bool)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(g.shape[:-1], dtype=torch.float32, device=g.device)
                        state["exp_avg_sq_col"] = torch.zeros(g.shape[:-2] + g.shape[-1:], dtype=torch.float32,
                                                              device=g.device)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(g)
                state["step"] += 1
                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                sq = g * g + eps1
                if factored:
                    row, col = state["exp_avg_sq_row"], state["exp_avg_sq_col"]
                    row.mul_(beta2t).add_(sq.mean(dim=-1), alpha=1 - beta2t)
                    col.mul_(beta2t).add_(sq.mean(dim=-2), alpha=1 - beta2t)
                    update = self._approx_sq_grad(row, col).mul_(g)
                else:
                    v = state["exp_avg_sq"]
                    v.mul_(beta2t).add_(sq, alpha=1 - beta2t)
                    update = v.rsqrt().mul_(g)
                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

                # 방향이 유지되면 학습률 ↑, 바뀌면 ↓ (원소별)
                polarity = update > 0
                mask = state["lr_mask"]
                if state["step"] > 1:
                    same = polarity == state["last_polarity"]
                    mask.add_(torch.where(same, group["lr_bump"], -group["lr_bump"]))
                    mask.clamp_(group["min_lr"], group["max_lr"])
                state["last_polarity"] = polarity

                lr_t = mask * scale
                if group["weight_decay"]:
                    p.sub_((p.float() * lr_t * group["weight_decay"]).to(p.dtype))
                p.sub_((update * lr_t).to(p.dtype))
        return loss

    @torch.no_grad()
    def effective_lr(self) -> float:
        """전체 원소 평균 학습률 × 현재 배율 (화면 표시용)."""
        total, n = 0.0, 0
        for group in self.param_groups:
            scale = self._scale(group)
            for p in group["params"]:
                st = self.state.get(p)
                if st and "lr_mask" in st:
                    total += float(st["lr_mask"].sum()) * scale
                    n += st["lr_mask"].numel()
        if n == 0:
            g = self.param_groups[0]
            return float(g["start_lr"]) * self._scale(g)
        return total / n
