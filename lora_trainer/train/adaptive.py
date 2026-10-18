"""실시간 적응형 학습 컨트롤러.

학습 루프가 매 스텝 호출하며 아래 기능을 담당한다. 모든 조정은 "adaptive" 이벤트로 GUI 에 알린다.

  1. 정체 감지   : 기준 loss 가 일정 스텝 동안 개선되지 않으면 학습률 배율을 낮춘다
  2. 발산 복구   : loss 급등·NaN 이 이어지면 직전 정상 스냅샷(LoRA+옵티마이저)으로 되돌리고 학습률을 낮춘다
  3. 자동 클리핑 : 최근 그래디언트 노름의 중앙값 × k 로 클리핑 값을 정한다
  4. 검증 평가   : 고정 노이즈·고정 타임스텝으로 loss 를 측정 (학습 loss 보다 훨씬 안정적인 추세 지표)
  5. 조기 종료   : 검증 loss 가 계속 나빠지면 최고 지점 가중치로 되돌리고 학습을 끝낸다
  6. 실시간 제어 : GUI 가 만든 .lr_scale 파일로 학습 중 학습률 배율을 즉시 변경 (적응형 사용 여부와 무관)

학습률 = 기본 학습률 × 스케줄 × scale()  (scale = 자동 배율 × 사용자 배율)
"""

from __future__ import annotations

import math
import os
import statistics
from collections import deque
from typing import Any, Callable, Dict, Optional

import torch

from .optim import is_self_tuning, scale_self_tuned_lr

LR_SCALE_FILE = ".lr_scale"


def _to_cpu(obj: Any) -> Any:
    """state_dict 를 CPU 로 깊은 복사 (스냅샷용)."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


class AdaptiveController:
    def __init__(self, cfg, network, optimizer, sink, run_dir: str):
        self.cfg = cfg
        self.on = bool(cfg.enabled)
        self.network = network
        self.optimizer = optimizer
        self.sink = sink
        self.run_dir = run_dir
        self.auto_scale = 1.0
        self.user_scale = 1.0
        self.last_grad_norm: Optional[float] = None
        # Automagic3 처럼 학습률을 스스로 관리하는 옵티마이저: 배율을 내부 학습률에 직접 곱하고, 정체 감지는 끈다
        self.self_tuned = is_self_tuning(optimizer)
        if self.on and self.self_tuned and cfg.plateau:
            sink.log("[적응형] 이 옵티마이저는 학습률을 스스로 정하므로 '정체 감지' 는 끄고 나머지 기능만 사용합니다.")
        self._norms: deque = deque(maxlen=200)
        # 정체 감지
        self._ema: Optional[float] = None
        self._best_metric = math.inf
        self._best_step = 0
        # 발산 감지
        self._fast: Optional[float] = None
        self._baseline: Optional[float] = None
        self._spike_run = 0
        self._recovers = 0
        self._snap: Optional[Dict[str, Any]] = None
        # 검증
        self._eval_fn: Optional[Callable[[], float]] = None
        self._best_val = math.inf
        self._best_val_step = 0
        self._bad_evals = 0
        self._best_weights: Optional[Dict[str, torch.Tensor]] = None
        p = os.path.join(run_dir, LR_SCALE_FILE)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------
    def scale(self) -> float:
        return self.auto_scale * self.user_scale

    def set_eval_fn(self, fn: Callable[[], float]) -> None:
        self._eval_fn = fn

    @property
    def eval_enabled(self) -> bool:
        return self.on and self.cfg.eval_every > 0 and self._eval_fn is not None

    def poll_controls(self, step: int) -> None:
        """매 스텝 시작 시 호출: GUI 제어 파일 확인 + 최초 스냅샷."""
        if self.on and self.cfg.divergence and self._snap is None:
            self._snapshot(step)
        p = os.path.join(self.run_dir, LR_SCALE_FILE)
        if not os.path.exists(p):
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                v = float(f.read().strip())
            os.remove(p)
        except (OSError, ValueError):
            return
        v = min(max(v, 0.01), 100.0)
        if abs(v - self.user_scale) > 1e-9:
            if self.self_tuned:
                scale_self_tuned_lr(self.optimizer, v / self.user_scale)
            self.user_scale = v
            self._notify(step, "user_lr", f"사용자 설정: 학습률 배율 ×{v:.2f} (총 배율 ×{self.scale():.3f})")

    def clip_value(self, base: float) -> float:
        """이번 스텝에 쓸 그래디언트 클리핑 값 (0 = 클리핑 안 함)."""
        if self.on and self.cfg.auto_clip and len(self._norms) >= 20:
            return max(1e-6, statistics.median(self._norms) * self.cfg.auto_clip_k)
        return base

    def record_grad_norm(self, norm) -> None:
        if norm is None:
            return
        v = float(norm)
        self.last_grad_norm = v
        if math.isfinite(v):
            self._norms.append(v)

    def after_step(self, step: int, loss: float) -> str:
        """옵티마이저 스텝 직후 호출. 'ok' | 'recover' | 'stop' 반환."""
        finite = math.isfinite(loss)
        if not self.on or not self.cfg.divergence:
            if not finite:
                raise RuntimeError("loss 가 NaN/Inf 입니다. 학습률을 낮추거나 dtype 을 확인하세요."
                                   " (적응형 학습의 '발산 복구'를 켜면 자동으로 되돌립니다)")
        if not self.on:
            return "ok"

        # 1) 발산 감지 -------------------------------------------------------
        if self.cfg.divergence:
            spiking = not finite
            if finite:
                self._fast = loss if self._fast is None else self._fast * 0.8 + loss * 0.2
                if self._baseline is None:
                    self._baseline = loss
                if step > 20 and self._fast > self.cfg.spike_ratio * self._baseline:
                    self._spike_run += 1
                else:
                    self._spike_run = 0
                    self._baseline = self._baseline * 0.99 + loss * 0.01
                spiking = self._spike_run >= 5
            if spiking:
                return self._recover(step, "NaN/Inf" if not finite else f"loss 급등 (×{self._fast / self._baseline:.1f})")
            if self._spike_run == 0 and (self._snap is None or step - self._snap["step"] >= self.cfg.snapshot_every):
                self._snapshot(step)

        # 2) 검증 평가 --------------------------------------------------------
        metric = None
        if self.eval_enabled and step % self.cfg.eval_every == 0:
            val = float(self._eval_fn())
            if math.isfinite(val):
                metric = val
                if val < self._best_val:
                    self._best_val, self._best_val_step, self._bad_evals = val, step, 0
                    if self.cfg.early_stop:
                        self._best_weights = _to_cpu(self.network.state_dict())
                else:
                    self._bad_evals += 1
                self.sink.emit("val", step=step, loss=val, best=self._best_val, best_step=self._best_val_step)
                self.sink.log(f"[검증] 스텝 {step}: loss {val:.5f} (최고 {self._best_val:.5f} @ {self._best_val_step})")
                if self.cfg.early_stop and self._bad_evals >= self.cfg.early_stop_patience:
                    return self._early_stop(step)

        # 3) 정체 감지 --------------------------------------------------------
        if self.cfg.plateau and finite and not self.self_tuned:
            self._ema = loss if self._ema is None else self._ema * 0.98 + loss * 0.02
            if self.eval_enabled:
                cur = metric  # 검증할 때만 판단
            else:
                cur = self._ema if step % 10 == 0 else None
            if cur is not None:
                if cur < self._best_metric * (1 - self.cfg.plateau_threshold):
                    self._best_metric, self._best_step = cur, step
                elif step - self._best_step >= self.cfg.plateau_patience:
                    floor = self.cfg.min_lr_ratio
                    if self.auto_scale > floor + 1e-9:
                        self.auto_scale = max(floor, self.auto_scale * self.cfg.plateau_factor)
                        self._notify(step, "plateau",
                                     f"{self.cfg.plateau_patience} 스텝 동안 개선 없음 → 학습률 배율 ×{self.scale():.3f}")
                    self._best_step = step
                    self._best_metric = min(self._best_metric, cur)
        return "ok"

    # ------------------------------------------------------------------
    # 이어학습용 상태
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {"auto_scale": self.auto_scale, "user_scale": self.user_scale, "recovers": self._recovers,
                "best_metric": self._best_metric, "best_step": self._best_step,
                "best_val": self._best_val, "best_val_step": self._best_val_step}

    def load_state_dict(self, st: Dict[str, Any]) -> None:
        self.auto_scale = float(st.get("auto_scale", 1.0))
        self.user_scale = float(st.get("user_scale", 1.0))
        self._recovers = int(st.get("recovers", 0))
        self._best_metric = float(st.get("best_metric", math.inf))
        self._best_step = int(st.get("best_step", 0))
        self._best_val = float(st.get("best_val", math.inf))
        self._best_val_step = int(st.get("best_val_step", 0))

    # ------------------------------------------------------------------
    # 내부
    # ------------------------------------------------------------------
    def _notify(self, step: int, action: str, message: str) -> None:
        self.sink.log(f"[적응형] 스텝 {step}: {message}")
        self.sink.emit("adaptive", step=step, action=action, message=message, lr_scale=self.scale())

    def _snapshot(self, step: int) -> None:
        self._snap = {
            "step": step,
            "lora": _to_cpu(self.network.state_dict()),
            "optimizer": _to_cpu(self.optimizer.state_dict()),
        }

    def _recover(self, step: int, reason: str) -> str:
        self._recovers += 1
        if self._recovers > self.cfg.max_recover:
            raise RuntimeError(f"발산이 {self.cfg.max_recover}회 넘게 반복되었습니다 ({reason}). "
                               "기본 학습률을 낮추거나 다른 옵티마이저를 사용하세요.")
        snap = self._snap
        if snap is not None:
            self.network.load_state_dict(snap["lora"], strict=False)
            self.optimizer.load_state_dict(snap["optimizer"])
        self.optimizer.zero_grad(set_to_none=True)
        old = self.auto_scale
        self.auto_scale = max(self.cfg.min_lr_ratio * 0.5, self.auto_scale * self.cfg.recover_factor)
        if self.self_tuned:
            scale_self_tuned_lr(self.optimizer, self.auto_scale / old)
        self._fast, self._spike_run, self._ema = None, 0, None
        at = f"스텝 {snap['step']}" if snap is not None else "시작"
        self._notify(step, "recover",
                     f"{reason} 감지 → {at} 시점으로 복구, 학습률 배율 ×{self.scale():.3f} "
                     f"({self._recovers}/{self.cfg.max_recover}회)")
        return "recover"

    def _early_stop(self, step: int) -> str:
        if self._best_weights is not None:
            self.network.load_state_dict(self._best_weights, strict=False)
        self._notify(step, "early_stop",
                     f"검증 loss 가 {self.cfg.early_stop_patience}회 연속 개선되지 않아 조기 종료 → "
                     f"최고 지점(스텝 {self._best_val_step}, loss {self._best_val:.5f}) 가중치로 저장합니다")
        return "stop"
