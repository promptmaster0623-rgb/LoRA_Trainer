"""Flow-matching 샘플러 (속도 v 예측 모델 공용).

규약: x_t = (1 - σ) x0 + σ ε, 모델 출력 v = ε - x0  →  x0 = x_t - σ v
  - euler          : x ← x + (σ_next - σ) v
  - res_multistep  : ComfyUI k_diffusion `sample_res_multistep` (eta 0) 이식.
                     지수 적분 2차 다단계 (arXiv 2308.02157). 첫 스텝·마지막 스텝은 Euler.
                     Fizgig v8.5.4~ MOSS-Image 미리보기 · ComfyUI MOSS 템플릿 기본 샘플러.
스케줄
  - comfy_simple_sigmas : ComfyUI 'simple' 스케줄러 (shift 적용된 1000 격자에서 균등 간격 선택)
"""

from __future__ import annotations

import math
from typing import Callable, List

import torch

Velocity = Callable[[torch.Tensor, int, float], torch.Tensor]  # (x, step_index, sigma) -> v


def time_snr_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
    if shift == 1.0:
        return t
    return shift * t / (1 + (shift - 1) * t)


def comfy_simple_sigmas(steps: int, shift: float = 1.0, total: int = 1000) -> torch.Tensor:
    """ComfyUI ModelSamplingDiscreteFlow(shift) + 'simple' 스케줄러. 끝에 0 포함 (steps+1 개)."""
    grid = time_snr_shift(shift, torch.arange(1, total + 1, dtype=torch.float64) / total)
    ss = total / steps
    sigs = [float(grid[-(1 + int(i * ss))]) for i in range(steps)]
    return torch.tensor(sigs + [0.0], dtype=torch.float32)


def sample_euler(x: torch.Tensor, sigmas: torch.Tensor, velocity: Velocity) -> torch.Tensor:
    x = x.float()
    for i in range(len(sigmas) - 1):
        s, sn = float(sigmas[i]), float(sigmas[i + 1])
        v = velocity(x, i, s).float()
        x = x + (sn - s) * v
    return x


def sample_res_multistep(x: torch.Tensor, sigmas: torch.Tensor, velocity: Velocity) -> torch.Tensor:
    x = x.float()
    old_denoised = None
    n = len(sigmas) - 1
    for i in range(n):
        s, sn = float(sigmas[i]), float(sigmas[i + 1])
        v = velocity(x, i, s).float()
        denoised = x - s * v
        if sn == 0.0 or old_denoised is None:
            x = x + (sn - s) * v  # Euler (d = (x - denoised) / σ = v)
        else:
            t, t_next, t_prev = -math.log(s), -math.log(sn), -math.log(float(sigmas[i - 1]))
            t_old = t  # eta 0: 직전 스텝의 sigma_down = 현재 sigma
            h = t_next - t
            c2 = (t_prev - t_old) / h
            phi1 = math.expm1(-h) / -h
            phi2 = (phi1 - 1.0) / -h
            b1 = phi1 - phi2 / c2 if c2 != 0 else phi1
            b2 = phi2 / c2 if c2 != 0 else 0.0
            x = math.exp(-h) * x + h * (b1 * denoised + b2 * old_denoised)
        old_denoised = denoised
    return x


SAMPLERS = {"euler": sample_euler, "res_multistep": sample_res_multistep}
SAMPLER_LABELS: List[tuple] = [
    ("auto", "자동 (모델 권장: Krea 2 Euler · MOSS res_multistep)"),
    ("euler", "Euler"),
    ("res_multistep", "res_multistep (2차 다단계, ComfyUI 동일)"),
]


def run_sampler(name: str, x: torch.Tensor, sigmas: torch.Tensor, velocity: Velocity) -> torch.Tensor:
    fn = SAMPLERS.get(name)
    if fn is None:
        raise ValueError(f"알 수 없는 샘플러: {name}")
    return fn(x, sigmas, velocity)
