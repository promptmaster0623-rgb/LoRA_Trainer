"""옵티마이저 / 학습률 스케줄러 / 타임스텝 샘플링.

옵티마이저는 OPTIMIZER_SPECS 사양표 하나로 관리한다.
GUI 는 이 표를 읽어 그룹별 목록 · 권장 학습률 · 설명 · 설치 여부를 보여주고,
학습 엔진은 build_optimizer() 로 실제 객체를 만든다.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from ..core.config import TimestepConfig, TrainConfig


# ---------------------------------------------------------------------------
# 옵티마이저 사양표
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OptimizerSpec:
    key: str                    # 설정 파일에 저장되는 이름
    label: str                  # 목록 표시 이름
    group: str                  # 목록 그룹
    desc: str                   # 한 줄 설명
    lr: float                   # 권장 학습률
    weight_decay: float = 0.01  # 권장 weight decay
    betas: Tuple[float, ...] = (0.9, 0.999)
    memory: str = "보통"         # 옵티마이저 상태 메모리 (LoRA 기준 상대값)
    lr_free: bool = False       # 학습률 자동 (lr=1.0 기준)
    schedule_free: bool = False  # 스케줄러 불필요 (내부 평균화)
    self_tuning: bool = False   # 학습률을 옵티마이저가 직접 관리 (스케줄러 · 적응형 배율 무시, 학습률 칸은 시작값)
    module: str = ""            # 필요한 파이썬 모듈 (import 이름)
    pip: str = ""               # 설치 명령에 쓸 패키지 이름


OPTIMIZER_GROUPS: List[str] = ["AdamW 계열", "AdEMAMix 계열", "Lion 계열", "학습률 자동", "메모리 최소 · 기타"]

_SPECS: List[OptimizerSpec] = [
    # AdamW 계열 -------------------------------------------------------------
    OptimizerSpec("adamw8bit", "AdamW 8bit", "AdamW 계열",
                  "가장 무난한 기본값. 상태를 8bit 로 저장해 메모리를 아낍니다.",
                  1e-4, memory="낮음", module="bitsandbytes", pip="bitsandbytes"),
    OptimizerSpec("adamw", "AdamW", "AdamW 계열",
                  "PyTorch 기본 AdamW. 추가 설치가 필요 없고 결과가 가장 예측 가능합니다.",
                  1e-4, memory="보통"),
    OptimizerSpec("pagedadamw8bit", "Paged AdamW 8bit", "AdamW 계열",
                  "VRAM 이 모자라면 옵티마이저 상태를 CPU 메모리로 자동으로 옮겨 OOM 을 막습니다.",
                  1e-4, memory="매우 낮음", module="bitsandbytes", pip="bitsandbytes"),
    OptimizerSpec("adamw_schedulefree", "AdamW Schedule-Free", "AdamW 계열",
                  "스케줄러 없이 내부 평균화로 수렴. 총 스텝을 정하기 어려울 때 좋습니다.",
                  2e-4, memory="보통", schedule_free=True, module="schedulefree", pip="schedulefree"),
    # AdEMAMix 계열 ----------------------------------------------------------
    OptimizerSpec("ademamix8bit", "AdEMAMix 8bit", "AdEMAMix 계열",
                  "느린 EMA 를 하나 더 써서 오래된 그래디언트까지 활용. 긴 학습에서 안정적이고 잊힘이 적습니다.",
                  1e-4, betas=(0.9, 0.999, 0.9999), memory="낮음", module="bitsandbytes", pip="bitsandbytes"),
    OptimizerSpec("pagedademamix8bit", "Paged AdEMAMix 8bit", "AdEMAMix 계열",
                  "AdEMAMix 8bit + CPU 페이징. 12GB 이하에서 AdEMAMix 를 쓸 때 권장합니다.",
                  1e-4, betas=(0.9, 0.999, 0.9999), memory="매우 낮음", module="bitsandbytes", pip="bitsandbytes"),
    # Lion 계열 ---------------------------------------------------------------
    OptimizerSpec("lion8bit", "Lion 8bit", "Lion 계열",
                  "부호만 쓰는 가벼운 옵티마이저. 학습률은 AdamW 의 1/3~1/10, weight decay 는 더 크게.",
                  3e-5, weight_decay=0.05, betas=(0.9, 0.99), memory="매우 낮음", module="bitsandbytes", pip="bitsandbytes"),
    OptimizerSpec("lion", "Lion", "Lion 계열",
                  "내장 Lion (추가 설치 불필요). 상태가 모멘텀 하나뿐이라 메모리가 적습니다.",
                  3e-5, weight_decay=0.05, betas=(0.9, 0.99), memory="낮음"),
    # 학습률 자동 ---------------------------------------------------------------
    OptimizerSpec("automagic3", "Automagic3 (Ostris v3)", "학습률 자동",
                  "AI-Toolkit 원본 Automagic v3 (Fizgig 의 automagic3 와 동일). 업데이트 부호의 일관성을 보고 "
                  "학습률을 스스로 올리고 내립니다. 학습률 칸은 시작값이며 1e-6 권장. 상태 메모리가 매우 작습니다.",
                  1e-6, weight_decay=0.0, memory="매우 낮음", self_tuning=True),
    OptimizerSpec("automagic", "Automagic (v1 방식)", "학습률 자동",
                  "원소마다 학습률을 따로 조정하는 초기 Automagic 방식 (내장 구현). 스케줄러 · 적응형 배율도 함께 적용됩니다.",
                  1e-4, weight_decay=0.0, memory="낮음"),
    OptimizerSpec("prodigy", "Prodigy", "학습률 자동",
                  "최적 학습률을 스스로 추정합니다. 학습률 칸은 1.0 (배율)로 두세요.",
                  1.0, betas=(0.9, 0.99), memory="보통", lr_free=True, module="prodigyopt", pip="prodigyopt"),
    OptimizerSpec("prodigy_schedulefree", "Prodigy + Schedule-Free", "학습률 자동",
                  "학습률 추정 + 스케줄러 불필요. 설정할 것이 가장 적은 조합입니다.",
                  1.0, betas=(0.9, 0.99), memory="보통", lr_free=True, schedule_free=True,
                  module="prodigyplus", pip="prodigy-plus-schedule-free"),
    OptimizerSpec("dadaptadam", "D-Adaptation Adam", "학습률 자동",
                  "Prodigy 의 전신. 학습률을 자동 추정하며 Prodigy 보다 보수적입니다.",
                  1.0, memory="보통", lr_free=True, module="dadaptation", pip="dadaptation"),
    # 메모리 최소 · 기타 --------------------------------------------------------
    OptimizerSpec("adafactor", "Adafactor", "메모리 최소 · 기타",
                  "2차 모멘트를 행·열로 분해해 메모리가 가장 적습니다.",
                  1e-4, memory="최소", module="transformers", pip="transformers"),
    OptimizerSpec("came", "CAME", "메모리 최소 · 기타",
                  "Adafactor 수준 메모리에 Adam 수준 안정성. 큰 rank 에서 유리합니다.",
                  1e-4, betas=(0.9, 0.999, 0.9999), memory="최소", module="came_pytorch", pip="came-pytorch"),
]

OPTIMIZER_SPECS: Dict[str, OptimizerSpec] = {s.key: s for s in _SPECS}

# 이전 버전 호환 (키 -> 표시 이름)
OPTIMIZERS: Dict[str, str] = {s.key: s.label for s in _SPECS}

SCHEDULERS = {
    "constant": "일정",
    "cosine": "코사인 감소",
    "linear": "선형 감소",
    "cosine_restarts": "코사인 재시작 (3회)",
}

_BNB_FALLBACK = {
    "adamw8bit": "adamw",
    "pagedadamw8bit": "adamw",
    "ademamix8bit": "adamw",
    "pagedademamix8bit": "adamw",
    "lion8bit": "lion",
}


def optimizer_available(key: str) -> bool:
    """필요한 패키지가 설치되어 있는지 (import 하지 않고 확인)."""
    spec = OPTIMIZER_SPECS.get(key)
    if spec is None:
        return False
    return not spec.module or importlib.util.find_spec(spec.module) is not None


def _betas(cfg: TrainConfig, spec: OptimizerSpec) -> Tuple[float, ...]:
    """사용자가 기본값(0.9, 0.999)을 바꾸지 않았다면 옵티마이저 권장 betas 를 쓴다."""
    user = tuple(float(b) for b in (cfg.betas or []))
    if not user or user[:2] == (0.9, 0.999):
        return spec.betas
    if len(spec.betas) == 3 and len(user) == 2:
        return (user[0], user[1], spec.betas[2])
    return user[: len(spec.betas)]


def _need(spec: OptimizerSpec):
    try:
        return importlib.import_module(spec.module)
    except ImportError as e:
        raise RuntimeError(
            f"{spec.label} 를 쓰려면 'pip install {spec.pip}' 가 필요합니다. (install.bat 을 다시 실행해도 됩니다)"
        ) from e


def build_optimizer(params: List[torch.nn.Parameter], cfg: TrainConfig, log: Callable[[str], None] = print):
    name = cfg.optimizer
    if name not in OPTIMIZER_SPECS:
        raise ValueError(f"알 수 없는 옵티마이저: {name}")
    spec = OPTIMIZER_SPECS[name]
    lr, wd, eps = cfg.learning_rate, cfg.weight_decay, cfg.eps
    betas = _betas(cfg, spec)

    # bitsandbytes 계열 (설치 안 됐거나 GPU 미지원이면 가장 가까운 옵티마이저로 대체)
    if name in _BNB_FALLBACK:
        try:
            import bitsandbytes as bnb

            o = bnb.optim
            if name == "adamw8bit":
                return o.AdamW8bit(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
            if name == "pagedadamw8bit":
                return o.PagedAdamW8bit(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
            if name in ("ademamix8bit", "pagedademamix8bit"):
                cls = o.PagedAdEMAMix8bit if name.startswith("paged") else o.AdEMAMix8bit
                # 짧은 LoRA 학습에서는 느린 EMA 를 서서히 켜는 것이 안정적 (논문 권장)
                warm = max(1, cfg.max_steps)
                return cls(params, lr=lr, betas=betas, alpha=5.0, t_alpha=warm, t_beta3=warm, eps=eps, weight_decay=wd)
            if name == "lion8bit":
                return o.Lion8bit(params, lr=lr, betas=betas[:2], weight_decay=wd)
        except Exception as e:  # pragma: no cover - 환경 의존
            alt = _BNB_FALLBACK[name]
            log(f"[경고] {spec.label} 사용 불가 ({e}). {OPTIMIZER_SPECS[alt].label} 로 대체합니다.")
            name, spec = alt, OPTIMIZER_SPECS[alt]
            betas = _betas(cfg, spec)

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas[:2], eps=eps, weight_decay=wd)
    if name == "lion":
        from .custom_optimizers import Lion

        return Lion(params, lr=lr, betas=betas[:2], weight_decay=wd)
    if name == "automagic3":
        from .automagic3 import Automagic3

        if lr > 1e-3:
            log(f"[안내] Automagic3 시작 학습률 {lr:g} 는 높습니다 (권장 1e-6). 옵티마이저가 스스로 낮춰갑니다.")
        # fused=False: 클리핑 · 누적 · 발산 복구가 backward 와 step 사이에 일어나야 함
        return Automagic3(params, lr=lr, weight_decay=wd, fused=False)
    if name == "automagic":
        from .custom_optimizers import Automagic

        return Automagic(params, lr=lr, weight_decay=wd)
    if name == "adamw_schedulefree":
        sf = _need(spec)
        return sf.AdamWScheduleFree(params, lr=lr, betas=betas[:2], eps=eps, weight_decay=wd,
                                    warmup_steps=max(0, cfg.warmup_steps))
    if name == "prodigy":
        mod = _need(spec)
        return mod.Prodigy(params, lr=lr, betas=betas[:2], weight_decay=wd,
                           use_bias_correction=True, safeguard_warmup=True)
    if name == "prodigy_schedulefree":
        _need(spec)
        try:
            from prodigyplus import ProdigyPlusScheduleFree
        except ImportError:  # 구버전 경로
            from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree
        return ProdigyPlusScheduleFree(params, lr=lr, betas=betas[:2], weight_decay=wd)
    if name == "dadaptadam":
        mod = _need(spec)
        return mod.DAdaptAdam(params, lr=lr, betas=betas[:2], eps=eps, weight_decay=wd, decouple=True)
    if name == "adafactor":
        from transformers.optimization import Adafactor

        return Adafactor(params, lr=lr, scale_parameter=False, relative_step=False,
                         warmup_init=False, weight_decay=wd)
    if name == "came":
        mod = _need(spec)
        b = betas if len(betas) == 3 else (betas[0], betas[1], 0.9999)
        return mod.CAME(params, lr=lr, weight_decay=wd, betas=b, eps=(1e-30, 1e-16))
    raise ValueError(f"알 수 없는 옵티마이저: {name}")


# ---------------------------------------------------------------------------
# Schedule-Free 옵티마이저 모드 전환 / 실제 학습률 표시
# ---------------------------------------------------------------------------
def set_optimizer_mode(optimizer, train: bool) -> None:
    """Schedule-Free 계열은 학습 시 train(), 저장·샘플 시 eval() 이 필요하다. 그 외는 아무것도 안 함."""
    if optimizer is None:
        return
    fn = getattr(optimizer, "train" if train else "eval", None)
    if callable(fn) and hasattr(optimizer, "eval") and hasattr(optimizer, "train"):
        try:
            fn()
        except Exception:  # pragma: no cover
            pass


def effective_lr(optimizer, scheduler=None) -> float:
    """화면에 표시할 실제 학습률 (Prodigy/D-Adapt 는 d×lr, Automagic 은 평균 학습률)."""
    if hasattr(optimizer, "effective_lr"):
        return float(optimizer.effective_lr())
    if hasattr(optimizer, "get_avg_learning_rate"):  # Automagic3
        return float(optimizer.get_avg_learning_rate())
    g = optimizer.param_groups[0]
    lr = float(g.get("lr", 0.0))
    if "d" in g:
        try:
            return float(g["d"]) * lr
        except (TypeError, ValueError):
            pass
    return lr


def is_self_tuning(optimizer) -> bool:
    """학습률을 옵티마이저 상태에 직접 보관하는 경우 (Automagic3). group['lr'] 변경이 반영되지 않는다."""
    return hasattr(optimizer, "get_avg_learning_rate") and hasattr(optimizer, "_apply_group_votes")


@torch.no_grad()
def scale_self_tuned_lr(optimizer, factor: float) -> None:
    """Automagic3 의 내부 학습률에 배율을 곱한다 (발산 복구 · 사용자 배율 변경용)."""
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p)
            if st is not None and torch.is_tensor(st.get("lr")):
                st["lr"].mul_(factor).clamp_(min=group.get("min_lr", 0.0), max=group.get("max_lr", 1e9))
    for group in optimizer.param_groups:  # 아직 상태가 없는 경우 시작값에 반영
        group["lr"] = float(group["lr"]) * factor


# ---------------------------------------------------------------------------
# 스케줄러
# ---------------------------------------------------------------------------
def build_scheduler(optimizer, cfg: TrainConfig, scale_fn: Optional[Callable[[], float]] = None,
                    log: Optional[Callable[[str], None]] = None):
    """스케줄 × (적응형 학습 배율). scale_fn 은 매 스텝 호출되어 즉시 반영된다."""
    total = max(1, cfg.max_steps)
    warm = max(0, cfg.warmup_steps)
    kind = cfg.lr_scheduler
    spec = OPTIMIZER_SPECS.get(cfg.optimizer)
    if spec is not None and spec.self_tuning:
        if (kind != "constant" or warm) and log:
            log(f"[안내] {spec.label} 는 학습률을 스스로 정하므로 스케줄러 · 웜업을 쓰지 않습니다.")
        kind, warm = "constant", 0
    elif spec is not None and spec.schedule_free:
        if kind != "constant" or warm:
            if log:
                log(f"[안내] {spec.label} 는 스케줄러가 필요 없어 '일정' 으로 학습합니다 (웜업은 옵티마이저가 처리).")
        kind, warm = "constant", 0

    def factor(step: int) -> float:
        if warm and step < warm:
            base = (step + 1) / warm
        else:
            progress = (step - warm) / max(1, total - warm)
            progress = min(max(progress, 0.0), 1.0)
            if kind == "cosine":
                base = 0.5 * (1 + math.cos(math.pi * progress))
            elif kind == "linear":
                base = 1.0 - progress
            elif kind == "cosine_restarts":
                cycles = 3
                base = 0.5 * (1 + math.cos(math.pi * ((progress * cycles) % 1.0)))
            else:
                base = 1.0
        return base * (scale_fn() if scale_fn is not None else 1.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def sample_timesteps(batch: int, cfg: TimestepConfig, shift: Optional[Callable], generator=None) -> torch.Tensor:
    """학습용 t ∈ (0, 1) 샘플 (t=1 노이즈)."""
    if cfg.sampling == "uniform":
        u = torch.rand(batch, generator=generator)
    else:  # logit_normal
        u = torch.sigmoid(torch.randn(batch, generator=generator) * cfg.logit_std + cfg.logit_mean)
    u = u.clamp(1e-4, 1 - 1e-4)
    if shift is not None:
        u = shift(u)
    return u.clamp(1e-4, 1 - 1e-4)
