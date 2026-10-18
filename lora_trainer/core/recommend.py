"""시스템 · 데이터셋 기반 추천 옵션 (퀄리티 중시 / 최적 / 속도 중시) 과 속도 · 시간 추정.

판단 순서
  1. 사양(VRAM / 시스템 RAM / CPU / 가상메모리)과 데이터셋(장수, 해상도, 캡션)을 읽는다.
  2. 모델별 메모리 모델로 "이 PC 에서 가능한 조합" 범위를 만든다
     (DiT 정밀도 bf16·FP8·NF4 × 블록 스왑 × 해상도 × 배치 × 체크포인팅).
  3. 그 범위 안에서 목적에 맞는 3가지 조합을 고르고 예상 VRAM · s/it · 총 시간을 계산한다.

속도 추정은 GPU 상대 성능표 × 모델별 기준 속도(RTX 4090, 1024px, 배치1) 로 계산한 근사치다.
실제 학습이 끝나면 측정 속도로 보정 계수를 저장(user_data/speed_calibration.json)해
같은 PC 의 다음 추정부터 반영한다.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .sysinfo import SystemSpec

# ---------------------------------------------------------------------------
# 모델별 상수
# ---------------------------------------------------------------------------
MODEL_SPECS = {
    # weights_gb: DiT 가중치 크기, act_gb: 1024px·배치1·체크포인팅 활성화 메모리,
    # lora_mparams_per_rank: rank 1 당 LoRA 파라미터 수(백만, attn+mlp 기준),
    # base_sec: RTX 4090 · 1024px · 배치1 · bf16 연산 기준 1스텝 시간(초)
    "krea2": {
        "blocks": 28,
        "weights_gb": {"none": 25.8, "fp8": 13.4, "nf4": 7.4},
        "act_gb": 1.5,
        "lora_mparams_per_rank": 3.6,
        "base_sec": 1.9,
        "te_gb": {"none": 8.8, "int8": 4.8, "nf4": 3.0},
        "te_load_sec": 40,
    },
    "moss_image": {
        "blocks": 42,
        "weights_gb": {"none": 10.2, "fp8": 5.4, "nf4": 3.1},
        "act_gb": 1.0,
        "lora_mparams_per_rank": 2.4,
        "base_sec": 1.0,
        "te_gb": {"none": 18.0, "int8": 9.6, "nf4": 6.0},
        "te_load_sec": 70,
    },
}

QUANT_SPEED = {"none": 1.0, "fp8": 1.12, "nf4": 1.35}
QUANT_LABEL = {"none": "bf16 (무손실)", "fp8": "FP8", "nf4": "NF4 4bit"}
CUDA_OVERHEAD_GB = 0.9

# GPU 상대 성능 (RTX 4090 = 1.0, 학습용 bf16 텐서 연산 기준 근사)
GPU_FACTORS = [
    (r"H100", 2.0), (r"H200", 2.2), (r"B200", 3.0), (r"A100", 1.0), (r"L40S?", 0.95), (r"A6000", 0.6),
    (r"5090", 1.45), (r"5080", 0.88), (r"5070\s*Ti", 0.75), (r"5070", 0.58), (r"5060\s*Ti", 0.4), (r"5060", 0.33),
    (r"4090", 1.0), (r"4080\s*SUPER", 0.78), (r"4080", 0.75), (r"4070\s*Ti\s*SUPER", 0.65), (r"4070\s*Ti", 0.6),
    (r"4070\s*SUPER", 0.55), (r"4070", 0.47), (r"4060\s*Ti", 0.34), (r"4060", 0.27),
    (r"3090\s*Ti", 0.66), (r"3090", 0.6), (r"3080\s*Ti", 0.58), (r"3080", 0.52), (r"3070\s*Ti", 0.42),
    (r"3070", 0.38), (r"3060\s*Ti", 0.33), (r"3060", 0.28), (r"2080\s*Ti", 0.22), (r"2080", 0.17),
]


def gpu_factor(name: str, vram_gb: float) -> float:
    for pat, f in GPU_FACTORS:
        if re.search(pat, name or "", re.I):
            return f
    if vram_gb >= 40:
        return 0.9
    if vram_gb >= 20:
        return 0.55
    if vram_gb >= 14:
        return 0.4
    return 0.25 if vram_gb > 0 else 0.02  # CPU 는 사실상 학습 불가 수준


# ---------------------------------------------------------------------------
# 데이터셋 통계
# ---------------------------------------------------------------------------
@dataclass
class DatasetStats:
    images: int = 0
    captioned: int = 0
    min_short: int = 0
    median_short: int = 0
    weighted: int = 0              # 반복(repeats) 포함 장수
    folders: int = 0

    @property
    def empty(self) -> bool:
        return self.images == 0


def analyze_dataset(folders: Sequence, max_images: int = 2000) -> DatasetStats:
    """folders: DatasetFolder 목록 (path, repeats, caption_ext)."""
    from PIL import Image

    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    st = DatasetStats(folders=len(folders))
    shorts = []
    for fd in folders:
        path = getattr(fd, "path", fd)
        reps = max(1, int(getattr(fd, "repeats", 1) or 1))
        cap_ext = getattr(fd, "caption_ext", ".txt") or ".txt"
        if not os.path.isdir(path):
            continue
        for name in sorted(os.listdir(path)):
            if not name.lower().endswith(exts):
                continue
            full = os.path.join(path, name)
            st.images += 1
            st.weighted += reps
            if os.path.isfile(os.path.splitext(full)[0] + cap_ext):
                st.captioned += 1
            if len(shorts) < max_images:
                try:
                    with Image.open(full) as im:  # 헤더만 읽음
                        shorts.append(min(im.size))
                except Exception:
                    pass
    if shorts:
        st.min_short = min(shorts)
        st.median_short = int(statistics.median(shorts))
    return st


# ---------------------------------------------------------------------------
# 보정 계수 저장소
# ---------------------------------------------------------------------------
class SpeedCalibration:
    """실측 s/it ÷ 추정 s/it 비율을 GPU·모델별로 저장 (지수이동평균)."""

    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, dict] = {}
        try:
            with open(path, encoding="utf-8") as f:
                self.data = json.load(f)
        except (OSError, json.JSONDecodeError):
            self.data = {}

    @staticmethod
    def key(spec: SystemSpec, arch: str) -> str:
        return f"{spec.gpu_name}|{arch}"

    def ratio(self, spec: SystemSpec, arch: str) -> tuple:
        d = self.data.get(self.key(spec, arch))
        if not d:
            return 1.0, 0
        return float(d.get("ratio", 1.0)), int(d.get("n", 0))

    def record(self, spec: SystemSpec, arch: str, measured_sec: float, estimated_sec: float) -> None:
        if measured_sec <= 0 or estimated_sec <= 0:
            return
        r = measured_sec / estimated_sec
        r = min(max(r, 0.2), 5.0)
        k = self.key(spec, arch)
        old = self.data.get(k)
        if old:
            n = int(old.get("n", 0)) + 1
            ratio = old["ratio"] * 0.5 + r * 0.5
        else:
            n, ratio = 1, r
        self.data[k] = {"ratio": round(ratio, 4), "n": n}
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 메모리 / 속도 모델
# ---------------------------------------------------------------------------
def _res_mem(res: Sequence[int]) -> float:
    return max((r / 1024) ** 2 for r in res)


def _res_cost(res: Sequence[int]) -> float:
    vals = [0.85 * (r / 1024) ** 2 + 0.15 * (r / 1024) ** 4 for r in res]
    return sum(vals) / len(vals)


@dataclass
class TrainShape:
    """추정에 필요한 학습 조합."""

    arch: str
    quant: str
    block_swap: int
    resolutions: List[int]
    batch: int = 1
    grad_accum: int = 1
    grad_ckpt: bool = True
    rank: int = 16
    optimizer: str = "adamw8bit"
    max_steps: int = 2000
    sample_every: int = 250
    sample_prompts: int = 1
    sample_steps: int = 20
    sample_res: int = 768
    sample_enabled: bool = True
    te_quant: str = "none"
    images: int = 25


def estimate_vram(s: TrainShape) -> float:
    m = MODEL_SPECS[s.arch]
    w = m["weights_gb"][s.quant]
    blocks = m["blocks"]
    swap = min(max(0, s.block_swap), blocks - 2) if s.quant != "nf4" else 0
    w_gpu = w * (1 - swap / blocks) + (w / blocks if swap else 0)
    act = m["act_gb"] * _res_mem(s.resolutions) * s.batch * (1.0 if s.grad_ckpt else 3.5)
    lora_params = m["lora_mparams_per_rank"] * s.rank * 1e6
    opt_bytes = {"adamw": 8, "adamw8bit": 2, "prodigy": 12, "adafactor": 4}.get(s.optimizer, 8)
    lora_gb = lora_params * (4 + 4 + opt_bytes) / 1024 ** 3
    # 샘플 생성은 학습 활성화가 해제된 뒤라 더 큰 쪽만 반영
    sample_act = m["act_gb"] * 0.5 * (s.sample_res / 1024) ** 2 if s.sample_enabled else 0
    return w_gpu + max(act, sample_act) + lora_gb + CUDA_OVERHEAD_GB


def estimate_sec_per_step(s: TrainShape, spec: SystemSpec, calib: float = 1.0) -> float:
    m = MODEL_SPECS[s.arch]
    g = gpu_factor(spec.gpu_name, spec.vram_gb)
    sec = m["base_sec"] / g * QUANT_SPEED[s.quant] * _res_cost(s.resolutions)
    if not s.grad_ckpt:
        sec *= 0.78
    sec *= s.batch * (0.92 if s.batch > 1 else 1.0) * max(1, s.grad_accum)
    if s.block_swap > 0 and s.quant != "nf4":
        per_block = m["weights_gb"][s.quant] / m["blocks"]
        sec += s.block_swap * per_block * 2 / 8.0 * 0.8 * max(1, s.grad_accum)  # PCIe 실효 ≈8GB/s, 일부만 연산과 겹침
    return sec * calib


def estimate_prep_sec(s: TrainShape, spec: SystemSpec) -> float:
    """캐싱 + 모델 로딩 시간 (처음 1회, 캐시가 있으면 짧아짐)."""
    m = MODEL_SPECS[s.arch]
    g = max(gpu_factor(spec.gpu_name, spec.vram_gb), 0.05)
    cpu_slow = 1.0 if spec.cpu_threads >= 8 else 1.4
    te = m["te_load_sec"] + s.images * len(s.resolutions) * 0.25 / g
    lat = s.images * len(s.resolutions) * 0.5 / g * cpu_slow
    dit = m["weights_gb"]["none"] / 1.2 + (60 if s.quant == "nf4" else 20 if s.quant == "fp8" else 0)
    return te + lat + dit


def estimate_sample_sec(s: TrainShape, spec: SystemSpec, calib: float = 1.0) -> float:
    if not s.sample_enabled or s.sample_every <= 0:
        return 0.0
    m = MODEL_SPECS[s.arch]
    g = gpu_factor(spec.gpu_name, spec.vram_gb)
    per_img = (m["base_sec"] / g * QUANT_SPEED[s.quant] * _res_cost([s.sample_res]) / 3.2 * s.sample_steps) * calib + 3
    events = s.max_steps // s.sample_every
    return events * (per_img * s.sample_prompts + 8)


# ---------------------------------------------------------------------------
# 추천
# ---------------------------------------------------------------------------
@dataclass
class Recommendation:
    key: str                         # quality | balanced | speed
    title: str
    subtitle: str
    settings: Dict[str, object]      # 적용할 값 (apply_to_job 참고)
    vram_gb: float
    sec_per_step: float
    train_sec: float
    prep_sec: float
    sample_sec: float
    calibrated: int = 0              # 보정에 쓰인 실측 횟수
    fits: bool = True
    notes: List[str] = field(default_factory=list)

    @property
    def total_sec(self) -> float:
        return self.train_sec + self.prep_sec + self.sample_sec

    @property
    def it_per_sec(self) -> float:
        return 1.0 / self.sec_per_step if self.sec_per_step > 0 else 0.0


@dataclass
class Analysis:
    spec: SystemSpec
    stats: DatasetStats
    arch: str
    purpose: str
    options: List[Recommendation]
    warnings: List[str]
    usable_vram: float


PURPOSES = {"character": "캐릭터 · 인물", "style": "스타일 · 화풍", "general": "범용 · 사물/컨셉"}


def base_steps(purpose: str, images: int) -> int:
    n = images or 25
    if purpose == "character":
        s = min(max(n * 80, 1200), 3000)
    elif purpose == "style":
        s = min(max(n * 60, 2000), 4000)
    else:
        s = min(max(n * 70, 1500), 3500)
    return int(round(s / 50) * 50)


def _swap_needed(arch: str, quant: str, res, batch, ckpt, rank, opt, usable, sample_res, max_ram_gb) -> Optional[int]:
    """해당 정밀도에서 VRAM 에 맞추기 위한 최소 블록 스왑 수 (불가하면 None)."""
    m = MODEL_SPECS[arch]
    limit = 0 if quant == "nf4" else int(m["blocks"] * 0.6)
    per_block = m["weights_gb"][quant] / m["blocks"]
    for sw in range(0, limit + 1):
        s = TrainShape(arch, quant, sw, list(res), batch, 1, ckpt, rank, opt, sample_res=sample_res)
        if estimate_vram(s) <= usable:
            if sw * per_block > max_ram_gb:
                return None
            return sw
    return None


def recommend(arch: str, spec: SystemSpec, stats: DatasetStats, purpose: str = "character",
              calibration: Optional[SpeedCalibration] = None) -> Analysis:
    warnings: List[str] = []
    vram = spec.vram_gb if spec.gpu_available else 0.0
    usable = max(0.0, vram * 0.93 - 0.3)          # 드라이버/화면 출력 여유
    # 블록 스왑에 쓸 수 있는 RAM: 전체 RAM 에서 OS·앱 몫(약 10GB)을 뺀 값
    swap_ram = max(0.0, spec.ram_gb - 10.0)
    calib, n_cal = (calibration.ratio(spec, arch) if calibration else (1.0, 0))

    if not spec.gpu_available:
        warnings.append("CUDA GPU 가 감지되지 않았습니다. 학습은 NVIDIA GPU 가 필요합니다 (CPU 는 테스트용).")
    if spec.ram_gb and spec.ram_gb < 24:
        warnings.append(f"시스템 RAM {spec.ram_gb:.0f}GB — 모델 로딩 중 부족할 수 있습니다. 32GB 이상 권장.")
    te_need = MODEL_SPECS[arch]["te_gb"]["none"]
    if spec.ram_gb and spec.ram_gb + spec.swap_gb < te_need + 12:
        warnings.append(f"RAM+가상메모리({spec.ram_gb + spec.swap_gb:.0f}GB)가 작아 텍스트 인코더 로딩 중 멈출 수 있습니다. "
                        f"가상메모리를 {math.ceil(te_need + 16 - spec.ram_gb)}GB 이상으로 늘리세요.")
    elif spec.ram_gb and spec.ram_gb < 32 and spec.swap_gb < 16:
        warnings.append("가상메모리(페이지 파일)가 16GB 미만입니다. 대용량 체크포인트 로딩 안정성을 위해 늘리는 것을 권장합니다.")
    if spec.cpu_threads and spec.cpu_threads < 8:
        warnings.append(f"CPU {spec.cpu_threads}스레드 — 캐싱 단계가 다소 느릴 수 있습니다 (학습 속도에는 영향 적음).")
    if stats.empty:
        warnings.append("데이터셋이 비어 있어 이미지 25장 기준으로 계산했습니다. 데이터셋 폴더를 추가하면 더 정확해집니다.")
    else:
        if stats.images < 8:
            warnings.append(f"이미지 {stats.images}장 — 너무 적습니다. 캐릭터 15장+, 스타일 30장+ 권장.")
        if stats.captioned < stats.images:
            warnings.append(f"캡션 없는 이미지 {stats.images - stats.captioned}장 — 데이터셋 페이지의 자동 캡션을 권장합니다.")
        if stats.median_short and stats.median_short < 768:
            warnings.append(f"이미지 짧은 변 중앙값 {stats.median_short}px — 1024 학습 시 업스케일로 흐려집니다 (768 권장).")

    images = stats.images or 25
    steps = base_steps(purpose, images)
    small_data = bool(stats.median_short) and stats.median_short < 900
    max_res = 768 if small_data else 1024
    if arch == "krea2" and vram < 10:
        max_res = 768
    rank_main = 16 if purpose == "character" else 32
    lr_main = 1e-4 if purpose == "character" else 1.5e-4

    order = ["none", "fp8", "nf4"]

    def build(key, title, subtitle, candidates, opt, ckpt, steps_n, lr, sample_every, quant_pref, allow_swap, notes):
        """candidates: [(해상도 목록, rank), ...] 앞에서부터 VRAM 에 맞는 첫 조합 사용."""
        chosen = None
        for res, rank in candidates:
            for q in quant_pref:
                sw = _swap_needed(arch, q, res, 1, ckpt, rank, opt, usable, 768, swap_ram if allow_swap else 0.0)
                if sw is None or (sw > 0 and not allow_swap):
                    continue
                chosen = (q, sw, list(res), rank)
                break
            if chosen:
                break
        fits = chosen is not None
        if not fits:
            chosen = ("nf4", 0, [768], 16)
            notes = notes + ["VRAM 이 부족해 실행이 어려울 수 있습니다 (최소 설정으로 계산)."]
        q, sw, res, rank = chosen
        if fits and (res, rank) != (list(candidates[0][0]), candidates[0][1]):
            notes = notes + [f"VRAM 에 맞추기 위해 해상도 {'+'.join(map(str, res))} · rank {rank} 로 조정"]
        te_q = "none" if vram >= MODEL_SPECS[arch]["te_gb"]["none"] + 1.5 else (
            "int8" if vram >= MODEL_SPECS[arch]["te_gb"]["int8"] + 1.5 else "nf4")
        shape = TrainShape(arch, q, sw, list(res), 1, 1, ckpt, rank, opt, steps_n, sample_every, 2, 20, 768,
                           True, te_q, images)
        sps = estimate_sec_per_step(shape, spec, calib)
        settings = {
            "transformer_quant": q, "text_encoder_quant": te_q, "block_swap": sw,
            "gradient_checkpointing": ckpt, "resolutions": list(res), "rank": rank, "alpha": rank,
            "optimizer": opt, "learning_rate": lr, "max_steps": steps_n, "batch_size": 1, "grad_accum": 1,
            "save_every": max(100, (steps_n // 8) // 50 * 50), "sample_every": sample_every,
        }
        if sw:
            gb = sw * MODEL_SPECS[arch]["weights_gb"][q] / MODEL_SPECS[arch]["blocks"]
            notes = notes + [f"블록 스왑 {sw}개 (RAM {gb:.1f}GB 사용, 속도 감소)"]
        return Recommendation(
            key, title, subtitle, settings, estimate_vram(shape), sps, sps * steps_n,
            estimate_prep_sec(shape, spec), estimate_sample_sec(shape, spec, calib), n_cal, fits, notes,
        )

    opts = []
    # 퀄리티: 가장 높은 정밀도 (블록 스왑 허용), 멀티 해상도, rank 상향, 스텝 +30%
    q_res = [768, max_res] if max_res > 768 else [768]
    q_opt = "adamw" if usable >= 20 else "adamw8bit"
    q_cands = [(q_res, 32), (q_res, rank_main), ([max_res], rank_main), ([768], rank_main)]
    opts.append(build("quality", "퀄리티 중시", "정밀도·해상도 우선", q_cands, q_opt, True,
                      int(round(steps * 1.3 / 50) * 50), lr_main, max(250, steps // 8 // 50 * 50),
                      order, True, ["멀티 해상도로 다양한 크기에 강함"] if len(q_res) > 1 else []))
    # 최적: 스왑 없이 들어가는 가장 높은 정밀도, 기본 해상도
    opts.append(build("balanced", "최적", "품질·속도 균형 (권장)", [([max_res], rank_main), ([max_res], 16), ([768], rank_main), ([768], 16)],
                      "adamw8bit", True,
                      steps, lr_main, 250, order, False, []))
    # 속도: 스왑 없이 가장 빠른 조합, 768, 스텝 -25%, 샘플 간격 넓게
    fast = [q for q in ("fp8", "none", "nf4")]
    sp = build("speed", "속도 중시", "빠른 결과 확인용", [([768], 16), ([512], 16)], "adamw8bit", True,
               int(round(steps * 0.75 / 50) * 50), lr_main * 1.25, 500, fast, False,
               ["768 해상도 · 스텝 25% 감소 (학습률 소폭 상향)"])
    # 여유가 있으면 체크포인팅 끄기 (더 빠름)
    shape_off = TrainShape(arch, sp.settings["transformer_quant"], 0, list(sp.settings["resolutions"]), 1, 1,
                           False, 16, "adamw8bit")
    if sp.fits and sp.settings["block_swap"] == 0 and estimate_vram(shape_off) <= usable:
        sp.settings["gradient_checkpointing"] = False
        shape_off.max_steps = sp.settings["max_steps"]
        sp.sec_per_step = estimate_sec_per_step(shape_off, spec, calib)
        sp.train_sec = sp.sec_per_step * sp.settings["max_steps"]
        sp.vram_gb = estimate_vram(shape_off)
        sp.notes.append("VRAM 여유가 있어 그래디언트 체크포인팅 끔 (약 20% 빠름)")
    opts.append(sp)
    if not any(o.fits for o in opts):
        other = "MOSS-Image" if arch == "krea2" else "Krea 2"
        warnings.insert(0, f"이 GPU 의 VRAM({vram:.0f}GB)으로는 {'Krea 2' if arch == 'krea2' else 'MOSS-Image'} 학습이 어려울 수 있습니다. "
                           f"{other} 를 고려하거나 더 큰 VRAM 의 GPU 를 사용하세요.")
    return Analysis(spec, stats, arch, purpose, opts, warnings, usable)


# ---------------------------------------------------------------------------
# 적용 / 현재 설정 추정
# ---------------------------------------------------------------------------
def apply_to_job(job, rec: Recommendation) -> None:
    s = rec.settings
    m, n, t, d = job.model, job.network, job.train, job.dataset
    m.transformer_quant = s["transformer_quant"]
    m.text_encoder_quant = s["text_encoder_quant"]
    m.block_swap = int(s["block_swap"])
    m.gradient_checkpointing = bool(s["gradient_checkpointing"])
    d.resolutions = list(s["resolutions"])
    n.rank = int(s["rank"])
    n.alpha = float(s["alpha"])
    t.optimizer = s["optimizer"]
    t.learning_rate = float(s["learning_rate"])
    t.max_steps = int(s["max_steps"])
    t.batch_size = int(s["batch_size"])
    t.grad_accum = int(s["grad_accum"])
    t.save_every = int(s["save_every"])
    job.sample.every = int(s["sample_every"])


def shape_from_job(job, spec: SystemSpec, images: int = 25) -> TrainShape:
    """현재 설정을 추정용 조합으로 변환 ('auto' 는 실제 학습과 같은 규칙으로 해석)."""
    from .device import auto_memory_plan

    plan = auto_memory_plan(job.model.arch, spec.vram_gb if spec.gpu_available else 0.0)
    q = plan.transformer_quant if job.model.transformer_quant == "auto" else job.model.transformer_quant
    teq = plan.text_encoder_quant if job.model.text_encoder_quant == "auto" else job.model.text_encoder_quant
    sw = plan.block_swap if job.model.block_swap < 0 else job.model.block_swap
    if q == "nf4":
        sw = 0
    sc = job.sample
    return TrainShape(
        job.model.arch, q, sw, list(job.dataset.resolutions or [1024]), job.train.batch_size,
        job.train.grad_accum, job.model.gradient_checkpointing, job.network.rank, job.train.optimizer,
        job.train.max_steps, sc.every, max(1, len([p for p in sc.prompts if p.strip()])), sc.steps,
        max(sc.width, sc.height), sc.enabled, teq, images,
    )


def fmt_duration(sec: float) -> str:
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m = rem // 60
    if h:
        return f"{h}시간 {m:02d}분"
    return f"{max(1, m)}분"


def fmt_speed(sec_per_step: float) -> str:
    if sec_per_step <= 0:
        return "-"
    if sec_per_step >= 1:
        return f"{sec_per_step:.2f} s/it  ({1 / sec_per_step:.2f} it/s)"
    return f"{1 / sec_per_step:.2f} it/s  ({sec_per_step:.2f} s/it)"
