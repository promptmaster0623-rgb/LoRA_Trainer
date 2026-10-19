"""학습 설정(Config) 정의.

모든 설정은 dataclass로 정의되며 YAML 파일로 저장/불러오기가 가능하다.
GUI 와 CLI 가 동일한 설정 구조를 공유한다.
"""

from __future__ import annotations

import copy
import dataclasses
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# 개별 섹션
# ---------------------------------------------------------------------------
@dataclass
class ProjectConfig:
    name: str = "my_lora"                 # LoRA 이름 (출력 파일명에 사용)
    output_dir: str = "output"            # 출력 루트 폴더 (프로젝트별 하위 폴더 생성)
    models_dir: str = "models"            # 허깅페이스에서 내려받은 모델 저장 위치
    cache_dir: str = ""                   # 비워두면 <output_dir>/<name>/cache 사용
    hf_token: str = ""                    # 게이트 모델(Krea 2 등) 다운로드용 토큰


@dataclass
class ModelConfig:
    arch: str = "krea2"                   # krea2 | moss_image
    transformer_path: str = ""            # 파일/폴더/허깅페이스 경로 (비우면 기본값)
    text_encoder_path: str = ""
    tokenizer_path: str = ""
    vae_path: str = ""
    dtype: str = "bf16"                   # 연산 dtype: bf16 | fp16
    transformer_quant: str = "auto"       # auto | none | fp8 | nf4
    text_encoder_quant: str = "auto"      # auto | none | int8 | nf4
    block_swap: int = -1                  # -1 = 자동, 0 = 사용 안 함, N = CPU로 내릴 블록 수
    gradient_checkpointing: bool = True
    attention: str = "sdpa"               # sdpa (PyTorch 기본)


@dataclass
class DatasetFolder:
    path: str = ""
    repeats: int = 1                      # 에폭당 반복 횟수
    trigger_word: str = ""                # 폴더별 트리거 단어 (비우면 전역값)
    caption_ext: str = ".txt"


@dataclass
class DatasetConfig:
    folders: List[DatasetFolder] = field(default_factory=list)
    resolutions: List[int] = field(default_factory=lambda: [1024])
    trigger_word: str = ""                # 전역 트리거 단어
    trigger_position: str = "prefix"      # prefix | replace_token (캡션 속 [trigger] 치환)
    default_caption: str = ""             # 캡션 파일이 없을 때 사용
    caption_dropout: float = 0.05         # 빈 캡션으로 학습할 확률
    flip_horizontal: bool = False         # 좌우반전 증강 (캐시에 반전본도 저장)
    bucket_min: int = 512                 # 버킷 최소 변 길이
    bucket_max: int = 2048                # 버킷 최대 변 길이
    bucket_step: int = 64                 # 버킷 해상도 단위 (16의 배수)
    num_workers: int = 0


@dataclass
class NetworkConfig:
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0                  # LoRA 입력 드롭아웃
    rank_dropout: float = 0.0
    target: str = "attn_mlp"              # attn | attn_mlp | all_linear | custom
    include_regex: str = ""               # target=custom 일 때 포함 정규식
    exclude_regex: str = ""               # 제외 정규식 (항상 적용)
    rs_lora: bool = False                 # Rank-Stabilized LoRA 스케일
    init_weights: str = ""                # 기존 LoRA 에서 이어 학습


@dataclass
class TimestepConfig:
    sampling: str = "logit_normal"        # uniform | logit_normal
    logit_mean: float = 0.0
    logit_std: float = 1.0
    shift_mode: str = "auto"              # auto | none | fixed | dynamic
    shift: float = 3.0                    # fixed 모드 shift 값
    # MOSS-Image 전용 (shift_mode=auto): sigma < 0.5 (깨끗한 쪽)에 배정할 학습 스텝 비율 %.
    # Fizgig v8.7.0 기본 60. 0 이면 모델 자체 스케줄(약 11%).
    clean_share: float = 60.0


@dataclass
class AdaptiveConfig:
    """실시간 적응형 학습 (train/adaptive.py)."""
    enabled: bool = False                 # 전체 사용 여부
    plateau: bool = True                  # 정체 감지 → 학습률 자동 감소
    plateau_patience: int = 300           # 이 스텝 수 동안 개선이 없으면 감소
    plateau_factor: float = 0.5           # 감소 비율
    plateau_threshold: float = 0.01       # 1% 이상 낮아져야 개선으로 인정
    min_lr_ratio: float = 0.1             # 자동 배율 하한
    divergence: bool = True               # loss 급등·NaN 시 직전 정상 지점으로 복구
    spike_ratio: float = 3.0              # 단기 평균 loss 가 기준의 몇 배면 급등으로 판단
    max_recover: int = 3                  # 최대 복구 횟수 (넘으면 학습 중단)
    recover_factor: float = 0.5           # 복구 시 학습률 배율
    snapshot_every: int = 100             # 복구용 스냅샷 주기 (스텝)
    auto_clip: bool = False               # 자동 그래디언트 클리핑 (최근 중앙값 × k)
    auto_clip_k: float = 3.0
    eval_every: int = 100                 # 검증 평가 주기 (0 = 끔)
    eval_samples: int = 4                 # 평가에 쓸 이미지 수
    val_holdout: int = 0                  # 학습에서 빼서 검증 전용으로 쓸 이미지 수
    early_stop: bool = False              # 검증 loss 가 개선되지 않으면 조기 종료
    early_stop_patience: int = 5          # 연속 미개선 평가 횟수


@dataclass
class TrainConfig:
    max_steps: int = 2000
    batch_size: int = 1
    grad_accum: int = 1
    learning_rate: float = 1e-4
    optimizer: str = "adamw8bit"          # 목록: train/optim.py 의 OPTIMIZER_SPECS
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    lr_scheduler: str = "constant"        # constant | cosine | linear | cosine_restarts
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    ema_decay: float = 0.98               # LoRA 가중치 EMA (저장·샘플에 사용). 0 = 끔. Fizgig 기본 0.98
    loss: str = "mse"                     # mse | huber
    huber_delta: float = 0.1
    timestep: TimestepConfig = field(default_factory=TimestepConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    seed: int = 42
    save_every: int = 250
    keep_last: int = 4                    # 남겨둘 체크포인트 수 (0 = 전부)
    save_dtype: str = "bf16"              # bf16 | fp16 | fp32
    save_format: str = "kohya"            # kohya | comfy (diffusion_model.*.lora_A/B)
    save_state: bool = True               # 이어학습용 옵티마이저 상태 저장
    resume_state: str = ""                # 이어학습할 state 폴더
    comfy_lora_dir: str = ""              # 저장 시 자동 복사할 ComfyUI loras 폴더


@dataclass
class SampleConfig:
    enabled: bool = True
    every: int = 250
    at_start: bool = False
    prompts: List[str] = field(default_factory=lambda: ["[trigger], portrait photo"])
    negative_prompt: str = ""
    width: int = 768
    height: int = 768
    steps: int = 20
    guidance: float = -1.0                # -1 = 모델 기본값
    seed: int = 1234
    sampler: str = "auto"                 # auto | euler | res_multistep (auto: Krea 2 Euler, MOSS res_multistep)
    # Krea 2 전용: 공식 Turbo LoRA 를 미리보기에만 덧붙여 빠르게 생성 (Fizgig 기본: 0.7, 10스텝, CFG 1, mu 1.15)
    turbo: bool = False
    turbo_lora: str = ""                  # 비우면 Comfy-Org/Krea-2 의 krea2_turbo_lora_rank_64_bf16 자동 다운로드
    turbo_strength: float = 0.7
    turbo_steps: int = 10


@dataclass
class TrainingJob:
    """GUI/CLI 가 다루는 전체 작업 설정."""

    version: int = 1
    project: ProjectConfig = field(default_factory=ProjectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)

    # ------------------------------------------------------------------
    # 경로 헬퍼
    # ------------------------------------------------------------------
    @property
    def run_dir(self) -> str:
        return os.path.join(self.project.output_dir, self.project.name)

    @property
    def cache_root(self) -> str:
        return self.project.cache_dir or os.path.join(self.run_dir, "cache")

    # ------------------------------------------------------------------
    # 직렬화
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingJob":
        return _from_dict(cls, data or {})

    def save_yaml(self, path: str, hide_token: bool = True) -> None:
        data = self.to_dict()
        if hide_token:
            data["project"]["hf_token"] = ""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str) -> "TrainingJob":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def copy(self) -> "TrainingJob":
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # 검증
    # ------------------------------------------------------------------
    def validate(self) -> List[str]:
        """문제 목록을 한글 메시지로 반환한다 (빈 리스트 = 정상)."""
        errors: List[str] = []
        if self.model.arch not in ("krea2", "moss_image"):
            errors.append(f"지원하지 않는 모델: {self.model.arch}")
        if not self.project.name.strip():
            errors.append("LoRA 이름이 비어 있습니다.")
        if not self.dataset.folders:
            errors.append("데이터셋 폴더를 하나 이상 추가하세요.")
        for f in self.dataset.folders:
            if not os.path.isdir(f.path):
                errors.append(f"데이터셋 폴더가 없습니다: {f.path}")
        if self.network.rank <= 0:
            errors.append("LoRA rank 는 1 이상이어야 합니다.")
        if self.train.max_steps <= 0:
            errors.append("학습 스텝은 1 이상이어야 합니다.")
        if self.train.batch_size <= 0:
            errors.append("배치 크기는 1 이상이어야 합니다.")
        if self.dataset.bucket_step % 16 != 0:
            errors.append("버킷 단위는 16의 배수여야 합니다.")
        for r in self.dataset.resolutions:
            if r % 16 != 0 or r < 64:
                errors.append(f"해상도 {r} 는 64 이상, 16의 배수여야 합니다.")
        from ..train.optim import OPTIMIZER_SPECS  # 순환 import 방지용 지연 import

        if self.train.optimizer not in OPTIMIZER_SPECS:
            errors.append(f"알 수 없는 옵티마이저: {self.train.optimizer}")
        if self.sample.sampler not in ("auto", "euler", "res_multistep"):
            errors.append(f"알 수 없는 샘플러: {self.sample.sampler}")
        if not 0.0 <= self.train.ema_decay < 1.0:
            errors.append("EMA 감쇠값은 0 이상 1 미만이어야 합니다 (0 = 끔).")
        if not 0.0 <= self.train.timestep.clean_share < 100.0:
            errors.append("깨끗한 쪽 비율(clean share)은 0 이상 100 미만이어야 합니다.")
        ad = self.train.adaptive
        if ad.enabled:
            if not 0 < ad.plateau_factor < 1:
                errors.append("적응형 학습: 감소 비율은 0~1 사이여야 합니다.")
            if ad.early_stop and ad.eval_every <= 0:
                errors.append("적응형 학습: 조기 종료를 쓰려면 검증 평가 주기를 1 이상으로 설정하세요.")
        if self.model.transformer_quant not in ("auto", "none", "fp8", "nf4"):
            errors.append("트랜스포머 양자화 옵션이 잘못되었습니다.")
        if self.model.text_encoder_quant not in ("auto", "none", "int8", "nf4"):
            errors.append("텍스트 인코더 양자화 옵션이 잘못되었습니다.")
        return errors


# ---------------------------------------------------------------------------
# 내부: dict -> dataclass 재귀 변환 (알 수 없는 키는 무시, 누락 키는 기본값)
# ---------------------------------------------------------------------------
def _from_dict(cls, data: Dict[str, Any]):
    kwargs = {}
    type_hints = {f.name: f for f in fields(cls)}
    for name, f in type_hints.items():
        if name not in data:
            continue
        value = data[name]
        default_obj = (
            f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        )
        if is_dataclass(default_obj) and isinstance(value, dict):
            kwargs[name] = _from_dict(type(default_obj), value)
        elif name == "folders" and isinstance(value, list):
            kwargs[name] = [
                _from_dict(DatasetFolder, v) if isinstance(v, dict) else DatasetFolder(path=str(v))
                for v in value
            ]
        else:
            kwargs[name] = value
    return cls(**kwargs)
