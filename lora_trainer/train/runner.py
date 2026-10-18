"""학습 실행 진입점 (GUI 가 별도 프로세스로 실행하거나 CLI 로 직접 실행).

사용법
  python -m lora_trainer.train.runner --config presets/krea2_12gb_character.yaml
  python -m lora_trainer.train.runner --config my.yaml --steps 3000 --name test
허깅페이스 토큰은 환경변수 HF_TOKEN 또는 설정 파일의 project.hf_token 으로 전달한다.
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="LoRA Trainer 학습 실행")
    parser.add_argument("--config", required=True, help="학습 설정 YAML 경로")
    parser.add_argument("--steps", type=int, default=None, help="max_steps 덮어쓰기")
    parser.add_argument("--name", default=None, help="LoRA 이름 덮어쓰기")
    args = parser.parse_args(argv)

    # Windows 콘솔/파이프에서 한글 출력이 깨지지 않도록
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", line_buffering=True)
        except Exception:
            pass

    if os.name != "nt":  # Windows 는 expandable_segments 미지원
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from ..core.config import TrainingJob
    from ..core.events import EventSink
    from .trainer import Trainer

    job = TrainingJob.load_yaml(args.config)
    if args.steps:
        job.train.max_steps = args.steps
    if args.name:
        job.project.name = args.name
    if not job.project.hf_token:
        job.project.hf_token = os.environ.get("HF_TOKEN", "")

    trainer = Trainer(job, EventSink())
    try:
        result = trainer.run()
    except Exception:
        return 1
    return 0 if result is not None else 2


if __name__ == "__main__":
    sys.exit(main())
