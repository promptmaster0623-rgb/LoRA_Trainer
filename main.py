"""LoRA Trainer 실행 파일.

  python main.py            -> GUI 실행
  python main.py --config x.yaml  -> GUI 없이 바로 학습 (CLI)
"""

import sys


def _main() -> int:
    if "--config" in sys.argv:
        from lora_trainer.train.runner import main as run

        return run(sys.argv[1:])
    from lora_trainer.gui.app import main as gui

    return gui()


if __name__ == "__main__":
    sys.exit(_main())
