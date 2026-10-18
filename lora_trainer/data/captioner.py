"""Qwen3-VL 자동 캡션 (별도 프로세스로 실행).

python -m lora_trainer.data.captioner --folder <폴더> [--style detailed|short|tags] [--overwrite]
"""

from __future__ import annotations

import argparse
import os
import sys

STYLE_PROMPTS = {
    "detailed": (
        "Describe this image in one detailed English paragraph for training an image generation model. "
        "Cover the subject, appearance, clothing, pose, expression, background, lighting, colors, "
        "camera angle and art style. Do not start with 'The image shows'. No line breaks."
    ),
    "short": "Describe this image in one concise English sentence for an image generation caption.",
    "tags": (
        "List comma-separated English tags describing this image (subject, attributes, clothing, pose, "
        "background, lighting, style). Output only the tags."
    ),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Qwen3-VL 자동 캡션")
    ap.add_argument("--folder", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--style", default="detailed", choices=list(STYLE_PROMPTS))
    ap.add_argument("--quant", default="none", choices=["none", "nf4", "int8"])
    ap.add_argument("--extra", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", line_buffering=True)
        except Exception:
            pass

    import torch
    from PIL import Image

    from ..core.events import EventSink
    from ..models.hf_loading import load_hf_model
    from ..utils.hub import resolve_path
    from .dataset import list_images, read_caption, write_caption

    sink = EventSink()
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        paths = [p for p in list_images(args.folder) if args.overwrite or not read_caption(p).strip()]
        if not paths:
            sink.log("캡션이 필요한 이미지가 없습니다.")
            return 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model_path = resolve_path(args.model, args.models_dir, os.environ.get("HF_TOKEN"), log=sink.log)
        sink.log(f"캡션 모델 로딩: {model_path}")
        processor = AutoProcessor.from_pretrained(model_path)
        model = load_hf_model(Qwen3VLForConditionalGeneration, model_path, dtype, device, args.quant)
        instruction = STYLE_PROMPTS[args.style] + (" " + args.extra if args.extra else "")
        for i, path in enumerate(paths):
            img = Image.open(path).convert("RGB")
            img.thumbnail((1024, 1024))
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": instruction}]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            text = " ".join(text.strip().split())
            write_caption(path, text)
            sink.emit("cache", kind="caption", done=i + 1, total=len(paths), file=os.path.basename(path))
            sink.log(f"[{i + 1}/{len(paths)}] {os.path.basename(path)}: {text[:120]}")
        return 0
    except Exception as e:  # pragma: no cover
        import traceback

        sink.emit("error", message=str(e), traceback=traceback.format_exc())
        sink.log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
