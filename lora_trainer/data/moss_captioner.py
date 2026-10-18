"""MOSS-Image 공식 형식 자동 캡션 (Ollama · Gemma 4, 별도 프로세스로 실행).

python -m lora_trainer.data.moss_captioner --folder <폴더> [--mode reformat|missing|all]
    [--url http://localhost:11434] [--model gemma4:e4b] [--variant long|medium|short]

  mode
    missing   캡션이 없는 이미지만 새로 생성
    reformat  (기본) 없는 것은 새로 만들고, 공식 형식이 아닌 캡션은 공식 형식으로 다시 생성
    all       모든 이미지를 다시 생성
  기존 캡션은 덮어쓰기 전에 <폴더>/_caption_backup/<시각>/ 에 백업하고,
  다시 만들 때 공식 가이드처럼 reference 로 넘겨 참고하게 한다.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time


def main(argv=None) -> int:
    from .moss_caption import (
        DEFAULT_OLLAMA_MODEL,
        DEFAULT_OLLAMA_URL,
        EDITED,
        JSON_IN_TXT,
        MISSING,
        OTHER,
        STATUS_LABELS,
        VARIANTS,
        OllamaCaptionConfig,
        OllamaError,
        caption_image,
        caption_status,
        extract_json,
        list_models,
        model_capabilities,
        needs_work,
        normalize_record,
        pick_model,
        sidecar_path,
        strip_trigger,
        training_text,
        unload_model,
        write_record,
    )

    ap = argparse.ArgumentParser(description="MOSS-Image 공식 형식 자동 캡션 (Ollama Gemma 4)")
    ap.add_argument("--folder", required=True)
    ap.add_argument("--mode", default="reformat", choices=["missing", "reformat", "all"])
    ap.add_argument("--url", default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    ap.add_argument("--variant", default="long", choices=list(VARIANTS))
    ap.add_argument("--trigger", default="", help="기존 캡션에서 뺄 트리거 단어 (학습 시 다시 붙음)")
    ap.add_argument("--subject", default="", help="주 피사체를 부를 일반 표현 (예: a young woman)")
    ap.add_argument("--extra", default="")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--no-unload", action="store_true", help="끝난 뒤 Ollama 모델을 VRAM 에 남겨둠")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", line_buffering=True)
        except Exception:
            pass

    from ..core.events import EventSink
    from .dataset import caption_path, list_images, read_caption, write_caption

    sink = EventSink()
    cfg = OllamaCaptionConfig(url=args.url, model=args.model, think=not args.no_think,
                              temperature=args.temperature, top_p=args.top_p, num_ctx=args.num_ctx,
                              timeout=args.timeout, max_side=args.max_side)
    try:
        images = list_images(args.folder)
        statuses = {p: caption_status(p, trigger=args.trigger) for p in images}
        counts = {}
        for s in statuses.values():
            counts[s] = counts.get(s, 0) + 1
        sink.log("캡션 상태: " + ", ".join(f"{STATUS_LABELS[k]} {v}" for k, v in counts.items()) if counts else "이미지가 없습니다.")
        todo = [p for p in images if needs_work(statuses[p], args.mode)]
        if not todo:
            sink.log("작업할 이미지가 없습니다. 모든 캡션이 이미 공식 형식입니다.")
            return 0

        backup_dir = ""
        if not args.no_backup and any(statuses[p] != MISSING for p in todo):
            backup_dir = os.path.join(args.folder, "_caption_backup", time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(backup_dir, exist_ok=True)
            sink.log(f"기존 캡션 백업 폴더: {backup_dir}")

        # VLM 이 필요 없는 항목 (.txt 안에 공식 JSON) 먼저 변환
        need_vlm = []
        done = 0
        for p in todo:
            if statuses[p] == JSON_IN_TXT and args.mode != "all":
                rec = normalize_record(extract_json(read_caption(p)))
                _backup(p, backup_dir, caption_path, sidecar_path)
                write_record(p, rec, "(txt 변환)", args.variant)
                write_caption(p, training_text(rec, args.variant))
                done += 1
                sink.emit("cache", kind="caption", done=done, total=len(todo), file=os.path.basename(p))
            else:
                need_vlm.append(p)

        if need_vlm:
            installed = list_models(cfg.url)
            model = pick_model(cfg.model, installed)
            if installed and model not in installed:
                raise OllamaError(f"Ollama 에 '{cfg.model}' 모델이 없습니다. 설치된 모델: {', '.join(installed) or '없음'}\n"
                                  f"명령 프롬프트에서 'ollama pull {cfg.model}' 로 받으세요.")
            if model != cfg.model:
                sink.log(f"설정한 모델 '{cfg.model}' 이 없어 설치된 '{model}' 을 사용합니다.")
            cfg.model = model
            caps = model_capabilities(cfg.url, model)
            if caps and "vision" not in caps:
                raise OllamaError(f"'{model}' 은 이미지 입력(vision)을 지원하지 않습니다. Gemma 4 비전 모델을 선택하세요.")
            if caps and "thinking" not in caps and cfg.think:
                cfg.think = False
                sink.log(f"'{model}' 은 thinking 을 지원하지 않아 끄고 진행합니다.")
            sink.log(f"Ollama 캡션 모델: {model} · thinking {'켬' if cfg.think else '끔'} · 학습용 변형: {args.variant}")

        failed = []
        t0 = time.time()
        for idx, p in enumerate(need_vlm):
            name = os.path.basename(p)
            st = statuses[p]
            reference = ""
            if st == JSON_IN_TXT:
                old = normalize_record(extract_json(read_caption(p)))
                reference = strip_trigger(old["caption"]["long"], args.trigger) if old else ""
            elif st != MISSING:
                reference = strip_trigger(read_caption(p), args.trigger)
            try:
                rec, _raw = caption_image(p, cfg, reference=reference, subject_hint=args.subject,
                                          extra=args.extra, log=sink.log)
            except OllamaError as e:
                failed.append(name)
                sink.log(f"[실패] {name}: {e}")
                if "연결할 수 없습니다" in str(e):
                    raise
                continue
            _backup(p, backup_dir, caption_path, sidecar_path)
            write_record(p, rec, cfg.model, args.variant)
            text = training_text(rec, args.variant)
            write_caption(p, text)
            done += 1
            avg = (time.time() - t0) / (idx + 1)
            sink.emit("cache", kind="caption", done=done, total=len(todo), file=name)
            tag = {MISSING: "새로 생성", OTHER: "형식 변환", EDITED: "다시 생성", JSON_IN_TXT: "다시 생성"}.get(st, "다시 생성")
            sink.log(f"[{done}/{len(todo)}] {tag} · {name} ({avg:.1f}s/장): {text[:110]}")

        if need_vlm and not args.no_unload:
            unload_model(cfg.url, cfg.model)
            sink.log("Ollama 모델을 VRAM 에서 내렸습니다 (학습에 VRAM 확보).")
        if failed:
            sink.log(f"실패 {len(failed)}장: {', '.join(failed[:10])}{' …' if len(failed) > 10 else ''}")
            return 2 if done == 0 else 0
        sink.log(f"완료: {done}장 공식 형식으로 저장")
        return 0
    except Exception as e:  # pragma: no cover
        import traceback

        sink.emit("error", message=str(e), traceback=traceback.format_exc())
        sink.log(str(e))
        return 1


def _backup(image_path: str, backup_dir: str, caption_path, sidecar_path) -> None:
    if not backup_dir:
        return
    for src in (caption_path(image_path), sidecar_path(image_path)):
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(backup_dir, os.path.basename(src)))


if __name__ == "__main__":
    sys.exit(main())
