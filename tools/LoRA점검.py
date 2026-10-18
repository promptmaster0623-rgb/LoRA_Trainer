"""LoRA 파일 점검: 학습이 실제로 되었는지 / ComfyUI 에서 안 먹는 이유가 무엇인지 진단.

사용: LoRA점검.bat 에 .safetensors 파일을 끌어다 놓거나, 인자 없이 실행하면 output 폴더의 최근 LoRA 를 검사.
  - lora_up 가중치는 0 에서 시작하므로, 0 에 가깝다면 '학습 자체가 안 된 것'
  - 값이 충분히 크다면 학습은 된 것 → ComfyUI 적용 단계(키 이름 · 모델 종류 · 강도) 문제
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def say(s=""):
    print(s, flush=True)


def inspect(path: str) -> None:
    import torch
    from safetensors import safe_open

    say("=" * 70)
    say(f"파일: {path}")
    with safe_open(path, framework="pt") as f:
        meta = f.metadata() or {}
        keys = list(f.keys())
        tensors = {k: f.get_tensor(k) for k in keys}
    say(f"키 {len(keys)}개 · 크기 {os.path.getsize(path) / 1024 ** 2:.1f} MB")

    # 형식 판별 -------------------------------------------------------------
    fmt = "알 수 없음"
    if any(k.startswith("lora_unet_") for k in keys):
        fmt = "kohya (lora_unet_*)"
    elif any(k.startswith("diffusion_model.") and ".lora_A." in k for k in keys):
        fmt = "ComfyUI (diffusion_model.*.lora_A/B)"
    say(f"형식: {fmt}")

    # 메타데이터 ---------------------------------------------------------------
    cfg = {}
    if meta.get("lora_trainer_config"):
        try:
            cfg = json.loads(meta["lora_trainer_config"])
        except json.JSONDecodeError:
            pass
    say(f"베이스 모델: {meta.get('ss_base_model_version', '?')} · 학습 스텝: {meta.get('ss_training_steps', '?')} · "
        f"트레이너 v{meta.get('lora_trainer_version', '?')} · 트리거: {meta.get('lora_trainer_trigger', '') or '(없음)'}")
    if cfg:
        m, t, n = cfg.get("model", {}), cfg.get("train", {}), cfg.get("network", {})
        say(f"설정: arch={m.get('arch')} quant={m.get('transformer_quant')} swap={m.get('block_swap')} dtype={m.get('dtype')} | "
            f"rank={n.get('rank')} alpha={n.get('alpha')} target={n.get('target')} | "
            f"opt={t.get('optimizer')} lr={t.get('learning_rate')} steps={t.get('max_steps')} "
            f"accum={t.get('grad_accum')} sched={t.get('lr_scheduler')}")
        ds = cfg.get("dataset", {})
        say(f"데이터셋: 폴더 {len(ds.get('folders', []))}개 "
            + ", ".join(f"{os.path.basename(str(x.get('path', '')))}×{x.get('repeats')}" for x in ds.get("folders", []))
            + f" · 해상도 {ds.get('resolutions')} · 캡션드롭 {ds.get('caption_dropout')}")

    # 모듈별 가중치 --------------------------------------------------------------
    mods = defaultdict(dict)
    for k, v in tensors.items():
        for suf, kind in ((".lora_up.weight", "up"), (".lora_B.weight", "up"),
                          (".lora_down.weight", "down"), (".lora_A.weight", "down"), (".alpha", "alpha")):
            if k.endswith(suf):
                mods[k[: -len(suf)]][kind] = v
    say(f"LoRA 모듈: {len(mods)}개")
    if not mods:
        say("[문제] LoRA 가중치 키가 없습니다.")
        return
    ups, deltas, zero, nan = [], [], 0, 0
    per_block = defaultdict(list)
    for name, d in mods.items():
        if "up" not in d or "down" not in d:
            continue
        up, down = d["up"].float(), d["down"].float()
        if not torch.isfinite(up).all() or not torch.isfinite(down).all():
            nan += 1
            continue
        rank = down.shape[0]
        alpha = float(d["alpha"]) if "alpha" in d else rank
        a = up.abs().mean().item()
        ups.append(a)
        if up.abs().max().item() == 0:
            zero += 1
        # 실제 가중치 변화량 ||ΔW|| (rank 이 작아 저렴하게 계산)
        delta = (torch.linalg.matrix_norm(up @ down) * alpha / rank).item() if up.dim() == 2 else 0.0
        deltas.append(delta)
        m = re.search(r"(layers|blocks)[._](\d+)", name)
        per_block[int(m.group(2)) if m else -1].append(delta)
    if nan:
        say(f"[문제] NaN/Inf 가 들어 있는 모듈 {nan}개 → 학습이 발산했습니다. 학습률을 낮추거나 적응형 '발산 복구' 를 켜세요.")
    if not ups:
        return
    mean_up = sum(ups) / len(ups)
    mean_delta = sum(deltas) / len(deltas)
    say(f"lora_up 평균 |w|: {mean_up:.2e} · 0 인 모듈 {zero}/{len(ups)} · 평균 ||ΔW||: {mean_delta:.3e}")
    if per_block:
        rows = sorted(per_block.items())
        line = " ".join(f"{b}:{sum(v) / len(v):.2e}" for b, v in rows[:: max(1, len(rows) // 8)])
        say(f"블록별 ||ΔW|| (일부): {line}")
    sample = sorted(mods)[:3]
    say("키 예시: " + " | ".join(sample))

    # 판정 -------------------------------------------------------------------
    say("")
    if zero == len(ups) or mean_up < 1e-6:
        say("▶ 판정: 학습이 되지 않았습니다 (가중치가 처음 상태 그대로).")
        say("  원인 후보: 학습률이 0에 가까움 / 그래디언트가 LoRA 에 흐르지 않음 / 저장 전에 중지됨.")
        say("  → 이 결과를 그대로 복사해 Claude 에게 보내주세요.")
    elif mean_up < 1e-4:
        say("▶ 판정: 학습이 매우 약합니다. ComfyUI 에서 강도 1 로 거의 차이가 없을 수 있습니다.")
        say("  → 스텝 · 학습률을 늘리거나, 강도를 2 이상으로 올려 차이가 나는지 확인하세요.")
    else:
        say("▶ 판정: 가중치는 정상적으로 학습되었습니다. ComfyUI 적용 단계를 확인하세요.")
        say("  1) ComfyUI 콘솔에 'lora key not loaded' 가 뜨는지 (뜨면 키 형식 문제 → 저장 형식을 ComfyUI 로 바꿔 재저장)")
        say("  2) 베이스 모델이 학습 때와 같은 MOSS-Image(Base/SFT) 인지")
        say("  3) 프롬프트에 트리거 단어를 넣었는지 · LoRA 강도 1.0 이상인지")


def main(argv) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    paths = [a for a in argv if a.lower().endswith(".safetensors")]
    if not paths:
        found = glob.glob(os.path.join(ROOT, "output", "**", "*.safetensors"), recursive=True)
        found = [p for p in found if os.sep + "cache" + os.sep not in p and os.sep + "state" + os.sep not in p]
        paths = sorted(found, key=os.path.getmtime, reverse=True)[:5]
        if not paths:
            say("검사할 LoRA 가 없습니다. .safetensors 파일을 LoRA점검.bat 위로 끌어다 놓으세요.")
            return 1
        say(f"output 폴더의 최근 LoRA {len(paths)}개를 검사합니다.")
    for p in paths:
        try:
            inspect(p)
        except Exception as e:  # pragma: no cover
            say(f"[오류] {p}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
