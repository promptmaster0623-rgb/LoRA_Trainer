"""MOSS-Image 공식 가이드 형식 캡션 + Ollama(Gemma 4) 로컬 연결.

공식 가이드(OpenMOSS/MOSS-Image USERGUIDE.md · Image Captioning) 요약
  - VLM(Gemma 4 / Qwen3.5) 으로 이미지를 다시 캡션하며, 기존 캡션은 {{reference}} 로 넘겨
    '참고만 하고 이미지에 맞지 않는 부분은 고치도록' 한다.
  - 출력은 JSON 하나: caption.long / caption.medium / caption.short, watermark[], borders[]
  - 보이는 글자는 정확히 옮기고(워터마크 제외), 글자 배치(줄바꿈·정렬·크기)까지 서술
  - 워터마크 글자는 어떤 캡션에도 넣지 않는다.
  - 요청 설정: temperature 1.0, top_p 0.95, 최대 16384 토큰, thinking 켬

저장 방식 (이미지 img001.png 기준)
  img001.moss.json  공식 형식 원본 JSON (long/medium/short/watermark/borders + 메타)
  img001.txt        학습에 쓰는 캡션 = 선택한 변형(long/medium/short) 텍스트
                    (Krea 2 · 기존 학습 코드와 그대로 호환)
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

SIDECAR_EXT = ".moss.json"
VARIANTS = ("long", "medium", "short")
BORDER_VALUES = ("left", "right", "up", "down", "none")
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:e4b"
FORMAT_VERSION = 1

# 공식 가이드 권장 요청 설정
OFFICIAL_OPTIONS = {"temperature": 1.0, "top_p": 0.95, "num_predict": 16384}

# ---------------------------------------------------------------------------
# 시스템 프롬프트 (공식 가이드의 리캡셔너 규칙을 그대로 따르도록 작성)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an image recaptioner that writes training captions for the MOSS-Image text-to-image model.
Look at the image carefully and return exactly ONE JSON object with this schema:

{
  "caption": {
    "long": "<detailed description>",
    "medium": "<one-paragraph description>",
    "short": "<one-sentence description>"
  },
  "watermark": [ {"box_2d": [y_min, x_min, y_max, x_max], "label": "<watermark text>"} ],
  "borders": ["left" | "right" | "up" | "down"]  or  ["none"]
}

Rules for "caption":
- Write in natural, fluent English. Describe only what is actually visible.
- "long": a detailed description covering overall composition and framing, every main subject
  (appearance, face and expression, hair, clothing, accessories, pose, action), their positions
  and relationships, the setting and background, colors, lighting, camera angle / lens feel,
  and the medium or art style (photo, illustration, 3D render, anime, painting ...).
- "medium": one paragraph with the most important content of "long".
- "short": one sentence that summarizes the image.
- Visible text that is NOT a watermark must be transcribed exactly, inside double quotes, keeping
  the original language, spelling and capitalization. In "long", also describe the text layout:
  line breaks, alignment, relative size / hierarchy, font style, color and where it is placed.
- Never mention watermark text, signatures used as watermarks, or logos overlaid as watermarks
  in any caption field.
- Do not start with phrases like "The image shows" or "This is a picture of". Start directly
  with the content.
- Do not guess names of real people or characters unless their name is visibly written in the image.
- Do not add opinions, guesses about what is outside the frame, or meta commentary.

Rules for "watermark":
- List every watermark (semi-transparent overlay text, stock-photo marks, site URLs, artist
  handles stamped over the image). "box_2d" uses coordinates normalized to 0-1000 in the order
  [y_min, x_min, y_max, x_max]. "label" is the watermark text, or a short description if unreadable.
- Use an empty list [] when there is no watermark.

Rules for "borders":
- List the sides that have an evident border, frame, letterbox or padding bar:
  any of "left", "right", "up", "down".
- Use ["none"] when there are no evident borders. "none" must never be combined with other values.

Reference caption:
- The user may give a reference caption (an older caption of this image). Use it only as a hint
  for details that are hard to see; if it conflicts with the image, follow the image and correct it.

Output only the JSON object. No markdown, no code fences, no explanation."""

# Ollama 구조화 출력용 JSON 스키마
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {
            "type": "object",
            "properties": {v: {"type": "string"} for v in VARIANTS},
            "required": list(VARIANTS),
        },
        "watermark": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "box_2d": {"type": "array", "items": {"type": "number"}},
                    "label": {"type": "string"},
                },
                "required": ["box_2d", "label"],
            },
        },
        "borders": {"type": "array", "items": {"type": "string", "enum": list(BORDER_VALUES)}},
    },
    "required": ["caption", "watermark", "borders"],
}


def build_user_prompt(reference: str = "", subject_hint: str = "", extra: str = "") -> str:
    """사용자 메시지. 기존 캡션은 공식 가이드처럼 reference 로 전달."""
    parts = ["Recaption this image following the system rules."]
    if subject_hint:
        parts.append(f"The main subject is: {subject_hint}. Refer to it with this generic description.")
    if extra:
        parts.append(f"Additional instruction: {extra}")
    ref = reference.strip() or "(none)"
    parts.append(f"Reference caption:\n{ref}")
    parts.append("Return only the JSON object.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 형식 검사 · 정규화
# ---------------------------------------------------------------------------
def sidecar_path(image_path: str) -> str:
    return os.path.splitext(image_path)[0] + SIDECAR_EXT


def _clean_text(s) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n").strip()
    # 연속 공백 정리 (줄바꿈은 글자 배치 정보라서 유지)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def extract_json(text: str) -> Optional[dict]:
    """모델 출력에서 JSON 객체 하나를 꺼낸다 (코드펜스·앞뒤 잡담 허용)."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(t[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
                    break
        start = t.find("{", start + 1)
    return None


def normalize_record(obj) -> Optional[dict]:
    """공식 스키마로 정규화. 필수 항목(caption.long/medium/short)이 없으면 None."""
    if not isinstance(obj, dict):
        return None
    cap = obj.get("caption")
    if not isinstance(cap, dict):
        # 평평한 형태 {"long":..,"medium":..,"short":..} 도 허용
        if all(k in obj for k in VARIANTS):
            cap = {k: obj[k] for k in VARIANTS}
        else:
            return None
    caption = {k: _clean_text(cap.get(k)) for k in VARIANTS}
    if not all(caption.values()):
        return None
    marks = []
    for w in obj.get("watermark") or []:
        if not isinstance(w, dict):
            continue
        box = w.get("box_2d") or []
        try:
            box = [max(0, min(1000, int(round(float(v))))) for v in box][:4]
        except (TypeError, ValueError):
            box = []
        if len(box) != 4:
            continue
        marks.append({"box_2d": box, "label": _clean_text(w.get("label")) or "watermark"})
    borders = [b for b in (obj.get("borders") or []) if isinstance(b, str)]
    borders = [b.strip().lower() for b in borders if b.strip().lower() in BORDER_VALUES]
    borders = list(dict.fromkeys(borders))
    if not borders or ("none" in borders and len(borders) > 1):
        borders = [b for b in borders if b != "none"] or ["none"]
    # 워터마크 글자가 캡션에 섞였으면 제거 (공식 규칙: 캡션에 워터마크 텍스트 금지)
    for m in marks:
        lab = m["label"]
        if len(lab) >= 4:
            for k in VARIANTS:
                caption[k] = _clean_text(caption[k].replace(f'"{lab}"', "").replace(lab, ""))
    return {"caption": caption, "watermark": marks, "borders": borders}


def is_official_record(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    cap = obj.get("caption")
    if not isinstance(cap, dict) or not all(isinstance(cap.get(k), str) and cap.get(k).strip() for k in VARIANTS):
        return False
    if not isinstance(obj.get("watermark"), list):
        return False
    b = obj.get("borders")
    if not isinstance(b, list) or not b or any(x not in BORDER_VALUES for x in b):
        return False
    return not ("none" in b and len(b) > 1)


def read_record(image_path: str) -> Optional[dict]:
    p = sidecar_path(image_path)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return obj if is_official_record(obj) else None


def write_record(image_path: str, record: dict, model: str = "", variant: str = "long") -> None:
    data = dict(record)
    data["_meta"] = {"format": "moss-image-recaption", "version": FORMAT_VERSION,
                     "captioner": model, "train_variant": variant}
    with open(sidecar_path(image_path), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def training_text(record: dict, variant: str = "long") -> str:
    variant = variant if variant in VARIANTS else "long"
    return record["caption"][variant].strip()


def _norm_cmp(s: str) -> str:
    return " ".join(s.replace("[trigger]", " ").split()).strip(" ,").lower()


def strip_trigger(caption: str, trigger: str = "") -> str:
    """기존 캡션을 reference 로 넘길 때 트리거 표기는 뺀다 (학습 시 다시 붙음)."""
    c = caption.replace("[trigger]", " ")
    t = (trigger or "").strip()
    if t:
        c = re.sub(re.escape(t), " ", c, flags=re.IGNORECASE)
    return " ".join(c.split()).strip(" ,;")


# 상태 값
MISSING = "missing"          # 캡션 없음 → 새로 생성
OFFICIAL = "official"        # 공식 형식 (.moss.json 정상 + .txt 가 그 변형과 일치)
EDITED = "edited"            # 공식 JSON 은 있으나 .txt 를 사용자가 직접 고침 → 보존
JSON_IN_TXT = "json_in_txt"  # .txt 안에 공식 JSON 이 들어 있음 → VLM 없이 변환
OTHER = "other"              # 다른 형식 (태그·짧은 문장 등) → 공식 형식으로 다시 생성

STATUS_LABELS = {
    MISSING: "캡션 없음",
    OFFICIAL: "공식 형식",
    EDITED: "직접 수정됨",
    JSON_IN_TXT: "JSON 변환 필요",
    OTHER: "다른 형식",
}


def caption_status(image_path: str, txt: Optional[str] = None, trigger: str = "") -> str:
    from .dataset import read_caption

    txt = read_caption(image_path) if txt is None else txt
    rec = read_record(image_path)
    if not txt.strip():
        return MISSING
    if txt.lstrip().startswith("{"):
        obj = normalize_record(extract_json(txt))
        if obj is not None:
            return JSON_IN_TXT
    if rec is None:
        return OTHER
    cur = _norm_cmp(strip_trigger(txt, trigger))
    if any(cur == _norm_cmp(rec["caption"][k]) for k in VARIANTS):
        return OFFICIAL
    return EDITED


def needs_work(status: str, mode: str) -> bool:
    """mode: missing(없는 것만) | reformat(없는 것 + 다른 형식, 기본) | all(전부 다시)."""
    if mode == "all":
        return True
    if mode == "missing":
        return status in (MISSING, JSON_IN_TXT)
    return status in (MISSING, JSON_IN_TXT, OTHER)


def scan_status(paths: List[str], trigger: str = "") -> Dict[str, str]:
    return {p: caption_status(p, trigger=trigger) for p in paths}


# ---------------------------------------------------------------------------
# Ollama 클라이언트 (표준 라이브러리만 사용)
# ---------------------------------------------------------------------------
class OllamaError(RuntimeError):
    pass


def _url(base: str, path: str) -> str:
    base = (base or DEFAULT_OLLAMA_URL).strip().rstrip("/")
    if not re.match(r"^https?://", base):
        base = "http://" + base
    return base + path


def _request(base: str, path: str, payload: Optional[dict] = None, timeout: float = 10.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(_url(base, path), data=data, method="GET" if data is None else "POST",
                                 headers={"Content-Type": "application/json"})
    # 로컬 연결은 시스템 프록시를 타지 않게 한다
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:400]
        raise OllamaError(f"Ollama 오류 {e.code}: {detail}") from e
    except (urllib.error.URLError, OSError) as e:
        raise OllamaError(f"Ollama 에 연결할 수 없습니다 ({_url(base, '')}): {getattr(e, 'reason', e)}\n"
                          "Ollama 가 실행 중인지 확인하세요 (작업 표시줄 아이콘 또는 'ollama serve').") from e
    try:
        return json.loads(body) if body.strip() else {}
    except json.JSONDecodeError as e:
        raise OllamaError(f"Ollama 응답을 해석할 수 없습니다: {body[:200]}") from e


def ollama_version(base: str, timeout: float = 4.0) -> str:
    return str(_request(base, "/api/version", timeout=timeout).get("version", "?"))


def list_models(base: str, timeout: float = 6.0) -> List[str]:
    data = _request(base, "/api/tags", timeout=timeout)
    return sorted(m.get("name") or m.get("model") for m in data.get("models", []) if (m.get("name") or m.get("model")))


def gemma4_models(names: List[str]) -> List[str]:
    return [n for n in names if n.lower().startswith(("gemma4", "gemma-4")) or "/gemma4" in n.lower()
            or "gemma-4" in n.lower()]


def pick_model(preferred: str, installed: List[str]) -> str:
    """설정한 모델이 설치돼 있으면 그대로, 없으면 설치된 Gemma 4 중 하나."""
    if not installed:
        return preferred
    if preferred in installed:
        return preferred
    if preferred and ":" not in preferred and f"{preferred}:latest" in installed:
        return f"{preferred}:latest"
    g = gemma4_models(installed)
    return g[0] if g else preferred


def model_capabilities(base: str, model: str, timeout: float = 8.0) -> List[str]:
    try:
        data = _request(base, "/api/show", {"model": model}, timeout=timeout)
    except OllamaError:
        return []
    return [str(c) for c in data.get("capabilities") or []]


def unload_model(base: str, model: str) -> None:
    try:
        _request(base, "/api/generate", {"model": model, "keep_alive": 0}, timeout=20)
    except OllamaError:
        pass


def encode_image(path: str, max_side: int = 1280) -> str:
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class OllamaCaptionConfig:
    url: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_OLLAMA_MODEL
    think: bool = True
    temperature: float = OFFICIAL_OPTIONS["temperature"]
    top_p: float = OFFICIAL_OPTIONS["top_p"]
    num_predict: int = OFFICIAL_OPTIONS["num_predict"]
    num_ctx: int = 16384
    timeout: float = 600.0
    max_side: int = 1280
    retries: int = 2


def caption_image(path: str, cfg: OllamaCaptionConfig, reference: str = "", subject_hint: str = "",
                  extra: str = "", log: Optional[Callable[[str], None]] = None) -> Tuple[dict, str]:
    """이미지 하나를 공식 형식으로 캡션. (정규화된 record, 원문 응답) 반환."""
    img = encode_image(path, cfg.max_side)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(reference, subject_hint, extra), "images": [img]},
    ]
    options = {"temperature": cfg.temperature, "top_p": cfg.top_p, "num_predict": cfg.num_predict,
               "num_ctx": cfg.num_ctx}
    fmt: object = OUTPUT_SCHEMA
    think = cfg.think
    last_err = ""
    for attempt in range(cfg.retries + 1):
        payload = {"model": cfg.model, "messages": messages, "stream": False, "format": fmt,
                   "options": options, "keep_alive": "10m"}
        if think:
            payload["think"] = True
        try:
            resp = _request(cfg.url, "/api/chat", payload, timeout=cfg.timeout)
        except OllamaError as e:
            msg = str(e)
            # 구버전 Ollama / thinking 미지원 모델 대비: 기능을 하나씩 낮춰 재시도
            if think and ("think" in msg.lower()):
                think = False
                if log:
                    log("이 모델/Ollama 버전은 thinking 을 지원하지 않아 끄고 다시 시도합니다.")
                continue
            if fmt is not None and fmt != "json" and ("format" in msg.lower() or "schema" in msg.lower()):
                fmt = "json"
                continue
            raise
        content = (resp.get("message") or {}).get("content", "") or resp.get("response", "")
        rec = normalize_record(extract_json(content))
        if rec is not None:
            return rec, content
        last_err = content[:300]
        if log:
            log(f"JSON 형식이 아니라서 다시 시도합니다 ({attempt + 1}/{cfg.retries + 1}).")
    raise OllamaError(f"공식 형식 JSON 을 받지 못했습니다. 응답 앞부분: {last_err}")
