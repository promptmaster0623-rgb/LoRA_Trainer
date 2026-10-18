"""학습 프로세스 -> GUI 이벤트 전달.

학습은 별도 프로세스에서 실행되며, 표준출력에 `@@EVENT {json}` 형태의 줄을
출력해 GUI 에 진행 상황을 알린다. 일반 로그 줄은 그대로 로그창에 표시된다.

이벤트 종류
  status   : {"phase": str, "message": str}
  progress : {"step": int, "total": int, "loss": float, "lr": float,
              "it_per_sec": float, "eta_sec": float, "vram_gb": float}
  cache    : {"kind": "text"|"latent", "done": int, "total": int}
  sample   : {"step": int, "paths": [str]}
  saved    : {"step": int, "path": str}
  error    : {"message": str, "traceback": str}
  done     : {"path": str}
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Callable, Optional

EVENT_PREFIX = "@@EVENT "


class EventSink:
    """이벤트를 내보내는 기본 구현 (stdout JSON 라인)."""

    def __init__(self, stream=None, callback: Optional[Callable[[str, dict], None]] = None):
        self.stream = stream if stream is not None else sys.stdout
        self.callback = callback

    def emit(self, event_type: str, /, **data: Any) -> None:
        payload = {**data, "type": event_type, "time": time.time()}
        if self.callback is not None:
            self.callback(event_type, payload)
            return
        try:
            self.stream.write(EVENT_PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
            self.stream.flush()
        except Exception:
            pass

    def log(self, message: str) -> None:
        if self.callback is not None:
            self.callback("log", {"type": "log", "message": message})
            return
        try:
            self.stream.write(message + "\n")
            self.stream.flush()
        except Exception:
            pass

    # 편의 함수 ---------------------------------------------------------
    def status(self, phase: str, message: str) -> None:
        self.log(f"[{phase}] {message}")
        self.emit("status", phase=phase, message=message)


def parse_event_line(line: str):
    """GUI 측: 한 줄을 (kind, payload) 또는 None 으로 변환."""
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        payload = json.loads(line[len(EVENT_PREFIX):])
        return payload.get("type", "unknown"), payload
    except json.JSONDecodeError:
        return None
