"""학습/캡션 작업을 별도 파이썬 프로세스로 실행하고 이벤트를 GUI 시그널로 변환."""

from __future__ import annotations

import os
import sys
from typing import List, Optional

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from ..core.events import parse_event_line
from .app_settings import ROOT


class WorkerProcess(QObject):
    event = Signal(str, dict)
    log = Signal(str)
    finished = Signal(int)
    started = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc: Optional[QProcess] = None
        self._buf = b""
        self.control_dir: str = ""

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def start_module(self, module: str, args: List[str], hf_token: str = "", control_dir: str = "") -> None:
        if self.running:
            raise RuntimeError("이미 실행 중인 작업이 있습니다.")
        self.control_dir = control_dir
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(ROOT)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        if hf_token:
            env.insert("HF_TOKEN", hf_token)
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        self.proc.started.connect(self.started.emit)
        self._buf = b""
        self.proc.start(sys.executable, ["-u", "-m", module, *args])

    def _read(self) -> None:
        self._buf += bytes(self.proc.readAllStandardOutput())
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            text = line.decode("utf-8", errors="replace").rstrip("\r")
            ev = parse_event_line(text)
            if ev is None:
                if text.strip():
                    self.log.emit(text)
            else:
                self.event.emit(ev[0], ev[1])

    def _done(self, code: int, _status=None) -> None:
        if self._buf:
            self.log.emit(self._buf.decode("utf-8", errors="replace"))
            self._buf = b""
        self.finished.emit(int(code))

    # 제어 파일 ---------------------------------------------------------
    def _touch(self, name: str) -> None:
        if self.control_dir:
            os.makedirs(self.control_dir, exist_ok=True)
            with open(os.path.join(self.control_dir, name), "w") as f:
                f.write("1")

    def _remove(self, name: str) -> None:
        p = os.path.join(self.control_dir, name)
        if self.control_dir and os.path.exists(p):
            os.remove(p)

    def request_stop(self) -> None:
        self._remove(".pause")
        self._touch(".stop")

    def set_paused(self, paused: bool) -> None:
        (self._touch if paused else self._remove)(".pause")

    def request_sample(self) -> None:
        self._touch(".sample_now")

    def request_save(self) -> None:
        self._touch(".save_now")

    def request_lr_scale(self, scale: float) -> None:
        """학습 중 학습률 배율 변경 (다음 스텝부터 반영)."""
        if self.control_dir:
            os.makedirs(self.control_dir, exist_ok=True)
            with open(os.path.join(self.control_dir, ".lr_scale"), "w", encoding="utf-8") as f:
                f.write(f"{scale:.6g}")

    def kill(self) -> None:
        if self.running:
            self.proc.kill()
