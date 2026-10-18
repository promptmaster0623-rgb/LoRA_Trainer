"""Ollama 연결 위젯: 모델 선택 콤보(직접 입력 가능) + 목록 새로고침 + 연결 테스트 (GUI 안 멈추게 백그라운드 실행)."""

from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QWidget

from ...data import moss_caption as mc


class _ProbeSignals(QObject):
    done = Signal(dict)


class _ProbeJob(QRunnable):
    """Ollama 버전·설치 모델·선택 모델 기능(vision/thinking)을 조회."""

    def __init__(self, url: str, model: str, signals: _ProbeSignals):
        super().__init__()
        self.url, self.model, self.signals = url, model, signals

    def run(self):
        res = {"ok": False, "url": self.url, "version": "", "models": [], "gemma4": [], "caps": [], "error": ""}
        try:
            res["version"] = mc.ollama_version(self.url)
            res["models"] = mc.list_models(self.url)
            res["gemma4"] = mc.gemma4_models(res["models"])
            pick = mc.pick_model(self.model, res["models"])
            res["picked"] = pick
            if pick in res["models"]:
                res["caps"] = mc.model_capabilities(self.url, pick)
            res["ok"] = True
        except mc.OllamaError as e:
            res["error"] = str(e)
        except Exception as e:  # pragma: no cover
            res["error"] = f"{type(e).__name__}: {e}"
        try:
            self.signals.done.emit(res)
        except RuntimeError:
            pass  # 결과가 오기 전에 창이 닫힘


def probe_async(url: str, model: str, callback) -> _ProbeSignals:
    sig = _ProbeSignals()
    sig.done.connect(callback)
    QThreadPool.globalInstance().start(_ProbeJob(url, model, sig))
    return sig  # 호출 쪽에서 참조를 잡고 있어야 시그널이 살아 있음


def describe(res: dict, model: str) -> str:
    """조회 결과를 사람이 읽을 문장으로."""
    if not res.get("ok"):
        return "✕ " + (res.get("error") or "연결 실패")
    lines = [f"✓ Ollama {res.get('version')} 연결됨 · {res.get('url')}"]
    g = res.get("gemma4") or []
    lines.append(f"설치된 Gemma 4: {', '.join(g)}" if g else
                 "⚠ 설치된 Gemma 4 모델이 없습니다 — 명령 프롬프트에서  ollama pull gemma4:e4b")
    picked = res.get("picked") or model
    caps = res.get("caps") or []
    if picked not in (res.get("models") or []):
        lines.append(f"⚠ '{model}' 이(가) 설치돼 있지 않습니다.")
    else:
        if picked != model:
            lines.append(f"'{model}' 이 없어 '{picked}' 을 사용합니다.")
        if caps:
            v = "예" if "vision" in caps else "아니오 (캡션 불가)"
            t = "예" if "thinking" in caps else "아니오 (자동으로 끄고 진행)"
            lines.append(f"{picked} · 이미지 입력 {v} · thinking {t}")
    return "\n".join(lines)


class OllamaModelPicker(QWidget):
    """[모델 콤보 ▾] [↻ 목록] — 콤보는 직접 입력도 가능."""

    def __init__(self, url_getter, parent=None):
        super().__init__(parent)
        self.url_getter = url_getter
        self._sig = None
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        self.combo.setEditable(True)
        self.combo.setMinimumWidth(240)
        self.combo.setToolTip("Ollama 에 설치된 모델 이름 (예: gemma4:e4b, gemma4:26b). 직접 입력도 됩니다.")
        self.refresh_btn = QPushButton("↻ 목록")
        self.refresh_btn.setToolTip("Ollama 에 설치된 모델 목록 불러오기")
        self.refresh_btn.clicked.connect(self.refresh)
        lay.addWidget(self.combo, 1)
        lay.addWidget(self.refresh_btn)

    def text(self) -> str:
        return self.combo.currentText().strip()

    def setText(self, t: str) -> None:
        if t and self.combo.findText(t) < 0:
            self.combo.addItem(t)
        self.combo.setCurrentText(t)

    def refresh(self, on_done=None):
        self.refresh_btn.setEnabled(False)
        cur = self.text()

        def done(res):
            self.refresh_btn.setEnabled(True)
            if res.get("ok"):
                models = res.get("models") or []
                g = res.get("gemma4") or []
                others = [m for m in models if m not in g]
                self.combo.blockSignals(True)
                self.combo.clear()
                for m in g + others:
                    self.combo.addItem(m)
                self.combo.blockSignals(False)
                # 사용자가 고른 값은 유지 (설치 안 된 값이면 실행 시 설치된 Gemma 4 로 자동 대체)
                self.combo.setCurrentText(cur or (res.get("picked") or ""))
            if on_done:
                on_done(res)

        self._sig = probe_async(self.url_getter(), cur, done)
