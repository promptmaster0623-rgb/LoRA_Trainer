"""학습 대기열 저장소.

대기열 항목 = "추가한 시점의 전체 학습 설정 스냅샷(YAML)" + LoRA 이름 + 데이터셋 폴더.
user_data/queue/ 에 저장되므로 프로그램을 다시 켜도 유지된다 (깃에 올라가지 않음).

상태 흐름: waiting(대기) → running(진행) → done(완료) / failed(실패) / stopped(중지)
프로그램이 진행 중에 종료되면 다음 실행 때 running 항목은 stopped 로 바뀐다.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import List, Optional

WAITING, RUNNING, DONE, FAILED, STOPPED = "waiting", "running", "done", "failed", "stopped"
STATUS_KO = {WAITING: "대기", RUNNING: "진행 중", DONE: "완료", FAILED: "실패", STOPPED: "중지됨"}


@dataclass
class QueueItem:
    id: str
    name: str                      # LoRA 이름
    folders: List[str]             # 데이터셋 이미지 폴더
    config: str                    # 스냅샷 YAML 경로
    arch: str = ""
    trigger: str = ""              # 표시용 (스냅샷에 반영됨)
    status: str = WAITING
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0
    result: str = ""               # 완료된 LoRA 경로
    message: str = ""              # 오류/중지 사유
    step: int = 0
    total: int = 0

    @property
    def status_ko(self) -> str:
        return STATUS_KO.get(self.status, self.status)


def safe_name(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", name.strip())
    return name.strip(" .") or "my_lora"


class JobQueue:
    def __init__(self, root: str):
        self.root = root
        self.index = os.path.join(root, "queue.json")
        self.items: List[QueueItem] = []
        self.continue_on_error = True
        self.load()

    # 저장/불러오기 -------------------------------------------------------
    def load(self) -> None:
        try:
            with open(self.index, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = {}
        known = set(QueueItem.__dataclass_fields__)
        self.items = []
        for d in data.get("items", []):
            try:
                it = QueueItem(**{k: v for k, v in d.items() if k in known})
            except TypeError:
                continue
            if it.status == RUNNING:  # 비정상 종료
                it.status, it.message = STOPPED, "프로그램 종료로 중단됨"
            self.items.append(it)
        self.continue_on_error = bool(data.get("continue_on_error", True))

    def save(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        tmp = self.index + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"continue_on_error": self.continue_on_error,
                       "items": [asdict(i) for i in self.items]}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.index)

    # 조작 ---------------------------------------------------------------
    def add(self, job, name: str, folders: List[str], trigger: Optional[str] = None) -> QueueItem:
        """job(TrainingJob) 의 복사본에 이름/폴더만 바꿔 스냅샷으로 저장."""
        from .config import DatasetFolder

        snap = job.copy()
        snap.project.name = safe_name(name)
        old = list(snap.dataset.folders)
        repeats = old[0].repeats if old else 1
        caption_ext = old[0].caption_ext if old else ".txt"
        snap.dataset.folders = [DatasetFolder(path=p, repeats=repeats, caption_ext=caption_ext) for p in folders]
        if trigger is not None and trigger.strip():
            snap.dataset.trigger_word = trigger.strip()
        snap.train.resume_state = ""          # 이어학습 경로는 항목마다 달라야 하므로 비움
        item_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        cfg = os.path.join(self.root, f"{item_id}.yaml")
        os.makedirs(self.root, exist_ok=True)
        snap.save_yaml(cfg)
        it = QueueItem(item_id, snap.project.name, list(folders), cfg, snap.model.arch, snap.dataset.trigger_word)
        self.items.append(it)
        self.save()
        return it

    def get(self, item_id: str) -> Optional[QueueItem]:
        return next((i for i in self.items if i.id == item_id), None)

    def remove(self, item_id: str) -> None:
        it = self.get(item_id)
        if it is None or it.status == RUNNING:
            return
        self.items.remove(it)
        try:
            os.remove(it.config)
        except OSError:
            pass
        self.save()

    def move(self, item_id: str, delta: int) -> None:
        it = self.get(item_id)
        if it is None:
            return
        i = self.items.index(it)
        j = min(max(0, i + delta), len(self.items) - 1)
        if i != j:
            self.items.insert(j, self.items.pop(i))
            self.save()

    def reset(self, item_id: str) -> None:
        """완료/실패/중지 항목을 다시 대기로."""
        it = self.get(item_id)
        if it and it.status != RUNNING:
            it.status, it.message, it.step, it.result = WAITING, "", 0, ""
            self.save()

    def clear_finished(self) -> None:
        for it in [i for i in self.items if i.status == DONE]:
            self.remove(it.id)

    def next_waiting(self) -> Optional[QueueItem]:
        return next((i for i in self.items if i.status == WAITING), None)

    def mark(self, item_id: str, status: str, **kw) -> None:
        it = self.get(item_id)
        if it is None:
            return
        it.status = status
        for k, v in kw.items():
            setattr(it, k, v)
        if status == RUNNING:
            it.started = time.time()
        elif status in (DONE, FAILED, STOPPED):
            it.finished = time.time()
        self.save()

    def names(self, active_only: bool = True) -> List[str]:
        return [i.name.lower() for i in self.items if not active_only or i.status in (WAITING, RUNNING)]

    def unique_name(self, name: str, output_dir: str) -> str:
        """대기열/출력 폴더와 겹치지 않는 이름 (겹치면 _2, _3 ...)."""
        base = safe_name(name)
        cand, n = base, 2
        while cand.lower() in self.names() or _has_output(output_dir, cand):
            cand, n = f"{base}_{n}", n + 1
        return cand

    @property
    def counts(self) -> dict:
        c = {k: 0 for k in STATUS_KO}
        for i in self.items:
            c[i.status] = c.get(i.status, 0) + 1
        return c


def _has_output(output_dir: str, name: str) -> bool:
    d = os.path.join(output_dir or "output", name)
    if not os.path.isdir(d):
        return False
    return any(f.endswith(".safetensors") for f in os.listdir(d))
