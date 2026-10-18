"""대기열 페이지: 같은 설정으로 LoRA 이름 · 데이터셋 폴더만 바꿔 여러 학습을 예약하고 순서대로 실행."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ...core.job_queue import DONE, FAILED, RUNNING, STOPPED, WAITING, JobQueue
from ...core.model_files import ARCH_TITLES
from ...core.recommend import fmt_duration
from ..theme import COLORS
from ..widgets.common import Card, FormGrid, Page, PathEdit, hint

STATUS_COLOR = {
    WAITING: COLORS["muted"], RUNNING: COLORS["accent_hover"], DONE: COLORS["success"],
    FAILED: COLORS["danger"], STOPPED: COLORS["warning"],
}
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def count_images(folder: str) -> int:
    if not os.path.isdir(folder):
        return -1
    return sum(1 for n in os.listdir(folder) if n.lower().endswith(IMAGE_EXTS))


class AddQueueDialog(QDialog):
    """현재 설정 그대로, 이름과 데이터셋 폴더만 입력."""

    def __init__(self, summary: str, suggest_name, default_trigger: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("대기열에 추가")
        self.setMinimumWidth(620)
        self.suggest_name = suggest_name
        lay = QVBoxLayout(self)
        lay.addWidget(hint("지금 화면의 모든 학습 설정을 그대로 복사해 예약합니다. 아래 두 가지만 바꾸세요."))
        info = QLabel(summary)
        info.setWordWrap(True)
        info.setStyleSheet(f"background:{COLORS['panel']}; border:1px solid {COLORS['border']}; border-radius:8px; padding:8px;")
        lay.addWidget(info)
        form = FormGrid()
        self.name = QLineEdit()
        self.name.setPlaceholderText("예: character_b_v1")
        form.row("LoRA 이름", self.name, "출력 파일 이름 (겹치면 자동으로 _2 를 붙입니다)")
        self.folder = PathEdit("dir", placeholder="학습 이미지 폴더")
        form.row("데이터셋 이미지 폴더", self.folder)
        self.trigger = QLineEdit()
        self.trigger.setPlaceholderText(f"비우면 현재 설정 사용: {default_trigger or '(없음)'}")
        form.row("트리거 단어 (선택)", self.trigger, "캐릭터마다 다르면 입력하세요")
        lay.addLayout(form)
        self.check = QLabel()
        self.check.setWordWrap(True)
        lay.addWidget(self.check)
        self.bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.bb.button(QDialogButtonBox.Ok).setText("대기열에 추가")
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        lay.addWidget(self.bb)
        self.folder.changed.connect(self._folder_changed)
        self.name.textChanged.connect(self._validate)
        self._validate()

    def _folder_changed(self, path: str):
        if path and not self.name.text().strip():
            self.name.setText(os.path.basename(os.path.normpath(path)))
        self._validate()

    def _validate(self):
        msgs, ok = [], True
        name = self.name.text().strip()
        folder = self.folder.text()
        if not name:
            ok = False
        else:
            final = self.suggest_name(name)
            if final != name:
                msgs.append(f"<span style='color:{COLORS['warning']}'>같은 이름이 있어 <b>{final}</b> 으로 저장됩니다.</span>")
        if folder:
            n = count_images(folder)
            if n < 0:
                ok = False
                msgs.append(f"<span style='color:{COLORS['danger']}'>폴더를 찾을 수 없습니다.</span>")
            elif n == 0:
                ok = False
                msgs.append(f"<span style='color:{COLORS['danger']}'>이미지가 없습니다.</span>")
            else:
                msgs.append(f"<span style='color:{COLORS['success']}'>이미지 {n}장</span>")
        else:
            ok = False
        self.check.setText("<br>".join(msgs))
        self.bb.button(QDialogButtonBox.Ok).setEnabled(ok)

    def values(self):
        return self.name.text().strip(), self.folder.text(), self.trigger.text().strip()


class QueuePage(Page):
    add_requested = Signal()
    start_requested = Signal()
    stop_after_current = Signal(bool)

    COLS = ["#", "상태", "LoRA 이름", "모델", "데이터셋 폴더", "진행", "결과 / 메모"]

    def __init__(self, queue: JobQueue):
        super().__init__("대기열", "같은 학습 설정으로 LoRA 이름과 데이터셋 폴더만 바꿔 여러 개를 예약하고, 위에서부터 순서대로 학습합니다.")
        self.queue = queue
        self.active = False

        ctrl = QHBoxLayout()
        self.add_btn = QPushButton("+ 현재 설정으로 추가")
        self.add_btn.clicked.connect(self.add_requested.emit)
        self.start_btn = QPushButton("▶  대기열 시작")
        self.start_btn.setObjectName("Primary")
        self.start_btn.setMinimumHeight(36)
        self.start_btn.clicked.connect(self.start_requested.emit)
        self.stop_after = QPushButton("현재 항목 후 정지")
        self.stop_after.setCheckable(True)
        self.stop_after.setToolTip("진행 중인 학습은 끝까지 하고, 다음 항목은 시작하지 않습니다")
        self.stop_after.toggled.connect(self.stop_after_current.emit)
        self.stop_after.setEnabled(False)
        ctrl.addWidget(self.add_btn)
        ctrl.addSpacing(8)
        ctrl.addWidget(self.start_btn)
        ctrl.addWidget(self.stop_after)
        ctrl.addStretch(1)
        self.cont = QCheckBox("실패해도 다음 항목 계속")
        self.cont.setChecked(queue.continue_on_error)
        self.cont.toggled.connect(self._cont_changed)
        ctrl.addWidget(self.cont)
        self.add(ctrl)

        self.status = QLabel()
        self.status.setObjectName("Hint")
        self.add(self.status)

        card = Card("예약 목록", "항목을 더블클릭하면 출력 폴더를 엽니다. 우클릭으로 다시 대기 / 삭제할 수 있습니다.")
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        h = self.table.horizontalHeader()
        for i, mode in enumerate([QHeaderView.ResizeToContents, QHeaderView.ResizeToContents, QHeaderView.Interactive,
                                  QHeaderView.ResizeToContents, QHeaderView.Stretch, QHeaderView.ResizeToContents,
                                  QHeaderView.Interactive]):
            h.setSectionResizeMode(i, mode)
        self.table.setColumnWidth(2, 180)
        self.table.setColumnWidth(6, 220)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setMinimumHeight(320)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._menu)
        self.table.itemDoubleClicked.connect(self._open_item)
        card.add(self.table)
        row = QHBoxLayout()
        self.up_btn = QPushButton("▲ 위로")
        self.down_btn = QPushButton("▼ 아래로")
        self.retry_btn = QPushButton("다시 대기")
        self.del_btn = QPushButton("삭제")
        self.del_btn.setObjectName("Danger")
        self.clear_btn = QPushButton("완료 항목 정리")
        self.clear_btn.setObjectName("Ghost")
        self.up_btn.clicked.connect(lambda: self._move(-1))
        self.down_btn.clicked.connect(lambda: self._move(1))
        self.retry_btn.clicked.connect(lambda: self._act("reset"))
        self.del_btn.clicked.connect(lambda: self._act("remove"))
        self.clear_btn.clicked.connect(self._clear_done)
        for b in (self.up_btn, self.down_btn, self.retry_btn, self.del_btn):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(self.clear_btn)
        card.add(row)
        self.add(card)
        self.add(hint("· 각 항목은 추가한 순간의 설정을 저장합니다 (이후 화면 설정을 바꿔도 이미 예약된 항목은 그대로).\n"
                      "· 진행 상황은 '실행 · 모니터' 페이지에서 그대로 볼 수 있습니다. 대기열 진행 중에도 다음 항목을 계속 추가할 수 있습니다.\n"
                      "· 데이터셋마다 캐시가 따로 만들어지므로 첫 단계(캐싱)는 항목마다 진행됩니다."))
        self.content.addStretch(1)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.refresh()

    # ------------------------------------------------------------------
    def _cont_changed(self, on: bool):
        self.queue.continue_on_error = on
        self.queue.save()

    def selected_id(self) -> str:
        r = self.table.currentRow()
        if r < 0:
            return ""
        it = self.table.item(r, 0)
        return it.data(Qt.UserRole) if it else ""

    def _move(self, d: int):
        sid = self.selected_id()
        if sid:
            self.queue.move(sid, d)
            self.refresh(select=sid)

    def _act(self, what: str):
        sid = self.selected_id()
        if not sid:
            return
        getattr(self.queue, what)(sid)
        self.refresh(select=sid if what != "remove" else None)

    def _clear_done(self):
        self.queue.clear_finished()
        self.refresh()

    def _menu(self, pos):
        sid = self.selected_id()
        if not sid:
            return
        it = self.queue.get(sid)
        m = QMenu(self)
        a_retry = m.addAction("다시 대기")
        a_open = m.addAction("출력 폴더 열기")
        m.addSeparator()
        a_del = m.addAction("삭제")
        running = it.status == RUNNING
        a_retry.setEnabled(not running and it.status != WAITING)
        a_del.setEnabled(not running)
        act = m.exec(self.table.viewport().mapToGlobal(pos))
        if act == a_retry:
            self._act("reset")
        elif act == a_del:
            self._act("remove")
        elif act == a_open:
            self._open_item(None)

    def _open_item(self, _item):
        sid = self.selected_id()
        it = self.queue.get(sid) if sid else None
        if not it:
            return
        from ...core.config import TrainingJob

        try:
            d = TrainingJob.load_yaml(it.config).run_dir
        except Exception:
            return
        if os.path.isdir(d):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(d)))

    def _update_buttons(self):
        sid = self.selected_id()
        it = self.queue.get(sid) if sid else None
        running = bool(it and it.status == RUNNING)
        for b in (self.up_btn, self.down_btn, self.del_btn):
            b.setEnabled(bool(it) and not running)
        self.retry_btn.setEnabled(bool(it) and not running and it.status != WAITING)

    def set_active(self, active: bool):
        self.active = active
        self.stop_after.setEnabled(active)
        self.start_btn.setEnabled(not active)
        self.start_btn.setText("대기열 진행 중…" if active else "▶  대기열 시작")
        if not active:
            self.stop_after.blockSignals(True)
            self.stop_after.setChecked(False)
            self.stop_after.blockSignals(False)
        self.refresh()

    def update_progress(self, item_id: str, step: int, total: int):
        it = self.queue.get(item_id)
        if not it:
            return
        it.step, it.total = step, total
        for r in range(self.table.rowCount()):
            cell = self.table.item(r, 0)
            if cell and cell.data(Qt.UserRole) == item_id:
                pct = int(step / total * 100) if total else 0
                self.table.setItem(r, 5, QTableWidgetItem(f"{step:,} / {total:,}  ({pct}%)"))
                break

    def refresh(self, select: str = None):
        keep = select or self.selected_id()
        self.table.setRowCount(0)
        for i, it in enumerate(self.queue.items):
            r = self.table.rowCount()
            self.table.insertRow(r)
            num = QTableWidgetItem(str(i + 1))
            num.setData(Qt.UserRole, it.id)
            st = QTableWidgetItem(("● " if it.status == RUNNING else "") + it.status_ko)
            st.setForeground(QColor(STATUS_COLOR.get(it.status, COLORS["text"])))
            folders = "; ".join(it.folders)
            prog = ""
            if it.total:
                prog = f"{it.step:,} / {it.total:,}"
            if it.status == DONE and it.finished and it.started:
                prog = f"{fmt_duration(it.finished - it.started)} 소요"
            note = it.message or (os.path.basename(it.result) if it.result else "")
            cells = [num, st, QTableWidgetItem(it.name), QTableWidgetItem(ARCH_TITLES.get(it.arch, it.arch)),
                     QTableWidgetItem(folders), QTableWidgetItem(prog), QTableWidgetItem(note)]
            cells[4].setToolTip(folders)
            cells[6].setToolTip(it.result or it.message)
            for c, w in enumerate(cells):
                self.table.setItem(r, c, w)
            if it.id == keep:
                self.table.selectRow(r)
        c = self.queue.counts
        self.status.setText(
            f"대기 {c[WAITING]} · 진행 {c[RUNNING]} · 완료 {c[DONE]} · 실패 {c[FAILED]} · 중지 {c[STOPPED]}"
            + ("   —   대기열 실행 중" if self.active else "")
        )
        self.start_btn.setEnabled(not self.active and c[WAITING] > 0)
        self._update_buttons()

    # Page 인터페이스 (설정 파일과 무관)
    def load(self, job):
        pass

    def store(self, job):
        pass
