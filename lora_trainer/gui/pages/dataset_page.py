"""데이터셋 페이지: 폴더 관리, 썸네일 브라우저, 캡션 편집, 일괄 편집, 자동 캡션."""

from __future__ import annotations

import os
from typing import List

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.config import DatasetConfig, DatasetFolder
from ...data import moss_caption as mc
from ...data.dataset import apply_trigger, dataset_summary, list_images, read_caption, scan_dataset, write_caption
from ..process import WorkerProcess
from ..widgets.common import Card, FormGrid, Page, combo, dspin, hint, set_combo, spin
from ..widgets.ollama import OllamaModelPicker, describe
from .settings_page import VARIANT_ITEMS

# 썸네일 표시용 상태 마크 (MOSS-Image)
STATUS_MARK = {mc.MISSING: "  ⚠", mc.OTHER: "  ◇", mc.JSON_IN_TXT: "  ◇", mc.EDITED: "  ✎", mc.OFFICIAL: "  ✓"}

RES_CHOICES = [512, 768, 1024, 1280, 1536]
THUMB = 132


class _ThumbSignals(QObject):
    done = Signal(str, QImage)


class _ThumbJob(QRunnable):
    def __init__(self, path: str, signals: _ThumbSignals):
        super().__init__()
        self.path, self.signals = path, signals

    def run(self):
        reader = QImageReader(self.path)
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid():
            reader.setScaledSize(size.scaled(THUMB * 2, THUMB * 2, Qt.KeepAspectRatio))
        img = reader.read()
        if not img.isNull():
            self.signals.done.emit(self.path, img)


class BatchEditDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("캡션 일괄 편집")
        self.setMinimumWidth(460)
        lay = QVBoxLayout(self)
        form = FormGrid()
        self.prefix, self.suffix = QLineEdit(), QLineEdit()
        self.find, self.replace = QLineEdit(), QLineEdit()
        self.prefix.setPlaceholderText("예: [trigger], ")
        form.row("앞에 추가", self.prefix)
        form.row("뒤에 추가", self.suffix)
        form.row("찾기", self.find)
        form.row("바꾸기", self.replace)
        self.only_empty = QCheckBox("캡션이 비어 있는 이미지에만 적용")
        lay.addLayout(form)
        lay.addWidget(self.only_empty)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


class AutoCaptionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("자동 캡션 (Qwen3-VL)")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)
        lay.addWidget(hint("Krea 2 텍스트 인코더와 같은 Qwen3-VL-4B-Instruct 비전 모델로 캡션을 생성합니다. "
                           "처음 실행 시 모델(약 9GB)을 내려받습니다."))
        form = FormGrid()
        self.model = QLineEdit("Qwen/Qwen3-VL-4B-Instruct")
        self.style = QComboBox()
        for v, l in (("detailed", "상세한 문장 (권장)"), ("short", "짧은 한 문장"), ("tags", "태그 나열")):
            self.style.addItem(l, v)
        self.quant = QComboBox()
        for v, l in (("none", "bf16 (VRAM 10GB+)"), ("nf4", "NF4 4bit (저VRAM)")):
            self.quant.addItem(l, v)
        self.overwrite = QCheckBox("기존 캡션 덮어쓰기")
        self.extra = QLineEdit()
        self.extra.setPlaceholderText("예: 인물의 이름은 언급하지 말 것")
        form.row("모델", self.model)
        form.row("캡션 형식", self.style)
        form.row("정밀도", self.quant)
        form.row("추가 지시", self.extra)
        lay.addLayout(form)
        lay.addWidget(self.overwrite)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("캡션 생성")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


class MossCaptionDialog(QDialog):
    """MOSS-Image 공식 가이드 형식 캡션 (Ollama · Gemma 4)."""

    USE_QWEN = 2  # exec() 반환값: Qwen3-VL 방식으로 전환

    def __init__(self, settings, counts: dict, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("MOSS-Image 자동 캡션 (Ollama · Gemma 4)")
        self.setMinimumWidth(860)
        lay = QVBoxLayout(self)
        lay.addWidget(hint("MOSS-Image 공식 가이드의 리캡션 형식으로 만듭니다: 상세(long) · 한 문단(medium) · 한 문장(short) "
                           "+ 워터마크 위치 + 테두리. 원본 JSON 은 <이미지>.moss.json, 학습용 캡션은 .txt 로 저장됩니다. "
                           "기존 캡션은 참고(reference)로 넘기고, 덮어쓰기 전에 _caption_backup 폴더에 백업합니다."))
        tot = sum(counts.values())
        need = counts.get(mc.MISSING, 0) + counts.get(mc.OTHER, 0) + counts.get(mc.JSON_IN_TXT, 0)
        summary = "  ·  ".join(f"{mc.STATUS_LABELS[k]} {counts.get(k, 0)}"
                                for k in (mc.MISSING, mc.OTHER, mc.JSON_IN_TXT, mc.EDITED, mc.OFFICIAL) if counts.get(k))
        self.counts_label = QLabel(f"<b>이미지 {tot}장</b>  —  {summary or '없음'}")
        lay.addWidget(self.counts_label)

        form = FormGrid()
        self.model = OllamaModelPicker(lambda: settings.ollama_url or mc.DEFAULT_OLLAMA_URL)
        self.model.setText(settings.ollama_model or mc.DEFAULT_OLLAMA_MODEL)
        form.row("Gemma 4 모델", self.model, f"Ollama {settings.ollama_url}")
        self.make_default = QCheckBox("이 모델을 기본값으로 저장")
        form.row(" ", self.make_default)

        mode_box = QVBoxLayout()
        self.mode_reformat = QRadioButton(f"없는 것 새로 + 다른 형식은 공식 형식으로 다시  ({need}장 · 권장)")
        self.mode_missing = QRadioButton(f"캡션 없는 이미지만 새로  ({counts.get(mc.MISSING, 0) + counts.get(mc.JSON_IN_TXT, 0)}장)")
        self.mode_all = QRadioButton(f"전부 다시  ({tot}장 · 직접 고친 캡션도 덮어씀)")
        self.mode_reformat.setChecked(True)
        for w in (self.mode_reformat, self.mode_missing, self.mode_all):
            mode_box.addWidget(w)
        form.row("작업 범위", mode_box, "✎ 직접 수정한 캡션은 '전부 다시' 에서만 바뀜")
        self.variant = combo(VARIANT_ITEMS)
        set_combo(self.variant, settings.moss_caption_variant)
        form.row("학습에 쓸 캡션", self.variant)
        self.subject = QLineEdit()
        self.subject.setPlaceholderText("예: a young woman  (비워도 됨)")
        form.row("주 피사체 표현", self.subject, "트리거는 캡션에서 빼고 학습 때 앞에 자동으로 붙음")
        self.extra = QLineEdit()
        self.extra.setPlaceholderText("예: Do not describe the hairstyle.")
        form.row("추가 지시 (영어)", self.extra)
        self.think = QCheckBox("thinking 켜기 (공식 권장)")
        self.think.setChecked(settings.ollama_think)
        form.row("생각 모드", self.think)
        lay.addLayout(form)
        self.conn = hint("Ollama 연결 확인 중…")
        self.conn.setWordWrap(True)
        lay.addWidget(self.conn)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.ok_btn = bb.button(QDialogButtonBox.Ok)
        self.ok_btn.setText("캡션 생성")
        self.ok_btn.setObjectName("Primary")
        qwen = bb.addButton("Qwen3-VL 방식으로…", QDialogButtonBox.ResetRole)
        qwen.setToolTip("공식 형식 대신 기존 Qwen3-VL(transformers) 캡션 사용")
        qwen.clicked.connect(lambda: self.done(self.USE_QWEN))
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.model.refresh(self._probed)

    def _probed(self, res: dict):
        self.conn.setText(describe(res, self.model.text()))

    def mode(self) -> str:
        if self.mode_all.isChecked():
            return "all"
        if self.mode_missing.isChecked():
            return "missing"
        return "reformat"


class DatasetPage(Page):
    def __init__(self, settings_getter=None, arch_getter=None):
        super().__init__("데이터셋", "학습 이미지 폴더를 추가하고 캡션을 확인하세요. 캡션은 이미지와 같은 이름의 .txt 파일로 저장됩니다.")
        self.settings_getter = settings_getter
        self.arch_getter = arch_getter
        self.signals = _ThumbSignals()
        self.signals.done.connect(self._thumb_ready)
        self.pool = QThreadPool.globalInstance()
        self._items_by_path = {}
        self._current = None
        self._dirty = False
        self.worker = WorkerProcess(self)
        self.worker.event.connect(self._caption_event)
        self.worker.log.connect(lambda s: self.status.setText(s[-160:]))
        self.worker.finished.connect(self._caption_done)

        # 폴더 표 --------------------------------------------------------
        folders = Card("이미지 폴더", "폴더별로 반복 횟수와 트리거 단어를 다르게 줄 수 있습니다 (반복 = 에폭당 사용 횟수).")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["폴더", "반복", "트리거 (비우면 전역)", "이미지"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setColumnWidth(1, 60)
        self.table.setColumnWidth(2, 190)
        self.table.setColumnWidth(3, 70)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(130)
        self.table.itemSelectionChanged.connect(self._folder_selected)
        folders.add(self.table)
        row = QHBoxLayout()
        add = QPushButton("+ 폴더 추가")
        add.setObjectName("Primary")
        add.clicked.connect(self._add_folder)
        rem = QPushButton("선택 삭제")
        rem.clicked.connect(self._remove_folder)
        self.analyze_btn = QPushButton("데이터셋 분석")
        self.analyze_btn.clicked.connect(self._analyze)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        row.addWidget(self.analyze_btn)
        folders.add(row)
        self.summary = hint("")
        folders.add(self.summary)
        self.add(folders)

        # 전역 설정 ------------------------------------------------------
        opts = Card("캡션 · 해상도")
        form = FormGrid()
        self.trigger = QLineEdit()
        self.trigger.setPlaceholderText("예: ohwx woman  (캡션의 [trigger] 를 치환하거나 앞에 붙임)")
        form.row("트리거 단어", self.trigger, "LoRA 를 부를 고유 단어. 흔하지 않은 조합 권장")
        self.default_caption = QLineEdit()
        form.row("기본 캡션", self.default_caption, "캡션 파일이 없는 이미지에 사용")
        res_row = QHBoxLayout()
        res_row.setSpacing(18)
        self.res_checks = {}
        for r in RES_CHOICES:
            cb = QCheckBox(str(r))
            self.res_checks[r] = cb
            res_row.addWidget(cb)
        res_row.addStretch(1)
        form.row("학습 해상도", res_row, "여러 개 선택 시 멀티 해상도 학습. Krea 2 는 1024 이하 권장")
        self.dropout = dspin(0, 0.5, 0.01, 2)
        form.row("캡션 드롭아웃", self.dropout, "빈 캡션으로 학습할 확률 (0.05 권장)")
        self.flip = QCheckBox("좌우 반전 증강 (글자/비대칭 피사체에는 끄세요)")
        form.row("증강", self.flip)
        bucket_row = QHBoxLayout()
        bucket_row.setSpacing(8)
        self.bmin, self.bmax, self.bstep = spin(256, 4096, 64), spin(256, 4096, 64), spin(16, 256, 16)
        for lab, w in (("최소", self.bmin), ("최대", self.bmax), ("단위", self.bstep)):
            bucket_row.addWidget(QLabel(lab))
            w.setFixedWidth(110)
            bucket_row.addWidget(w)
            bucket_row.addSpacing(10)
        bucket_row.addStretch(1)
        form.row("버킷 (종횡비)", bucket_row, "이미지 비율별로 자동 분류해 자르지 않고 학습")
        opts.add(form)
        self.add(opts)

        # 브라우저 + 캡션 편집기 ----------------------------------------
        browser = Card("이미지 · 캡션 편집", "썸네일을 클릭해 캡션을 수정하세요. 다른 이미지로 이동하면 자동 저장됩니다.")
        tools = QHBoxLayout()
        self.batch_btn = QPushButton("일괄 편집")
        self.batch_btn.clicked.connect(self._batch_edit)
        self.auto_btn = QPushButton("자동 캡션 생성")
        self.auto_btn.clicked.connect(self._auto_caption)
        self.refresh_btn = QPushButton("새로고침")
        self.refresh_btn.clicked.connect(self._folder_selected)
        tools.addWidget(self.batch_btn)
        tools.addWidget(self.auto_btn)
        tools.addWidget(self.refresh_btn)
        tools.addStretch(1)
        self.count_label = hint("")
        tools.addWidget(self.count_label)
        browser.add(tools)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.status = hint("")
        browser.add(self.progress)
        browser.add(self.status)

        split = QSplitter(Qt.Horizontal)
        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.IconMode)
        self.grid.setIconSize(QSize(THUMB, THUMB))
        self.grid.setGridSize(QSize(THUMB + 18, THUMB + 34))
        self.grid.setResizeMode(QListWidget.Adjust)
        self.grid.setMovement(QListWidget.Static)
        self.grid.setUniformItemSizes(True)
        self.grid.setWordWrap(True)
        self.grid.currentItemChanged.connect(self._image_selected)
        split.addWidget(self.grid)

        editor = QWidget()
        el = QVBoxLayout(editor)
        el.setContentsMargins(10, 0, 0, 0)
        self.preview = QLabel("이미지를 선택하세요")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(300, 300)
        self.preview.setStyleSheet("background:#151821; border:1px solid #2a3040; border-radius:10px; color:#8a93a8;")
        el.addWidget(self.preview, 1)
        self.file_label = hint("")
        el.addWidget(self.file_label)
        self.caption = QPlainTextEdit()
        self.caption.setPlaceholderText("캡션 입력 (트리거는 [trigger] 로 표기 가능)")
        self.caption.setMaximumHeight(130)
        self.caption.textChanged.connect(self._mark_dirty)
        el.addWidget(self.caption)
        self.final_caption = hint("")
        el.addWidget(self.final_caption)
        # MOSS 공식 캡션 변형 전환 (.moss.json 이 있을 때만)
        self.variant_row = QWidget()
        vr = QHBoxLayout(self.variant_row)
        vr.setContentsMargins(0, 0, 0, 0)
        self.variant_info = hint("")
        vr.addWidget(self.variant_info, 1)
        self.variant_btns = {}
        for v in mc.VARIANTS:
            b = QPushButton(v)
            b.setToolTip(f"공식 캡션의 {v} 변형을 이 이미지의 학습 캡션(.txt)으로 사용")
            b.clicked.connect(lambda _=False, v=v: self._use_variant(v))
            vr.addWidget(b)
            self.variant_btns[v] = b
        self.variant_row.setVisible(False)
        el.addWidget(self.variant_row)
        nav = QHBoxLayout()
        prev_btn, next_btn, save_btn = QPushButton("◀ 이전"), QPushButton("다음 ▶"), QPushButton("저장")
        save_btn.setObjectName("Primary")
        prev_btn.clicked.connect(lambda: self._move(-1))
        next_btn.clicked.connect(lambda: self._move(1))
        save_btn.clicked.connect(self._save_caption)
        nav.addWidget(prev_btn)
        nav.addWidget(next_btn)
        nav.addStretch(1)
        nav.addWidget(save_btn)
        el.addLayout(nav)
        split.addWidget(editor)
        split.setSizes([620, 420])
        split.setMinimumHeight(520)
        browser.add(split)
        self.add(browser)

        self._autosave = QTimer(self)
        self._autosave.setSingleShot(True)
        self._autosave.setInterval(1200)
        self._autosave.timeout.connect(self._save_caption)

    # ------------------------------------------------------------------
    # 폴더 표
    # ------------------------------------------------------------------
    def _add_folder(self):
        p = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택")
        if p:
            self._append_row(DatasetFolder(path=p))
            self.table.selectRow(self.table.rowCount() - 1)

    def _append_row(self, f: DatasetFolder):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(f.path))
        rep = spin(1, 1000)
        rep.setValue(f.repeats)
        self.table.setCellWidget(r, 1, rep)
        self.table.setItem(r, 2, QTableWidgetItem(f.trigger_word))
        n = QTableWidgetItem(str(len(list_images(f.path))))
        n.setFlags(n.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(r, 3, n)

    def _remove_folder(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)
        self.grid.clear()

    def _folders(self) -> List[DatasetFolder]:
        out = []
        for r in range(self.table.rowCount()):
            path = self.table.item(r, 0).text().strip()
            rep = self.table.cellWidget(r, 1).value()
            trig = (self.table.item(r, 2).text() if self.table.item(r, 2) else "").strip()
            if path:
                out.append(DatasetFolder(path=path, repeats=rep, trigger_word=trig))
        return out

    def _selected_folder(self):
        rows = {i.row() for i in self.table.selectedIndexes()}
        if not rows:
            return None
        r = min(rows)
        return self.table.item(r, 0).text().strip(), (self.table.item(r, 2).text() if self.table.item(r, 2) else "")

    # ------------------------------------------------------------------
    # 썸네일
    # ------------------------------------------------------------------
    def _folder_selected(self):
        self._save_caption()
        sel = self._selected_folder()
        self.grid.clear()
        self._items_by_path = {}
        self._current = None
        if not sel:
            return
        folder, _ = sel
        paths = list_images(folder)
        missing = 0
        moss = self._is_moss()
        trig = (sel[1].strip() if sel else "") or self.trigger.text().strip()
        counts = {}
        for p in paths:
            cap = read_caption(p)
            has_cap = bool(cap)
            missing += 0 if has_cap else 1
            if moss:
                st = mc.caption_status(p, cap, trig)
                counts[st] = counts.get(st, 0) + 1
                mark, tip = STATUS_MARK.get(st, ""), mc.STATUS_LABELS.get(st, "")
            else:
                mark, tip = ("" if has_cap else "  ⚠"), ("" if has_cap else "캡션 없음")
            item = QListWidgetItem(os.path.basename(p) + mark)
            item.setData(Qt.UserRole, p)
            item.setToolTip(p + (f"\n{tip}" if tip else ""))
            item.setSizeHint(QSize(THUMB + 14, THUMB + 30))
            self.grid.addItem(item)
            self._items_by_path[p] = item
            self.pool.start(_ThumbJob(p, self.signals))
        if moss:
            other = counts.get(mc.OTHER, 0) + counts.get(mc.JSON_IN_TXT, 0)
            self.count_label.setText(f"{len(paths)}장 · ✓ 공식 {counts.get(mc.OFFICIAL, 0)} · ⚠ 없음 {missing} · "
                                     f"◇ 다른 형식 {other} · ✎ 직접 수정 {counts.get(mc.EDITED, 0)}")
        else:
            self.count_label.setText(f"{len(paths)}장 · 캡션 없음 {missing}장")
        if paths:
            self.grid.setCurrentRow(0)

    def _thumb_ready(self, path: str, img: QImage):
        item = self._items_by_path.get(path)
        if item is not None:
            item.setIcon(QIcon(QPixmap.fromImage(img)))

    # ------------------------------------------------------------------
    # 캡션 편집
    # ------------------------------------------------------------------
    def _image_selected(self, cur, _prev=None):
        self._save_caption()
        if cur is None:
            return
        path = cur.data(Qt.UserRole)
        self._current = path
        pm = QPixmap(path)
        if not pm.isNull():
            self.preview.setPixmap(pm.scaled(self.preview.size() * 0.98, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.file_label.setText(f"{os.path.basename(path)}  ·  {pm.width()}×{pm.height()}")
        self.caption.blockSignals(True)
        self.caption.setPlainText(read_caption(path))
        self.caption.blockSignals(False)
        self._dirty = False
        self._update_final()
        self._update_variant_row()

    # ------------------------------------------------------------------
    # MOSS-Image 공식 캡션
    # ------------------------------------------------------------------
    def _is_moss(self) -> bool:
        return bool(self.arch_getter) and self.arch_getter() == "moss_image"

    def set_arch(self, arch: str):
        self.auto_btn.setText("자동 캡션 생성 (MOSS 공식 형식)" if arch == "moss_image" else "자동 캡션 생성")
        self.auto_btn.setToolTip("Ollama · Gemma 4 로 MOSS-Image 공식 가이드 형식 캡션 생성" if arch == "moss_image"
                                 else "Qwen3-VL 로 캡션 생성")
        if self._selected_folder():
            self._folder_selected()

    def _update_variant_row(self):
        rec = mc.read_record(self._current) if self._current else None
        self.variant_row.setVisible(rec is not None)
        if rec is None:
            return
        cur = " ".join(self.caption.toPlainText().split()).lower()
        active = next((v for v in mc.VARIANTS if " ".join(rec["caption"][v].split()).lower() == cur), "")
        extras = []
        if rec.get("watermark"):
            extras.append(f"워터마크 {len(rec['watermark'])}개")
        if rec.get("borders") and rec["borders"] != ["none"]:
            extras.append("테두리: " + ", ".join(rec["borders"]))
        self.variant_info.setText("공식 캡션 · 사용 중: " + (active or "직접 수정") + (" · " + " · ".join(extras) if extras else ""))
        for v, b in self.variant_btns.items():
            b.setObjectName("Primary" if v == active else "")
            b.style().unpolish(b)
            b.style().polish(b)

    def _use_variant(self, v: str):
        rec = mc.read_record(self._current) if self._current else None
        if rec is None:
            return
        self.caption.setPlainText(mc.training_text(rec, v))  # textChanged → 자동 저장
        self._save_caption()
        item = self._items_by_path.get(self._current)
        if item is not None:
            item.setText(os.path.basename(self._current) + STATUS_MARK[mc.OFFICIAL])
        self._update_variant_row()

    def _mark_dirty(self):
        self._dirty = True
        self._update_final()
        self._autosave.start()

    def _update_final(self):
        sel = self._selected_folder()
        trig = (sel[1].strip() if sel else "") or self.trigger.text().strip()
        final = apply_trigger(self.caption.toPlainText(), trig)
        self.final_caption.setText(f"학습에 쓰일 캡션: {final}" if final else "")

    def _save_caption(self):
        if self._current and self._dirty:
            write_caption(self._current, self.caption.toPlainText())
            self._dirty = False
            item = self._items_by_path.get(self._current)
            if item is not None:
                mark = ""
                if self._is_moss():
                    sel = self._selected_folder()
                    trig = (sel[1].strip() if sel else "") or self.trigger.text().strip()
                    mark = STATUS_MARK.get(mc.caption_status(self._current, trigger=trig), "")
                item.setText(os.path.basename(self._current) + mark)

    def _move(self, d: int):
        r = self.grid.currentRow() + d
        if 0 <= r < self.grid.count():
            self.grid.setCurrentRow(r)

    def _batch_edit(self):
        sel = self._selected_folder()
        if not sel:
            QMessageBox.information(self, "일괄 편집", "먼저 폴더를 선택하세요.")
            return
        dlg = BatchEditDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        self._save_caption()
        n = 0
        for p in list_images(sel[0]):
            cap = read_caption(p)
            if dlg.only_empty.isChecked() and cap.strip():
                continue
            new = cap
            if dlg.find.text():
                new = new.replace(dlg.find.text(), dlg.replace.text())
            new = dlg.prefix.text() + new + dlg.suffix.text()
            if new != cap:
                write_caption(p, new)
                n += 1
        self.status.setText(f"{n}개 캡션을 수정했습니다.")
        self._folder_selected()

    def _auto_caption(self):
        sel = self._selected_folder()
        if not sel:
            QMessageBox.information(self, "자동 캡션", "먼저 폴더를 선택하세요.")
            return
        if self.worker.running:
            return
        self._save_caption()
        settings = self.settings_getter() if self.settings_getter else None
        if self._is_moss() and settings is not None:
            trig = sel[1].strip() or self.trigger.text().strip()
            st = mc.scan_status(list_images(sel[0]), trig)
            counts = {}
            for v in st.values():
                counts[v] = counts.get(v, 0) + 1
            mdlg = MossCaptionDialog(settings, counts, self)
            r = mdlg.exec()
            if r == QDialog.Accepted:
                self._start_moss_caption(sel[0], trig, mdlg, settings)
                return
            if r != MossCaptionDialog.USE_QWEN:
                return
        dlg = AutoCaptionDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        args = ["--folder", sel[0], "--model", dlg.model.text().strip(), "--style", dlg.style.currentData(),
                "--quant", dlg.quant.currentData(), "--extra", dlg.extra.text().strip()]
        if settings is not None:
            args += ["--models-dir", settings.models_dir]
        if dlg.overwrite.isChecked():
            args.append("--overwrite")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.auto_btn.setEnabled(False)
        self.status.setText("자동 캡션 준비 중 (모델 로딩)…")
        self.worker.start_module("lora_trainer.data.captioner", args, settings.hf_token if settings else "")

    def _start_moss_caption(self, folder: str, trigger: str, dlg: "MossCaptionDialog", settings):
        model = dlg.model.text() or settings.ollama_model
        if dlg.make_default.isChecked():
            settings.ollama_model = model
        settings.moss_caption_variant = dlg.variant.currentData()
        settings.save()
        args = ["--folder", folder, "--mode", dlg.mode(), "--url", settings.ollama_url or mc.DEFAULT_OLLAMA_URL,
                "--model", model, "--variant", dlg.variant.currentData(), "--trigger", trigger,
                "--subject", dlg.subject.text().strip(), "--extra", dlg.extra.text().strip(),
                "--temperature", str(settings.ollama_temperature), "--top-p", str(settings.ollama_top_p),
                "--num-ctx", str(settings.ollama_num_ctx), "--timeout", str(settings.ollama_timeout)]
        if not dlg.think.isChecked():
            args.append("--no-think")
        if not settings.ollama_unload_after:
            args.append("--no-unload")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.auto_btn.setEnabled(False)
        self.status.setText(f"MOSS 공식 형식 캡션 준비 중 (Ollama · {model})…")
        self.worker.start_module("lora_trainer.data.moss_captioner", args, settings.hf_token)

    def _caption_event(self, kind: str, data: dict):
        if kind == "cache":
            self.progress.setMaximum(max(1, data.get("total", 1)))
            self.progress.setValue(data.get("done", 0))
            self.status.setText(f"캡션 생성 {data.get('done')}/{data.get('total')}  ·  {data.get('file', '')}")
        elif kind == "error":
            self.status.setText("오류: " + data.get("message", ""))

    def _caption_done(self, code: int):
        self.auto_btn.setEnabled(True)
        self.progress.setVisible(False)
        if code == 0:
            self.status.setText("자동 캡션 완료")
        elif code == 2:
            self.status.setText("캡션 생성 실패 — 로그를 확인하세요 (Ollama 실행 · 모델 설치 여부)")
        self._folder_selected()

    # ------------------------------------------------------------------
    def _analyze(self):
        cfg = DatasetConfig()
        self._store_into(cfg)
        if not cfg.folders:
            self.summary.setText("폴더가 없습니다.")
            return
        items, samples = scan_dataset(cfg)
        self.summary.setText(dataset_summary(items, samples))

    def _store_into(self, d: DatasetConfig):
        d.folders = self._folders()
        d.trigger_word = self.trigger.text().strip()
        d.default_caption = self.default_caption.text().strip()
        d.resolutions = [r for r, cb in self.res_checks.items() if cb.isChecked()] or [1024]
        d.caption_dropout = self.dropout.value()
        d.flip_horizontal = self.flip.isChecked()
        d.bucket_min, d.bucket_max, d.bucket_step = self.bmin.value(), self.bmax.value(), self.bstep.value()

    def load(self, job):
        d = job.dataset
        self.table.setRowCount(0)
        for f in d.folders:
            self._append_row(f)
        self.trigger.setText(d.trigger_word)
        self.default_caption.setText(d.default_caption)
        for r, cb in self.res_checks.items():
            cb.setChecked(r in d.resolutions)
        self.dropout.setValue(d.caption_dropout)
        self.flip.setChecked(d.flip_horizontal)
        self.bmin.setValue(d.bucket_min)
        self.bmax.setValue(d.bucket_max)
        self.bstep.setValue(d.bucket_step)
        if d.folders:
            self.table.selectRow(0)

    def store(self, job):
        self._save_caption()
        self._store_into(job.dataset)
