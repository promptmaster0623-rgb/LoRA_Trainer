"""메인 윈도우: 사이드바 내비게이션 + 시스템 상태 + 페이지 스택 + 학습 프로세스 · 대기열 관리."""

from __future__ import annotations

import os
import statistics

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME, __version__
from ..core.config import TrainingJob
from ..core.device import gpu_info
from ..core.job_queue import DONE, FAILED, RUNNING, STOPPED, JobQueue
from ..core.model_files import ARCH_TITLES, COMPONENTS, FIELD_OF, MISMATCH, MISSING, check_path
from ..core.recommend import (
    SpeedCalibration,
    analyze_dataset,
    apply_to_job,
    estimate_sec_per_step,
    fmt_duration,
    recommend,
    shape_from_job,
)
from ..core.sysinfo import detect_system
from .app_settings import QUEUE_DIR, ROOT, SPEED_CALIB_PATH, AppSettings
from .pages.dataset_page import DatasetPage
from .pages.model_page import ModelPage
from .pages.project_page import ProjectPage
from .pages.queue_page import AddQueueDialog, QueuePage
from .pages.run_page import RunPage
from .pages.sample_page import SamplePage
from .pages.settings_page import SettingsPage
from .pages.training_page import TrainingPage
from .process import WorkerProcess
from .theme import COLORS
from .widgets.sys_monitor import SysMonitor

NAV = [
    ("project", "①  시작하기"),
    ("dataset", "②  데이터셋"),
    ("model", "③  모델"),
    ("training", "④  학습 설정"),
    ("sample", "⑤  샘플"),
    ("run", "▶  실행 · 모니터"),
    ("queue", "⏱  대기열"),
    ("settings", "⚙  설정"),
]
# 설정 파일(job)과 무관한 페이지
NON_JOB_PAGES = ("run", "settings", "queue")


class _AnalysisSignals(QObject):
    done = Signal(object)


class _AnalysisJob(QRunnable):
    """데이터셋 스캔(이미지 헤더 읽기)이 있어 백그라운드에서 추천 계산."""

    def __init__(self, arch, folders, purpose, calib, signals):
        super().__init__()
        self.arch, self.folders, self.purpose, self.calib, self.signals = arch, folders, purpose, calib, signals

    def run(self):
        try:
            spec = detect_system()
            stats = analyze_dataset(self.folders)
            self.signals.done.emit(recommend(self.arch, spec, stats, self.purpose, self.calib))
        except Exception as e:  # pragma: no cover
            self.signals.done.emit(e)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.resize(1400, 920)
        self.settings = AppSettings.load()
        self.job = TrainingJob()
        self.config_path = ""
        self.queue = JobQueue(QUEUE_DIR)
        self.calib = SpeedCalibration(SPEED_CALIB_PATH)
        self.spec = detect_system()

        # 학습 실행 상태
        self._run_job = None
        self._run_item = None
        self._last_done = None
        self._last_error = ""
        self._speed = []
        self._queue_active = False
        self._queue_stop_after = False
        self._queue_stop_all = False

        # 추천 상태
        self._analysis = None
        self._analysis_sig = None
        self._applied_rec = ""
        self._pending_purpose = None
        self.pool = QThreadPool.globalInstance()
        self._an_signals = _AnalysisSignals()
        self._an_signals.done.connect(self._analysis_done)
        self._an_timer = QTimer(self)
        self._an_timer.setSingleShot(True)
        self._an_timer.setInterval(400)
        self._an_timer.timeout.connect(lambda: self.run_recommend())

        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # 사이드바 --------------------------------------------------------
        side = QWidget()
        side.setObjectName("Sidebar")
        side.setFixedWidth(236)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(14, 18, 14, 14)
        sl.setSpacing(4)
        title = QLabel("LoRA Trainer")
        title.setObjectName("AppTitle")
        sub = QLabel("ComfyUI LoRA 학습 · Krea 2 / MOSS-Image")
        sub.setObjectName("AppSub")
        sub.setWordWrap(True)
        sl.addWidget(title)
        sl.addWidget(sub)
        sl.addSpacing(6)
        self.model_badge = QLabel()
        self.model_badge.setToolTip("현재 학습 대상 베이스 모델 (시작하기 · 모델 페이지에서 변경)")
        self.model_badge.setStyleSheet(
            f"background: {COLORS['accent_soft']}; color: {COLORS['text']}; border-radius: 8px; "
            f"padding: 6px 10px; font-weight: 700;"
        )
        sl.addWidget(self.model_badge)
        sl.addSpacing(10)

        self.stack = QStackedWidget()
        self.pages = {
            "project": ProjectPage(),
            "dataset": DatasetPage(lambda: self.settings, lambda: self.pages["model"].arch),
            "model": ModelPage(lambda: self.settings),
            "training": TrainingPage(),
            "sample": SamplePage(),
            "run": RunPage(),
            "queue": QueuePage(self.queue),
            "settings": SettingsPage(self.settings),
        }
        self.nav_group = QButtonGroup(self)
        self.nav_buttons = {}
        self.nav_labels = dict(NAV)
        for key, label in NAV:
            b = QPushButton(label)
            b.setObjectName("NavButton")
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=key: self.show_page(k))
            self.nav_group.addButton(b)
            self.nav_buttons[key] = b
            sl.addWidget(b)
            self.stack.addWidget(self.pages[key])
            if key == "sample":
                sl.addSpacing(10)
        sl.addStretch(1)

        self.sysmon = SysMonitor()
        sl.addWidget(self.sysmon)
        sl.addSpacing(10)

        cfg_label = QLabel("설정 파일")
        cfg_label.setObjectName("AppSub")
        sl.addWidget(cfg_label)
        self.cfg_name = QLabel("(저장 안 됨)")
        self.cfg_name.setObjectName("Hint")
        self.cfg_name.setWordWrap(True)
        sl.addWidget(self.cfg_name)
        row = QHBoxLayout()
        for text, fn in (("새로", self.new_job), ("열기", self.open_job), ("저장", self.save_job)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        sl.addLayout(row)
        save_as = QPushButton("다른 이름으로 저장")
        save_as.setObjectName("Ghost")
        save_as.clicked.connect(lambda: self.save_job(as_new=True))
        sl.addWidget(save_as)
        lay.addWidget(side)
        lay.addWidget(self.stack, 1)

        info = gpu_info()
        self.statusBar().showMessage(f"{info.describe()}   ·   {ROOT}")

        # 연결 --------------------------------------------------------------
        p = self.pages
        p["project"].arch_changed.connect(self.change_arch)
        p["model"].arch_changed.connect(self.change_arch)
        p["project"].preset_requested.connect(self.apply_preset)
        p["project"].recommend_requested.connect(self.run_recommend)
        p["project"].recommend_apply.connect(self.apply_recommendation)
        p["project"].queue_add_requested.connect(self.add_to_queue)
        p["project"].recommend.set_purpose(self.settings.purpose)
        q: QueuePage = p["queue"]
        q.add_requested.connect(self.add_to_queue)
        q.start_requested.connect(self.start_queue)
        q.stop_after_current.connect(self._set_stop_after)
        run: RunPage = p["run"]
        self.worker = WorkerProcess(self)
        self.worker.event.connect(run.on_event)
        self.worker.event.connect(self._on_event)
        self.worker.log.connect(run.append_log)
        self.worker.finished.connect(self._finished)
        run.start_clicked.connect(self.start_training)
        run.stop_clicked.connect(self.stop_training)
        run.pause_toggled.connect(self.worker.set_paused)
        run.sample_clicked.connect(self.worker.request_sample)
        run.save_clicked.connect(self.worker.request_save)
        run.lr_scale_requested.connect(self.worker.request_lr_scale)

        # 초기 작업 불러오기
        if self.settings.last_config and os.path.isfile(self.settings.last_config):
            self._load_from(self.settings.last_config)
        else:
            preset = os.path.join(ROOT, "presets", "krea2_12gb_character.yaml")
            if os.path.isfile(preset):
                self.job = TrainingJob.load_yaml(preset)
            self.load_pages()
        self._update_queue_nav()
        self.show_page("project")
        QTimer.singleShot(300, lambda: self.run_recommend())

    # ------------------------------------------------------------------
    # 페이지 / 모델 전환
    # ------------------------------------------------------------------
    def show_page(self, key: str):
        self.nav_buttons[key].setChecked(True)
        self.stack.setCurrentWidget(self.pages[key])
        if key == "project":
            self._update_file_status()
            if self._analysis_sig is not None and self._analysis_sig != self._signature():
                self._an_timer.start()
        elif key == "queue":
            self.pages["queue"].refresh()

    def _arch_ui(self, arch: str):
        title = ARCH_TITLES.get(arch, arch)
        self.model_badge.setText(f"●  현재 모델 · {title}")
        self.nav_buttons["model"].setText(f"{self.nav_labels['model']}  ·  {title}")
        self.pages["project"].set_arch(arch)
        self.pages["dataset"].set_arch(arch)
        self._update_file_status()

    def change_arch(self, arch: str):
        """시작하기/모델 페이지 어디서 바꾸든 같은 경로로 처리 (모델별 파일 경로 교체)."""
        model: ModelPage = self.pages["model"]
        old = model.arch
        if arch == old:
            self._arch_ui(arch)
            return
        self.settings.remember_paths(old, model.get_paths())
        paths = self.settings.paths_for(arch)
        self.job.model.arch = arch
        model.set_arch(arch, paths)
        self.settings.save()
        self._arch_ui(arch)
        n = sum(1 for v in paths.values() if v)
        title = ARCH_TITLES.get(arch, arch)
        self.statusBar().showMessage(
            f"베이스 모델 → {title}. 모델 페이지에 {title} 용 경로"
            + (f" {n}개를 불러왔습니다." if n else "가 비어 있어 기본값을 자동으로 내려받습니다."), 6000)
        self._an_timer.start()

    def _update_file_status(self):
        """시작하기 모델 카드에 모델별 파일 준비 상태 표시."""
        model: ModelPage = self.pages["model"]
        for arch in ARCH_TITLES:
            paths = model.get_paths() if arch == model.arch else self.settings.paths_for(arch)
            set_n = sum(1 for c in COMPONENTS[arch] if paths.get(c.key))
            bad = [c for c in COMPONENTS[arch]
                   if check_path(arch, c.key, paths.get(c.key, "")).state in (MISSING, MISMATCH)]
            if bad:
                txt = f"⚠ 모델 파일 확인 필요: {', '.join(c.label for c in bad)}"
            elif set_n:
                txt = f"✓ 모델 파일 {set_n}개 지정됨 (나머지 자동 다운로드)" if set_n < len(COMPONENTS[arch]) else "✓ 모델 파일 모두 지정됨"
            else:
                txt = "⬇ 모델 파일 미지정 — 처음 학습 시 자동 다운로드"
            self.pages["project"].set_files_status(arch, txt)

    def load_pages(self):
        for k, page in self.pages.items():
            if k not in NON_JOB_PAGES:
                page.load(self.job)
        model: ModelPage = self.pages["model"]
        arch = self.job.model.arch
        paths = model.get_paths()
        if any(paths.values()):
            self.settings.remember_paths(arch, paths)
        else:
            lib = self.settings.paths_for(arch)
            if any(lib.values()):
                model.set_paths(lib)  # 프리셋/새 설정이라 경로가 비어 있으면 이 모델용으로 기억해 둔 경로 사용
        self._arch_ui(arch)

    def collect(self) -> TrainingJob:
        job = self.job.copy()
        for k, page in self.pages.items():
            if k not in NON_JOB_PAGES:
                page.store(job)
        job.project.output_dir = self.settings.output_dir or job.project.output_dir
        job.project.models_dir = self.settings.models_dir or job.project.models_dir
        if not job.train.comfy_lora_dir:
            job.train.comfy_lora_dir = self.settings.comfy_lora_dir
        self.settings.remember_paths(job.model.arch, self.pages["model"].get_paths())
        self.job = job
        return job

    # ------------------------------------------------------------------
    # 시스템 맞춤 추천
    # ------------------------------------------------------------------
    def _signature(self):
        job = self.job
        return (job.model.arch, tuple(f.path for f in job.dataset.folders), self.settings.purpose)

    def run_recommend(self, purpose: str = None):
        if purpose:
            self.settings.purpose = purpose
            self.settings.save()
        job = self.collect()
        self.pages["project"].recommend.set_busy(True)
        self.pool.start(_AnalysisJob(job.model.arch, list(job.dataset.folders), self.settings.purpose,
                                     self.calib, self._an_signals))

    def _analysis_done(self, result):
        panel = self.pages["project"].recommend
        panel.set_busy(False)
        if isinstance(result, Exception):
            self.statusBar().showMessage(f"추천 계산 실패: {result}", 8000)
            return
        self.spec = result.spec
        sig = self._signature()
        if sig != self._analysis_sig:
            self._applied_rec = ""
        self._analysis, self._analysis_sig = result, sig
        panel.show_analysis(result, self._applied_rec)

    def apply_recommendation(self, key: str):
        if not self._analysis:
            return
        rec = next((r for r in self._analysis.options if r.key == key), None)
        if rec is None:
            return
        if not rec.fits:
            r = QMessageBox.question(self, "VRAM 부족", "이 조합은 VRAM 이 부족할 수 있습니다. 그래도 적용할까요?")
            if r != QMessageBox.Yes:
                return
        job = self.collect()
        apply_to_job(job, rec)
        self.job = job
        self.load_pages()
        self._applied_rec = key
        self.pages["project"].recommend.mark_applied(key)
        self.statusBar().showMessage(
            f"추천 설정 적용: {rec.title} — 예상 {fmt_duration(rec.total_sec)} (학습 설정 · 모델 페이지에 반영됨)", 8000)

    # ------------------------------------------------------------------
    # 설정 파일
    # ------------------------------------------------------------------
    def new_job(self):
        self.job = TrainingJob()
        self.config_path = ""
        self.cfg_name.setText("(저장 안 됨)")
        self.load_pages()

    def open_job(self):
        path, _ = QFileDialog.getOpenFileName(self, "설정 열기", os.path.join(ROOT, "presets"), "YAML (*.yaml *.yml)")
        if path:
            self._load_from(path)

    def _load_from(self, path: str):
        try:
            self.job = TrainingJob.load_yaml(path)
        except Exception as e:
            QMessageBox.warning(self, "열기 실패", str(e))
            return
        self.config_path = path
        self.cfg_name.setText(os.path.basename(path))
        self.settings.push_recent(path)
        self.settings.save()
        self.load_pages()

    def apply_preset(self, path: str):
        if not path:
            return
        cur = self.collect()
        try:
            preset = TrainingJob.load_yaml(path)
        except Exception as e:
            QMessageBox.warning(self, "프리셋 오류", str(e))
            return
        # 사용자가 입력한 이름/데이터셋은 유지, 모델 경로는 그 모델용으로 기억해 둔 경로 사용
        preset.project = cur.project
        preset.dataset.folders = cur.dataset.folders
        if cur.dataset.trigger_word:
            preset.dataset.trigger_word = cur.dataset.trigger_word
        if preset.model.arch == cur.model.arch:
            src = {k: getattr(cur.model, f) for k, f in FIELD_OF.items()}
        else:
            self.settings.remember_paths(cur.model.arch, self.pages["model"].get_paths())
            src = self.settings.paths_for(preset.model.arch)
        for k, f in FIELD_OF.items():
            setattr(preset.model, f, src.get(k, ""))
        self.job = preset
        self.load_pages()
        self._applied_rec = ""
        self.statusBar().showMessage(f"프리셋 적용: {os.path.basename(path)}", 5000)
        self._an_timer.start()

    def save_job(self, as_new: bool = False):
        job = self.collect()
        path = self.config_path
        if as_new or not path or os.path.dirname(os.path.abspath(path)) == os.path.join(ROOT, "presets"):
            default = os.path.join(ROOT, "configs", f"{job.project.name}.yaml")
            os.makedirs(os.path.dirname(default), exist_ok=True)
            path, _ = QFileDialog.getSaveFileName(self, "설정 저장", default, "YAML (*.yaml)")
            if not path:
                return
        job.save_yaml(path)
        self.config_path = path
        self.cfg_name.setText(os.path.basename(path))
        self.settings.push_recent(path)
        self.settings.save()
        self.statusBar().showMessage(f"저장됨: {path}", 5000)

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------
    @staticmethod
    def _path_problems(job) -> list:
        out = []
        for c in COMPONENTS.get(job.model.arch, []):
            st = check_path(job.model.arch, c.key, getattr(job.model, FIELD_OF[c.key]))
            if st.state in (MISSING, MISMATCH):
                out.append(f"{c.label}: {st.message}")
        return out

    def start_training(self):
        if self.worker.running:
            return
        if self._queue_active:
            QMessageBox.information(self, "대기열 진행 중", "대기열이 진행 중입니다. 대기열 페이지에서 확인하세요.")
            return
        job = self.collect()
        errors = job.validate()
        if errors:
            QMessageBox.warning(self, "설정 확인", "\n".join(errors))
            return
        problems = self._path_problems(job)
        if problems:
            r = QMessageBox.question(
                self, "모델 파일 확인",
                f"{ARCH_TITLES.get(job.model.arch)} 모델 파일에 문제가 있습니다:\n\n" + "\n".join(problems)
                + "\n\n그래도 시작할까요?",
            )
            if r != QMessageBox.Yes:
                self.show_page("model")
                return
        if job.model.arch == "krea2" and not job.model.transformer_path and not self.settings.hf_token:
            r = QMessageBox.question(
                self, "허깅페이스 토큰",
                "Krea 2 Raw 자동 다운로드에는 허깅페이스 토큰이 필요합니다 (설정 페이지).\n그래도 시작할까요?",
            )
            if r != QMessageBox.Yes:
                return
        self._launch(job)

    def _launch(self, job, item=None):
        run_dir = job.run_dir
        os.makedirs(run_dir, exist_ok=True)
        cfg = os.path.join(run_dir, "job.yaml")
        job.save_yaml(cfg)
        run: RunPage = self.pages["run"]
        run.reset(job.train.max_steps, run_dir)
        run.set_running(True)
        self._run_job, self._run_item = job, item
        self._last_done, self._last_error, self._speed = None, "", []
        if item is not None:
            idx = self.queue.items.index(item) + 1
            self.queue.mark(item.id, RUNNING, message="", step=0, total=job.train.max_steps)
            self.pages["queue"].refresh()
            run.append_log(f"[대기열 {idx}/{len(self.queue.items)}] '{item.name}' 학습을 시작합니다.")
            self.statusBar().showMessage(f"대기열: '{item.name}' 학습 중")
        else:
            self.show_page("run")
        self.worker.start_module("lora_trainer.train.runner", ["--config", cfg], self.settings.hf_token, run_dir)

    def stop_training(self):
        if not self.worker.running:
            return
        if self._run_item is not None:
            box = QMessageBox(self)
            box.setWindowTitle("중지")
            box.setText(f"대기열 항목 '{self._run_item.name}' 을 중지합니다. 현재까지의 LoRA 는 저장됩니다.")
            only = box.addButton("이 항목만 중지 (다음 진행)", QMessageBox.AcceptRole)
            all_ = box.addButton("대기열 전체 중지", QMessageBox.DestructiveRole)
            box.addButton("취소", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() not in (only, all_):
                return
            self._queue_stop_all = box.clickedButton() is all_
        else:
            r = QMessageBox.question(self, "중지", "학습을 중지할까요? 현재까지의 LoRA 는 저장됩니다.")
            if r != QMessageBox.Yes:
                return
        self.worker.request_stop()
        self.pages["run"].status.setText("중지 요청됨 · 현재 스텝이 끝나면 저장 후 종료합니다")

    def _on_event(self, kind: str, data: dict):
        if kind == "done":
            self._last_done = data
            if data.get("path"):
                self.statusBar().showMessage(f"LoRA 저장: {data['path']}")
        elif kind == "error":
            self._last_error = data.get("message", "")
        elif kind == "progress":
            step, total = data.get("step", 0), data.get("total", 0)
            sps = data.get("sec_per_step", 0)
            if step >= 10 and sps > 0:
                self._speed = (self._speed + [sps])[-40:]
            if self._run_item is not None:
                self.pages["queue"].update_progress(self._run_item.id, step, total)

    def _record_speed(self):
        job = self._run_job
        if job is None or len(self._speed) < 3 or not self.spec.gpu_available:
            return
        try:
            measured = statistics.median(self._speed)
            est = estimate_sec_per_step(shape_from_job(job, self.spec), self.spec, 1.0)
            self.calib.record(self.spec, job.model.arch, measured, est)
        except Exception:
            pass

    def _finished(self, code: int):
        run: RunPage = self.pages["run"]
        run.set_running(False)
        if code not in (0, 2):
            run.append_log(f"[프로세스 종료 코드 {code}] 로그를 확인하세요.")
        self._record_speed()
        item, self._run_item = self._run_item, None
        if item is None:
            if self._queue_active:  # 수동 학습이 끝나기를 기다리던 대기열
                QTimer.singleShot(1500, self._queue_next)
            return
        done = self._last_done or {}
        if done.get("interrupted"):
            self.queue.mark(item.id, STOPPED, message="사용자 중지", result=done.get("path", ""))
            status = STOPPED
        elif code == 0 and done.get("path"):
            self.queue.mark(item.id, DONE, result=done["path"], message="")
            status = DONE
        else:
            self.queue.mark(item.id, FAILED, message=self._last_error or f"종료 코드 {code}")
            status = FAILED
        self._update_queue_nav()
        self.pages["queue"].refresh()
        if not self._queue_active:
            return
        if self._queue_stop_all or self._queue_stop_after:
            self._end_queue("사용자 요청으로 대기열을 멈췄습니다.")
        elif status == FAILED and not self.queue.continue_on_error:
            self._end_queue(f"'{item.name}' 실패로 대기열을 멈췄습니다.")
        else:
            QTimer.singleShot(2500, self._queue_next)

    # ------------------------------------------------------------------
    # 대기열
    # ------------------------------------------------------------------
    def _update_queue_nav(self):
        n = sum(1 for i in self.queue.items if i.status in ("waiting", "running"))
        self.nav_buttons["queue"].setText(self.nav_labels["queue"] + (f"  ({n})" if n else ""))

    def _set_stop_after(self, on: bool):
        self._queue_stop_after = on

    def _job_summary(self, job) -> str:
        t, n, m = job.train, job.network, job.model
        res = "+".join(str(r) for r in job.dataset.resolutions)
        parts = [
            f"<b>{ARCH_TITLES.get(m.arch, m.arch)}</b>",
            f"DiT {m.transformer_quant}" + (f" · 스왑 {m.block_swap}" if m.block_swap > 0 else ""),
            f"해상도 {res}", f"rank {n.rank}", f"{t.max_steps:,} 스텝", f"lr {t.learning_rate:.1e}", t.optimizer,
        ]
        txt = " · ".join(parts)
        try:
            sps = estimate_sec_per_step(shape_from_job(job, self.spec), self.spec,
                                        self.calib.ratio(self.spec, m.arch)[0])
            txt += f"<br>예상 학습 시간 약 {fmt_duration(sps * t.max_steps)} (캐싱 제외)"
        except Exception:
            pass
        return txt

    def add_to_queue(self):
        job = self.collect()
        errors = [e for e in job.validate() if "데이터셋" not in e and "LoRA 이름" not in e]
        if errors:
            QMessageBox.warning(self, "설정 확인", "현재 설정에 문제가 있어 예약할 수 없습니다:\n" + "\n".join(errors))
            return
        out_dir = job.project.output_dir

        def suggest(name):
            final = self.queue.unique_name(name, out_dir)
            if self.worker.running and self._run_job and final.lower() == self._run_job.project.name.lower():
                final = self.queue.unique_name(final + "_2", out_dir)
            return final

        dlg = AddQueueDialog(self._job_summary(job), suggest, job.dataset.trigger_word, self)
        if dlg.exec() != QDialog.Accepted:
            return
        name, folder, trigger = dlg.values()
        final = suggest(name)
        it = self.queue.add(job, final, [folder], trigger or None)
        self._update_queue_nav()
        self.pages["queue"].refresh(select=it.id)
        self.statusBar().showMessage(f"대기열에 추가: {final} (총 대기 {self.queue.counts['waiting']}개)", 6000)

    def start_queue(self):
        if self._queue_active:
            return
        if not self.queue.next_waiting():
            QMessageBox.information(self, "대기열", "대기 중인 항목이 없습니다.")
            return
        self._queue_active = True
        self._queue_stop_all = False
        self.pages["queue"].set_active(True)
        if self.worker.running:
            self.statusBar().showMessage("현재 학습이 끝나면 대기열을 시작합니다.", 8000)
            return
        self._queue_next()

    def _end_queue(self, msg: str):
        self._queue_active = False
        self._queue_stop_all = False
        self._queue_stop_after = False
        self.pages["queue"].set_active(False)
        self._update_queue_nav()
        self.pages["run"].append_log(f"[대기열] {msg}")
        self.statusBar().showMessage(msg, 10000)

    def _queue_next(self):
        if not self._queue_active:
            return
        if self.worker.running:
            QTimer.singleShot(3000, self._queue_next)
            return
        item = self.queue.next_waiting()
        if item is None:
            c = self.queue.counts
            self._end_queue(f"대기열 완료 — 완료 {c['done']} · 실패 {c['failed']} · 중지 {c['stopped']}")
            return
        try:
            job = TrainingJob.load_yaml(item.config)
        except Exception as e:
            self.queue.mark(item.id, FAILED, message=f"설정 파일 오류: {e}")
            QTimer.singleShot(500, self._queue_next)
            return
        job.project.output_dir = self.settings.output_dir or job.project.output_dir
        job.project.models_dir = self.settings.models_dir or job.project.models_dir
        errors = job.validate() + self._path_problems(job)
        if errors:
            self.queue.mark(item.id, FAILED, message=errors[0])
            self.pages["queue"].refresh()
            if self.queue.continue_on_error:
                QTimer.singleShot(500, self._queue_next)
            else:
                self._end_queue(f"'{item.name}' 설정 오류로 대기열을 멈췄습니다.")
            return
        self._launch(job, item)

    # ------------------------------------------------------------------
    def closeEvent(self, e):
        if self.worker.running:
            extra = "\n(대기열은 저장되며, 진행 중 항목은 '중지됨'으로 표시됩니다)" if self._queue_active else ""
            r = QMessageBox.question(self, "종료", "학습이 진행 중입니다. 중지하고 종료할까요?" + extra)
            if r != QMessageBox.Yes:
                e.ignore()
                return
            self.worker.request_stop()
            self.worker.proc.waitForFinished(30000)
            self.worker.kill()
        try:
            self.collect()
            self.settings.save()
        except Exception:
            pass
        self.sysmon.shutdown()
        e.accept()
