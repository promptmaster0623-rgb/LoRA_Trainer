"""LoRA 학습 엔진.

단계
  1. 데이터셋 스캔 / 버킷 결정
  2. 텍스트 임베딩 캐시  (텍스트 인코더만 GPU 에 → 끝나면 해제)
  3. latent 캐시         (VAE 만 GPU 에 → 끝나면 해제)
  4. DiT 로딩 (FP8/NF4 양자화, 블록 스왑) + LoRA 주입
  5. 학습 루프 (그래디언트 누적, 체크포인트 저장, 샘플 생성, 일시정지/중지)
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import shutil
import time
import traceback
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .. import __version__
from ..core.config import TrainingJob
from ..core.device import auto_memory_plan, free_memory, gpu_info, torch_dtype
from ..core.events import EventSink
from ..data.cache import BucketBatcher, FeatureCache, build_latent_cache, build_text_cache
from ..data.dataset import apply_trigger, dataset_summary, scan_dataset
from ..models import get_adapter_class
from ..network.ema import LoRAEma
from ..network.lora import LoRANetwork, build_target_filter
from .adaptive import LR_SCALE_FILE, AdaptiveController
from .optim import build_optimizer, build_scheduler, effective_lr, sample_timesteps, set_optimizer_mode

STOP_FILE = ".stop"
PAUSE_FILE = ".pause"
SAMPLE_NOW_FILE = ".sample_now"
SAVE_NOW_FILE = ".save_now"


class Trainer:
    def __init__(self, job: TrainingJob, sink: Optional[EventSink] = None):
        self.job = job
        self.sink = sink or EventSink()
        self.run_dir = job.run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        for f in (STOP_FILE, PAUSE_FILE, SAMPLE_NOW_FILE, SAVE_NOW_FILE, LR_SCALE_FILE):
            p = os.path.join(self.run_dir, f)
            if os.path.exists(p):
                os.remove(p)
        self.adapter = None
        self.network: Optional[LoRANetwork] = None
        self.step = 0
        self.optimizer = None
        self.adaptive: Optional[AdaptiveController] = None

    # ------------------------------------------------------------------
    # 제어 파일 (GUI 가 생성)
    # ------------------------------------------------------------------
    def _flag(self, name: str, consume: bool = False) -> bool:
        p = os.path.join(self.run_dir, name)
        if os.path.exists(p):
            if consume:
                try:
                    os.remove(p)
                except OSError:
                    pass
            return True
        return False

    def should_stop(self) -> bool:
        return self._flag(STOP_FILE)

    def log(self, msg: str) -> None:
        self.sink.log(msg)

    # ------------------------------------------------------------------
    def run(self) -> Optional[str]:
        try:
            return self._run()
        except InterruptedError:
            self.sink.status("중지", "사용자가 학습을 중지했습니다.")
            if self.network is not None and self.step > 0:
                path = self.save_lora(self.step, final=False, tag="interrupted")
                self.sink.emit("done", path=path, interrupted=True)
            else:
                self.sink.emit("done", path="", interrupted=True)
            return None
        except Exception as e:  # pragma: no cover - 런타임 오류 보고
            tb = traceback.format_exc()
            self.log(tb)
            self.sink.emit("error", message=str(e), traceback=tb)
            raise

    def _run(self) -> Optional[str]:
        job = self.job
        errors = job.validate()
        if errors:
            raise ValueError("설정 오류:\n" + "\n".join(errors))
        job.save_yaml(os.path.join(self.run_dir, "config.yaml"))

        seed = job.train.seed
        random.seed(seed)
        torch.manual_seed(seed)

        info = gpu_info()
        device = torch.device("cuda:0" if info.available else "cpu")
        dtype = torch_dtype(job.model.dtype)
        if dtype == torch.bfloat16 and info.available and not info.supports_bf16:
            self.log("[경고] 이 GPU 는 bf16 을 지원하지 않아 fp16 으로 전환합니다.")
            dtype = torch.float16
        self.sink.status("준비", f"GPU: {info.describe()}")
        plan = auto_memory_plan(job.model.arch, info.total_gb if info.available else 0)
        t_quant = plan.transformer_quant if job.model.transformer_quant == "auto" else job.model.transformer_quant
        te_quant = plan.text_encoder_quant if job.model.text_encoder_quant == "auto" else job.model.text_encoder_quant
        swap = plan.block_swap if job.model.block_swap < 0 else job.model.block_swap
        if t_quant == "nf4" and swap:
            self.log("[알림] NF4 는 블록 스왑과 함께 쓸 수 없어 블록 스왑을 끕니다.")
            swap = 0
        if not info.available:
            t_quant, te_quant, swap = ("none" if t_quant == "nf4" else t_quant), "none", 0
        self.sink.status("준비", f"메모리 설정: DiT={t_quant}, 텍스트인코더={te_quant}, 블록스왑={swap}")

        cls = get_adapter_class(job.model.arch)
        adapter = cls(job.model, job.project.models_dir, job.project.hf_token, device, dtype, log=self.log)
        self.adapter = adapter

        # 1. 데이터셋 ---------------------------------------------------------
        self.sink.status("데이터셋", "이미지 스캔 중")
        items, samples = scan_dataset(job.dataset, adapter.spatial_divisor)
        if not samples:
            raise ValueError("학습할 이미지가 없습니다.")
        self.log(dataset_summary(items, samples))

        cache = FeatureCache(job.cache_root, adapter.arch)
        te_tag = f"{adapter.path_or_default('text_encoder')}|{te_quant}"
        vae_tag = adapter.path_or_default("vae")

        trigger = job.dataset.trigger_word
        sample_prompts = [apply_trigger(p, trigger) for p in job.sample.prompts if p.strip()]
        neg = job.sample.negative_prompt
        prompts = [s.caption for s in samples] + [""] + sample_prompts + [neg]

        # 2. 텍스트 캐시 ------------------------------------------------------
        need_text = [p for p in dict.fromkeys(prompts) if not cache.has_text(cache.text_key(p, te_tag))]
        if need_text:
            self.sink.status("텍스트 캐시", f"{len(need_text)}개 프롬프트 인코딩")
            adapter.load_text_encoder(te_quant)
            text_map = build_text_cache(adapter, cache, prompts, te_tag,
                                        progress=lambda d, t: self.sink.emit("cache", kind="text", done=d, total=t),
                                        should_stop=self.should_stop)
            adapter.unload_text_encoder()
        else:
            text_map = {p: cache.text_key(p, te_tag) for p in prompts}
            self.log("텍스트 캐시: 모두 캐시됨")

        # 3. latent 캐시 -----------------------------------------------------
        lat_keys = [cache.latent_key(s, vae_tag) for s in samples]
        missing = sum(1 for k in lat_keys if not cache.has_latent(k))
        if missing:
            self.sink.status("Latent 캐시", f"{missing}개 이미지 인코딩")
            adapter.load_vae()
            build_latent_cache(adapter, cache, samples, vae_tag,
                               progress=lambda d, t: self.sink.emit("cache", kind="latent", done=d, total=t),
                               should_stop=self.should_stop)
            adapter.unload_vae()
        else:
            self.log("Latent 캐시: 모두 캐시됨")
        free_memory()

        # 4. DiT + LoRA ------------------------------------------------------
        self.sink.status("모델", "DiT 로딩 중 (시간이 걸릴 수 있습니다)")
        model = adapter.load_transformer(t_quant, swap)
        net_cfg = job.network
        target = net_cfg.target or adapter.default_lora_target
        filt = build_target_filter(adapter.lora_presets, target, net_cfg.include_regex)
        network = LoRANetwork(model, filt, net_cfg.rank, net_cfg.alpha, net_cfg.dropout,
                              net_cfg.rank_dropout, net_cfg.rs_lora, net_cfg.exclude_regex)
        network.to(device=device, dtype=torch.float32)
        if net_cfg.init_weights:
            loaded, total = network.load_weights(net_cfg.init_weights)
            self.log(f"기존 LoRA 불러옴: {loaded}/{total} 모듈")
        network.apply()
        self.network = network
        self.log(f"LoRA: {len(network.loras)}개 레이어, 학습 파라미터 {network.num_params() / 1e6:.2f}M")

        tc = job.train
        self.ema = LoRAEma(network, tc.ema_decay)
        if self.ema.enabled:
            self.log(f"EMA: 저장·샘플에 가중치 평균 사용 (감쇠 {tc.ema_decay})")
        optimizer = build_optimizer(network.trainable_params(), tc, self.log)
        self.optimizer = optimizer
        self.log(f"옵티마이저: {type(optimizer).__name__} (설정 '{tc.optimizer}')")
        adaptive = AdaptiveController(tc.adaptive, network, optimizer, self.sink, self.run_dir)
        self.adaptive = adaptive
        scheduler = build_scheduler(optimizer, tc, adaptive.scale, self.log)
        if tc.resume_state:
            self._load_state(tc.resume_state, optimizer, scheduler)

        entries = [(k, text_map[s.caption], s.bucket, s.item.repeats) for s, k in zip(samples, lat_keys)]
        # 적응형 학습: 검증용 이미지 분리 / 평가용 고정 샘플
        val_entries = []
        ad_cfg = tc.adaptive
        if ad_cfg.enabled and ad_cfg.eval_every > 0:
            rng = random.Random(seed)
            if ad_cfg.val_holdout > 0:
                img_paths = sorted({s.item.path for s in samples})
                if len(img_paths) > ad_cfg.val_holdout:
                    hold = set(rng.sample(img_paths, ad_cfg.val_holdout))
                    val_entries = [e for e, s in zip(entries, samples) if s.item.path in hold]
                    entries = [e for e, s in zip(entries, samples) if s.item.path not in hold]
                    self.log(f"검증용 이미지 {len(hold)}장 분리 (학습에서 제외): "
                             + ", ".join(os.path.basename(p) for p in sorted(hold)))
                else:
                    self.log("[경고] 이미지 수가 너무 적어 검증용 분리를 건너뜁니다 (학습 이미지로 평가).")
            if not val_entries:
                val_entries = rng.sample(entries, min(len(entries), ad_cfg.eval_samples))
            val_entries = val_entries[: max(1, ad_cfg.eval_samples)]
        batcher = BucketBatcher(entries, tc.batch_size, seed)
        empty_key = text_map[""]
        empty_text = cache.load_text(empty_key)
        self.log(f"에폭당 스텝: {batcher.steps_per_epoch()} (배치 {tc.batch_size}, 누적 {tc.grad_accum})")

        use_scaler = dtype == torch.float16 and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_scaler else None
        autocast = device.type == "cuda"

        # 샘플/저장 준비
        sample_conds = [(p, text_map[p]) for p in sample_prompts]
        neg_text = cache.load_text(text_map[neg])

        def eval_loss() -> float:
            """고정 노이즈 · 고정 타임스텝 loss (적응형 학습의 검증 지표)."""
            if not val_entries:
                return float("nan")
            model.eval()
            network.eval()
            g = torch.Generator().manual_seed(20240607)
            total, n = 0.0, 0
            with torch.no_grad():
                for lk, tk, _bucket, _rep in val_entries:
                    lat = cache.load_latent(lk).unsqueeze(0).to(device, torch.float32)
                    txt = adapter.collate_text([cache.load_text(tk)])
                    for tv in (0.25, 0.5, 0.75):
                        noise = torch.randn(lat.shape, generator=g).to(device)
                        tt_ = torch.full((1,), tv, device=device)
                        noisy_ = (1 - tv) * lat + tv * noise
                        if adapter.swapper is not None:
                            adapter.swapper.offload_all()
                        with torch.autocast(device.type, dtype=dtype, enabled=autocast):
                            pred_ = adapter.predict(noisy_.to(dtype), tt_, txt)
                        total += F.mse_loss(pred_.float(), noise - lat).item()
                        n += 1
            model.train()
            network.train()
            return total / max(1, n)

        adaptive.set_eval_fn(eval_loss)

        # 5. 학습 루프 -------------------------------------------------------
        if job.sample.enabled and job.sample.at_start and self.step == 0:
            self.generate_samples(sample_conds, neg_text, cache)

        self.sink.status("학습", f"총 {tc.max_steps} 스텝 시작")
        model.train()
        network.train()
        it = iter(batcher)
        gen = torch.Generator().manual_seed(seed + self.step)
        t_last = time.time()
        ema_loss = None
        step_times: List[float] = []
        while self.step < tc.max_steps:
            if self.should_stop():
                raise InterruptedError()
            while self._flag(PAUSE_FILE):
                if self.should_stop():
                    raise InterruptedError()
                time.sleep(0.5)
            t0 = time.time()
            set_optimizer_mode(optimizer, train=True)
            adaptive.poll_controls(self.step)
            if adapter.swapper is not None:
                adapter.swapper.offload_all()
            total_loss = 0.0
            for _ in range(tc.grad_accum):
                batch = next(it)
                latents = torch.stack([cache.load_latent(lk) for lk, _ in batch]).to(device, torch.float32)
                texts = []
                for _lk, tk in batch:
                    if job.dataset.caption_dropout > 0 and random.random() < job.dataset.caption_dropout:
                        texts.append(empty_text)
                    else:
                        texts.append(cache.load_text(tk))
                text = adapter.collate_text(texts)
                b = latents.shape[0]
                noise = torch.randn(latents.shape, generator=gen).to(device)
                shift = adapter.shift_for(tuple(latents.shape[-2:]), tc.timestep)
                t = adapter.sample_train_timesteps(b, tc.timestep, shift, gen)
                if t is None:
                    t = sample_timesteps(b, tc.timestep, shift, gen)
                t = t.to(device)
                tt = t.view(-1, 1, 1, 1)
                noisy = (1 - tt) * latents + tt * noise
                target = noise - latents
                with torch.autocast(device.type, dtype=dtype, enabled=autocast):
                    pred = adapter.predict(noisy.to(dtype), t, text)
                if tc.loss == "huber":
                    loss = F.huber_loss(pred.float(), target, delta=tc.huber_delta)
                else:
                    loss = F.mse_loss(pred.float(), target)
                loss = loss / tc.grad_accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                total_loss += loss.item()
            if scaler is not None:
                scaler.unscale_(optimizer)
            clip = adaptive.clip_value(tc.max_grad_norm)
            if clip > 0:
                adaptive.record_grad_norm(torch.nn.utils.clip_grad_norm_(network.trainable_params(), clip))
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            self.step += 1
            self.ema.update()

            action = adaptive.after_step(self.step, total_loss)  # 발산 복구 · 정체 감지 · 검증 · 조기 종료
            if action == "recover":
                ema_loss = None
                self.ema.reset()
                continue
            if action == "stop":
                break
            ema_loss = total_loss if ema_loss is None else ema_loss * 0.95 + total_loss * 0.05
            step_times.append(time.time() - t0)
            step_times = step_times[-20:]
            if time.time() - t_last > 0.5 or self.step == tc.max_steps:
                t_last = time.time()
                avg = sum(step_times) / len(step_times)
                vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
                self.sink.emit("progress", step=self.step, total=tc.max_steps, loss=total_loss,
                               loss_avg=ema_loss, lr=effective_lr(optimizer, scheduler),
                               lr_scale=adaptive.scale(), grad_norm=adaptive.last_grad_norm,
                               sec_per_step=avg, eta_sec=avg * (tc.max_steps - self.step), vram_gb=vram)

            if self.step < tc.max_steps:
                if (tc.save_every and self.step % tc.save_every == 0) or self._flag(SAVE_NOW_FILE, True):
                    self.save_lora(self.step)
                    if tc.save_state:
                        self._save_state(optimizer, scheduler)
                if job.sample.enabled and ((job.sample.every and self.step % job.sample.every == 0)
                                           or self._flag(SAMPLE_NOW_FILE, True)):
                    self.generate_samples(sample_conds, neg_text, cache)
                    model.train()
                    network.train()

        final = self.save_lora(self.step, final=True)
        if tc.save_state:
            self._save_state(optimizer, scheduler)
        if job.sample.enabled:
            self.generate_samples(sample_conds, neg_text, cache)
        self.sink.status("완료", f"학습 완료: {final}")
        self.sink.emit("done", path=final, interrupted=False)
        return final

    # ------------------------------------------------------------------
    # 샘플
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_samples(self, sample_conds, neg_text, cache: FeatureCache) -> None:
        set_optimizer_mode(self.optimizer, train=False)  # Schedule-Free: 평균 가중치로 샘플
        if not sample_conds:
            return
        sc = self.job.sample
        adapter = self.adapter
        self.sink.status("샘플", f"{len(sample_conds)}장 생성 중 (스텝 {self.step})")
        adapter.transformer.eval()
        self.network.eval()
        guidance = sc.guidance if sc.guidance >= 0 else adapter.default_guidance
        steps = sc.steps
        div = adapter.spatial_divisor
        w, h = (sc.width // div) * div, (sc.height // div) * div
        uncond = adapter.collate_text([neg_text])
        adapter.sample_options = {"sampler": sc.sampler}
        turbo = self._attach_turbo()
        if turbo:
            steps, guidance = max(1, sc.turbo_steps), 0.0      # Turbo: CFG 없이 (CFG 1)
            adapter.sample_options["mu"] = 1.15
        latents = []
        try:
            with self._ema_weights():
                for i, (prompt, key) in enumerate(sample_conds):
                    cond = adapter.collate_text([cache.load_text(key)])
                    latents.append(adapter.sample(cond, uncond, w, h, steps, guidance, sc.seed + i).cpu())
        finally:
            if turbo:
                self._turbo.detach()
            adapter.sample_options = {}
        free_memory()
        adapter.load_vae()
        out_dir = os.path.join(self.run_dir, "samples")
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for i, lat in enumerate(latents):
            img = adapter.decode_latents(lat)[0]
            p = os.path.join(out_dir, f"step{self.step:06d}_{i:02d}.png")
            img.save(p)
            paths.append(p)
        adapter.unload_vae()
        free_memory()
        prompts = [p for p, _ in sample_conds]
        self.sink.emit("sample", step=self.step, paths=paths, prompts=prompts)

    # ------------------------------------------------------------------
    # 저장
    # ------------------------------------------------------------------
    def _metadata(self) -> Dict[str, str]:
        job = self.job
        cfg = job.to_dict()
        cfg["project"]["hf_token"] = ""
        return {
            "ss_network_module": "lora_trainer.lora",
            "ss_network_dim": str(job.network.rank),
            "ss_network_alpha": str(job.network.alpha),
            "ss_base_model_version": self.adapter.base_model_version(),
            "ss_training_steps": str(self.step),
            "ss_output_name": job.project.name,
            "lora_trainer_version": __version__,
            "lora_trainer_trigger": job.dataset.trigger_word,
            "lora_trainer_config": json.dumps(cfg, ensure_ascii=False),
        }

    def save_lora(self, step: int, final: bool = False, tag: str = "") -> str:
        set_optimizer_mode(self.optimizer, train=False)  # Schedule-Free: 평균 가중치로 저장
        job = self.job
        name = job.project.name
        fname = f"{name}.safetensors" if final else f"{name}_{tag or f'{step:06d}'}.safetensors"
        path = os.path.join(self.run_dir, fname)
        with self._ema_weights():
            self.network.save(path, job.train.save_format, torch_dtype(job.train.save_dtype), self._metadata())
            try:  # 학습이 진행되는지 바로 알 수 있도록 (lora_up 은 0 에서 시작)
                ups = [l.lora_up.weight.detach().float().abs().mean().item() for l in self.network.loras]
                up_mean = sum(ups) / len(ups)
            except Exception:
                up_mean = None
        self.sink.emit("saved", step=step, path=path)
        self.log(f"LoRA 저장: {path}" + (" (EMA 평균 가중치)" if self._ema_active() else ""))
        if up_mean is not None:
            self.log(f"  LoRA 변화량: up 평균 |w| = {up_mean:.2e}"
                     + ("  ← 0 에 가까움: 학습이 거의 안 되는 중!" if up_mean < 1e-6 else ""))
        if job.train.comfy_lora_dir and os.path.isdir(job.train.comfy_lora_dir):
            try:
                shutil.copy2(path, os.path.join(job.train.comfy_lora_dir, fname))
                self.log(f"ComfyUI loras 폴더로 복사: {job.train.comfy_lora_dir}")
            except OSError as e:
                self.log(f"[경고] ComfyUI 폴더 복사 실패: {e}")
        if not final and job.train.keep_last > 0:
            self._cleanup_checkpoints(name, job.train.keep_last)
        return path

    def _attach_turbo(self) -> bool:
        """Krea 2 Turbo LoRA 를 미리보기 동안만 덧붙임. 실패하면 경고 후 일반 샘플링."""
        sc, adapter = self.job.sample, self.adapter
        if not sc.turbo or "turbo_lora" not in adapter.defaults:
            return False
        try:
            if getattr(self, "_turbo", None) is None:
                from ..network.extra_lora import StaticLoRA
                from ..utils.hub import resolve_path

                spec = sc.turbo_lora.strip() or adapter.defaults["turbo_lora"]
                path = resolve_path(spec, self.job.project.models_dir, self.job.project.hf_token, log=self.log)
                self._turbo = StaticLoRA(path, adapter.dtype)
                self.log(f"Turbo LoRA (미리보기 전용): {path}")
            n, total = self._turbo.attach(adapter.transformer, sc.turbo_strength)
            if n == 0:
                self.log("[경고] Turbo LoRA 가 모델과 맞지 않습니다 (적용 0개). 일반 샘플링으로 진행합니다.")
                return False
            if not getattr(self, "_turbo_logged", False):
                self._turbo_logged = True
                self.log(f"  Turbo LoRA 적용 {n}/{total}개 모듈, 강도 {sc.turbo_strength}, {sc.turbo_steps}스텝, CFG 1")
            return True
        except Exception as e:  # 다운로드 실패 등
            self.log(f"[경고] Turbo LoRA 를 쓰지 못했습니다 ({e}). 일반 샘플링으로 진행합니다.")
            sc.turbo = False
            return False

    def _ema_active(self) -> bool:
        ema = getattr(self, "ema", None)
        return bool(ema is not None and ema.enabled and ema.updates > 0)

    def _ema_weights(self):
        ema = getattr(self, "ema", None)
        return ema.swap() if ema is not None else contextlib.nullcontext()

    def _cleanup_checkpoints(self, name: str, keep: int) -> None:
        files = sorted(
            f for f in os.listdir(self.run_dir)
            if f.startswith(name + "_") and f.endswith(".safetensors") and f[len(name) + 1:-12].isdigit()
        )
        for f in files[:-keep]:
            try:
                os.remove(os.path.join(self.run_dir, f))
            except OSError:
                pass

    def _save_state(self, optimizer, scheduler) -> None:
        state_dir = os.path.join(self.run_dir, "state")
        os.makedirs(state_dir, exist_ok=True)
        torch.save(
            {
                "step": self.step,
                "lora": {k: v.detach().cpu() for k, v in self.network.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": torch.get_rng_state(),
                "adaptive": self.adaptive.state_dict() if self.adaptive is not None else {},
                "ema": self.ema.state_dict() if getattr(self, "ema", None) is not None else {},
            },
            os.path.join(state_dir, "state.pt"),
        )

    def _load_state(self, state_dir: str, optimizer, scheduler) -> None:
        path = state_dir if state_dir.endswith(".pt") else os.path.join(state_dir, "state.pt")
        st = torch.load(path, map_location="cpu", weights_only=False)
        self.network.load_state_dict(st["lora"], strict=False)
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        torch.set_rng_state(st["rng"])
        if self.adaptive is not None and st.get("adaptive"):
            self.adaptive.load_state_dict(st["adaptive"])
        if getattr(self, "ema", None) is not None:
            if not (st.get("ema") and self.ema.load_state_dict(st["ema"])):
                self.ema.reset()
        self.step = int(st["step"])
        self.log(f"이어학습: 스텝 {self.step} 부터 재개")
