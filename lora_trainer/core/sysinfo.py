"""시스템 사양 감지 및 실시간 사용량 측정 (CPU / RAM / GPU / VRAM / 가상메모리).

- 사양(SystemSpec): 추천 옵션 계산에 사용 (VRAM, 시스템 RAM, CPU 코어, 가상메모리)
- 실시간 사용량(sample_usage): 사이드바 상태 표시에 사용
  GPU 는 nvidia-ml-py(pynvml) 가 있으면 사용하고, 없으면 torch 로 VRAM 만 측정합니다.
  학습은 별도 프로세스에서 돌기 때문에 장치 전체 기준(드라이버 기준) 값을 읽습니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import psutil
except ImportError:  # pragma: no cover - requirements 에 포함
    psutil = None


GB = 1024 ** 3


@dataclass
class SystemSpec:
    gpu_available: bool = False
    gpu_name: str = "CPU"
    vram_gb: float = 0.0
    gpu_capability: tuple = (0, 0)
    ram_gb: float = 0.0
    ram_avail_gb: float = 0.0
    cpu_name: str = ""
    cpu_cores: int = 0              # 물리 코어
    cpu_threads: int = 0            # 논리 코어
    swap_gb: float = 0.0            # 가상메모리(페이지 파일) 크기

    def describe(self) -> str:
        gpu = f"{self.gpu_name} {self.vram_gb:.0f}GB" if self.gpu_available else "CUDA GPU 없음"
        return (f"{gpu} · RAM {self.ram_gb:.0f}GB · CPU {self.cpu_cores}코어/{self.cpu_threads}스레드 · "
                f"가상메모리 {self.swap_gb:.0f}GB")


def _cpu_name() -> str:
    import platform

    name = platform.processor() or ""
    if os.name == "nt":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except Exception:
            pass
    elif os.path.isfile("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    return name


def detect_system() -> SystemSpec:
    """현재 PC 사양을 읽는다 (GPU 는 0번 장치 기준)."""
    spec = SystemSpec()
    try:
        from .device import gpu_info

        info = gpu_info()
        spec.gpu_available = info.available
        spec.gpu_name = info.name
        spec.vram_gb = info.total_gb
        spec.gpu_capability = info.capability
    except Exception:
        pass
    if psutil is not None:
        vm = psutil.virtual_memory()
        spec.ram_gb = vm.total / GB
        spec.ram_avail_gb = vm.available / GB
        spec.swap_gb = psutil.swap_memory().total / GB
        spec.cpu_cores = psutil.cpu_count(logical=False) or 0
        spec.cpu_threads = psutil.cpu_count(logical=True) or 0
    else:  # pragma: no cover
        spec.cpu_threads = os.cpu_count() or 0
        spec.cpu_cores = spec.cpu_threads
    spec.cpu_name = _cpu_name()
    return spec


# ---------------------------------------------------------------------------
# 실시간 사용량
# ---------------------------------------------------------------------------
@dataclass
class Usage:
    cpu_pct: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    gpu_ok: bool = False
    gpu_pct: Optional[float] = None       # None = 측정 불가
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    gpu_temp: Optional[float] = None
    extra: dict = field(default_factory=dict)


class UsageSampler:
    """주기적으로 호출해 사용량을 읽는다. (백그라운드 스레드에서 사용)"""

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self._nvml = None
        self._handle = None
        self._torch_ok = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self._nvml = pynvml
        except Exception:
            self._nvml = None
        if psutil is not None:
            psutil.cpu_percent(interval=None)  # 첫 호출은 기준값 설정용

    @property
    def gpu_source(self) -> str:
        if self._nvml is not None:
            return "nvml"
        return "torch" if self._torch_ok else "none"

    def sample(self) -> Usage:
        u = Usage()
        if psutil is not None:
            u.cpu_pct = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            u.ram_total_gb = vm.total / GB
            u.ram_used_gb = (vm.total - vm.available) / GB
        if self._nvml is not None:
            try:
                n = self._nvml
                mem = n.nvmlDeviceGetMemoryInfo(self._handle)
                util = n.nvmlDeviceGetUtilizationRates(self._handle)
                u.gpu_ok = True
                u.vram_total_gb = mem.total / GB
                u.vram_used_gb = mem.used / GB
                u.gpu_pct = float(util.gpu)
                try:
                    u.gpu_temp = float(n.nvmlDeviceGetTemperature(self._handle, n.NVML_TEMPERATURE_GPU))
                except Exception:
                    pass
                return u
            except Exception:
                self._nvml = None
        if self._torch_ok is not False:
            try:
                import torch

                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info(self.gpu_index)
                    u.gpu_ok = True
                    u.vram_total_gb = total / GB
                    u.vram_used_gb = (total - free) / GB
                    self._torch_ok = True
                else:
                    self._torch_ok = False
            except Exception:
                self._torch_ok = False
        return u

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None
