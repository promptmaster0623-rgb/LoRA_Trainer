"""모델 경로 해석 및 허깅페이스 다운로드.

지원 형식
  - 로컬 파일 경로           : D:/ComfyUI/models/diffusion_models/raw.safetensors
  - 로컬 폴더 경로           : D:/models/Qwen3-VL-4B-Instruct
  - 허깅페이스 저장소        : Qwen/Qwen3-VL-4B-Instruct
  - 저장소 + 하위폴더        : Qwen/Qwen-Image::vae
  - 저장소 내 단일 파일      : krea/Krea-2-Raw/raw.safetensors
다운로드 파일은 <models_dir>/<org>/<repo>/ 아래에 그대로 저장된다.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional

Log = Optional[Callable[[str], None]]


def _is_repo_id(spec: str) -> bool:
    parts = spec.split("/")
    return (
        len(parts) == 2
        and all(parts)
        and not os.path.exists(spec)
        and ":" not in parts[0]
        and "\\" not in spec
    )


def _local_dir(models_dir: str, repo_id: str) -> str:
    return os.path.join(models_dir, *repo_id.split("/"))


def resolve_path(
    spec: str,
    models_dir: str = "models",
    token: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    log: Log = None,
) -> str:
    """spec 을 실제 로컬 경로(파일 또는 폴더)로 변환한다. 필요하면 다운로드."""
    spec = (spec or "").strip().strip('"')
    if not spec:
        raise ValueError("모델 경로가 비어 있습니다.")
    if os.path.exists(spec):
        return spec

    token = token or os.environ.get("HF_TOKEN") or None

    # 저장소::하위폴더
    if "::" in spec:
        repo_id, sub = spec.split("::", 1)
        local = _local_dir(models_dir, repo_id)
        target = os.path.join(local, sub)
        if _dir_ready(target):
            return target
        patterns = [f"{sub}/*"] + [f"{sub}/**"]
        _snapshot(repo_id, local, patterns, token, log)
        return target

    # 저장소 내 단일 파일
    parts = spec.replace("\\", "/").split("/")
    if len(parts) >= 3 and "." in parts[-1]:
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        local = _local_dir(models_dir, repo_id)
        target = os.path.join(local, filename)
        if os.path.isfile(target):
            return target
        from huggingface_hub import hf_hub_download

        if log:
            log(f"다운로드: {repo_id} / {filename}")
        return hf_hub_download(repo_id, filename, local_dir=local, token=token)

    # 저장소 전체 (패턴 제한 가능)
    if _is_repo_id(spec):
        local = _local_dir(models_dir, spec)
        if _dir_ready(local) and not allow_patterns:
            return local
        _snapshot(spec, local, allow_patterns, token, log)
        return local

    raise FileNotFoundError(f"경로를 찾을 수 없습니다: {spec}")


def _dir_ready(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for _root, _dirs, files in os.walk(path):
        if any(f.endswith((".json", ".safetensors")) for f in files):
            return True
    return False


def _snapshot(repo_id, local, patterns, token, log) -> None:
    from huggingface_hub import snapshot_download

    if log:
        log(f"다운로드: {repo_id} {patterns or ''} -> {local}")
    snapshot_download(repo_id, local_dir=local, allow_patterns=patterns, token=token)


def find_safetensors(path: str) -> List[str]:
    """파일이면 그 파일, 폴더면 내부의 .safetensors 목록(정렬)."""
    if os.path.isfile(path):
        return [path]
    out = []
    for name in sorted(os.listdir(path)):
        if name.endswith(".safetensors"):
            out.append(os.path.join(path, name))
    return out
