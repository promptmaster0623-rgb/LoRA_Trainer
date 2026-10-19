"""배포판 업데이트 — 배포 저장소(promptmaster0623-rgb/LoRA_Trainer)의 최신 버전으로 바꾸기 (표준 라이브러리만 사용).

  업데이트.bat               최신 버전 확인 → 바뀐 내용 보여주기 → 백업 후 업데이트
  python tools/update.py --check   확인만 (아무것도 바꾸지 않음)
  python tools/update.py --yes     질문 없이 진행

- 토큰 필요 없음 (공개 저장소)
- git 이 있고 이 폴더가 배포 저장소에서 받은 것이면 git 으로, 아니면 zip 으로 받습니다.
- 절대 건드리지 않는 것: venv, models, output, configs, user_data, _backup, 데이터셋 등 (배포 파일이 아닌 것 전부)
- 직접 고친 배포 파일은 덮어쓰기 전에 _backup/업데이트_<시각>/ 에 보관합니다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OWNER, REPO, BRANCH = "promptmaster0623-rgb", "LoRA_Trainer", "main"
REPO_URL = f"https://github.com/{OWNER}/{REPO}.git"
RAW = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}"
ZIP_URLS = (f"https://codeload.github.com/{OWNER}/{REPO}/zip/refs/heads/{BRANCH}",
            f"https://github.com/{OWNER}/{REPO}/archive/refs/heads/{BRANCH}.zip")
MANIFEST = os.path.join(ROOT, "user_data", "update_manifest.json")   # 지난 업데이트로 깔린 파일 목록
# 사용자 데이터 — 저장소에 같은 이름이 있어도 절대 덮어쓰지 않음
PROTECT = ("venv/", ".venv/", "models/", "output/", "configs/", "user_data/", "_backup/", "00_DOCUMENT/", ".git/")


def say(msg: str = "") -> None:
    print(msg, flush=True)


def ask(q: str, yes: bool, default: bool = True) -> bool:
    if yes:
        return default
    try:
        a = input(f"{q} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    except EOFError:
        return default
    return default if not a else a in ("y", "yes", "ㅛ", "예", "네")


def ver_of(text: str) -> str:
    m = re.search(r'__version__\s*=\s*"([^"]+)"', text or "")
    return m.group(1) if m else ""


def local_version() -> str:
    p = os.path.join(ROOT, "lora_trainer", "__init__.py")
    return ver_of(open(p, encoding="utf-8").read()) if os.path.isfile(p) else ""


def vkey(v: str):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3]) or (0,)


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "LoRA-Trainer-update"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def changelog_between(text: str, old: str, new: str) -> str:
    out = []
    for m in re.finditer(r"^## \[([^\]]+)\][^\n]*\n(.*?)(?=^## \[|\Z)", text, re.S | re.M):
        v = m.group(1)
        if vkey(old) < vkey(v) <= vkey(new):
            out.append(f"■ v{v}\n{m.group(2).strip()}")
    return "\n\n".join(out)


def is_dev_folder() -> bool:
    """개발 폴더에서 실수로 실행하는 것 방지 (개발 도구가 있거나 원격이 개발 저장소)."""
    if os.path.isfile(os.path.join(ROOT, "tools", "github.py")):
        return True
    if os.path.isdir(os.path.join(ROOT, ".git")) and shutil.which("git"):
        r = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True)
        url = (r.stdout or "").strip().lower()
        if url and f"{OWNER}/{REPO}".lower() not in url:
            return True
    return False


def protected(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    return any(rel.startswith(p) for p in PROTECT)


def backup(paths, tag: str) -> str:
    paths = [p for p in paths if os.path.isfile(os.path.join(ROOT, p))]
    if not paths:
        return ""
    dst = os.path.join(ROOT, "_backup", f"업데이트_{tag}_{dt.datetime.now():%Y%m%d_%H%M%S}")
    for p in paths:
        os.makedirs(os.path.dirname(os.path.join(dst, p)), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, p), os.path.join(dst, p))
    return dst


# ---------------------------------------------------------------------------
# git 방식
# ---------------------------------------------------------------------------
def git(*args, check=True):
    r = subprocess.run(["git", "-c", "core.quotepath=false", *args], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
    if check and r.returncode != 0:
        raise SystemExit("[실패] git " + " ".join(args[:2]) + "\n" + (r.stderr or r.stdout))
    return (r.stdout or "").strip()


def update_git(yes: bool, check_only: bool, new_ver: str) -> bool:
    git("fetch", "origin", BRANCH, "--tags")
    head, remote = git("rev-parse", "HEAD"), git("rev-parse", f"origin/{BRANCH}")
    if head == remote:
        return False
    say("\n· 받을 내용:\n" + git("log", "--format=  %h  %ad  %s", "--date=short", f"HEAD..origin/{BRANCH}"))
    if check_only:
        return True
    if not ask(f"\nv{new_ver} 로 업데이트할까요?", yes):
        raise SystemExit("취소했습니다.")
    dirty = [l[3:].strip('"') for l in git("status", "--porcelain", "--untracked-files=no").splitlines() if l.strip()]
    if dirty:
        b = backup(dirty, f"v{local_version()}")
        say(f"· 직접 고친 파일 {len(dirty)}개를 백업했습니다: {b}")
    old = head
    git("reset", "-q", "--hard", f"origin/{BRANCH}")
    changed = git("diff", "--name-only", old, "HEAD").splitlines()
    report(changed)
    return True


# ---------------------------------------------------------------------------
# zip 방식 (git 없이 압축 파일로 설치한 경우)
# ---------------------------------------------------------------------------
def update_zip(yes: bool, check_only: bool, new_ver: str) -> bool:
    say("· 최신 배포 파일 받는 중 (zip)...")
    data, err = None, None
    for url in ZIP_URLS:
        try:
            data = fetch(url, timeout=180)
            break
        except Exception as e:  # 다음 주소로 재시도
            err = e
    if data is None:
        raise RuntimeError(f"zip 다운로드 실패: {err}")
    z = zipfile.ZipFile(io.BytesIO(data))
    prefix = z.namelist()[0].split("/")[0] + "/"
    files = {}
    for n in z.namelist():
        if n.endswith("/") or not n.startswith(prefix):
            continue
        rel = n[len(prefix):]
        if rel and not protected(rel):
            files[rel] = n
    changed, added = [], []
    for rel, n in files.items():
        dst = os.path.join(ROOT, rel)
        if not os.path.isfile(dst):
            added.append(rel)
        elif open(dst, "rb").read() != z.read(n):
            changed.append(rel)
    old_list = []
    if os.path.isfile(MANIFEST):
        try:
            old_list = json.load(open(MANIFEST, encoding="utf-8")).get("files", [])
        except Exception:
            old_list = []
    removed = [p for p in old_list if p not in files and os.path.isfile(os.path.join(ROOT, p)) and not protected(p)]
    if not (changed or added or removed):
        return False
    say(f"\n· 바뀌는 파일: 추가 {len(added)} · 수정 {len(changed)} · 삭제 {len(removed)}")
    if check_only:
        return True
    if not ask(f"\nv{new_ver} 로 업데이트할까요? (바뀌는 파일은 먼저 백업합니다)", yes):
        raise SystemExit("취소했습니다.")
    b = backup(changed + removed, f"v{local_version() or '이전'}")
    if b:
        say(f"· 기존 파일 백업: {b}")
    for rel in added + changed:
        dst = os.path.join(ROOT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(z.read(files[rel]))
    for rel in removed:
        os.remove(os.path.join(ROOT, rel))
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    json.dump({"version": new_ver, "date": f"{dt.datetime.now():%Y-%m-%d %H:%M}", "files": sorted(files)},
              open(MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    report(added + changed + removed)
    return True


def report(changed) -> None:
    say(f"\n· 업데이트한 파일 {len(changed)}개")
    for p in changed[:40]:
        say("    " + p)
    if "requirements.txt" in changed:
        say("\n[알림] 필요한 패키지가 바뀌었습니다. install.bat 을 한 번 실행하세요.")


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="LoRA Trainer 업데이트")
    ap.add_argument("--check", action="store_true", help="확인만")
    ap.add_argument("--yes", action="store_true", help="질문 없이 진행")
    ap.add_argument("--zip", action="store_true", help="git 이 있어도 zip 방식 사용")
    a = ap.parse_args(argv)
    say("=" * 60)
    say("  LoRA Trainer 업데이트")
    say("=" * 60)
    if is_dev_folder():
        say("[중단] 개발 폴더입니다. 개발 폴더는 tools\\깃허브에서다운받기.bat 를 사용하세요.")
        return 1
    cur = local_version()
    try:
        new = ver_of(fetch(f"{RAW}/lora_trainer/__init__.py", 30).decode("utf-8"))
        log_text = fetch(f"{RAW}/CHANGELOG.md", 30).decode("utf-8")
    except Exception as e:
        say(f"[오류] 깃허브에 연결하지 못했습니다: {e}")
        return 1
    say(f"· 설치된 버전 : v{cur or '?'}")
    say(f"· 최신 버전   : v{new or '?'}")
    if cur and new and vkey(new) > vkey(cur):
        notes = changelog_between(log_text, cur, new)
        if notes:
            say("\n[새 버전에서 바뀐 점]\n" + notes)
    use_git = (not a.zip and shutil.which("git") and os.path.isdir(os.path.join(ROOT, ".git")))
    try:
        did = update_git(a.yes, a.check, new) if use_git else update_zip(a.yes, a.check, new)
    except SystemExit:
        raise
    except Exception as e:
        say(f"[오류] 업데이트 실패: {e}")
        return 1
    if not did:
        say("\n· 이미 최신 버전입니다.")
    elif a.check:
        say("\n(확인만 했습니다. 업데이트하려면 업데이트.bat 를 실행하세요)")
    else:
        say(f"\n완료! 현재 버전 v{local_version()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
