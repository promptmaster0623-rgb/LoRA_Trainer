@echo off
chcp 65001 > nul
REM 배포판 업데이트: 깃허브 배포 저장소의 최신 버전을 받아 바뀐 파일만 교체합니다 (모델·결과물·설정은 유지)
title LoRA Trainer 업데이트
cd /d "%~dp0"
set "PY=python"
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
"%PY%" tools\update.py %*
echo.
pause
