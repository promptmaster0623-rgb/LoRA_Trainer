@echo off
chcp 65001 > nul
title LoRA 점검
cd /d "%~dp0"
set "PY=python"
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
"%PY%" tools\LoRA점검.py %*
pause
