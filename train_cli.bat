@echo off
chcp 65001 > nul
REM GUI 없이 학습: train_cli.bat configs\my_lora.yaml
cd /d "%~dp0"
if "%~1"=="" (echo 사용법: train_cli.bat 설정파일.yaml & pause & exit /b 1)
venv\Scripts\python.exe -m lora_trainer.train.runner --config "%~1"
pause
