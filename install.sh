#!/usr/bin/env bash
# 리눅스 설치 스크립트
set -e
cd "$(dirname "$0")"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
echo "설치 완료: ./run.sh"
