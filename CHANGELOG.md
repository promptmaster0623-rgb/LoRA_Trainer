# 변경 이력

형식: [유의적 버전](https://semver.org/lang/ko/) · 최신이 위

## [0.3.1] - 2026-10-18
### 수정
- **배포판 실행 오류 수정**: `run.bat` 실행 시 `ModuleNotFoundError: No module named 'lora_trainer.models'`
  - 원인: `.gitignore` 의 `models/` 가 루트의 모델 저장 폴더뿐 아니라 소스 폴더 `lora_trainer/models/` (Krea 2 · MOSS-Image 어댑터)까지 제외해 깃허브에 올라가지 않았음
  - `.gitignore` 의 로컬 폴더 항목을 루트 전용(`/models/`, `/output/`, `/configs/`, `/user_data/` 등)으로 변경
### 추가
- 배포 도구: `tools/배포저장소에올리기.bat` (배포 저장소에 사용자용 파일만 커밋·푸시, 실행 시 토큰 입력)
  - 본체 소스(`lora_trainer/**/*.py`)가 하나라도 빠지면 배포를 중단하는 검사 추가

## [0.3.0] - 2026-10-18
### 추가
- **MOSS-Image 공식 가이드 형식 자동 캡션 (Ollama · Gemma 4)**
  - 공식 USERGUIDE 의 리캡션 규칙으로 시스템 프롬프트 작성: `caption.long/medium/short` + `watermark(box_2d 0~1000, yxyx)` + `borders`
  - 공식 권장 요청값 기본 적용: thinking 켬, temperature 1.0, top_p 0.95, 최대 16384 토큰, JSON 스키마 구조화 출력
  - 캡션 없는 이미지는 새로 생성, 형식이 다른 캡션(태그·짧은 문장 등)은 기존 캡션을 reference 로 넘겨 공식 형식으로 다시 생성
  - `.txt` 안에 공식 JSON 이 들어 있으면 모델 없이 바로 변환, 직접 고친 캡션은 보존
  - 원본은 `<이미지>.moss.json`, 학습용은 `.txt`(long/medium/short 선택), 덮어쓰기 전 `_caption_backup/` 백업
  - 설정한 모델이 없으면 설치된 Gemma 4 로 자동 대체, vision/thinking 지원 자동 확인, JSON 오류 시 재시도
  - 끝나면 Ollama 모델을 VRAM 에서 내려 학습 VRAM 확보
- 데이터셋: 썸네일에 MOSS 캡션 상태 표시(✓ 공식 · ⚠ 없음 · ◇ 다른 형식 · ✎ 직접 수정), 편집기에서 이미지별 long/medium/short 전환
- 설정: **MOSS-Image 캡션 · Ollama (Gemma 4) 로컬 연결** — 주소, 모델 선택(설치 목록 불러오기), 연결 테스트, thinking·샘플링·대기시간, 학습용 변형, VRAM 해제
- 테스트: `tests/test_moss_caption.py` (가짜 Ollama 서버로 형식 판별·요청 내용·전체 흐름 검증)

## [0.2.0] - 2026-10-17
### 추가
- 시작하기: **시스템 맞춤 추천 설정** — VRAM·RAM·CPU·가상메모리와 데이터셋을 분석해 퀄리티 중시 / 최적 / 속도 중시 3가지 조합 제시
  - 조합별 예상 속도(s/it · it/s), 총 시간(학습·준비·샘플), 예상 VRAM 표시, 클릭 한 번으로 적용
  - 학습이 끝나면 실측 속도로 추정치를 자동 보정 (`user_data/speed_calibration.json`)
- **대기열**: 현재 설정 그대로 LoRA 이름·데이터셋 폴더만 바꿔 예약 → 순서대로 자동 학습
  - 순서 변경·다시 대기·삭제, 이름 중복 자동 회피, "현재 항목 후 정지", "실패해도 계속", 재시작해도 유지
- 사이드바 하단 **실시간 시스템 상태** (CPU / RAM / GPU 사용률·온도 / VRAM, 임계값 색상 경고)
### 변경
- 모델 페이지: 맨 위 **베이스 모델 전환(Krea 2 | MOSS-Image)** 추가, 선택한 모델에 필요한 파일 칸만 표시
  - 모델별 경로를 따로 기억 (전환해도 각자 경로 유지), 사이드바·메뉴에 현재 모델 표시
  - 경로 상태 표시(확인됨 / 자동 다운로드 / 없음 / 다른 모델용 파일 경고), 폴더에서 자동 찾기
  - 학습 시작 전 잘못된 모델 파일 경고
- 프리셋 적용 시 모델이 바뀌면 그 모델용으로 기억해 둔 경로를 사용
- 의존성: `psutil`, `nvidia-ml-py` 추가

## [0.1.0] - 2026-10-17
### 추가
- 최초 버전: Krea 2 / MOSS-Image LoRA 학습 엔진
  - Krea 2 DiT (공식 코드 이식), Qwen3-VL-4B 12층 스택 텍스트 인코딩, Qwen-Image VAE
  - MOSS-Image DiT (ComfyUI 레이아웃), Qwen3.5-9B 텍스트 인코딩, FLUX.2 VAE
- 저VRAM: 텍스트/latent 디스크 캐시, FP8·NF4 베이스 양자화, 블록 스왑, 스트리밍 로더, 그래디언트 체크포인팅
- LoRA: 대상 프리셋/정규식, rsLoRA, 드롭아웃, kohya·ComfyUI 형식 저장, 이어학습
- 학습: AdamW/AdamW8bit/Prodigy/Adafactor, 스케줄러, 로짓-정규 타임스텝 + 모델별 시간이동, 버킷·멀티 해상도
- PySide6 데스크톱 GUI: 시작·데이터셋(캡션 편집/일괄 편집/자동 캡션)·모델·학습 설정·샘플·실행 모니터·설정
- 프리셋 6종, CLI 실행, CPU 테스트 24개
- 깃허브 도구: `깃허브에올리기.bat`(한글 커밋 메시지·업데이트기록 자동 작성, 토큰 유출 검사, 버전 태그) / `깃허브에서다운받기.bat`
