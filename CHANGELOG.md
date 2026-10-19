# 변경 이력

형식: [유의적 버전](https://semver.org/lang/ko/) · 최신이 위

## [0.5.0] - 2026-10-19
### 추가
- **Krea 2 빠른 미리보기 (Turbo LoRA)**: 공식 `krea2_turbo_lora_rank_64_bf16` 을 샘플 생성 동안만 덧붙여 10스텝 · CFG 1 · mu 1.15 로 생성 (Fizgig 기본 0.7). 저장되는 LoRA 에는 섞이지 않음. Krea 2 프리셋은 기본 켬
- **res_multistep 샘플러** (ComfyUI 이식, 2차 다단계): MOSS-Image 미리보기 기본값 — ComfyUI MOSS 템플릿(res_multistep · simple · shift 5 · 30스텝 · CFG 4 · SLG 22)과 같은 결과
- 샘플 페이지: 샘플러 선택, Turbo LoRA 켜기·경로·강도·스텝
- **배포판 업데이트**: 루트 `업데이트.bat` (+ `tools/update.py`) — 배포 저장소 최신 버전 확인 · 바뀐 점 표시 · 바뀐 파일만 교체 (모델·결과물·설정 유지, 고친 파일 백업). git/zip 설치 모두 지원
- 버전 관리: `tools/버전올리기.bat` (patch/minor/major + CHANGELOG 새 항목), `tools/배포저장소에서다운받기.bat` (배포 상태 확인·비교)
- 깃허브 **Releases 페이지 릴리스 노트 자동 작성** (개발 · 배포 저장소 모두, CHANGELOG 해당 버전 내용)
- 배포 저장소 `docs/배포기록.md` 자동 누적 (버전 · 원본 커밋 · 바뀐 파일)
- 올리기 직전 소스 백업 `_backup/업로드/v버전_dev_시각.zip` (최근 20개)
### 변경
- `깃허브에올리기.bat`: 새 버전이면 "배포 저장소에도 올릴까요?" 로 개발 → 배포를 이어서 진행 (`--release`)
- `배포저장소에올리기.bat`: 토큰을 `00_DOCUMENT/release_git_token.txt` 에서 읽음 (없으면 입력)
- MOSS-Image 프리셋: 샘플 30스텝 + 공식 네거티브 (`worst quality, bad anatomy, watermark, jpeg artifact`)

## [0.4.0] - 2026-10-19
### 수정
- **Krea 2 VAE 로딩 오류 수정**: `FromOriginalModelMixin is currently only compatible with ...`
  - 원인: VAE 경로를 파일(`.../Qwen-Image/vae/diffusion_pytorch_model.safetensors`)로 지정하면 diffusers `from_single_file` 을 호출했는데, Qwen-Image VAE 는 이 기능을 지원하지 않음
  - 새 로더 `models/krea2/vae_loader.py`: 설정으로 VAE 를 직접 만들고 가중치를 넣음. 폴더 · diffusers 단일 파일 · ComfyUI `qwen_image_vae.safetensors`(Wan 형식 키, 자동 변환) 모두 지원
### 변경 (Fizgig v8.7.0 기준 정렬)
- **Krea 2 LoRA 대상 = DiT 의 모든 Linear 264개** (28 블록 + 텍스트 융합 4블록·프로젝터 + 입력/타임스텝/출력층). Krea 2 기본값 `all_linear`
- Krea 2 양자화는 28개 블록의 Linear 만 (텍스트 융합부는 bf16 유지, Fizgig 와 동일)
- Krea 2 텍스트 토큰 길이를 공식/Fizgig 와 같은 512 로 맞춤 (접미 5토큰 포함)
- **EMA 평균 가중치 (기본 0.98)**: 학습은 원래 가중치, 체크포인트 저장 · 샘플은 최근 스텝 평균 가중치. 이어학습 상태에도 저장. 학습 설정에 `EMA 평균` 항목
- **MOSS-Image 학습 타임스텝**: 자동 모드에서 공식 사전학습 분포(잘린 로짓-정규, 1000 격자) + **깨끗한 쪽 비율 60%** (Fizgig v8.7.0 Clean-end share)
  - 이전 기본값(로짓-정규 + shift 3)은 sigma<0.5 스텝이 약 13% 뿐이라 얼굴·디테일 학습이 매우 느렸음 → MOSS LoRA 가 잘 학습되지 않던 원인 후보
  - 학습 설정에 `깨끗한 쪽 비율` 항목 (0 = 모델 자체 스케줄 약 11%, 구도가 사진에 붙으면 30 권장)
- MOSS-Image 샘플: CFG 3 → 4, 네거티브 22번 블록 스킵은 앞 절반 스텝만 (ComfyUI MOSS 템플릿 SLG 와 동일)
- 프리셋 · 추천 설정: Krea 2 캐릭터 rank 32 · 1e-4 · 모든 Linear, 스타일 rank 16. MOSS 캐릭터 rank 32 · 1e-4, 스타일 rank 16 · 5e-5 · EMA 끔
### 추가
- 테스트 `tests/test_v040_fizgig.py` (단일 파일/ComfyUI VAE 로딩, 264 Linear, 깨끗한 쪽 비율, EMA), E2E 에 VAE 단일 파일 경로 회귀 테스트

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
