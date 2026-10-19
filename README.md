# LoRA Trainer

ComfyUI 에서 바로 쓸 수 있는 LoRA 를 학습하는 **파이썬 데스크톱 툴**입니다.
**Krea 2** 와 **MOSS-Image** 를 기본 지원하며, **VRAM 12GB 이하 GPU** 에서도 학습할 수 있게 설계했습니다.

| | Krea 2 | MOSS-Image |
|---|---|---|
| DiT | 12.9B 단일 스트림 (28블록) | 5B 단일 스트림 (42블록) |
| 텍스트 인코더 | Qwen3-VL-4B-Instruct (12개 레이어 스택) | Qwen3.5-9B (최종 hidden state) |
| VAE | Qwen-Image VAE (16ch) | FLUX.2 VAE (128ch 패킹) |
| 권장 학습 대상 | **Raw** → Turbo 에서 사용 | **Base** → Base/SFT 에서 사용 |
| 12GB 기본 설정 | DiT NF4 · 텍스트 인코더 bf16 | DiT FP8 · 텍스트 인코더 NF4 |

## 주요 기능

- **저VRAM 파이프라인**: 텍스트 임베딩·latent 를 먼저 디스크에 캐시 → 학습 중에는 DiT 만 GPU 에 상주
- **양자화 베이스 + 고정밀 LoRA**: FP8(행별 스케일) / NF4(bitsandbytes) 베이스 위에서 fp32 LoRA 학습
- **블록 스왑**: bf16/FP8 베이스의 일부 블록을 CPU 메모리에 두고 필요할 때만 GPU 로 (그래디언트 체크포인팅 연동)
- **스트리밍 로더**: 26GB 체크포인트도 RAM 에 통째로 올리지 않고 텐서 단위로 로드·양자화
- **ComfyUI 호환 저장**: kohya(`lora_unet_*`) 또는 ComfyUI(`diffusion_model.*`) 형식, 저장 시 loras 폴더 자동 복사
- **데이터셋 도구**: 썸네일 브라우저, 캡션 편집기(자동 저장), 일괄 편집, Qwen3-VL 자동 캡션, **MOSS-Image 공식 형식 캡션 (Ollama Gemma 4)**, 종횡비 버킷, 멀티 해상도
- **실시간 모니터**: loss 그래프, 속도/ETA/VRAM, 학습 중 샘플 갤러리, 일시정지·지금 샘플·지금 저장·중지
- **이어학습**: 옵티마이저 상태 포함 저장/재개, 기존 LoRA 에서 시작
- **프리셋**: 모델 × 목적(캐릭터/스타일) × VRAM 별 추천 설정
- **시스템 맞춤 추천**: VRAM·RAM·CPU·가상메모리·데이터셋을 분석해 퀄리티/최적/속도 3가지 조합과 예상 속도·총 시간 제시 (실측으로 자동 보정)
- **대기열**: 같은 설정으로 LoRA 이름·데이터셋만 바꿔 여러 개를 예약하고 순서대로 자동 학습
- **실시간 시스템 상태**: 사이드바에 CPU / RAM / GPU / VRAM 사용량 표시
- **모델별 파일 관리**: Krea 2 / MOSS-Image 전환 시 각 모델용 경로 자동 교체, 파일 형식 확인·자동 찾기

## 설치 (Windows)

1. Python 3.10~3.13, NVIDIA 드라이버(CUDA 12.8 지원) 준비
2. `install.bat` 실행 → `venv` 생성, PyTorch(cu128) 및 패키지 설치
3. `run.bat` 실행

리눅스는 `install.sh` / `run.sh` 를 사용합니다.

### 업데이트

`업데이트.bat` 실행 → 최신 버전과 바뀐 점을 보여주고, 바뀐 파일만 교체합니다.
`venv`·`models`·`output`·`configs`·`user_data` 와 데이터셋은 건드리지 않으며, 직접 고친 파일은 `_backup/` 에 먼저 보관합니다.
버전별 변경 내용은 [CHANGELOG.md](CHANGELOG.md), 깃허브 [Releases](https://github.com/promptmaster0623-rgb/LoRA_Trainer/releases) 에서 볼 수 있습니다.

## 빠른 시작

1. **설정** 페이지에서 허깅페이스 토큰 입력 (Krea 2 Raw 는 게이트 모델: 허깅페이스에서 라이선스 동의 필요)
2. **시작하기** → LoRA 이름, 베이스 모델, 프리셋 선택
3. **데이터셋** → 이미지 폴더 추가, 캡션 확인(없으면 *자동 캡션 생성*), 트리거 단어 입력
4. **모델** → 비워두면 기본 모델 자동 다운로드. ComfyUI 에 있는 파일 경로를 지정해도 됨
5. **실행 · 모니터** → 학습 시작

결과물: `output/<LoRA 이름>/<LoRA 이름>.safetensors` (중간 저장본 `_000250` 등, 샘플은 `samples/`)

### ComfyUI 에서 사용

- `LoraLoaderModelOnly` (또는 `LoraLoader`) 노드로 불러옵니다.
- Krea 2: Raw 로 학습한 LoRA 를 **Turbo** 체크포인트(8스텝, CFG 1)에 적용하는 것이 공식 권장 방식입니다.
- 트리거 단어를 프롬프트에 넣으세요.

### CLI (GUI 없이)

```bat
train_cli.bat configs\my_lora.yaml
:: 또는
venv\Scripts\python.exe -m lora_trainer.train.runner --config configs\my_lora.yaml --steps 3000
```

## VRAM 가이드

| VRAM | Krea 2 | MOSS-Image |
|---|---|---|
| 8~11GB | DiT NF4 + 텍스트 인코더 NF4, 해상도 768 권장 | DiT NF4 + 텍스트 인코더 NF4 |
| 12~16GB | DiT NF4, 1024 학습 가능 | DiT FP8 + 텍스트 인코더 NF4 |
| 24GB | DiT FP8 | bf16 전체 |
| 40GB+ | bf16 전체 | bf16 전체 |

- 메모리가 부족하면: 해상도 낮추기(768) → 배치 1 → 그래디언트 체크포인팅 확인 → 샘플 크기/스텝 줄이기
- 블록 스왑은 NF4 와 같이 쓸 수 없습니다 (bf16/FP8 품질을 원할 때 사용, 속도 감소)

## 폴더 구조

자세한 설명은 [docs/구조.md](docs/구조.md), 사용법은 [docs/사용가이드.md](docs/사용가이드.md) 를 참고하세요.

```
LoRA_Trainer/
├─ main.py              GUI 실행 (또는 --config 로 CLI 학습)
├─ lora_trainer/        본체 패키지 (core·data·models·network·quant·train·gui·utils)
├─ presets/             추천 설정 YAML
├─ tools/               업데이트 · LoRA 점검 도구 (업데이트.bat, LoRA점검.bat)
└─ docs/                한글 문서
```


## 참고 및 출처

- Krea 2 공식 코드: https://github.com/krea-ai/krea-2 (Apache-2.0) — DiT 구조 이식
- AI-Toolkit (Ostris): https://github.com/ostris/ai-toolkit (MIT) — MOSS-Image DiT/FLUX.2 VAE 포팅본 참고·이식
- Fizgig: https://github.com/shootthesound/Fizgig (Apache-2.0) — 저VRAM 전략·UX 참고
- MOSS-Image: https://github.com/OpenMOSS/MOSS-Image

이식한 코드의 라이선스 고지는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 에 있습니다.
모델 가중치는 각 모델의 라이선스(Krea 2 Community License 등)를 따릅니다.
