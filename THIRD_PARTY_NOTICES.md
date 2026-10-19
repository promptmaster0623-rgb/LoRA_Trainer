# 서드파티 고지

이 프로젝트는 아래 오픈소스 코드를 이식·수정하여 포함합니다.

| 포함 위치 | 원본 | 라이선스 | 변경 내용 |
|---|---|---|---|
| `lora_trainer/models/krea2/mmdit.py` | [krea-ai/krea-2](https://github.com/krea-ai/krea-2) `mmdit.py` (Copyright Krea AI, Inc.) | Apache License 2.0 (`LICENSES/Apache-2.0_krea-2.txt`) | torch.compile/CUDNN 강제 제거, key-padding 마스크, 256 패딩 제거, 그래디언트 체크포인팅 |
| `lora_trainer/models/krea2/adapter.py` 의 프롬프트 템플릿·샘플링 규칙 | krea-ai/krea-2 `encoder.py`, `sampling.py` | Apache License 2.0 | 프롬프트별 자연 길이 인코딩, 학습용 시간이동 |
| `lora_trainer/models/moss_image/transformer.py` | [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) MOSS-Image 포팅 (Copyright (c) 2024 Ostris, LLC) — 원 모델 [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) | MIT (`LICENSES/MIT_ai-toolkit.txt`) | 참조이미지/KV 캐시 제거, 재진입 체크포인팅 |
| `lora_trainer/models/moss_image/vae_flux2.py` | ostris/ai-toolkit `flux2_kl.py` | MIT | 로더 함수 추가, 믹스인 제거 |
| `lora_trainer/models/krea2/vae_loader.py` 의 ComfyUI→diffusers VAE 키 변환표 | [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) `qwen_image_utils.convert_comfyui_state_dict` ([Fizgig](https://github.com/shootthesound/Fizgig) `src/fizgig/krea2/vae_loader.py` 경유) | Apache License 2.0 | 표만 사용, 로더는 새로 작성 |
| `lora_trainer/models/moss_image/adapter.py` 의 학습 sigma 분포 · 깨끗한 쪽 비율 계산 | Fizgig `src/fizgig/moss_image/driver.py` (`_sample_t`, `mu_for_clean_share`) | Apache License 2.0 | 배치 단위 벡터화 |
| `lora_trainer/train/automagic3.py` | ostris/ai-toolkit `toolkit/optimizers/automagic3.py` (Copyright (c) 2024 Ostris, LLC) | MIT (`LICENSES/MIT_ai-toolkit.txt`) | 변경 없음 (머리말만 추가) |

설계 참고: [Fizgig](https://github.com/shootthesound/Fizgig) (Apache-2.0) — 저VRAM 전략, kohya 형식 저장, UX, Krea 2 · MOSS-Image 프리셋 값, EMA 기본값.

모델 가중치는 이 저장소에 포함되지 않으며 각 모델의 라이선스를 따릅니다
(Krea 2: Krea 2 Community License, MOSS-Image / Qwen: 각 저장소 라이선스).
