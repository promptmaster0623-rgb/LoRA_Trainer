# presets — 추천 설정

| 파일 | 모델 | 용도 | 대상 VRAM |
|---|---|---|---|
| `krea2_12gb_character.yaml` | Krea 2 | 캐릭터/인물 (rank 32, 모든 Linear 264개, 1e-4, EMA 0.98, 768) | 12GB |
| `krea2_12gb_style.yaml` | Krea 2 | 화풍/스타일 (rank 16, 모든 Linear, 1e-4, EMA 0.98, 768+1024) | 12GB |
| `krea2_24gb_quality.yaml` | Krea 2 | 고품질 (rank 32, 모든 Linear, EMA 0.98, 1024) | 24GB |
| `moss_12gb_character.yaml` | MOSS-Image | 캐릭터/인물 (rank 32, 1e-4, EMA 0.98, 깨끗한 쪽 60%, 1024) | 12GB |
| `moss_12gb_style.yaml` | MOSS-Image | 스타일 (rank 16, 5e-5, EMA 끔, 깨끗한 쪽 60%, 1024) | 12GB |
| `moss_24gb_quality.yaml` | MOSS-Image | 고품질 (bf16 전체, EMA 0.98, 깨끗한 쪽 60%) | 24GB |

값은 Fizgig v8.7.0 의 Krea 2 / MOSS-Image 프리셋을 기준으로 합니다.

GUI 의 *시작하기 → 프리셋 적용* 은 이름·데이터셋·모델 경로를 유지한 채 나머지 값만 바꿉니다.
양자화 항목이 `auto` 인 프리셋은 실제 GPU VRAM 에 맞춰 자동 결정됩니다.
