# presets — 추천 설정

| 파일 | 모델 | 용도 | 대상 VRAM |
|---|---|---|---|
| `krea2_12gb_character.yaml` | Krea 2 | 캐릭터/인물 (rank 16, 1e-4, 2000스텝, 1024) | 12GB |
| `krea2_12gb_style.yaml` | Krea 2 | 화풍/스타일 (rank 32, 1.5e-4, 3000스텝, 768+1024) | 12GB |
| `krea2_24gb_quality.yaml` | Krea 2 | 고품질 (FP8 DiT, rank 32) | 24GB |
| `moss_12gb_character.yaml` | MOSS-Image | 캐릭터/인물 | 12GB |
| `moss_12gb_style.yaml` | MOSS-Image | 스타일/포스터 | 12GB |
| `moss_24gb_quality.yaml` | MOSS-Image | 고품질 (bf16 전체) | 24GB |

GUI 의 *시작하기 → 프리셋 적용* 은 이름·데이터셋·모델 경로를 유지한 채 나머지 값만 바꿉니다.
양자화 항목이 `auto` 인 프리셋은 실제 GPU VRAM 에 맞춰 자동 결정됩니다.
