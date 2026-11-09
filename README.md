# lecture_cut — 강의 녹화본 자동 컷 편집

강의 녹화본에서 **더듬는 말, 숨소리, 긴 침묵, 멈춘 화면**을 잘라냅니다.
말과 말 사이를 바짝 붙이지 않고, **문장 끝은 약 0.65초 / 문장 중간은 약 0.35초**의 쉼을 남겨 자연스러운 흐름을 유지합니다.

## 설치

```bash
pip install -r requirements.txt
```

ffmpeg 가 필요합니다 (macOS: `brew install ffmpeg`, Windows: `winget install ffmpeg`).

## 사용법

```bash
python lecture_cut.py 강의.mp4
```

→ `강의_cut.mp4` 와 컷 내역 `강의_cut_report.json` 이 생성됩니다.

| 상황 | 명령 |
|---|---|
| 먼저 어디가 잘릴지만 확인 | `python lecture_cut.py 강의.mp4 --dry-run` |
| 판서·코딩 시연처럼 **말 없이 화면이 움직이는 구간은 남기기** | `python lecture_cut.py 강의.mp4 --keep-silent-motion` |
| 쉼을 더 여유 있게 | `python lecture_cut.py 강의.mp4 --pause 0.5 --sentence-pause 0.9` |
| 더 타이트하게 | `python lecture_cut.py 강의.mp4 --pause 0.25 --sentence-pause 0.45` |
| "그/저/아/뭐/이제"는 건드리지 않기 | `--no-soft-fillers` |
| GPU 사용 · 인식 정확도 향상 | `--device cuda --compute-type float16 --model large-v3` |
| 음성 인식 없이 침묵·숨소리만 정리 | `--no-asr` |
| 한국어 강의 (기본은 영어) | `--language ko` |

음성 인식 결과는 `강의.words.json` 에 캐시되므로, 쉼 길이 등을 바꿔 다시 돌릴 때는 인식을 다시 하지 않아 빠릅니다.

## 동작 방식

1. **음성 인식** (faster-whisper, 단어 단위 시간) — Whisper 가 "어/음"을 생략하지 않도록 필러가 섞인 프롬프트를 줍니다.
2. **더듬기 제거**
   - 항상 제거: 영어 `uh, um, uhm, er, erm, hmm, mm` (늘어진 `ummm`, `uhhh` 포함) / 한국어 `어, 음, 으, 엄, 흠`
   - 조건부 제거 (앞뒤에 쉼이 있어 단독으로 튀어나올 때만): `ah, oh, like, so, well, you know, I mean` / `아, 그, 저, 뭐, 이제`
   - 말하다 끊긴 단어: `wh- what`, `prog- program`
   - 반복: `we we will`, `the the`, `prog program`, `this is this is important` → 앞쪽 제거 (0.6초 이내만).
     `in into`, `the theory` 처럼 정상 단어는 건드리지 않습니다.
3. **자연스러운 쉼** — 긴 침묵을 0으로 만들지 않고 목표 길이로 줄입니다. 말 끝 여운을 조금 더 길게, 다음 말 직전 여유(0.15초)는 짧게 남겨 숨 들이쉬는 소리가 들어가지 않게 합니다.
4. **숨소리 회피** — 남겨야 할 쉼 자리에 숨소리가 있으면, 같은 틈에서 가장 조용한 '방 소리' 구간으로 대신 채웁니다. 쉼 길이는 그대로, 숨소리만 빠집니다.
5. **화면 멈춤** (`--keep-silent-motion`) — ffmpeg freezedetect 로 화면 변화를 분석합니다.

   | | 말 있음 | 말 없음 |
   |---|---|---|
   | 화면 멈춤 | 유지 (슬라이드 설명) | **잘라냄** |
   | 화면 움직임 | 유지 | **유지** (판서·코딩·시연) |

   옵션을 끄면(기본) 화면과 무관하게 말 없는 구간은 모두 쉼 길이로 줄어듭니다. 웹캠 얼굴이 나오는 강의는 화면이 항상 '움직이므로' 기본값을 쓰세요.
6. **렌더링** — 프레임 경계에 맞춰 한 번에 인코딩하므로 컷이 수백 개여도 음성/영상 싱크가 어긋나지 않습니다.

## 조정 팁

- 말 첫소리가 잘리면 `--lead-in 0.25`
- 작은 목소리가 침묵으로 취급되면 `--speech-db 10` (기본 14)
- 화면 멈춤이 잘 안 잡히면 `--freeze-noise -45` (높을수록 관대)
- 필러를 너무 많이/적게 잡으면 리포트의 `removed_words` 를 확인하고 `--no-soft-fillers` 또는 `--soft-gap` 조정
