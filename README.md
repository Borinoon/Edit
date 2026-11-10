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

## 프리미어 프로로 편집하기 (원본 화질 그대로)

원본은 업로드하지 않고, **저화질 사본으로 분석 → 원본 컷 정보(XML)만 프리미어로 가져오는** 방식입니다.

### 1. 분석용 저화질 사본 만들기 (원본 PC에서)

시간축만 같으면 되므로 해상도·fps를 크게 낮춰도 됩니다. 1시간 강의 기준 수십 MB 정도로 줄어듭니다.

```bash
ffmpeg -i lecture01.mp4 -vf "scale=-2:360,fps=5" -c:v libx264 -crf 32 -preset veryfast -c:a aac -b:a 96k -ac 1 lecture01_proxy.mp4
```

프리미어에서 만들려면: 파일 > 내보내기 > 미디어 → 형식 H.264, 해상도 640×360, 비트레이트 0.5 Mbps, 오디오 AAC 96kbps.
**앞뒤를 자르지 말고 전체를 그대로** 내보내야 시간이 맞습니다.

### 2. 분석해서 프리미어용 XML 만들기

```bash
python lecture_cut.py lecture01_proxy.mp4 --premiere --timeline-fps 29.97 --source-size 1920x1080 --source-path "D:/강의/lecture01.mp4"
```

- `--timeline-fps`: **원본** fps (사본 fps를 낮췄으므로 필수). 원본 정보는 프리미어에서 클립 우클릭 > 속성으로 확인
- `--source-path`: 원본 영상 위치. 생략하면 프리미어가 가져올 때 "미디어 연결" 창에서 원본을 지정하면 됨

결과물
- `lecture01_proxy_premiere.xml` — 컷이 적용된 시퀀스. 컷 지점마다 **무엇을 잘랐는지 마커**(예: `필러: um uh`, `숨소리+침묵 2.4초`)가 붙어 있음
- `lecture01_proxy_cuts.csv` — 잘린 구간 목록(원본 타임코드, 길이, 이유). 엑셀로 열어서 검토

### 3. 프리미어에서 가져오기

파일 > 가져오기 > `*_premiere.xml` → 새 시퀀스가 생깁니다. 원본에 연결된 상태라 화질 손실이 없고,
컷 하나하나를 일반 편집처럼 늘리거나 되살릴 수 있습니다 (리플 트리밍으로 마커 위치 확인하며 검토).
