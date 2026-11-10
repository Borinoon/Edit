#!/usr/bin/env python3
"""
lecture_cut.py — 강의 녹화본 자동 컷 편집기

하는 일
  1. 음성 인식(faster-whisper, 단어 단위 타임스탬프)으로 말한 구간을 찾는다.
  2. 더듬는 말(어, 음, 으…), 같은 말 반복("그 그래서", "프로 프로그램")을 찾아 지운다.
  3. 말과 말 사이의 숨소리·긴 침묵을 잘라내되, 쉼을 0으로 만들지 않고
     문장 끝/문장 중간에 맞는 자연스러운 길이(기본 0.65초 / 0.35초)만 남긴다.
  4. (선택) 화면 멈춤(freezedetect)을 분석해서
       - 화면이 멈춰 있고 말도 없는 구간  → 잘라냄
       - 말은 없지만 화면이 움직이는 구간 → 보존(판서·코딩 시연 등)
  5. ffmpeg로 한 번에 렌더링한다. (프레임 경계에 맞춰 잘라 음성/영상 싱크 유지)

사용 예
  python lecture_cut.py lecture.mp4                        # 기본 설정
  python lecture_cut.py lecture.mp4 --keep-silent-motion   # 말 없이 화면이 움직이는 시연 구간 보존
  python lecture_cut.py lecture.mp4 --dry-run              # 렌더링 없이 어디를 자를지만 리포트
  python lecture_cut.py lecture.mp4 --no-asr               # 음성 인식 없이 음량 기반으로만 침묵 컷
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict, field

import numpy as np

# ----------------------------------------------------------------------------
# 필러(더듬는 말) 사전
# ----------------------------------------------------------------------------
# 항상 지우는 필러: 의미가 거의 없는 소리. "어어어", "으음" 같은 반복형도 포함.
HARD_FILLER_RE = re.compile(
    r"^((어|음|으|엄|흠|에|읍)+"            # 한국어
    r"|u+h+|u+m+|u+h+m+|e+r+m*|h+m+|m+h*m+)$",  # 영어: uh, uhh, um, umm, uhm, er, erm, hmm, mm, mhm
    re.I)
# 앞뒤로 쉼이 있을 때(단독으로 튀어나올 때)만 지우는 필러.
# "그 사람", "저 책", "아 그렇구나"처럼 실제 의미가 있을 수 있어서 조건부로 처리한다.
SOFT_FILLERS = {"아", "그", "저", "뭐", "이제", "막", "그니까", "약간",
                "ah", "ahh", "oh", "like", "so", "well"}
# 두 단어짜리 조건부 필러 (앞뒤에 쉼이 있을 때만): "..., you know, ..."
SOFT_PHRASES = {("you", "know"), ("i", "mean")}

_PUNCT_RE = re.compile(r"[\s\.,!?~…·\-\"'“”‘’()\[\]]+")


def norm(word: str) -> str:
    return _PUNCT_RE.sub("", word).lower()


# ----------------------------------------------------------------------------
# 데이터 구조
# ----------------------------------------------------------------------------
@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float = 1.0
    drop: str = ""  # 지우는 이유 ("" 이면 유지)


@dataclass
class Plan:
    keep: list = field(default_factory=list)       # [(start, end)]
    removed_words: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# ffmpeg 유틸
# ----------------------------------------------------------------------------
def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def probe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if a is None:
        sys.exit("오디오 트랙이 없는 파일입니다.")
    fps = None
    if v is not None:
        for key in ("avg_frame_rate", "r_frame_rate"):
            num, _, den = v.get(key, "0/0").partition("/")
            try:
                f = float(num) / float(den)
                if 1 <= f <= 240:
                    fps = f
                    break
            except (ValueError, ZeroDivisionError):
                pass
    return {
        "duration": float(info["format"]["duration"]),
        "has_video": v is not None,
        "fps": fps,
        "sample_rate": int(a.get("sample_rate", 48000)),
        "channels": int(a.get("channels", 2)),
        "width": int(v["width"]) if v else 1920,
        "height": int(v["height"]) if v else 1080,
    }


def extract_audio(path: str, wav: str, sr: int = 16000):
    run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vn", "-ac", "1", "-ar", str(sr),
         "-c:a", "pcm_s16le", wav])


def load_wav(wav: str) -> tuple[np.ndarray, int]:
    import wave
    with wave.open(wav, "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0, sr


# ----------------------------------------------------------------------------
# 음량 분석 (숨소리 / 침묵 판정, 단어 경계 보정)
# ----------------------------------------------------------------------------
class Energy:
    """10ms 단위 RMS(dBFS). 잡음 바닥과 음성 임계값을 자동 추정한다."""

    HOP = 0.01

    def __init__(self, samples: np.ndarray, sr: int, speech_db_above_floor: float):
        hop = int(sr * self.HOP)
        n = len(samples) // hop
        frames = samples[: n * hop].reshape(n, hop)
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        self.db = 20 * np.log10(rms + 1e-9)
        # 잡음 바닥 = 하위 10% 분위, 음성 피크 = 상위 95% 분위
        self.floor = float(np.percentile(self.db, 10))
        self.peak = float(np.percentile(self.db, 95))
        # 숨소리는 잡음보다는 크지만 말소리보다는 작다. 임계값을 그 사이에 둔다.
        self.speech_thr = max(self.floor + speech_db_above_floor, self.peak - 30)

    def idx(self, t: float) -> int:
        return max(0, min(len(self.db) - 1, int(t / self.HOP)))

    def first_loud(self, a: float, b: float) -> float | None:
        i, j = self.idx(a), self.idx(b)
        hit = np.nonzero(self.db[i:j] >= self.speech_thr)[0]
        return None if len(hit) == 0 else (i + hit[0]) * self.HOP

    def last_loud(self, a: float, b: float) -> float | None:
        i, j = self.idx(a), self.idx(b)
        hit = np.nonzero(self.db[i:j] >= self.speech_thr)[0]
        return None if len(hit) == 0 else (i + hit[-1] + 1) * self.HOP

    def has_breath(self, a: float, b: float, above_floor_db: float = 8.0) -> bool:
        """[a,b] 안에 잡음보다 확실히 큰 소리(숨소리·입소리)가 있는가"""
        i, j = self.idx(a), self.idx(b)
        return j > i and float(np.max(self.db[i:j])) >= self.floor + above_floor_db

    def quietest(self, a: float, b: float, length: float) -> float | None:
        """[a,b] 안에서 길이 length 인 가장 조용한 구간의 시작 시각"""
        i, j, n = self.idx(a), self.idx(b), max(1, int(round(length / self.HOP)))
        if j - i < n:
            return None
        seg = self.db[i:j]
        win = np.convolve(np.maximum(seg, self.floor), np.ones(n), "valid") + 0.0
        # 평균만 보면 짧은 숨소리 피크가 묻힐 수 있어 최대값도 함께 고려
        peaks = np.array([seg[k:k + n].max() for k in range(len(seg) - n + 1)])
        k = int(np.argmin(win / n + peaks))
        return (i + k) * self.HOP

    def speech_regions(self, min_speech=0.15, bridge=0.18) -> list[tuple[float, float]]:
        """ASR 없이 쓸 때: 음량 기반 음성 구간."""
        loud = self.db >= self.speech_thr
        regions, start = [], None
        for i, on in enumerate(loud):
            if on and start is None:
                start = i
            elif not on and start is not None:
                regions.append([start * self.HOP, i * self.HOP])
                start = None
        if start is not None:
            regions.append([start * self.HOP, len(loud) * self.HOP])
        merged = []
        for r in regions:  # 짧은 틈(자음 사이 등)은 이어 붙인다
            if merged and r[0] - merged[-1][1] < bridge:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        return [(a, b) for a, b in merged if b - a >= min_speech]


# ----------------------------------------------------------------------------
# 음성 인식
# ----------------------------------------------------------------------------
# Whisper는 기본적으로 "어/음"을 생략하고 깔끔하게 받아 적는 경향이 있다.
# 필러가 섞인 문장을 프롬프트로 주면 필러도 받아 적게 유도할 수 있다.
FILLER_PROMPTS = {
    "ko": "음, 어, 그러니까 어... 이제 음 그 그래서, 으음 오늘은 어 이 부분을 볼 건데요.",
    "en": "Umm, so, uh, let me think, like, hmm... Okay, uh, so today we're gonna, um, look at this.",
}


def transcribe(audio, model_name: str, device: str, compute_type: str, language: str) -> list[Word]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper가 필요합니다:  pip install faster-whisper   (또는 --no-asr 사용)")
    print(f"[ASR] 모델 로드: {model_name} ({device}/{compute_type})", flush=True)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    # 파일 경로 대신 16kHz float 배열을 넘긴다 (PyAV 버전 차이로 인한 디코딩 오류 회피)
    segments, info = model.transcribe(
        audio,
        language=language or None,
        word_timestamps=True,
        initial_prompt=FILLER_PROMPTS.get(language or "", FILLER_PROMPTS["en"]),
        condition_on_previous_text=False,  # 긴 강의에서 환각/반복 방지
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 150},
        beam_size=5,
    )
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            if w.end > w.start:
                words.append(Word(float(w.start), float(w.end), w.word.strip(), float(w.probability)))
        print(f"\r[ASR] {seg.end / 60:6.1f}분 처리", end="", flush=True)
    print()
    return words


# ----------------------------------------------------------------------------
# 필러 / 더듬기 판정
# ----------------------------------------------------------------------------
# "the there", "in into" 처럼 정상적으로 이어지는 영어 단어는 '잘린 앞부분'으로 보지 않는다
COMMON_EN = set("""a an i to in on at of it is be by do go he me my no or so up us we
the and for are but not you all can her was one our out his has how its new now see two way who
did get may any use say she this that with have from they will what when then them some into more
time like just over also back only well even want here there these those than very much each""".split())


def is_fragment_prefix(w: str) -> bool:
    if re.fullmatch(r"[a-z']+", w):
        return len(w) >= 3 and w not in COMMON_EN
    return len(w) <= 3  # 한국어: "프로 프로그램"


def mark_disfluencies(words: list[Word], soft_gap: float, use_soft: bool, max_stutter_gap: float):
    n = len(words)

    def gap_before(i):
        return words[i].start - words[i - 1].end if i > 0 else 9.9

    def gap_after(i):
        return words[i + 1].start - words[i].end if i + 1 < n else 9.9

    # 1) 필러
    for i, w in enumerate(words):
        t = norm(w.text)
        if not t:
            continue
        if HARD_FILLER_RE.match(t):
            w.drop = "filler"
        elif use_soft and t in SOFT_FILLERS:
            isolated = gap_before(i) >= soft_gap and gap_after(i) >= soft_gap
            stretched = (w.end - w.start) >= 0.55  # "그~~~" 처럼 늘어진 소리
            if isolated or stretched:
                w.drop = "filler"

    # 1-a) 말하다 끊긴 단어: Whisper는 "prog-", "wh—" 처럼 하이픈으로 적는다
    for w in words:
        if not w.drop and re.search(r"[A-Za-z가-힣][-–—]$", w.text.strip()):
            w.drop = "fragment"

    # 1-b) 두 단어 필러 ("you know", "I mean")
    if use_soft:
        for i in range(n - 1):
            pair = (norm(words[i].text), norm(words[i + 1].text))
            if pair in SOFT_PHRASES and not words[i].drop and not words[i + 1].drop \
                    and gap_before(i) >= soft_gap and gap_after(i + 1) >= soft_gap:
                words[i].drop = words[i + 1].drop = "filler"

    # 2) 반복/더듬기: 지우지 않은 단어끼리 비교
    alive = [i for i, w in enumerate(words) if not w.drop]
    texts = [norm(words[i].text) for i in alive]

    def close(a_idx, b_idx):  # 두 단어 사이가 충분히 가까운가 (진짜 더듬기인지)
        return words[alive[b_idx]].start - words[alive[a_idx]].end <= max_stutter_gap

    k = 0
    while k < len(alive) - 1:
        handled = False
        # n-gram 반복 (3어절 → 1어절 순): "이것은 중요 이것은 중요한" → 앞쪽 제거
        for g in (3, 2, 1):
            if k + 2 * g > len(alive):
                continue
            a = texts[k:k + g]
            b = texts[k + g:k + 2 * g]
            if not all(a) or not close(k + g - 1, k + g):
                continue
            same = a[:-1] == b[:-1] and (a[-1] == b[-1] or (len(a[-1]) < len(b[-1]) and b[-1].startswith(a[-1])))
            if g == 1:
                # 한 단어: 완전히 같거나("그 그"), 잘린 앞부분("프로 프로그램")
                w0, w1 = a[0], b[0]
                same = w0 == w1 or (len(w0) < len(w1) and w1.startswith(w0) and is_fragment_prefix(w0))
            if same:
                for j in range(k, k + g):
                    words[alive[j]].drop = "repeat"
                k += g
                handled = True
                break
        if not handled:
            k += 1


# ----------------------------------------------------------------------------
# 화면 멈춤 감지
# ----------------------------------------------------------------------------
def detect_freeze(path: str, noise_db: float, min_dur: float) -> list[tuple[float, float]]:
    print("[화면] 멈춤 구간 분석 중...", flush=True)
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-an",
         "-vf", f"scale=480:-2,freezedetect=n={noise_db}dB:d={min_dur}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    spans, start = [], None
    for line in proc.stderr.splitlines():
        m = re.search(r"freeze_start: ([\d.]+)", line)
        if m:
            start = float(m.group(1))
        m = re.search(r"freeze_end: ([\d.]+)", line)
        if m and start is not None:
            spans.append((start, float(m.group(1))))
            start = None
    if start is not None:
        spans.append((start, float("inf")))
    return spans


def subtract(a: float, b: float, holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """[a,b] 에서 holes 를 뺀 구간들"""
    out, cur = [], a
    for h0, h1 in sorted(holes):
        if h1 <= cur or h0 >= b:
            continue
        if h0 > cur:
            out.append((cur, h0))
        cur = max(cur, h1)
    if cur < b:
        out.append((cur, b))
    return out


def moving_parts(a: float, b: float, freezes, min_len: float):
    """[a,b] 중 화면이 움직이는(멈추지 않은) 부분"""
    return [(x, y) for x, y in subtract(a, b, freezes) if y - x >= min_len]


# ----------------------------------------------------------------------------
# 컷 계획 (자연스러운 쉼 만들기)
# ----------------------------------------------------------------------------
def build_plan(speech: list[tuple[float, float, bool]], duration: float, energy: Energy | None,
               args, freezes, dropped: list[tuple[float, float]] = ()) -> list[tuple[float, float]]:
    """
    speech : 유지할 말 덩어리 [(start, end, sentence_end)] — 시간순
    dropped: 지울 단어(필러/반복) 구간 [(start, end)] — 이 소리는 쉼 여유로도 남기면 안 됨
    반환   : 유지할 구간 [(start, end)]
    """
    if not speech:
        return [(0.0, duration)]
    dropped = sorted(dropped)

    def prev_drop_end(t):
        ends = [d1 for d0, d1 in dropped if d1 <= t + 0.02]
        return max(ends) if ends else 0.0

    def next_drop_start(t):
        starts = [d0 for d0, d1 in dropped if d0 >= t - 0.02]
        return min(starts) if starts else duration

    # 단어 경계 보정: Whisper 타임스탬프는 ±0.1초쯤 흔들린다. 음량으로 실제 소리 시작/끝을 잡는다.
    # 단, 지운 단어 쪽으로는 넘어가지 않는다.
    refined = []
    for s, e, sent in speech:
        lo, hi = prev_drop_end(s), next_drop_start(e)
        if energy is not None:
            fs = energy.first_loud(max(lo, s - 0.12), e)
            le = energy.last_loud(s, min(hi, e + 0.15))
            if fs is not None and fs < s:
                s = fs
            if le is not None and le > e:
                e = le
        refined.append((s, e, sent))

    lead = args.lead_in
    first_s = refined[0][0]
    keep = [[max(0.0, prev_drop_end(first_s), first_s - lead), None]]

    for i in range(len(refined) - 1):
        _, e, sent = refined[i]
        s_next = refined[i + 1][0]
        gap = s_next - e
        inside = [(d0, d1) for d0, d1 in dropped if d0 >= e - 0.05 and d1 <= s_next + 0.05]
        long_gap = gap - sum(d1 - d0 for d0, d1 in inside) >= args.sentence_gap_hint
        target = args.sentence_pause if (sent or long_gap) else args.pause

        if gap <= target + args.tolerance:
            if not inside:
                continue  # 원래 쉼이 이미 자연스러움 → 그대로 둠
            cuts = [(max(e, inside[0][0]), min(s_next, inside[-1][1]))]  # 필러 소리만 뺀다
        else:
            # 남길 쉼을 앞(말 끝 여운)과 뒤(다음 말 들어가기 전)로 나눈다.
            t_keep = max(target - lead, target * 0.55)
            l_keep = target - t_keep
            cut_a, cut_b = e + t_keep, s_next - l_keep
            if inside:
                # 지운 단어가 '다음 말 앞 여유'를 잡아먹으면 그만큼 앞쪽 여운을 늘려 전체 쉼 길이를 유지
                if inside[-1][1] > cut_b:
                    cut_a += inside[-1][1] - cut_b
                    cut_b = inside[-1][1]
                cut_a = min(cut_a, inside[0][0])
            cuts = [(cut_a, cut_b)]
            if energy is not None and not args.keep_breath and not inside:
                # 말 끝 여운 자리에 숨소리가 섞여 있으면, 여운은 짧게(TAIL_MIN)만 남기고
                # 나머지 쉼은 이 틈에서 가장 조용한 '방 소리' 구간으로 채운다 → 쉼 길이는 그대로, 숨소리만 빠짐
                TAIL_MIN = 0.12
                if energy.has_breath(e + TAIL_MIN, cut_a):
                    fill = t_keep - TAIL_MIN
                    q = energy.quietest(e + TAIL_MIN, cut_b, fill)
                    if q is not None:
                        cuts = [(e + TAIL_MIN, q), (q + fill, cut_b)]
            if args.keep_silent_motion and freezes is not None:
                # 말은 없지만 화면이 움직이는 부분(판서, 코딩, 시연)은 남긴다
                # (지운 필러 소리는 화면이 움직여도 지움)
                moving = moving_parts(cut_a, cut_b, freezes, args.min_motion)
                moving = [m for mv in moving for m in subtract(mv[0], mv[1], inside)]
                cuts = [piece for a, b in cuts for piece in subtract(a, b, moving)]

        for a, b in cuts:
            if b - a < args.min_cut and not any(d0 < b and d1 > a for d0, d1 in inside):
                continue
            keep[-1][1] = a
            keep.append([b, None])

    last_e = refined[-1][1]
    keep[-1][1] = min(duration, next_drop_start(last_e), last_e + args.end_pad)
    return [(a, b) for a, b in keep if b - a > 0.02]


def find_untranscribed(words: list[Word], energy: "Energy", min_len: float = 0.15) -> list[Word]:
    """
    Whisper 는 uh/um, 헛기침, 웅얼거림을 받아쓰기에서 빼 버리는 경향이 강하다.
    → '목소리 크기의 소리가 나는데 어떤 단어에도 속하지 않는 구간'을 필러로 본다.
    단어 바로 앞뒤에 붙은 짧은 소리는 단어 끝/시작이 덜 잡힌 것일 수 있어 제외한다.
    """
    hop = energy.HOP
    loud = energy.db >= energy.speech_thr
    cov = np.zeros(len(loud), bool)
    for w in words:
        cov[int(w.start / hop):int(w.end / hop) + 1] = True
    un = loud & ~cov
    chunks, st = [], None
    for i, v in enumerate(np.append(un, False)):
        if v and st is None:
            st = i
        elif not v and st is not None:
            a, b = st * hop, i * hop
            if chunks and a - chunks[-1][1] < 0.1:
                chunks[-1][1] = b
            else:
                chunks.append([a, b])
            st = None
    starts = sorted(w.start for w in words)
    ends = sorted(w.end for w in words)
    import bisect

    def touches(a, b):
        i = bisect.bisect_left(ends, a - 0.06)
        after_word = i < len(ends) and ends[i] <= a + 0.02
        j = bisect.bisect_left(starts, b - 0.02)
        before_word = j < len(starts) and starts[j] <= b + 0.06
        return after_word or before_word

    out = []
    for a, b in chunks:
        dur = b - a
        if dur < min_len or (dur < 0.4 and touches(a, b)):
            continue
        out.append(Word(a, b, "[uh/um 등]", 0.0, "filler"))
    return out


def words_to_speech(words: list[Word]) -> list[tuple[float, float, bool]]:
    """남길 단어들을 '말 덩어리'로 묶는다. 지운 단어 자리는 덩어리를 끊는다."""
    chunks = []
    cur = None
    for w in words:
        if w.drop:
            if cur:
                chunks.append(cur)
                cur = None
            continue
        sent_end = bool(re.search(r"[.?!。]$", w.text)) or bool(re.search(r"(다|요|죠|까)[.?!]?$", w.text))
        if cur and w.start - cur[1] < 0.08:
            cur[1], cur[2] = w.end, sent_end
        else:
            if cur:
                chunks.append(cur)
            cur = [w.start, w.end, sent_end]
    if cur:
        chunks.append(cur)
    return [tuple(c) for c in chunks]


def snap(keep, fps: float | None):
    """영상 프레임 경계에 맞춘다 (음성/영상 싱크 누적 오차 방지)"""
    if not fps:
        return keep
    out = []
    for a, b in keep:
        a2, b2 = round(a * fps) / fps, round(b * fps) / fps
        if b2 - a2 >= 1 / fps:
            if out and a2 <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], b2))
            else:
                out.append((a2, b2))
    return out


# ----------------------------------------------------------------------------
# 렌더링
# ----------------------------------------------------------------------------
def render(src: str, dst: str, keep, meta: dict, fps: float | None, args):
    sr = meta["sample_rate"]
    eps_v = 0.5 / fps if fps else 0.0
    vexpr = "+".join(f"gte(t,{a - eps_v:.4f})*lt(t,{b - eps_v:.4f})" for a, b in keep)
    aexpr = "+".join(f"gte(t,{a - 0.0005:.4f})*lt(t,{b - 0.0005:.4f})" for a, b in keep)
    # 오디오를 작은 프레임으로 쪼개야 aselect 가 정밀하게 자른다.
    if fps and abs(sr / fps - round(sr / fps)) < 1e-6:
        nsamp = int(round(sr / fps))  # 영상 프레임과 정확히 정렬
    else:
        nsamp = max(1, sr // 1000)

    parts = []
    if meta["has_video"]:
        parts.append(f"[0:v]fps={fps:.6f},select='{vexpr}',setpts=N/FRAME_RATE/TB[v]")
    parts.append(f"[0:a]aresample={sr},asetnsamples=n={nsamp}:p=0,aselect='{aexpr}',asetpts=N/SR/TB[a]")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(";\n".join(parts))
        script = f.name
    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", src, "-filter_complex_script", script]
    if meta["has_video"]:
        cmd += ["-map", "[v]", "-c:v", args.vcodec, "-preset", args.preset, "-crf", str(args.crf),
                "-pix_fmt", "yuv420p"]
    cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", args.abitrate, "-movflags", "+faststart", dst]
    print("[렌더] ffmpeg 실행 중...", flush=True)
    try:
        run(cmd)
    finally:
        os.unlink(script)



# ----------------------------------------------------------------------------
# 프리미어 프로 내보내기 (FCP7 XML) + 컷 목록 CSV
# ----------------------------------------------------------------------------
def timebase_of(fps: float) -> tuple[int, bool, float]:
    """(timebase, ntsc 여부, 실제 fps). 29.97 → (30, True, 30000/1001)"""
    for base in (24, 30, 60, 120):
        ntsc = base * 1000 / 1001
        if abs(fps - ntsc) < 0.01:
            return base, True, ntsc
    return int(round(fps)), False, float(round(fps))


def tc(frames: int, base: int) -> str:
    f = frames % base
    s = frames // base
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}:{f:02d}"


def path_to_url(p: str) -> str:
    from urllib.parse import quote
    p = p.replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", p):           # Windows: C:/... → file://localhost/C%3a/...
        p = p[0] + "%3a" + quote(p[2:], safe="/")
        return "file://localhost/" + p
    return "file://localhost" + quote(p if p.startswith("/") else "/" + p, safe="/")


def describe_cuts(keep, duration, words, energy) -> list[dict]:
    """잘려나가는 구간마다 이유를 붙인다 (검토용)"""
    cuts, prev = [], 0.0
    for a, b in keep + [(duration, duration)]:
        if a - prev > 0.005:
            inside = [w for w in words if w.drop and w.start < a and w.end > prev]
            reasons = []
            fill = [w.text for w in inside if w.drop == "filler"]
            rep = [w.text for w in inside if w.drop in ("repeat", "fragment")]
            if fill:
                reasons.append("필러: " + " ".join(fill))
            if rep:
                reasons.append("반복/끊긴말: " + " ".join(rep))
            silence = (a - prev) - sum(min(w.end, a) - max(w.start, prev) for w in inside)
            if silence > 0.3:
                gaps = subtract(prev, a, [(w.start, w.end) for w in words])
                breath = energy is not None and any(energy.has_breath(x, y) for x, y in gaps if y - x > 0.05)
                kind = "숨소리+침묵" if breath else "침묵"
                reasons.append(f"{kind} {silence:.1f}초")
            cuts.append({"start": prev, "end": a, "reason": ", ".join(reasons) or "쉼 정리"})
        prev = b
    return cuts


def export_premiere(xml_path, csv_path, keep, cuts, fps, src_path, total_dur, width, height,
                    channels, sample_rate, seq_name):
    from xml.sax.saxutils import escape as esc
    base, ntsc, real = timebase_of(fps)
    fr = lambda t: int(round(t * real))  # noqa: E731
    rate = f"<rate><timebase>{base}</timebase><ntsc>{'TRUE' if ntsc else 'FALSE'}</ntsc></rate>"
    name = esc(os.path.basename(src_path))
    src_frames = fr(total_dur)

    file_full = f"""<file id="file-1"><name>{name}</name><pathurl>{esc(path_to_url(src_path))}</pathurl>{rate}
<duration>{src_frames}</duration><timecode>{rate}<string>00:00:00:00</string><frame>0</frame><displayformat>NDF</displayformat></timecode>
<media><video><samplecharacteristics>{rate}<width>{width}</width><height>{height}</height></samplecharacteristics></video>
<audio><samplecharacteristics><depth>16</depth><samplerate>{sample_rate}</samplerate></samplecharacteristics><channelcount>{channels}</channelcount></audio></media></file>"""

    v_items, a_items = [], [[] for _ in range(channels)]
    pos = 0
    for i, (a, b) in enumerate(keep):
        fin, fout = fr(a), fr(b)
        length = fout - fin
        if length <= 0:
            continue
        start, end = pos, pos + length
        ids = [f"clipitem-v{i}"] + [f"clipitem-a{c}-{i}" for c in range(channels)]
        links = "".join(
            f"<link><linkclipref>{cid}</linkclipref><mediatype>{'video' if k == 0 else 'audio'}</mediatype>"
            f"<trackindex>{1 if k == 0 else k}</trackindex><clipindex>{i + 1}</clipindex></link>"
            for k, cid in enumerate(ids))
        common = (f"<name>{name}</name><enabled>TRUE</enabled><duration>{src_frames}</duration>{rate}"
                  f"<start>{start}</start><end>{end}</end><in>{fin}</in><out>{fout}</out>")
        f_el = file_full if i == 0 else '<file id="file-1"/>'
        v_items.append(f'<clipitem id="{ids[0]}">{common}{f_el}{links}</clipitem>')
        for c in range(channels):
            a_items[c].append(
                f'<clipitem id="{ids[c + 1]}">{common}<file id="file-1"/>'
                f"<sourcetrack><mediatype>audio</mediatype><trackindex>{c + 1}</trackindex></sourcetrack>{links}</clipitem>")
        pos = end

    # 컷 지점마다 시퀀스 마커 (무엇을 잘랐는지 프리미어에서 바로 확인)
    markers, pos = [], 0
    cut_iter = iter(cuts)
    nxt = next(cut_iter, None)
    for a, b in keep:
        while nxt is not None and nxt["end"] <= a + 0.005:
            markers.append(f"<marker><name>{esc(nxt['reason'])}</name><comment>원본 {esc(tc(fr(nxt['start']), base))}"
                           f" ~ {esc(tc(fr(nxt['end']), base))}</comment><in>{pos}</in><out>-1</out></marker>")
            nxt = next(cut_iter, None)
        pos += fr(b) - fr(a)
    for c in ([nxt] if nxt else []) + list(cut_iter):  # 마지막 말 뒤에 잘린 부분
        markers.append(f"<marker><name>{esc(c['reason'])}</name><comment>원본 {esc(tc(fr(c['start']), base))}"
                       f" ~ {esc(tc(fr(c['end']), base))}</comment><in>{pos}</in><out>-1</out></marker>")

    audio_tracks = "".join(f"<track>{''.join(items)}<outputchannelindex>{c + 1}</outputchannelindex></track>"
                           for c, items in enumerate(a_items))
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE xmeml>
<xmeml version="4"><sequence id="sequence-1"><name>{esc(seq_name)}</name><duration>{pos}</duration>{rate}
<timecode>{rate}<string>00:00:00:00</string><frame>0</frame><displayformat>NDF</displayformat></timecode>
<media><video><format><samplecharacteristics>{rate}<width>{width}</width><height>{height}</height>
<pixelaspectratio>square</pixelaspectratio></samplecharacteristics></format>
<track>{''.join(v_items)}</track></video>
<audio><numOutputChannels>{channels}</numOutputChannels><format><samplecharacteristics><depth>16</depth>
<samplerate>{sample_rate}</samplerate></samplecharacteristics></format>{audio_tracks}</audio></media>
{''.join(markers)}</sequence></xmeml>
"""
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml)

    import csv
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: 엑셀에서 한글 안 깨짐
        w = csv.writer(f)
        w.writerow(["번호", "원본 시작", "원본 끝", "자른 길이(초)", "이유"])
        for k, c in enumerate(cuts, 1):
            w.writerow([k, tc(fr(c["start"]), base), tc(fr(c["end"]), base),
                        f"{c['end'] - c['start']:.2f}", c["reason"]])

# ----------------------------------------------------------------------------
def fmt(t: float) -> str:
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


def main():
    p = argparse.ArgumentParser(description="강의 녹화본에서 더듬는 말·숨소리·긴 침묵·멈춘 화면을 자연스럽게 잘라냅니다.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input")
    p.add_argument("-o", "--output", help="출력 파일 (기본: 입력이름_cut.mp4)")
    p.add_argument("--dry-run", action="store_true", help="렌더링하지 않고 컷 리포트만 저장")

    g = p.add_argument_group("음성 인식")
    g.add_argument("--no-asr", action="store_true", help="음성 인식 없이 음량만으로 침묵/숨소리 컷 (필러 제거 불가)")
    g.add_argument("--model", default="small", help="whisper 모델: tiny/base/small/medium/large-v3")
    g.add_argument("--device", default="auto", help="cpu / cuda / auto")
    g.add_argument("--compute-type", default="int8", help="int8(CPU 권장) / float16(GPU 권장)")
    g.add_argument("--language", default="en", help="강의 언어 (en / ko ...)")

    g = p.add_argument_group("더듬기 제거")
    g.add_argument("--no-fillers", action="store_true", help="필러/반복 제거 끄기 (침묵만 정리)")
    g.add_argument("--no-soft-fillers", action="store_true", help="'그/저/아/뭐/이제' 같은 조건부 필러는 지우지 않음")
    g.add_argument("--soft-gap", type=float, default=0.25, help="조건부 필러로 판정할 앞뒤 쉼(초)")
    g.add_argument("--keep-untranscribed", action="store_true",
                   help="받아쓰기에 없는 목소리 구간(대개 uh/um)을 지우지 않음")
    g.add_argument("--max-stutter-gap", type=float, default=0.6, help="이 간격 이내 반복만 더듬기로 판정(초)")

    g = p.add_argument_group("자연스러운 쉼 (값이 클수록 여유 있게)")
    g.add_argument("--pause", type=float, default=0.35, help="문장 중간 쉼을 이 길이로 줄임(초)")
    g.add_argument("--sentence-pause", type=float, default=0.65, help="문장 끝 쉼을 이 길이로 줄임(초)")
    g.add_argument("--sentence-gap-hint", type=float, default=1.2, help="원래 쉼이 이보다 길면 문장 끝으로 간주(초)")
    g.add_argument("--lead-in", type=float, default=0.15, help="말 시작 전에 남길 여유(초)")
    g.add_argument("--end-pad", type=float, default=0.8, help="영상 마지막 말 뒤에 남길 길이(초)")
    g.add_argument("--tolerance", type=float, default=0.1, help="목표 쉼보다 이만큼 이하로 길면 자르지 않음(초)")
    g.add_argument("--min-cut", type=float, default=0.12, help="이보다 짧은 컷은 하지 않음(초)")
    g.add_argument("--keep-breath", action="store_true", help="쉼 안의 숨소리를 피해서 고르지 않음")
    g.add_argument("--speech-db", type=float, default=14.0, help="잡음 바닥 대비 몇 dB 이상을 말소리로 볼지")

    g = p.add_argument_group("화면 멈춤")
    g.add_argument("--keep-silent-motion", action="store_true",
                   help="말이 없어도 화면이 움직이는 구간(판서·코딩 시연)은 보존. 멈춘 화면+무음만 컷")
    g.add_argument("--freeze-noise", type=float, default=-55.0, help="멈춤 판정 민감도(dB, 높을수록 관대)")
    g.add_argument("--freeze-min", type=float, default=1.0, help="이 시간 이상 변화 없으면 멈춤으로 판정(초)")
    g.add_argument("--min-motion", type=float, default=0.8, help="이보다 짧은 움직임은 무시(초)")

    g = p.add_argument_group("프리미어 프로로 내보내기 (원본은 그대로, 컷 정보만)")
    g.add_argument("--premiere", action="store_true",
                   help="영상을 렌더링하지 않고 프리미어용 XML(+컷 목록 CSV)만 만든다")
    g.add_argument("--source-path", help="원본 영상의 경로 (예: D:/강의/lecture01.mp4). "
                   "저화질 사본으로 분석할 때 XML이 원본을 가리키게 함. 생략 시 프리미어에서 '미디어 연결'로 지정")
    g.add_argument("--timeline-fps", type=float, help="원본 영상 fps (사본의 fps를 낮췄다면 반드시 지정. 예: 29.97)")
    g.add_argument("--source-size", default=None, help="원본 해상도 (예: 1920x1080, 기본: 입력 파일 해상도)")
    g.add_argument("--channels", type=int, default=None, help="원본 오디오 채널 수 (기본: 입력 파일)")

    g = p.add_argument_group("인코딩")
    g.add_argument("--fps", type=float, help="출력 fps (기본: 원본)")
    g.add_argument("--vcodec", default="libx264")
    g.add_argument("--preset", default="medium")
    g.add_argument("--crf", type=int, default=18)
    g.add_argument("--abitrate", default="192k")
    args = p.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} 를 찾을 수 없습니다. ffmpeg를 설치해 주세요.")

    src = args.input
    base = os.path.splitext(src)[0]
    # make_proxy.py 가 만든 원본 정보가 있으면 프리미어 내보내기 설정을 자동으로 채운다
    info_path = base + ".info.json"
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        args.source_path = args.source_path or info.get("source_path")
        args.timeline_fps = args.timeline_fps or info.get("source_fps")
        if not args.source_size and info.get("source_width"):
            args.source_size = f"{info['source_width']}x{info['source_height']}"
        args.channels = args.channels or info.get("source_channels")
        print(f"[원본 정보] {info_path} → {args.source_path}, {args.timeline_fps}fps, {args.source_size}")
    dst = args.output or base + "_cut.mp4"
    meta = probe(src)
    fps = args.fps or meta["fps"]
    if fps:
        fps = round(fps, 3)
    dur = meta["duration"]
    print(f"[입력] {src}  길이 {fmt(dur)}  fps {fps}  오디오 {meta['sample_rate']}Hz")

    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "a.wav")
        extract_audio(src, wav)
        samples, sr = load_wav(wav)
        energy = Energy(samples, sr, args.speech_db)
        print(f"[음량] 잡음 바닥 {energy.floor:.1f} dB, 말소리 임계 {energy.speech_thr:.1f} dB")

        words: list[Word] = []
        if args.no_asr:
            speech = [(a, b, False) for a, b in energy.speech_regions()]
        else:
            cache = base + ".words.json"
            if os.path.exists(cache):
                print(f"[ASR] 캐시 사용: {cache}  (다시 인식하려면 이 파일을 지우세요)")
                with open(cache, encoding="utf-8") as f:
                    words = [Word(**{k: d[k] for k in ("start", "end", "text", "prob")}) for d in json.load(f)]
            else:
                words = transcribe(samples, args.model, args.device, args.compute_type, args.language)
                with open(cache, "w", encoding="utf-8") as f:
                    json.dump([asdict(w) for w in words], f, ensure_ascii=False, indent=0)
            if not args.no_fillers:
                mark_disfluencies(words, args.soft_gap, not args.no_soft_fillers, args.max_stutter_gap)
            if not args.no_fillers and not args.keep_untranscribed:
                extra = find_untranscribed(words, energy)
                if extra:
                    print(f"[필러] 받아쓰기에 없는 목소리(uh/um·헛기침 등) {len(extra)}곳 추가 감지")
                words = sorted(words + extra, key=lambda w: w.start)
            speech = words_to_speech(words)

    freezes = None
    if args.keep_silent_motion:
        if not meta["has_video"]:
            print("[화면] 영상 트랙이 없어 화면 분석을 건너뜁니다.")
        else:
            freezes = detect_freeze(src, args.freeze_noise, args.freeze_min)
            frozen_total = sum(min(b, dur) - a for a, b in freezes)
            print(f"[화면] 멈춘 구간 {len(freezes)}개, 합계 {fmt(frozen_total)}")

    dropped = [(w.start, w.end) for w in words if w.drop]
    snap_fps = args.timeline_fps or (fps if meta["has_video"] else None)
    if args.premiere and not snap_fps:
        sys.exit("영상 트랙이 없습니다. --timeline-fps 로 원본 fps를 지정하세요 (예: --timeline-fps 29.97)")
    keep = snap(build_plan(speech, dur, energy, args, freezes, dropped), snap_fps)
    out_dur = sum(b - a for a, b in keep)

    removed = [w for w in words if w.drop]
    report = {
        "input": src,
        "original_duration": round(dur, 2),
        "output_duration": round(out_dur, 2),
        "removed_seconds": round(dur - out_dur, 2),
        "cut_count": max(0, len(keep) - 1),
        "removed_words": [{"time": fmt(w.start), "text": w.text, "reason": w.drop} for w in removed],
        "keep_segments": [[round(a, 3), round(b, 3)] for a, b in keep],
    }
    rpath = base + "_cut_report.json"
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print(f"\n[결과] {fmt(dur)} → {fmt(out_dur)}  ({dur - out_dur:.1f}초 / {100 * (dur - out_dur) / dur:.1f}% 감소)")
    print(f"       컷 {report['cut_count']}회, 제거한 필러/반복 {len(removed)}개")
    if removed:
        sample = ", ".join(f"{w.text}({fmt(w.start)[3:]})" for w in removed[:12])
        print(f"       예: {sample}{' ...' if len(removed) > 12 else ''}")
    print(f"       리포트: {rpath}")

    if args.premiere:
        cuts = describe_cuts(keep, dur, words, energy)
        if args.source_size:
            w_, h_ = (int(x) for x in args.source_size.lower().split("x"))
        else:
            w_, h_ = meta["width"], meta["height"]
        xml_path, csv_path = base + "_premiere.xml", base + "_cuts.csv"
        export_premiere(xml_path, csv_path, keep, cuts, snap_fps, args.source_path or os.path.abspath(src),
                        dur, w_, h_, args.channels or meta["channels"], meta["sample_rate"],
                        os.path.splitext(os.path.basename(args.source_path or src))[0] + " (자동 컷)")
        print(f"[프리미어] XML: {xml_path}")
        print(f"[프리미어] 컷 목록: {csv_path}  ({len(cuts)}개)")
        return
    if args.dry_run:
        return
    render(src, dst, keep, meta, fps if meta["has_video"] else None, args)
    print(f"[완료] {dst}")


if __name__ == "__main__":
    main()
