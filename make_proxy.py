#!/usr/bin/env python3
"""
make_proxy.py — 분석용 저화질 사본(프록시) 자동 생성기

원본 강의 영상을 넣으면 업로드하기 쉬운 작은 사본을 만듭니다.
  - 360p, 5fps, 모노 오디오 → 1시간 강의 기준 수십 MB
  - 시간축은 원본과 100% 동일 (앞뒤를 자르지 않음) → 분석 결과를 원본에 그대로 적용 가능
  - 원본 정보(fps·해상도·채널·경로)를 <사본>.info.json 에 같이 저장
    → lecture_cut.py 가 자동으로 읽어서 프리미어 XML을 원본 기준으로 만든다

사용법
  python make_proxy.py 강의.mp4                 # 파일 하나
  python make_proxy.py 강의1.mp4 강의2.mov      # 여러 개
  python make_proxy.py D:/강의폴더              # 폴더 안의 영상 전부
  python make_proxy.py --watch D:/강의폴더      # 폴더 감시: 새 영상이 들어오면 자동 변환
  Windows: 영상 파일을 make_proxy.bat 아이콘 위로 끌어다 놓기
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".wmv", ".mts", ".m2ts", ".webm", ".flv"}
PROXY_DIR = "proxy"
SUFFIX = "_proxy"


# ----------------------------------------------------------------------------
# ffmpeg 찾기: 설치된 것 → 없으면 pip 패키지 imageio-ffmpeg 에 들어있는 것
# ----------------------------------------------------------------------------
def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    sys.exit("ffmpeg 를 찾을 수 없습니다.\n"
             "  가장 쉬운 방법:  pip install imageio-ffmpeg\n"
             "  또는 직접 설치:  Windows → winget install ffmpeg   /   Mac → brew install ffmpeg")


FFMPEG = None


def probe(path: str) -> dict:
    """원본 정보. ffprobe 가 있으면 쓰고, 없으면 ffmpeg -i 출력에서 읽는다."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        out = subprocess.run([ffprobe, "-v", "error", "-print_format", "json", "-show_streams",
                              "-show_format", path], capture_output=True, text=True, encoding="utf-8")
        if out.returncode == 0:
            info = json.loads(out.stdout)
            v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
            a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
            fps = None
            if v:
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
                "duration": float(info["format"].get("duration", 0)),
                "fps": round(fps, 3) if fps else None,
                "width": int(v["width"]) if v else None,
                "height": int(v["height"]) if v else None,
                "channels": int(a.get("channels", 2)) if a else 0,
                "sample_rate": int(a.get("sample_rate", 48000)) if a else None,
            }
    # ffprobe 없음 → ffmpeg -i 로그 파싱
    err = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stderr
    d = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    vid = re.search(r"Stream.*Video:.*?(\d{2,5})x(\d{2,5})", err)
    fps = re.search(r"([\d.]+) fps", err)
    aud = re.search(r"Stream.*Audio:.*?(\d+) Hz, ([^,]+)", err)
    ch_map = {"mono": 1, "stereo": 2, "5.1": 6, "5.1(side)": 6, "7.1": 8}
    ch = 0
    if aud:
        lay = aud.group(2).strip()
        ch = ch_map.get(lay) or int((re.match(r"(\d+) channels", lay) or [0, 2])[1])
    return {
        "duration": int(d[1]) * 3600 + int(d[2]) * 60 + float(d[3]) if d else 0.0,
        "fps": round(float(fps[1]), 3) if fps else None,
        "width": int(vid[1]) if vid else None,
        "height": int(vid[2]) if vid else None,
        "channels": ch,
        "sample_rate": int(aud[1]) if aud else None,
    }


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def proxy_paths(src: str, out_dir: str | None):
    folder = out_dir or os.path.join(os.path.dirname(os.path.abspath(src)), PROXY_DIR)
    name = os.path.splitext(os.path.basename(src))[0] + SUFFIX
    return folder, os.path.join(folder, name + ".mp4"), os.path.join(folder, name + ".info.json")


def notify(title: str, msg: str):
    """Mac 알림 센터 알림 (백그라운드 감시 중에도 완료를 알 수 있게)"""
    if sys.platform == "darwin":
        msg, title = msg.replace('"', "'"), title.replace('"', "'")
        subprocess.run(["osascript", "-e", f'display notification "{msg}" with title "{title}"'],
                       capture_output=True)


def make_proxy(src: str, args) -> str | None:
    folder, dst, info_path = proxy_paths(src, args.out)
    if os.path.exists(info_path) and not args.force:
        print(f"[건너뜀] 이미 있음: {dst}")
        return dst
    os.makedirs(folder, exist_ok=True)

    meta = probe(src)
    if not meta["channels"]:
        print(f"[경고] 오디오가 없는 파일이라 건너뜁니다: {src}")
        return None
    dur = meta["duration"]
    print(f"\n[원본] {src}")
    print(f"       {meta['width']}x{meta['height']}  {meta['fps']}fps  오디오 {meta['channels']}ch  길이 {dur / 60:.1f}분"
          f"  크기 {human(os.path.getsize(src))}")

    cmd = [FFMPEG, "-hide_banner", "-nostats", "-loglevel", "error", "-y", "-i", src,
           "-map", "0:a:0", "-ac", "1", "-c:a", "aac", "-b:a", args.audio_bitrate]
    if args.audio_only or meta["width"] is None:
        dst_media = os.path.splitext(dst)[0] + ".m4a"
        cmd += ["-vn"]
    else:
        dst_media = dst
        cmd += ["-map", "0:v:0", "-vf", f"scale=-2:{args.height},fps={args.fps}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(args.crf), "-pix_fmt", "yuv420p"]
    cmd += ["-movflags", "+faststart", "-progress", "pipe:1", dst_media + ".part.mp4"]

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:
        m = re.match(r"out_time_us=(\d+)", line)
        if m and dur > 0:
            done = int(m[1]) / 1e6
            pct = min(100.0, 100 * done / dur)
            speed = done / max(time.time() - t0, 0.01)
            eta = (dur - done) / speed if speed > 0 else 0
            bar = "#" * int(pct / 4)
            print(f"\r       [{bar:<25}] {pct:5.1f}%  (남은 시간 약 {eta / 60:.1f}분)", end="", flush=True)
    proc.wait()
    print()
    if proc.returncode != 0:
        print(f"[실패] {src}\n{proc.stderr.read()}")
        if args.notify:
            notify("강의 사본 실패", os.path.basename(src))
        if args.notify:
            notify("강의 사본 실패", os.path.basename(src))
        try:
            os.remove(dst_media + ".part.mp4")
        except OSError:
            pass
        return None
    os.replace(dst_media + ".part.mp4", dst_media)

    info = {
        "source_path": os.path.abspath(src).replace("\\", "/"),
        "source_fps": meta["fps"],
        "source_width": meta["width"],
        "source_height": meta["height"],
        "source_channels": meta["channels"],
        "source_sample_rate": meta["sample_rate"],
        "source_duration": dur,
        "proxy": os.path.basename(dst_media),
    }
    info_path = os.path.splitext(dst_media)[0] + ".info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)

    size = os.path.getsize(dst_media)
    print(f"[완료] {dst_media}  ({human(size)}, {time.time() - t0:.0f}초 걸림)")
    if args.notify:
        notify("강의 사본 완료", f"{os.path.basename(dst_media)} ({human(size)})")
    print(f"       같이 올릴 정보 파일: {info_path}")
    return dst_media


SKIP_DIRS = {PROXY_DIR, "편집결과", "강의 XML"}


def collect(paths, recursive: bool = False) -> list[str]:
    """영상 파일 모으기. recursive 면 하위 폴더까지 (결과·사본·숨김 폴더는 제외)"""
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, names in os.walk(p):
                dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS)
                for name in sorted(names):
                    full = os.path.join(root, name)
                    if not name.startswith(".") and os.path.splitext(name)[1].lower() in VIDEO_EXT \
                            and not os.path.splitext(name)[0].endswith(SUFFIX):
                        files.append(full)
                if not recursive:
                    break
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"[경고] 없는 경로: {p}")
    return files


def results_dir(src: str, args) -> str:
    """결과 폴더. --results 가 있으면 감시 폴더 안의 하위 폴더 구조를 그대로 따라 만든다
    (예: 원본/3월 강의/01.mp4 → 강의 XML/3월 강의/01_라벨표시.xml)"""
    src_dir = os.path.dirname(os.path.abspath(src))
    if not args.results:
        return os.path.join(src_dir, "편집결과")
    root = os.path.abspath(args.watch) if args.watch else src_dir
    rel = os.path.relpath(src_dir, root)
    return os.path.normpath(os.path.join(args.results, "" if rel.startswith("..") else rel))


def analyze(src: str, args, status_cb=None) -> bool:
    """원본을 바로 분석해서 프리미어 XML + 컷 목록을 만든다 (사본·업로드 불필요)"""
    out = results_dir(src, args)
    stem = os.path.splitext(os.path.basename(src))[0]
    print(f"\n[자동 편집] {src}\n       결과 폴더: {out}", flush=True)
    if args.notify:
        notify("강의 자동 편집 시작", f"{stem} — 30분 강의 기준 10~30분 걸립니다")
    cmd = [sys.executable, "-u", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lecture_cut.py"),
           src, "--premiere", "--out-dir", out, "--model", args.model]
    if not args.no_motion:
        cmd.append("--keep-silent-motion")
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        cmd = ["caffeinate", "-i"] + cmd  # 처리 중에는 Mac 이 잠자기로 들어가지 않게
    t0 = time.time()
    # 진행 단계를 실시간으로 진행상황.txt 에 반영
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    total = None
    last_write = 0.0
    for line in proc.stdout:
        print(line, end="", flush=True)  # log.txt 에도 그대로
        step = None
        m_ = re.search(r"길이 (\d+):(\d+):([\d.]+)", line)
        if m_ and total is None:
            total = int(m_[1]) * 60 + int(m_[2]) + float(m_[3]) / 60
        if "[ASR] 모델 로드" in line:
            step = "음성 인식 준비 중"
        m_ = re.search(r"\[ASR\]\s+([\d.]+)분 처리", line)
        if m_:
            step = f"음성 인식 {m_[1]}분" + (f" / {total:.0f}분" if total else "")
        if "다시 듣는 중" in line:
            step = "uh/um 다시 듣는 중"
        m_ = re.search(r"다시 듣기 (\d+)/(\d+)", line)
        if m_:
            step = f"uh/um 다시 듣기 {m_[1]}/{m_[2]}"
        if "[화면] 멈춤" in line:
            step = "화면 분석 중"
        if step and time.time() - last_write > 20 and status_cb:
            status_cb(f"진행 중 — {step} ({(time.time() - t0) / 60:.0f}분 경과)")
            last_write = time.time()
    proc.wait()
    ok = proc.returncode == 0 and os.path.exists(os.path.join(out, stem + "_라벨표시.xml"))
    mins = (time.time() - t0) / 60
    if ok:
        print(f"[자동 편집 완료] {stem}  ({mins:.0f}분 걸림)", flush=True)
        if args.notify:
            notify("강의 자동 편집 완료", f"{stem} — 바탕화면 '강의 XML' 폴더를 확인하세요")
    else:
        print(f"[자동 편집 실패] {stem}", flush=True)
        if args.notify:
            notify("강의 자동 편집 실패", f"{stem} — log.txt 를 확인하세요")
    return ok


def is_done(f: str, args) -> bool:
    if args.analyze:
        stem = os.path.splitext(os.path.basename(f))[0]
        return os.path.exists(os.path.join(results_dir(f, args), stem + "_라벨표시.xml"))
    return os.path.exists(proxy_paths(f, args.out)[2])


def write_status(folder: str, args, states: dict[str, str]):
    """결과 폴더에 '진행상황.txt' — 어떤 영상이 완료/진행 중/대기/실패인지 한눈에"""
    target = args.results or os.path.join(folder, "편집결과")
    try:
        os.makedirs(target, exist_ok=True)
        lines = [f"업데이트: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
        order = ["진행 중", "실패", "대기", "완료"]

        def rank(st):
            return next((i for i, k in enumerate(order) if st.startswith(k)), 9)
        for f, st in sorted(states.items(), key=lambda kv: (rank(kv[1]), kv[0])):
            lines.append(f"[{st}] {os.path.relpath(f, folder)}")
        with open(os.path.join(target, "진행상황.txt"), "w", encoding="utf-8") as fp:
            fp.write("\n".join(lines) + "\n")
    except OSError:
        pass


def watch(folder: str, args):
    """폴더 감시: 새 영상이 들어오고 복사가 끝나면(크기 변화 멈춤) 하나씩 자동 처리"""
    what = "프리미어용 자동 편집(XML)을" if args.analyze else "저화질 사본을"
    print(f"[감시] {folder} 에 영상이 들어오면 자동으로 {what} 만듭니다. (종료: Ctrl+C)", flush=True)
    sizes: dict[str, tuple[int, int]] = {}
    failed: dict[str, tuple[int, float]] = {}  # 파일 → (실패 횟수, 마지막 실패 시각)
    MAX_TRY, RETRY_AFTER = 3, 600
    started = time.time()
    while True:
        files = collect([folder], recursive=True)
        states = {}
        todo = None
        for f in files:
            if is_done(f, args):
                states[f] = "완료"
                continue
            n, t = failed.get(f, (0, 0.0))
            if n >= MAX_TRY:
                states[f] = f"실패 {n}회 — log.txt 확인, 파일을 다시 넣으면 재시도"
                continue
            if n and time.time() - t < RETRY_AFTER:
                states[f] = f"실패 {n}회 — {int((RETRY_AFTER - (time.time() - t)) / 60) + 1}분 뒤 재시도"
                continue
            try:
                size = os.path.getsize(f)
            except OSError:
                continue
            prev, stable = sizes.get(f, (-1, 0))
            stable = stable + 1 if size == prev else 0
            sizes[f] = (size, stable)
            states[f] = "대기" if stable >= 2 else "대기 (복사 중)"
            if stable >= 2 and todo is None:  # 10초 동안 크기 변화 없음 → 복사 완료
                todo = f
        # 다시 넣은 파일(크기·시각이 바뀜)은 실패 기록을 지운다
        for f in list(failed):
            if f not in files:
                failed.pop(f)
        if todo:
            states[todo] = "진행 중"
            write_status(folder, args, states)

            def update(msg, _t=todo, _s=states):
                _s[_t] = msg
                write_status(folder, args, _s)
            ok = analyze(todo, args, update) if args.analyze else make_proxy(todo, args)
            if ok:
                failed.pop(todo, None)
            else:
                n, _ = failed.get(todo, (0, 0.0))
                failed[todo] = (n + 1, time.time())
            sizes.pop(todo, None)
            continue  # 바로 다음 파일로
        write_status(folder, args, states)
        if args.restart_hours and time.time() - started > args.restart_hours * 3600:
            # 할 일이 없을 때 주기적으로 종료 → 자동 재시작되면서 최신 버전으로 업데이트 (run_watch.sh)
            print("[감시] 업데이트 확인을 위해 재시작합니다.", flush=True)
            sys.exit(0)
        time.sleep(5)


def main():
    global FFMPEG
    p = argparse.ArgumentParser(description="분석용 저화질 사본(프록시)을 만듭니다. 시간축은 원본과 동일합니다.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("inputs", nargs="*", help="영상 파일 또는 폴더")
    p.add_argument("--watch", metavar="폴더", help="폴더를 감시해서 새 영상을 자동 변환")
    p.add_argument("--out", help="사본 저장 폴더 (기본: 원본 옆 proxy 폴더)")
    p.add_argument("--height", type=int, default=360, help="사본 세로 해상도")
    p.add_argument("--fps", type=float, default=5, help="사본 fps (분석에는 5면 충분)")
    p.add_argument("--crf", type=int, default=32, help="화질 (클수록 작고 흐림)")
    p.add_argument("--audio-bitrate", default="96k")
    p.add_argument("--audio-only", action="store_true", help="소리만 저장 (가장 작음, 화면 멈춤 분석 불가)")
    p.add_argument("--analyze", action="store_true",
                   help="사본 대신 원본을 바로 분석해 프리미어 XML·컷 목록을 만듦 (lecture_cut.py 실행)")
    p.add_argument("--results", help="--analyze 결과 폴더 (기본: 원본 옆 '편집결과')")
    p.add_argument("--model", default="medium", help="--analyze 음성 인식 모델")
    p.add_argument("--no-motion", action="store_true", help="--analyze 시 말 없이 화면이 움직이는 구간도 자름")
    p.add_argument("--restart-hours", type=float, default=0, help=argparse.SUPPRESS)
    p.add_argument("--notify", action="store_true", help="완료/실패 시 Mac 알림 표시")
    p.add_argument("--force", action="store_true", help="이미 사본이 있어도 다시 만듦")
    p.add_argument("--pause", action="store_true", help=argparse.SUPPRESS)  # .bat 용: 끝나고 창 유지
    args = p.parse_args()
    FFMPEG = find_ffmpeg()

    try:
        if args.watch:
            watch(args.watch, args)
        files = collect(args.inputs)
        if not files:
            p.print_help()
        results = [(analyze(f, args) and f) if args.analyze else make_proxy(f, args) for f in files]
        ok = [r for r in results if r]
        if files:
            print(f"\n총 {len(files)}개 중 {len(ok)}개 완료.")
            if ok:
                print("proxy 폴더의 *_proxy.mp4 와 *_proxy.info.json 을 함께 올려 주세요.")
    except KeyboardInterrupt:
        print("\n중단했습니다.")
    finally:
        if args.pause:
            input("\n엔터를 누르면 창이 닫힙니다...")


if __name__ == "__main__":
    main()
