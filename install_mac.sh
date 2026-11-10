#!/bin/bash
# =============================================================================
# 강의 자동 편집기 — Mac 설치 스크립트
#
# 터미널에 아래 한 줄을 붙여넣고 엔터:
#   curl -fsSL https://raw.githubusercontent.com/Borinoon/Edit/claude/compassionate-darwin-3r1ph6/install_mac.sh | bash
#
# 하는 일
#   1. 파이썬이 없으면 python.org 공식 설치 파일로 설치 (Mac 암호 입력 필요)
#   2. 프로그램을 ~/LectureCut 에 설치하고, 변환 도구(ffmpeg) 준비
#   3. 감시 폴더 ~/LectureCut/원본 을 만들고 바탕화면에 바로가기 생성
#   4. 로그인할 때마다 자동으로 폴더 감시 시작 (백그라운드, 창 없음)
#      → 영상을 폴더에 넣으면 uh/um·숨소리·침묵을 찾아 프리미어 XML 을 자동 생성, 완료 알림
# 삭제:  bash ~/LectureCut/app/uninstall_mac.sh
# =============================================================================
set -euo pipefail

BRANCH="claude/compassionate-darwin-3r1ph6"
ZIP_URL="https://github.com/Borinoon/Edit/archive/refs/heads/${BRANCH}.zip"
PY_PKG_URL="https://www.python.org/ftp/python/3.12.7/python-3.12.7-macos11.pkg"
HOME_DIR="$HOME/LectureCut"
APP="$HOME_DIR/app"
WATCH="$HOME_DIR/원본"
RESULTS="$HOME_DIR/강의 XML"
VENV="$HOME_DIR/venv"
LABEL="com.lecturecut.watch"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG="$HOME_DIR/log.txt"

say() { printf '\n\033[1;34m▶ %s\033[0m\n' "$1"; }
die() { printf '\n\033[1;31m✖ %s\033[0m\n' "$1"; exit 1; }

[ "$(uname)" = "Darwin" ] || die "Mac 전용 설치 스크립트입니다."

# --- 1. 파이썬 ---------------------------------------------------------------
say "1/4 파이썬 확인"
PY=""
for cand in /Library/Frameworks/Python.framework/Versions/3.1*/bin/python3 /opt/homebrew/bin/python3 /usr/local/bin/python3; do
  if [ -x "$cand" ] && "$cand" -c 'import sys; assert sys.version_info >= (3, 9)' 2>/dev/null; then PY="$cand"; break; fi
done
# /usr/bin/python3 는 개발자 도구가 설치돼 있을 때만 진짜 파이썬 (아니면 설치 창만 띄움)
if [ -z "$PY" ] && xcode-select -p >/dev/null 2>&1 && /usr/bin/python3 -c 'import sys; assert sys.version_info >= (3, 9)' 2>/dev/null; then
  PY=/usr/bin/python3
fi
if [ -z "$PY" ]; then
  echo "파이썬이 없어 공식 설치 파일을 받습니다 (약 45MB)."
  TMP_PKG="$(mktemp -d)/python.pkg"
  curl -fL --progress-bar "$PY_PKG_URL" -o "$TMP_PKG" || die "파이썬 다운로드 실패. 인터넷 연결을 확인하세요."
  echo "설치하려면 Mac 로그인 암호를 입력하세요 (입력해도 화면에 안 보이는 게 정상)."
  sudo installer -pkg "$TMP_PKG" -target / </dev/tty || die "파이썬 설치 실패"
  PY="$(ls -d /Library/Frameworks/Python.framework/Versions/3.1*/bin/python3 | tail -1)"
fi
echo "파이썬: $PY ($("$PY" --version))"

# --- 2. 프로그램 + 변환 도구 -------------------------------------------------
say "2/4 프로그램 설치 → $APP"
mkdir -p "$HOME_DIR"
TMP="$(mktemp -d)"
curl -fsSL "$ZIP_URL" -o "$TMP/app.zip" || die "프로그램 다운로드 실패"
unzip -q "$TMP/app.zip" -d "$TMP"
rm -rf "$APP"
mv "$TMP"/Edit-*/ "$APP"
rm -rf "$TMP"
chmod +x "$APP"/*.command "$APP"/*.sh 2>/dev/null || true

echo "변환 도구(ffmpeg) 준비 중..."
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q imageio-ffmpeg numpy faster-whisper
"$VENV/bin/python" -c "import imageio_ffmpeg; imageio_ffmpeg.get_ffmpeg_exe()" || die "ffmpeg 준비 실패"
echo "음성 인식 모델 받는 중 (약 1.5GB, 처음 한 번만)..."
"$VENV/bin/python" -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='int8')" \
  || die "음성 인식 모델 다운로드 실패. 인터넷 연결을 확인하고 다시 설치하세요."

# --- 3. 감시 폴더 + 바탕화면 바로가기 ----------------------------------------
say "3/4 감시 폴더 만들기"
mkdir -p "$WATCH"
# 바탕화면에 바로가기 (Finder 에서 이 폴더로 영상을 끌어다 넣으면 됨)
ln -sfn "$WATCH" "$HOME/Desktop/강의 원본 넣기" 2>/dev/null \
  || echo "(바탕화면 바로가기를 못 만들었습니다. Finder 에서 $WATCH 를 직접 열어 쓰세요)"
mkdir -p "$RESULTS"
rm -f "$HOME/Desktop/강의 사본 (올릴 파일)" "$HOME/Desktop/강의 편집결과 (프리미어)"   # 이전 버전 바로가기 정리
# 결과는 ~/LectureCut/강의 XML 에 저장하고 바탕화면에는 바로가기
# (macOS 는 백그라운드 프로그램이 바탕화면에 직접 쓰는 것을 막기 때문)
ln -sfn "$RESULTS" "$HOME/Desktop/강의 XML" 2>/dev/null || true

# --- 4. 로그인 시 자동 감시 --------------------------------------------------
say "4/4 자동 감시 등록 (로그인할 때마다 자동 시작)"
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${APP}/run_watch.sh</string>
    <string>--watch</string>
    <string>${WATCH}</string>
    <string>--analyze</string>
    <string>--results</string>
    <string>${RESULTS}</string>
    <string>--notify</string>
    <string>--restart-hours</string>
    <string>6</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>${LOG}</string>
  <key>StandardErrorPath</key><string>${LOG}</string>
  <key>EnvironmentVariables</key>
  <dict><key>LANG</key><string>ko_KR.UTF-8</string></dict>
</dict>
</plist>
PLIST
launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
sleep 2
if launchctl print "gui/$(id -u)/${LABEL}" >/dev/null 2>&1; then
  echo "자동 감시 실행 중."
else
  echo "자동 감시 등록에 실패했습니다. 대신 $APP/make_proxy.command 를 더블클릭해서 쓰세요."
fi

osascript -e 'display notification "바탕화면의 \"강의 원본 넣기\" 폴더에 영상을 넣으세요" with title "강의 자동 편집기 설치 완료"' >/dev/null 2>&1 || true
open "$WATCH"

cat <<DONE

==============================================================
 ✅ 설치 완료

 사용법
   1) 바탕화면 "강의 원본 넣기" 에 강의 영상이나 영상이 든 폴더를 넣기
   2) "자동 편집 완료" 알림이 뜨면 바탕화면 "강의 XML" 폴더의
      *_라벨표시.xml 을 프리미어에서 파일 > 가져오기
      → 원본 길이 그대로, 잘라낼 후보만 색 라벨 (아무것도 자르지 않음)
        빨강=말실수·반복  주황=필러(uh/um)  파랑=침묵·숨소리  노랑=확인 필요  초록=노이즈
      *_컷목록.csv 는 표시한 구간 목록 (엑셀)
   30분 강의 기준 10~30분 걸립니다.

 - Mac 을 껐다 켜도 자동으로 감시합니다. 프로그램은 자동으로 최신 버전으로 업데이트됩니다.
 - 진행 상황: 바탕화면 "강의 XML/진행상황.txt"
 - 진행 기록: $LOG
 - 삭제: bash $APP/uninstall_mac.sh
==============================================================
DONE
