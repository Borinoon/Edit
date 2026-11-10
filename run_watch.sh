#!/bin/bash
# 자동 감시 실행기 (LaunchAgent 가 실행). 시작할 때마다 최신 버전으로 자동 업데이트한 뒤 감시를 시작한다.
HOME_DIR="$HOME/LectureCut"
APP="$HOME_DIR/app"
BRANCH="claude/compassionate-darwin-3r1ph6"
ZIP_URL="https://github.com/Borinoon/Edit/archive/refs/heads/${BRANCH}.zip"

TMP="$(mktemp -d)"
if curl -fsSL --max-time 60 "$ZIP_URL" -o "$TMP/app.zip" && unzip -q "$TMP/app.zip" -d "$TMP" \
   && [ -f "$TMP"/Edit-*/lecture_cut.py ]; then
  rm -rf "$APP.new" && mv "$TMP"/Edit-*/ "$APP.new" && rm -rf "$APP" && mv "$APP.new" "$APP"
  chmod +x "$APP"/*.sh "$APP"/*.command 2>/dev/null
  echo "[업데이트] $(date '+%Y-%m-%d %H:%M') 최신 버전 적용"
else
  echo "[업데이트] 실패(인터넷 없음 등) — 기존 버전으로 실행"
fi
rm -rf "$TMP"
exec "$HOME_DIR/venv/bin/python" -u "$APP/make_proxy.py" "$@"
