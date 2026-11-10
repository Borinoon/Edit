#!/bin/bash
# 강의 사본 변환기 삭제 (감시 폴더의 영상/사본 파일은 지우지 않음)
LABEL="com.lecturecut.watch"
launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/${LABEL}.plist"
rm -f "$HOME/Desktop/강의 원본 넣기" "$HOME/Desktop/강의 사본 (올릴 파일)"   # 바로가기만 삭제
rm -rf "$HOME/LectureCut/app" "$HOME/LectureCut/venv" "$HOME/LectureCut/log.txt"
echo "삭제했습니다. 영상 파일은 $HOME/LectureCut/원본 에 그대로 남아 있습니다."
