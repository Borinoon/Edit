#!/bin/bash
# Mac: 더블클릭하면 이 파일 옆 "원본" 폴더를 감시합니다. 영상을 넣으면 자동으로 저화질 사본 생성.
cd "$(dirname "$0")"
mkdir -p "원본"
python3 make_proxy.py --watch "원본"
