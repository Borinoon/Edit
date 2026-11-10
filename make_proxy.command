#!/bin/bash
# Mac: 더블클릭하면 이 파일 옆 "원본" 폴더를 감시합니다.
# 영상을 "원본" 폴더에 넣으면 자동으로 "원본/proxy" 에 저화질 사본이 생깁니다.
cd "$(dirname "$0")" || exit 1

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 가 없습니다. https://www.python.org/downloads/ 에서 설치한 뒤 다시 실행하세요."
  read -r -p "엔터를 누르면 닫힙니다..."
  exit 1
fi

# ffmpeg 가 없고 imageio-ffmpeg 도 없으면 처음 한 번 자동 설치
if ! command -v ffmpeg >/dev/null 2>&1 && ! python3 -c "import imageio_ffmpeg" >/dev/null 2>&1; then
  echo "처음 실행: 변환 도구(imageio-ffmpeg)를 설치합니다..."
  python3 -m pip install --user imageio-ffmpeg || {
    echo "설치 실패. 터미널에서 'python3 -m pip install imageio-ffmpeg' 를 직접 실행해 보세요."
    read -r -p "엔터를 누르면 닫힙니다..."
    exit 1
  }
fi

mkdir -p "원본"
open "원본"   # Finder 에서 감시 폴더를 열어 줌 → 여기에 영상을 끌어다 넣으면 됨
python3 make_proxy.py --watch "원본"
