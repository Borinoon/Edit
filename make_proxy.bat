@echo off
chcp 65001 >nul
rem 영상 파일(또는 폴더)을 이 아이콘 위로 끌어다 놓으면 저화질 사본을 만듭니다.
python "%~dp0make_proxy.py" --pause %*
