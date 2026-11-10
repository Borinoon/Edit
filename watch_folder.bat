@echo off
chcp 65001 >nul
rem 더블클릭하면 이 파일이 있는 폴더의 "원본" 폴더를 감시합니다. 영상을 넣기만 하면 자동 변환.
if not exist "%~dp0원본" mkdir "%~dp0원본"
python "%~dp0make_proxy.py" --pause --watch "%~dp0원본"
