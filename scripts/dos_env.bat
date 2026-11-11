@echo off
cd /d "%~dp0\.."
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
echo Done. Activate with: .venv\Scripts\activate
pause
