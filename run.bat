@echo off
rem Start the server (settings in .env). Opens the UI in the browser; API docs at /docs.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo .venv not found - run install.bat first.
  exit /b 1
)
set PORT_UI=7860
for /f "tokens=1,* delims==" %%a in ('findstr /b "PORT=" .env 2^>nul') do set PORT_UI=%%b
start "" http://127.0.0.1:%PORT_UI%
.venv\Scripts\python app.py
