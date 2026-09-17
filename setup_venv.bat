@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0"

REM Setup Script for Virtual Environment without Docker
echo ========================================================
echo   Setting up Virtual Environment for Research AI Agent
echo ========================================================

REM 1. Check Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in PATH!
    echo Please install Python 3.11 or 3.12 and check "Add Python to PATH".
    pause
    exit /b 1
)

REM 2. Create venv if not existing
if not exist "venv" (
    echo [1/3] Creating virtual environment in .\venv ...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment .\venv already exists.
)

REM 3. Activate venv and upgrade pip
echo [2/3] Activating virtual environment and updating pip...
call .\venv\Scripts\activate.bat
python -m pip install --upgrade pip

REM 4. Install dependencies
echo [3/3] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Dependency installation encountered an issue.
    pause
    exit /b 1
)

echo ========================================================
echo   Setup Complete!
echo   You can now launch the application using:
echo     - run_all.bat      (Single click: launches both services)
echo     - run_backend.bat  (Terminal 1: FastAPI on http://127.0.0.1:8000)
echo     - run_frontend.bat (Terminal 2: Streamlit on http://127.0.0.1:8501)
echo ========================================================
pause
