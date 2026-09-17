@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0"

REM Launcher for FastAPI Backend using venv
echo ========================================================
echo   Starting Research Platform FastAPI Backend
echo   URL:  http://127.0.0.1:8000
echo   Docs: http://127.0.0.1:8000/docs
echo ========================================================

if not exist "venv\Scripts\activate.bat" (
    echo [WARNING] Virtual environment 'venv' not found!
    echo Launching setup_venv.bat to initialize environment...
    call setup_venv.bat
    if not exist "venv\Scripts\activate.bat" (
        echo [ERROR] Failed to initialize virtual environment.
        pause
        exit /b 1
    )
)

call .\venv\Scripts\activate.bat
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Backend exited with error code %ERRORLEVEL%.
)
pause
