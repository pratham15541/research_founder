@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0"

REM Launcher for Streamlit Frontend using venv
echo ========================================================
echo   Starting Research Platform Streamlit Frontend
echo   URL: http://127.0.0.1:8501
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
python -m streamlit run src/ui/app.py --server.port 8501 --server.address 127.0.0.1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Frontend exited with error code %ERRORLEVEL%.
)
pause
