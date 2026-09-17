@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0"

echo ========================================================
echo   ResearchGapAI — Launching Research Discovery Suite
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

echo [1/2] Launching FastAPI Backend on http://127.0.0.1:8000 ...
start "ResearchGapAI Backend (FastAPI)" cmd /k "cd /d "%~dp0" && set "PYTHONPATH=%~dp0" && call .\venv\Scripts\activate.bat && python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 2 /nobreak >nul 2>&1

echo [2/2] Launching Streamlit Frontend on http://127.0.0.1:8501 ...
start "ResearchGapAI Frontend (Streamlit)" cmd /k "cd /d "%~dp0" && set "PYTHONPATH=%~dp0" && call .\venv\Scripts\activate.bat && python -m streamlit run src/ui/app.py --server.port 8501 --server.address 127.0.0.1"

echo ========================================================
echo   All Services Launched!
echo   - Backend:  http://127.0.0.1:8000 (API Docs: /docs)
echo   - Frontend: http://127.0.0.1:8501
echo ========================================================
echo Close this window or press any key to exit (services keep running).
pause >nul
