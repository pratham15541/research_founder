@echo off
setlocal EnableDelayedExpansion
title ResearchGapAI - Streamlit UI (Port 8501)
cd /d "%~dp0"

:: Ensure project root is in PYTHONPATH
set "PYTHONPATH=%~dp0;!PYTHONPATH!"

:: Locate Virtual Environment
set "VENV_DIR="
if exist "%~dp0venv\Scripts\activate.bat" (
    set "VENV_DIR=%~dp0venv"
) else if exist "%~dp0.venv\Scripts\activate.bat" (
    set "VENV_DIR=%~dp0.venv"
)

if "%VENV_DIR%"=="" (
    echo.
    echo ===============================================================================
    echo [ERROR] Virtual environment not found in "%~dp0venv" or "%~dp0.venv"!
    echo ===============================================================================
    echo Please create a virtual environment first:
    echo     python -m venv venv
    echo     venv\Scripts\pip install -r requirements.txt
    echo ===============================================================================
    echo.
    pause
    exit /b 1
)

set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
call "%VENV_DIR%\Scripts\activate.bat"

echo ===============================================================================
echo            ResearchGapAI - Streamlit Intelligence Dashboard
echo ===============================================================================
echo  Virtual Env : %VENV_DIR%
echo  Local URL   : http://127.0.0.1:8501
echo ===============================================================================
echo.

"%PYTHON_EXE%" -m streamlit run src/ui/app.py --server.port=8501 --server.address=127.0.0.1

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Streamlit UI stopped unexpectedly with error code %ERRORLEVEL%.
    pause
)
