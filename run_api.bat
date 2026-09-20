@echo off
setlocal EnableDelayedExpansion
title ResearchGapAI - FastAPI Backend (Port 8000)
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
echo            ResearchGapAI - FastAPI Backend Service
echo ===============================================================================
echo  Virtual Env : %VENV_DIR%
echo  Host URL    : http://127.0.0.1:8000
echo  Swagger Docs: http://127.0.0.1:8000/docs
echo  Health Check: http://127.0.0.1:8000/health
echo ===============================================================================
echo.

"%PYTHON_EXE%" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] FastAPI backend stopped unexpectedly with error code %ERRORLEVEL%.
    pause
)
