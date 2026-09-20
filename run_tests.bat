@echo off
setlocal EnableDelayedExpansion
title ResearchGapAI - Test Suite
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
    echo [ERROR] Virtual environment not found in %~dp0venv or %~dp0.venv!
    pause
    exit /b 1
)

set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
call "%VENV_DIR%\Scripts\activate.bat"

echo ===============================================================================
echo            ResearchGapAI - Pytest Test Suite Runner
echo ===============================================================================
echo  Virtual Env : %VENV_DIR%
echo ===============================================================================
echo.

if "%~1"=="" (
    "%PYTHON_EXE%" -m pytest tests/ -v
) else (
    "%PYTHON_EXE%" -m pytest %*
)

echo.
echo ===============================================================================
echo  Tests finished with exit code %ERRORLEVEL%.
echo ===============================================================================
echo.
pause
