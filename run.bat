@echo off
setlocal EnableDelayedExpansion
title ResearchGapAI Launcher
cd /d "%~dp0"

:: -----------------------------------------------------------------------------
:: Locate Virtual Environment (checks "venv" then ".venv")
:: -----------------------------------------------------------------------------
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
set "PYTHONPATH=%~dp0;!PYTHONPATH!"
call "%VENV_DIR%\Scripts\activate.bat"

:: Ensure .env exists
if not exist "%~dp0.env" (
    if exist "%~dp0.env.example" (
        echo [INFO] .env not found. Creating default .env from .env.example...
        copy "%~dp0.env.example" "%~dp0.env" >nul
        echo [INFO] Created .env. Please configure any necessary API keys.
    )
)

:: -----------------------------------------------------------------------------
:: Command-Line Argument Dispatcher
:: -----------------------------------------------------------------------------
if /i "%~1"=="all" goto start_both
if /i "%~1"=="both" goto start_both
if /i "%~1"=="ui" goto start_ui
if /i "%~1"=="frontend" goto start_ui
if /i "%~1"=="streamlit" goto start_ui
if /i "%~1"=="api" goto start_api
if /i "%~1"=="backend" goto start_api
if /i "%~1"=="fastapi" goto start_api
if /i "%~1"=="test" goto run_tests
if /i "%~1"=="tests" goto run_tests
if /i "%~1"=="pytest" goto run_tests
if /i "%~1"=="e2e" goto run_e2e
if /i "%~1"=="docs" goto open_docs
if /i "%~1"=="browser" goto open_browser

:: -----------------------------------------------------------------------------
:: Interactive Selection Menu
:: -----------------------------------------------------------------------------
:menu
cls
echo ===============================================================================
echo            Research Landscape and Gap Matrix Platform (ResearchGapAI)
echo ===============================================================================
echo  Virtual Environment : %VENV_DIR%
echo  Python Interpreter  : %PYTHON_EXE%
echo.
echo   [1] Start Full Stack (FastAPI Backend + Streamlit UI in new windows) [Default in 5s]
echo   [2] Start Streamlit UI Only  (http://127.0.0.1:8501)
echo   [3] Start FastAPI Backend Only  (http://127.0.0.1:8000)
echo   [4] Run Full Test Suite  (all 47 tests via pytest)
echo   [5] Run End-to-End Workflow Test  (test_workflow_e2e.py)
echo   [6] Open Streamlit UI in Browser  (http://127.0.0.1:8501)
echo   [7] Open API Docs in Browser  (http://127.0.0.1:8000/docs)
echo   [8] Exit
echo ===============================================================================
echo.

where choice >nul 2>&1
if %ERRORLEVEL% equ 0 (
    choice /c 12345678 /t 5 /d 1 /m "Select option"
    set "CHOICE_CODE=!ERRORLEVEL!"
    if "!CHOICE_CODE!"=="1" goto start_both
    if "!CHOICE_CODE!"=="2" goto start_ui
    if "!CHOICE_CODE!"=="3" goto start_api
    if "!CHOICE_CODE!"=="4" goto run_tests
    if "!CHOICE_CODE!"=="5" goto run_e2e
    if "!CHOICE_CODE!"=="6" goto open_browser
    if "!CHOICE_CODE!"=="7" goto open_docs
    if "!CHOICE_CODE!"=="8" exit /b 0
) else (
    set /p "USER_CHOICE=Select an option [1-8] (default=1): "
    if "!USER_CHOICE!"=="" set "USER_CHOICE=1"
    if "!USER_CHOICE!"=="1" goto start_both
    if "!USER_CHOICE!"=="2" goto start_ui
    if "!USER_CHOICE!"=="3" goto start_api
    if "!USER_CHOICE!"=="4" goto run_tests
    if "!USER_CHOICE!"=="5" goto run_e2e
    if "!USER_CHOICE!"=="6" goto open_browser
    if "!USER_CHOICE!"=="7" goto open_docs
    if "!USER_CHOICE!"=="8" exit /b 0
    echo [WARNING] Invalid selection. Please enter 1, 2, 3, 4, 5, 6, 7, or 8.
    timeout /t 2 >nul
    goto menu
)

:: -----------------------------------------------------------------------------
:: Actions
:: -----------------------------------------------------------------------------
:start_both
echo.
echo ===============================================================================
echo  Launching Full Stack in Separate Windows...
echo ===============================================================================
echo  - Backend API : http://127.0.0.1:8000 (Docs: http://127.0.0.1:8000/docs)
echo  - Frontend UI : http://127.0.0.1:8501
echo ===============================================================================
echo.

echo [1/2] Launching FastAPI Backend (Port 8000)...
start "ResearchGapAI - FastAPI Backend (Port 8000)" "%~dp0run_api.bat"

echo Waiting 2 seconds for backend initialization...
timeout /t 2 /nobreak >nul 2>&1

echo [2/2] Launching Streamlit Frontend (Port 8501)...
start "ResearchGapAI - Streamlit UI (Port 8501)" "%~dp0run_ui.bat"

echo.
echo ===============================================================================
echo  [SUCCESS] Both services are running in dedicated terminal windows!
echo ===============================================================================
echo  - Interactive Dashboard: http://127.0.0.1:8501
echo  - FastAPI Swagger Docs:  http://127.0.0.1:8000/docs
echo ===============================================================================
echo You can close this launcher window or press any key to exit.
echo.
pause
exit /b 0

:start_ui
echo.
echo ===============================================================================
echo  Starting Streamlit UI (http://127.0.0.1:8501)
echo ===============================================================================
echo.
"%PYTHON_EXE%" -m streamlit run src/ui/app.py --server.port=8501 --server.address=127.0.0.1
pause
exit /b 0

:start_api
echo.
echo ===============================================================================
echo  Starting FastAPI Backend (http://127.0.0.1:8000)
echo  Swagger Docs: http://127.0.0.1:8000/docs
echo ===============================================================================
echo.
"%PYTHON_EXE%" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
pause
exit /b 0

:run_tests
echo.
echo ===============================================================================
echo  Running Pytest Test Suite
echo ===============================================================================
echo.
"%PYTHON_EXE%" -m pytest tests/ -v
echo.
pause
exit /b 0

:run_e2e
echo.
echo ===============================================================================
echo  Running End-to-End Workflow Integration Test
echo ===============================================================================
echo.
"%PYTHON_EXE%" -m pytest tests/test_workflow_e2e.py -v
echo.
pause
exit /b 0

:open_browser
echo Opening Streamlit UI in default browser...
start http://127.0.0.1:8501
timeout /t 2 >nul
goto menu

:open_docs
echo Opening Swagger API Documentation in default browser...
start http://127.0.0.1:8000/docs
timeout /t 2 >nul
goto menu
