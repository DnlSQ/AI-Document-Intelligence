@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ============================================
    echo   Setting up AI Document Intelligence
    echo   This only happens once - it may take a
    echo   few minutes. Please wait...
    echo ============================================
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERROR: Could not create a virtual environment.
        echo Make sure Python is installed and available as "python" in PATH.
        pause
        exit /b 1
    )

    call ".venv\Scripts\activate.bat"
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies.
        pause
        exit /b 1
    )
) else (
    call ".venv\Scripts\activate.bat"
)

echo Starting AI Document Intelligence...
start "AI Document Intelligence" cmd /k python -m src.webapp
timeout /t 3 /nobreak >nul
start "" http://127.0.0.1:5000

endlocal
