@echo off
title Live Audio Translator
cd /d "%~dp0"

REM Check if main.py exists
if not exist "main.py" (
    echo Error: main.py not found
    pause
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    pause
    exit /b 1
)

REM Run the application
echo Starting Live Audio Translator...
python main.py

if errorlevel 1 (
    echo.
    echo Application error occurred
    pause
)