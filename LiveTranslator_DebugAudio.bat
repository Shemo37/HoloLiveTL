@echo off
title Live Audio Translator (audio debug)
cd /d "%~dp0"

REM Same as LiveTranslator.bat, but dumps every ASR chunk to debug_audio\ as WAV
set HOLOTL_DEBUG_AUDIO=1

if not exist "main.py" (
    echo Error: main.py not found
    pause
    exit /b 1
)

python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    pause
    exit /b 1
)

echo Starting Live Audio Translator with audio debug dump...
python main.py

if errorlevel 1 (
    echo.
    echo Application error occurred
    pause
)
