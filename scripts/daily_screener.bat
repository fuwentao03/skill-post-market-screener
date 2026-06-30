@echo off
REM Daily Post-Market Screener — Windows Task Scheduler entry point
REM Schedule: weekdays at 15:37 Asia/Shanghai (30+ min after A-share close)
REM
REM To install:
REM   schtasks /create /tn "PostMarketScreener" /tr "%~dp0daily_screener.bat"
REM     /sc weekly /d MON,TUE,WED,THU,FRI /st 15:37 /f
REM
REM Log is written alongside this script: daily_screener.log

setlocal
set "SKILL_ROOT=%~dp0.."
set "LOG_FILE=%~dp0daily_screener.log"
set "PYTHON_EXE=python"

echo [%date% %time%] Starting post-market screener ... >> "%LOG_FILE%"

cd /d "%SKILL_ROOT%"

%PYTHON_EXE% run.py >> "%LOG_FILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% equ 0 (
    echo [%date% %time%] Screener completed successfully (exit 0) >> "%LOG_FILE%"
) else (
    echo [%date% %time%] Screener failed with exit code %EXIT_CODE% >> "%LOG_FILE%"
)

exit /b %EXIT_CODE%
