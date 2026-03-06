@echo off
cd /d "%~dp0\.."
echo ====================================
echo  AiVoice RVC WebSocket Server
echo ====================================

:: setup 여부 확인 - python/ 없으면 자동 설치
if not exist "python\python.exe" (
    echo [SETUP] Python environment not found. Running setup...
    echo.
    call "%~dp0setup_env.bat"
    if errorlevel 1 (
        echo [ERROR] Setup failed or cancelled.
        pause
        exit /b 1
    )
)

echo [BOOT] Starting server...
echo [BOOT] Server root: %cd%
python\python.exe core\rvc_ws_server.py

if errorlevel 1 (
    echo [ERROR] Server exited with error. Check logs above.
    pause
)