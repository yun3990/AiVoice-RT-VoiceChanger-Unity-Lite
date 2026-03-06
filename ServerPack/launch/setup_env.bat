@echo off
cd /d "%~dp0\.."
echo ====================================
echo  AiVoice Environment Setup
echo ====================================

if exist "python\python.exe" (
    echo [OK] Python already exists. Skipping download.
    goto install_packages
)

echo  This setup will download and install the following:
echo  - Python 3.10.11 (embedded^)
echo  - PyTorch 2.5.1 with CUDA 12.4
echo  - Additional open-source libraries (see OPEN_SOURCE_LICENSES.txt^)
echo.
echo  By proceeding, you agree to the license terms of all included
echo  third-party libraries listed in OPEN_SOURCE_LICENSES.txt.
echo.
set /p AGREE=Do you agree to the license terms? (Y/N): 
if /i not "%AGREE%"=="Y" (
    echo [CANCELLED] Setup cancelled by user.
    pause
    exit /b 1
)
echo.

echo [SETUP] Downloading Python 3.10.11 embedded...
mkdir python 2>nul
powershell -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip' -OutFile 'python_embed.zip' -UseBasicParsing"
if errorlevel 1 (
    echo [ERROR] Download failed. Check internet connection.
    pause
    exit /b 1
)

echo [SETUP] Extracting Python...
powershell -ExecutionPolicy Bypass -Command "Expand-Archive -Path 'python_embed.zip' -DestinationPath 'python' -Force"
del python_embed.zip

echo [SETUP] Enabling site-packages...
powershell -ExecutionPolicy Bypass -Command "(Get-Content 'python\python310._pth') -replace '#import site', 'import site' | Set-Content 'python\python310._pth'"

echo [SETUP] Installing pip...
powershell -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py' -UseBasicParsing"
python\python.exe get-pip.py --quiet --no-warn-script-location
del get-pip.py

echo [SETUP] Downgrading pip for compatibility...
python\python.exe -m pip install "pip==23.3.2" --quiet --no-warn-script-location

:install_packages
echo [SETUP] Installing torch with CUDA 12.4 (may take 5-10 min)...
python\python.exe -m pip install torch==2.5.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124 --quiet --no-warn-script-location
if errorlevel 1 (
    echo [ERROR] torch install failed.
    pause
    exit /b 1
)

echo [SETUP] Installing fairseq from wheels...
python\python.exe -m pip install wheels\fairseq-0.12.2-cp310-cp310-win_amd64.whl --no-deps --no-warn-script-location
if errorlevel 1 (
    echo [ERROR] fairseq install failed.
    pause
    exit /b 1
)

echo [SETUP] Installing onnxruntime-gpu...
python\python.exe -m pip install onnxruntime-gpu==1.23.2 --no-deps --no-warn-script-location
if errorlevel 1 (
    echo [ERROR] onnxruntime-gpu install failed.
    pause
    exit /b 1
)

echo [SETUP] Installing remaining packages from wheels...
python\python.exe -m pip install --no-index --find-links=wheels --no-warn-script-location -r core\requirements.txt
if errorlevel 1 (
    echo [ERROR] Package installation failed.
    pause
    exit /b 1
)

echo [SETUP] Checking base models...
if not exist "models\base\hubert_base.pt" (
    echo [ERROR] hubert_base.pt not found in models/base/.
    echo [ERROR] Please ensure base models are included in the ServerPack.
    pause
    exit /b 1
) else (
    echo [OK] hubert_base.pt found.
)

if not exist "models\base\rmvpe.onnx" (
    echo [ERROR] rmvpe.onnx not found in models/base/.
    echo [ERROR] Please ensure base models are included in the ServerPack.
    pause
    exit /b 1
) else (
    echo [OK] rmvpe.onnx found.
)

echo.
echo [OK] Setup complete. Run start_server.bat to start.
pause