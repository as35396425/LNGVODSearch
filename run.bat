@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

set VENV_DIR=.venv

echo ============================================
echo  Podcast STT - 語音辨識啟動腳本
echo ============================================
echo.

REM ---- 1. 檢查 / 建立虛擬環境 ----
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [1/4] 找不到虛擬環境，正在建立 %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo.
        echo [錯誤] 建立虛擬環境失敗，請確認系統已安裝 Python 並加入 PATH。
        pause
        exit /b 1
    )
) else (
    echo [1/4] 虛擬環境已存在，略過建立步驟。
)

echo.
echo [2/4] 啟動虛擬環境並檢查套件...
call "%VENV_DIR%\Scripts\activate.bat"

python -c "import faster_whisper" 2>nul
if errorlevel 1 (
    echo       未偵測到 faster-whisper，正在安裝套件（首次安裝會花一些時間，請耐心等候）...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [錯誤] 套件安裝失敗，請檢查網路連線或 requirements.txt 內容。
        pause
        exit /b 1
    )
) else (
    echo       套件已安裝，略過安裝步驟。
)

echo.
echo [3/4] 載入 CUDA 執行環境變數...

REM 透過 Python 抓取 nvidia-cublas 與 nvidia-cudnn 的 bin 目錄絕對路徑
FOR /F "delims=" %%i IN ('python -c "import os, nvidia.cublas, nvidia.cudnn; print(os.path.dirname(nvidia.cublas.__file__) + '\\bin;' + os.path.dirname(nvidia.cudnn.__file__) + '\\bin')" 2^>nul') DO SET "NVIDIA_DLL_DIR=%%i"

IF DEFINED NVIDIA_DLL_DIR (
    SET "PATH=%NVIDIA_DLL_DIR%;%PATH%"
    SET "DEVICE_FLAG=--device cuda"
    echo       已將 CUDA DLL 目錄掛載至系統 PATH，啟用 GPU 加速。
) ELSE (
    SET "DEVICE_FLAG=--device cpu"
    echo       [警告] 找不到 CUDA 套件目錄，將退回 CPU 模式。
)

echo.
echo [4/4] 啟動語音辨識程式...
echo.

REM 確保 video / SubTitle 兩個資料夾存在
if not exist "video" mkdir "video"
if not exist "SubTitle" mkdir "SubTitle"

set "INPUT_FILE=%~1"

if "%INPUT_FILE%"=="" (
    echo       未指定檔案，改用批次模式：
    echo       自動轉錄 video 資料夾內所有支援的音檔/影片檔，
    echo       完成後逐字稿會輸出到 SubTitle 資料夾。
    echo.
    python transcriber.py %DEVICE_FLAG% %2 %3 %4 %5
) else (
    python transcriber.py "%INPUT_FILE%" %DEVICE_FLAG% %2 %3 %4 %5
)

echo.
echo ============================================
echo  執行完畢，逐字稿已輸出到 SubTitle 資料夾
echo ============================================
pause
