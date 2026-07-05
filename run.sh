#!/bin/bash
set -e
cd "$(dirname "$0")"

VENV_DIR=".venv"

echo "============================================"
echo " Podcast STT - 語音辨識啟動腳本"
echo "============================================"
echo

# ---- 1. 檢查 / 建立虛擬環境 ----
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[1/3] 找不到虛擬環境，正在建立 $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/3] 虛擬環境已存在，略過建立步驟。"
fi

echo
echo "[2/3] 啟動虛擬環境並檢查套件..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if ! python -c "import faster_whisper" 2>/dev/null; then
    echo "      未偵測到 faster-whisper，正在安裝套件（首次安裝會花一些時間，請耐心等候）..."
    pip install -r requirements.txt
else
    echo "      套件已安裝，略過安裝步驟。"
fi

echo
echo "[3/3] 啟動語音辨識程式..."
echo

INPUT_FILE="$1"

if [ -z "$INPUT_FILE" ]; then
    read -r -p "請輸入要轉錄的 mp3/mp4 檔案完整路徑：" INPUT_FILE
fi

if [ -z "$INPUT_FILE" ]; then
    echo "未輸入檔案路徑，結束程式。"
    exit 1
fi

shift || true
python transcriber.py "$INPUT_FILE" "$@"

echo
echo "============================================"
echo " 執行完畢，逐字稿已輸出到同目錄的 .txt 檔"
echo "============================================"
