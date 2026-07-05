Podcast STT - 語音辨識模組使用說明
====================================

資料夾結構
----------
transcriber.py     語音辨識核心程式（faster-whisper + Breeze-ASR-25）
requirements.txt   Python 相依套件清單
run.bat             Windows 啟動腳本（雙擊執行）
run.sh              macOS / Linux 啟動腳本
video/              放要轉錄的 mp3/mp4 等音檔/影片檔（不存在會自動建立）
SubTitle/           轉錄完成的逐字稿 .txt 會輸出到這裡（不存在會自動建立）

兩個啟動腳本做的事情完全一樣：
  1. 檢查有沒有虛擬環境 (.venv)，沒有就自動建立
  2. 啟動虛擬環境，檢查 faster-whisper 是否已安裝，沒裝就自動 pip install
  3. 啟動 transcriber.py 進行轉錄

------------------------------------
Windows 使用方式
------------------------------------
方式一（推薦，批次處理）：
  1. 把要轉錄的 mp3/mp4 檔案都放進 video 資料夾
  2. 直接雙擊 run.bat（不用帶任何參數）
  3. 會自動依序轉錄 video 資料夾內的每一個檔案，
     逐字稿統一輸出到 SubTitle 資料夾，檔名與原始檔案同名

方式二（單一檔案）：把要轉錄的 mp3/mp4 檔案直接拖到 run.bat 圖示上放開，
  一樣會輸出到 SubTitle 資料夾。

第一次執行會比較久：
  - 建立虛擬環境 + 安裝套件約 1-2 分鐘
  - 下載 Breeze-ASR-25 模型約 3GB，視網速可能要幾分鐘到十幾分鐘
之後重複執行就不會再重複這兩步，因為 .venv 跟模型快取都已經建立好了。

------------------------------------
macOS / Linux 使用方式
------------------------------------
終端機內執行：

    chmod +x run.sh      # 第一次執行前，先給予執行權限

    # 批次模式：把檔案放進 video 資料夾後，不帶參數直接執行
    ./run.sh

    # 單一檔案模式
    ./run.sh /path/to/episode.mp4

------------------------------------
進階參數
------------------------------------
啟動腳本後面可以照樣加 transcriber.py 支援的參數，例如：

    run.bat "C:\podcast\ep01.mp4" --plain
    ./run.sh ./podcast/ep01.mp3 --device cpu

完整參數說明，可在虛擬環境啟動後執行：
    python transcriber.py --help

------------------------------------
注意事項
------------------------------------
- 執行環境需能連上網路下載模型（只有第一次需要）。如果是完全離線的機器，
  請先在其他能上網的機器跑過一次，再把模型快取資料夾整個複製過去：
    Windows: C:\Users\<你的帳號>\.cache\huggingface
    macOS/Linux: ~/.cache/huggingface

- 沒有 GPU 也能跑，只是速度比較慢。run.bat 會自動偵測有沒有裝 GPU
  函式庫（nvidia-cublas-cu12 / nvidia-cudnn-cu12），沒裝就自動以
  --device cpu 執行，不會再因為「偵測到 GPU 但缺 DLL」而崩潰
  （錯誤訊息類似 "Library cublas64_12.dll is not found"）。

- 想啟用 GPU 加速：打開 requirements.txt，取消註解
  nvidia-cublas-cu12 / nvidia-cudnn-cu12==9.* 這兩行，重新執行
  run.bat。腳本會自動安裝套件，並自動把對應的 DLL 路徑加進 PATH，
  不需要手動設定環境變數。注意 cuDNN 版本務必是 9.*，裝到 8.x
  會出現另一個類似的 DLL 找不到的錯誤。
