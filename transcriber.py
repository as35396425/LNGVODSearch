"""
transcriber.py
================
語音辨識模組 - Podcast STT 系統核心模組（對應規格書 services/transcriber.py）

使用 faster-whisper（CTranslate2 推論引擎）搭配 Breeze-ASR-25
（MediaTek Research 針對台灣國語 / 中英夾雜情境微調的 Whisper 模型），
將指定的音檔或影片檔轉錄為帶時間軸的繁體中文逐字稿 txt 檔。

支援格式：mp3 / mp4 / wav / m4a / flac / ogg / webm
（透過 PyAV 解碼，不需另外安裝 FFmpeg，也不需先把 mp4 的音軌抽出來）

安裝相依套件：
    pip install faster-whisper

預設資料夾（與本檔案同一層，不存在會自動建立）：
    video/      放要轉錄的 mp3/mp4 等音檔/影片檔
    SubTitle/   轉錄完成的逐字稿 .txt 會輸出到這裡（檔名與輸入檔同名）

CLI 使用範例：
    # 批次模式：不帶檔名參數，自動處理 video/ 資料夾內所有支援格式的檔案，
    # 逐字稿統一輸出到 SubTitle/
    python transcriber.py

    # 單檔模式：直接指定檔案路徑，輸出到 SubTitle/episode.txt
    python transcriber.py ./data/episode.mp4

    # 單檔模式 + 指定輸出路徑（不受 SubTitle/ 限制）
    python transcriber.py ./data/episode.mp3 -o ./data/episode_transcript.txt

    # 純文字輸出（不含時間軸標記）
    python transcriber.py ./data/episode.mp4 --plain

    # 指定裝置（無 GPU 環境請用 cpu）
    python transcriber.py ./data/episode.mp4 --device cpu

    # 暫時切回原生 Whisper 模型比較效果
    python transcriber.py ./data/episode.mp4 --model large-v3-turbo
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
import os

# -----------------------------------------------------------
logger = logging.getLogger(__name__)

# ---- 設定值集中管理，避免魔術值散落在程式各處 ----

SUPPORTED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".flac", ".ogg", ".webm"}
DEFAULT_MODEL_NAME = "SoybeanMilk/faster-whisper-Breeze-ASR-25"
DEFAULT_INITIAL_PROMPT = "以下是繁體中文的 Podcast 逐字稿。"
DEFAULT_BEAM_SIZE = 5
DEFAULT_LANGUAGE = "zh"

# ---- 預設輸入 / 輸出資料夾 ----
# 未指定 input_file 時，會自動掃描 VIDEO_DIR 底下所有支援格式的檔案進行批次轉錄；
# 逐字稿一律輸出到 SUBTITLE_DIR（找不到就自動建立），檔名與輸入檔同名、副檔名改為 .txt。
# 兩個資料夾都以本檔案（transcriber.py）所在目錄為基準，不受執行時的工作目錄影響。
BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "video"
SUBTITLE_DIR = BASE_DIR / "SubTitle"


class UnsupportedAudioFormatError(Exception):
    """輸入檔案的副檔名不在支援清單內時拋出。"""


class TranscriptionError(Exception):
    """語音辨識流程中發生不可恢復的錯誤時拋出，交由呼叫端決定如何處理
    （例如記錄、標記任務狀態為 FAILED、或往上層拋出讓背景任務重試）。
    """


@dataclass(frozen=True)
class TranscriptSegment:
    """對應規格書 `transcripts` 資料表的一筆逐字稿區段。"""

    start_time: float
    end_time: float
    text: str

    def to_timestamped_line(self) -> str:
        start = _format_timestamp(self.start_time)
        end = _format_timestamp(self.end_time)
        return f"[{start} - {end}] {self.text}"


def _format_timestamp(seconds: float) -> str:
    """將秒數轉為 HH:MM:SS，方便對照音檔播放位置。"""
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def validate_input_file(file_path: Path) -> None:
    """檔案不存在或格式不支援時盡早失敗，而不是讓底層解碼器拋出難懂的錯誤訊息。"""
    if not file_path.exists():
        raise FileNotFoundError(f"找不到檔案：{file_path}")

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = "、".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedAudioFormatError(
            f"不支援的檔案格式「{file_path.suffix}」，目前支援：{supported}"
        )


class WhisperTranscriber:
    """
    封裝 faster-whisper 模型的載入與推論邏輯。

    模型載入成本高（尤其 large 系列動輒數秒到十幾秒），設計為可重複使用的物件：
    處理多個檔案時只需建立一次 WhisperTranscriber，模型留在記憶體中重複呼叫 transcribe()，
    不要每個檔案都重新 new 一次。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError(
                "缺少 faster-whisper 套件，請先執行：pip install faster-whisper"
            ) from exc

        logger.info(
            "載入模型中 model=%s device=%s compute_type=%s"
            "（首次執行會從 Hugging Face 下載模型權重，請稍候）",
            model_name,
            device,
            compute_type,
        )

        try:
            self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as exc:
            raise TranscriptionError(f"模型載入失敗：{exc}") from exc

    def transcribe(
        self,
        file_path: Path,
        initial_prompt: str = DEFAULT_INITIAL_PROMPT,
        language: str = DEFAULT_LANGUAGE,
    ) -> list[TranscriptSegment]:
        """執行語音辨識，回傳依時間排序的逐字稿區段清單。

        :raises TranscriptionError: 當底層模型推論失敗時（例如音檔損毀、解碼失敗）。
        """
        try:
            segments, info = self._model.transcribe(
                str(file_path),
                language=language,
                initial_prompt=initial_prompt,
                vad_filter=True,
                beam_size=DEFAULT_BEAM_SIZE,
            )
            # segments 是 generator，必須消費完畢才會真正觸發推論（此處就是實際耗時的地方）
            segments = list(segments)
        except Exception as exc:
            raise TranscriptionError(f"轉錄失敗（{file_path.name}）：{exc}") from exc

        logger.info(
            "偵測語言=%s（信心度 %.2f），音檔長度約 %.1f 秒，共切出 %d 段",
            info.language,
            info.language_probability,
            info.duration,
            len(segments),
        )

        return [
            TranscriptSegment(start_time=seg.start, end_time=seg.end, text=seg.text.strip())
            for seg in segments
        ]


def write_transcript(
    segments: list[TranscriptSegment],
    output_path: Path,
    include_timestamps: bool = True,
) -> None:
    """將逐字稿區段寫入 txt 檔。"""
    lines = (
        [segment.to_timestamped_line() for segment in segments]
        if include_timestamps
        else [segment.text for segment in segments]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("逐字稿已輸出 -> %s（共 %d 段）", output_path, len(segments))


def resolve_output_path(input_path: Path, output_arg: str | None) -> Path:
    """決定輸出路徑。

    - 有指定 --output：直接使用該路徑。
    - 未指定：輸出到 SUBTITLE_DIR（不存在就自動建立），檔名與輸入檔同名、副檔名改為 .txt。
    """
    if output_arg:
        return Path(output_arg)

    SUBTITLE_DIR.mkdir(parents=True, exist_ok=True)
    return SUBTITLE_DIR / input_path.with_suffix(".txt").name


def collect_video_dir_files() -> list[Path]:
    """掃描 VIDEO_DIR，回傳所有支援格式的檔案（依檔名排序）。VIDEO_DIR 不存在則自動建立。"""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        p for p in VIDEO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="將 mp3/mp4 等音檔/影片檔轉錄為繁體中文逐字稿 txt 檔。",
    )
    parser.add_argument(
        "input_file", nargs="?", default=None,
        help=(
            "輸入的 mp3/mp4 檔案路徑；省略時會改用批次模式，"
            f"自動處理 {VIDEO_DIR.name}/ 資料夾內所有支援格式的檔案"
        ),
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help=(
            "輸出 txt 路徑（僅單檔模式適用）。"
            f"預設輸出到 {SUBTITLE_DIR.name}/ 資料夾，檔名與輸入檔同名"
        ),
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_NAME,
        help=f"faster-whisper 相容模型名稱或本地路徑，預設：{DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="推論裝置，預設自動偵測（優先使用 GPU）",
    )
    parser.add_argument(
        "--plain", action="store_true",
        help="輸出純文字逐字稿，不含時間軸標記",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="顯示詳細記錄（debug log）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.input_file:
        # 單檔模式：行為與原本一致，仍可用 -o 指定輸出路徑。
        input_paths = [Path(args.input_file)]
    else:
        # 批次模式：掃描 VIDEO_DIR，逐一轉錄後輸出到 SUBTITLE_DIR。
        input_paths = collect_video_dir_files()
        if not input_paths:
            logger.error(
                "「%s」資料夾內找不到任何支援的音檔/影片檔，"
                "請把檔案放進該資料夾，或改用參數指定單一檔案路徑。",
                VIDEO_DIR,
            )
            return 1
        logger.info(
            "批次模式：在 %s 找到 %d 個檔案，逐字稿將輸出到 %s",
            VIDEO_DIR, len(input_paths), SUBTITLE_DIR,
        )

    is_batch = args.input_file is None
    transcriber: WhisperTranscriber | None = None
    exit_code = 0

    for idx, input_path in enumerate(input_paths, start=1):
        try:
            validate_input_file(input_path)

            if transcriber is None:
                transcriber = WhisperTranscriber(model_name=args.model, device=args.device)

            if is_batch:
                logger.info("[%d/%d] 開始轉錄：%s", idx, len(input_paths), input_path.name)

            segments = transcriber.transcribe(input_path)

            if not segments:
                logger.warning("未偵測到任何語音內容：%s", input_path)

            # 批次模式一律輸出到 SUBTITLE_DIR，因此忽略 -o；單檔模式才套用 -o。
            output_path = resolve_output_path(input_path, None if is_batch else args.output)
            write_transcript(segments, output_path, include_timestamps=not args.plain)

        except (FileNotFoundError, UnsupportedAudioFormatError) as exc:
            logger.error(str(exc))
            exit_code = 1
        except TranscriptionError as exc:
            logger.error(str(exc))
            exit_code = 2

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
