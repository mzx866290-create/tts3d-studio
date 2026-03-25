from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path


LOGGER_NAME = "tts3d_app"
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
HRIR_FILE = ROOT_DIR / "hrir_spatial_map.npz"
MODEL_NAME = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
DEFAULT_SERVER_NAME = os.getenv("APP_SERVER_NAME", "0.0.0.0")
TARGET_HRIR_SAMPLE_RATE = 44_100
CLEAR_CUDA_CACHE_AFTER_GENERATE = os.getenv("CLEAR_CUDA_CACHE_AFTER_GENERATE", "0") == "1"
SOX_PATHS = (
    Path(r"C:\Program Files (x86)\sox-14-4-2"),
    Path(r"C:\Program Files\sox-14-4-2"),
)


def configure_runtime() -> None:
    """Prepare runtime warnings, logging and external tool paths."""
    warnings.filterwarnings("ignore", category=UserWarning, module="qwen_tts")
    warnings.filterwarnings("ignore", message=".*flash-attn.*")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for candidate in SOX_PATHS:
        if not candidate.exists():
            continue

        current_path = os.environ.get("PATH", "")
        candidate_text = str(candidate)
        if candidate_text not in current_path:
            os.environ["PATH"] = candidate_text + os.pathsep + current_path
        break

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def get_logger(name: str | None = None) -> logging.Logger:
    logger_name = LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}"
    return logging.getLogger(logger_name)
