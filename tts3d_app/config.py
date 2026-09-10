from __future__ import annotations

import importlib.util
import json
import logging
import os
import platform
import shutil
import warnings
from pathlib import Path
from typing import Any


LOGGER_NAME = "tts3d_app"
ROOT_DIR = Path(__file__).resolve().parent.parent
_OUTPUT_DIR_ENV = os.getenv("TTS3D_OUTPUT_DIR", "")
OUTPUT_DIR = Path(_OUTPUT_DIR_ENV).resolve() if _OUTPUT_DIR_ENV else ROOT_DIR / "outputs"
HRIR_FILE = ROOT_DIR / "hrir_spatial_map.npz"
LOCAL_FFMPEG_BIN = ROOT_DIR / "tools" / "ffmpeg" / "bin"
QWEN_MODEL_NAME = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
QWEN_BASE_MODEL_NAME = os.getenv("QWEN_TTS_BASE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
MODEL_NAME = QWEN_MODEL_NAME
DEFAULT_SERVER_NAME = os.getenv("APP_SERVER_NAME", "127.0.0.1")
TARGET_HRIR_SAMPLE_RATE = 44_100
CLEAR_CUDA_CACHE_AFTER_GENERATE = os.getenv("CLEAR_CUDA_CACHE_AFTER_GENERATE", "0") == "1"
ENABLE_TORCH_COMPILE = os.getenv("ENABLE_TORCH_COMPILE", "1") == "1"
TORCH_COMPILE_MODE = os.getenv("TORCH_COMPILE_MODE", "reduce-overhead")
DRY_AUDIO_CACHE_SIZE = max(int(os.getenv("DRY_AUDIO_CACHE_SIZE", "32")), 0)
ENABLE_GPU_FFT_CONVOLUTION = os.getenv("ENABLE_GPU_FFT_CONVOLUTION", "1") == "1"
ENABLE_NUMBA_DYNAMIC_HRIR = os.getenv("ENABLE_NUMBA_DYNAMIC_HRIR", "1") == "1"
ENABLE_GPU_PITCH_SHIFT = os.getenv("ENABLE_GPU_PITCH_SHIFT", "1") == "1"
# Long-form TTS: Qwen3-TTS one-shot generation is capped at max_new_tokens (default 2048, ~2.7 min).
MAX_TTS_CHUNK_CHARS = max(int(os.getenv("MAX_TTS_CHUNK_CHARS", "400")), 50)
TTS_MAX_NEW_TOKENS = max(int(os.getenv("TTS_MAX_NEW_TOKENS", "2048")), 256)
TTS_CHUNK_SENTENCE_PAUSE_MS = max(int(os.getenv("TTS_CHUNK_SENTENCE_PAUSE_MS", "300")), 0)
TTS_CHUNK_PARAGRAPH_PAUSE_MS = max(int(os.getenv("TTS_CHUNK_PARAGRAPH_PAUSE_MS", "700")), 0)
# VoiceDesign re-samples speaker per call. A fixed calibration sentence is
# synthesized once per (model, prompt, seed) and used as an ICL clone prompt
# for every content chunk, including single-chunk text.
# 0 disables speaker lock (each chunk stays independent VoiceDesign).
TTS_SPEAKER_REF_MAX_CHARS = max(int(os.getenv("TTS_SPEAKER_REF_MAX_CHARS", "60")), 0)
TTS_VOICE_CALIBRATION_TEXT = os.getenv(
    "TTS_VOICE_CALIBRATION_TEXT",
    "唉……她轻轻叹了口气，声音温柔得像耳语：今晚的月色真美啊……风也很温柔呢。",
)
# Hugging Face model cache directory. On Windows this defaults to E:\AI_Models\huggingface;
# on macOS/Linux it defaults to the standard HF cache under the user's home directory. Users can
# still override via HF_HOME / HF_HUB_CACHE / HUGGINGFACE_HUB_CACHE environment variables.
if os.name == "nt":
    HUGGINGFACE_CACHE_ROOT = Path(os.environ.get("HF_HOME", r"E:\AI_Models\huggingface"))
else:
    HUGGINGFACE_CACHE_ROOT = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
HUGGINGFACE_HUB_CACHE = HUGGINGFACE_CACHE_ROOT / "hub"
HF_HOME = HUGGINGFACE_CACHE_ROOT
HF_HUB_CACHE = HUGGINGFACE_HUB_CACHE

# SoX binary discovery. On Windows this looks for the packaged sox installs; elsewhere it relies
# on PATH lookups so self-built/Homebrew installs are found automatically.
if os.name == "nt":
    SOX_PATHS = (
        Path(r"C:\Program Files (x86)\sox-14-4-2"),
        Path(r"C:\Program Files\sox-14-4-2"),
    )
else:
    SOX_PATHS = (Path(shutil.which("sox") or ""),)
FLASH_ATTN_INSTALLED = importlib.util.find_spec("flash_attn") is not None
FASTMCP_INSTALLED = importlib.util.find_spec("fastmcp") is not None
NUMBA_INSTALLED = importlib.util.find_spec("numba") is not None
FASTER_WHISPER_INSTALLED = importlib.util.find_spec("faster_whisper") is not None
OPENAI_WHISPER_INSTALLED = importlib.util.find_spec("whisper") is not None
ASR_AVAILABLE = FASTER_WHISPER_INSTALLED or OPENAI_WHISPER_INSTALLED
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

_RUNTIME_CONFIGURED = False


def _prepend_to_path(path_entry: Path) -> None:
    if not path_entry.exists():
        return

    current_path = os.environ.get("PATH", "")
    path_text = str(path_entry)
    if path_text not in current_path:
        os.environ["PATH"] = path_text + os.pathsep + current_path


def _configure_pydub_via_env() -> None:
    """Configure pydub via environment variables instead of monkey-patching."""
    if not LOCAL_FFMPEG_BIN.exists():
        return

    exe_suffix = ".exe" if os.name == "nt" else ""
    ffmpeg_exe = LOCAL_FFMPEG_BIN / f"ffmpeg{exe_suffix}"
    ffprobe_exe = LOCAL_FFMPEG_BIN / f"ffprobe{exe_suffix}"
    if not ffmpeg_exe.exists() or not ffprobe_exe.exists():
        return

    os.environ["FFMPEG_BINARY"] = str(ffmpeg_exe)
    os.environ["FFPROBE_BINARY"] = str(ffprobe_exe)


def configure_runtime() -> None:
    """Prepare runtime warnings, logging, cache paths and external tool paths."""
    global _RUNTIME_CONFIGURED

    if _RUNTIME_CONFIGURED:
        return

    warnings.filterwarnings("ignore", category=UserWarning, module="qwen_tts")
    warnings.filterwarnings("ignore", message=".*flash-attn.*")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HUGGINGFACE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    HUGGINGFACE_HUB_CACHE.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HUB_CACHE)
    os.environ["TRANSFORMERS_CACHE"] = str(HF_HUB_CACHE)

    _prepend_to_path(LOCAL_FFMPEG_BIN)
    for candidate in SOX_PATHS:
        if not candidate.exists():
            continue
        _prepend_to_path(candidate)
        break

    _configure_pydub_via_env()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _RUNTIME_CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    logger_name = LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}"
    return logging.getLogger(logger_name)


USER_CONFIG_DIR = Path.home() / ".tts3d_studio"
USER_CONFIG_FILE = USER_CONFIG_DIR / "config.json"
PRESETS_FILE = USER_CONFIG_DIR / "presets.json"
FAVORITES_FILE = USER_CONFIG_DIR / "favorites.json"
VOICE_PROFILE_DIR = USER_CONFIG_DIR / "voices"


def _ensure_user_config_dir() -> None:
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_user_config() -> dict[str, Any]:
    """加载用户配置，无则返回空字典"""
    if not USER_CONFIG_FILE.exists():
        return {}
    try:
        with open(USER_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_user_config(config: dict[str, Any]) -> None:
    """保存用户配置"""
    try:
        _ensure_user_config_dir()
        with open(USER_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        get_logger("config").warning("Failed to save user config: %s", exc)


def load_user_presets() -> dict[str, dict[str, str]]:
    """加载用户自定义预设"""
    if not PRESETS_FILE.exists():
        return {}
    try:
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_user_presets(presets: dict[str, dict[str, str]]) -> None:
    """保存用户自定义预设"""
    try:
        _ensure_user_config_dir()
        with open(PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(presets, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        get_logger("config").warning("Failed to save user presets: %s", exc)


def load_user_favorites() -> dict[str, dict[str, Any]]:
    """加载用户收藏的声线，无则返回空字典"""
    if not FAVORITES_FILE.exists():
        return {}
    try:
        with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_user_favorites(favorites: dict[str, dict[str, Any]]) -> None:
    """保存用户收藏的声线"""
    try:
        _ensure_user_config_dir()
        with open(FAVORITES_FILE, "w", encoding="utf-8") as f:
            json.dump(favorites, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        get_logger("config").warning("Failed to save user favorites: %s", exc)

