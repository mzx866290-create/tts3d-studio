from __future__ import annotations

from pathlib import Path
from typing import Any

from tts3d_app.config import (
    FASTER_WHISPER_INSTALLED,
    HF_HUB_CACHE,
    OPENAI_WHISPER_INSTALLED,
    WHISPER_MODEL_SIZE,
    get_logger,
)


LOGGER = get_logger("asr")

_model: Any = None
_model_backend: str = ""


def _load_model() -> tuple[Any, str]:
    global _model, _model_backend
    if _model is not None:
        return _model, _model_backend

    if FASTER_WHISPER_INSTALLED:
        from faster_whisper import WhisperModel

        LOGGER.info("Loading faster-whisper model: %s", WHISPER_MODEL_SIZE)
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="auto",
            compute_type="auto",
            download_root=str(HF_HUB_CACHE),
        )
        _model_backend = "faster_whisper"
        return _model, _model_backend

    if OPENAI_WHISPER_INSTALLED:
        import whisper

        LOGGER.info("Loading openai-whisper model: %s", WHISPER_MODEL_SIZE)
        _model = whisper.load_model(WHISPER_MODEL_SIZE, download_root=str(HF_HUB_CACHE))
        _model_backend = "openai_whisper"
        return _model, _model_backend

    raise RuntimeError(
        "No ASR backend available. "
        "Install faster-whisper: pip install faster-whisper"
    )


def transcribe_audio(audio_path: str | Path, language: str | None = None) -> str:
    model, backend = _load_model()
    audio_path_str = str(Path(audio_path).resolve())

    if backend == "faster_whisper":
        segments, _info = model.transcribe(audio_path_str, language=language, beam_size=5)
        return "".join(seg.text for seg in segments).strip()

    # openai-whisper
    kwargs: dict = {}
    if language:
        kwargs["language"] = language
    result = model.transcribe(audio_path_str, **kwargs)
    return result["text"].strip()
