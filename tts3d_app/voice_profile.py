"""Persistent VoiceDesign reference clips used to lock timbre across texts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from tts3d_app.config import (
    TTS_SPEAKER_REF_MAX_CHARS,
    TTS_VOICE_CALIBRATION_TEXT,
    VOICE_PROFILE_DIR,
)
from tts3d_app.text_chunking import take_reference_sentence


SynthesizeFn = Callable[[], tuple[np.ndarray, int]]


def resolve_calibration_text(
    text: str | None = None,
    max_chars: int | None = None,
) -> str:
    source = (TTS_VOICE_CALIBRATION_TEXT if text is None else text).strip()
    limit = TTS_SPEAKER_REF_MAX_CHARS if max_chars is None else max_chars
    if not source or limit <= 0:
        return ""
    trimmed = take_reference_sentence(source, max_chars=limit)
    return trimmed or source[:limit]


def voice_clip_id(model_key: str, prompt: str, seed: int, calibration_text: str) -> str:
    payload = "\n".join(
        (
            model_key.strip(),
            prompt.strip(),
            str(int(seed) % (2**32)),
            calibration_text.strip(),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def reference_clip_wav_path(clip_id: str, profile_dir: Path | None = None) -> Path:
    return (profile_dir or VOICE_PROFILE_DIR) / f"{clip_id}.wav"


def reference_clip_meta_path(clip_id: str, profile_dir: Path | None = None) -> Path:
    return (profile_dir or VOICE_PROFILE_DIR) / f"{clip_id}.json"


def load_reference_clip(
    clip_id: str,
    profile_dir: Path | None = None,
) -> tuple[np.ndarray, int] | None:
    wav_path = reference_clip_wav_path(clip_id, profile_dir)
    if not wav_path.exists():
        return None
    audio, sample_rate = sf.read(str(wav_path), always_2d=True, dtype="float32")
    mono = np.mean(audio, axis=1).astype(np.float32, copy=False)
    if mono.size == 0:
        return None
    return mono, int(sample_rate)


def save_reference_clip(
    clip_id: str,
    audio: np.ndarray,
    sample_rate: int,
    metadata: dict[str, Any] | None = None,
    profile_dir: Path | None = None,
) -> Path:
    directory = profile_dir or VOICE_PROFILE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    wav_path = reference_clip_wav_path(clip_id, directory)
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    sf.write(str(wav_path), waveform, int(sample_rate))
    meta_path = reference_clip_meta_path(clip_id, directory)
    payload = {"clip_id": clip_id, "sample_rate": int(sample_rate), **(metadata or {})}
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return wav_path


def invalidate_reference_clip(clip_id: str, profile_dir: Path | None = None) -> bool:
    removed = False
    for path in (
        reference_clip_wav_path(clip_id, profile_dir),
        reference_clip_meta_path(clip_id, profile_dir),
    ):
        if path.exists():
            path.unlink()
            removed = True
    return removed


def get_or_create_reference_clip(
    *,
    model_key: str,
    prompt: str,
    seed: int,
    calibration_text: str,
    synthesize: SynthesizeFn,
    profile_dir: Path | None = None,
) -> tuple[np.ndarray, int, str]:
    clip_id = voice_clip_id(model_key, prompt, seed, calibration_text)
    cached = load_reference_clip(clip_id, profile_dir)
    if cached is not None:
        return cached[0], cached[1], clip_id

    audio, sample_rate = synthesize()
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    save_reference_clip(
        clip_id,
        waveform,
        int(sample_rate),
        metadata={
            "model_key": model_key,
            "prompt": prompt,
            "seed": int(seed) % (2**32),
            "calibration_text": calibration_text,
        },
        profile_dir=profile_dir,
    )
    return waveform, int(sample_rate), clip_id
