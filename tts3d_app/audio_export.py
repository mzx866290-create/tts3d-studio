from __future__ import annotations

from pathlib import Path

import numpy as np

from tts3d_app.config import get_logger


LOGGER = get_logger("audio_export")


SUPPORTED_FORMATS = ("wav", "mp3", "ogg", "flac")
DEFAULT_FORMAT = "wav"


def get_export_formats() -> list[str]:
    return list(SUPPORTED_FORMATS)


def export_audio(wav_path: Path, output_dir: Path, formats: list[str] | None = None) -> list[Path]:
    """Export a WAV file to multiple formats.

    Args:
        wav_path: Path to the source WAV file.
        output_dir: Directory to save exported files.
        formats: List of formats to export. Defaults to all supported formats.

    Returns:
        List of paths to the exported files.
    """
    if formats is None:
        formats = list(SUPPORTED_FORMATS)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: list[Path] = []

    for fmt in formats:
        fmt = fmt.lower().strip()
        if fmt not in SUPPORTED_FORMATS:
            continue
        if fmt == "wav":
            exported.append(wav_path)
            continue

        output_path = output_dir / f"{wav_path.stem}.{fmt}"

        try:
            from pydub import AudioSegment
        except Exception:  # pragma: no cover - pydub not installed
            continue

        try:
            audio = AudioSegment.from_wav(str(wav_path))
            audio.export(str(output_path), format=fmt)
            exported.append(output_path)
        except Exception as exc:  # pragma: no cover - conversion failed
            LOGGER.warning("Failed to export %s to %s: %s", wav_path.name, fmt, exc)
            continue

    return exported


def convert_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    output_path: Path,
    format: str = "wav",
) -> Path:
    """Convert audio data directly to a file format.

    Args:
        audio_data: Audio samples as numpy array.
        sample_rate: Sample rate in Hz.
        output_path: Path for the output file.
        format: Output format (wav, mp3, ogg, flac).

    Returns:
        Path to the output file.
    """
    import soundfile as sf

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "wav":
        sf.write(output_path, audio_data, sample_rate)
        return output_path

    sf.write(output_path.with_suffix(".wav"), audio_data, sample_rate)
    exported = export_audio(output_path.with_suffix(".wav"), output_path.parent, [format])
    if not exported:
        raise RuntimeError(f"Failed to export audio to format: {format}")
    return exported[0]
