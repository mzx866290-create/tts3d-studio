"""Measure per-chunk F0 and spectral drift in concatenated TTS audio."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
import torchaudio.transforms as AT


LOCK_F0_BUDGET_HZ = 10.0
_F0_MIN_HZ = 60.0
_F0_MAX_HZ = 400.0
_FRAME_MS = 20.0
_DEFAULT_MIN_SILENCE_MS = 200.0
_DEFAULT_SILENCE_RMS = 0.01


@dataclass(frozen=True, slots=True)
class ChunkTimbre:
    index: int
    start_s: float
    end_s: float
    f0_mean_hz: float
    f0_std_hz: float
    f0_delta_hz: float
    mfcc_distance: float


@dataclass(frozen=True, slots=True)
class TimbreReport:
    sample_rate: int
    chunks: tuple[ChunkTimbre, ...]
    max_f0_delta_hz: float
    within_lock_budget: bool


def load_mono_audio(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    mono = np.mean(audio, axis=1).astype(np.float32, copy=False)
    return mono, int(sample_rate)


def split_on_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_silence_ms: float = _DEFAULT_MIN_SILENCE_MS,
    silence_rms: float = _DEFAULT_SILENCE_RMS,
) -> list[tuple[int, int]]:
    """Return [start, end) sample spans for non-silent regions."""
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        return []

    frame_samples = max(int(sample_rate * _FRAME_MS / 1000.0), 1)
    min_silence_frames = max(int(math.ceil(min_silence_ms / _FRAME_MS)), 1)
    frame_count = int(math.ceil(waveform.size / frame_samples))
    silent_flags: list[bool] = []
    for frame_index in range(frame_count):
        start = frame_index * frame_samples
        frame = waveform[start : start + frame_samples]
        silent_flags.append(float(np.sqrt(np.mean(np.square(frame)))) < silence_rms)

    spans: list[tuple[int, int]] = []
    index = 0
    while index < frame_count:
        if silent_flags[index]:
            index += 1
            continue
        start_frame = index
        while index < frame_count:
            if not silent_flags[index]:
                index += 1
                continue
            silence_run = 0
            look = index
            while look < frame_count and silent_flags[look]:
                silence_run += 1
                look += 1
            if silence_run >= min_silence_frames:
                break
            index = look
        end_frame = index
        start_sample = start_frame * frame_samples
        end_sample = min(end_frame * frame_samples, waveform.size)
        if end_sample > start_sample:
            spans.append((start_sample, end_sample))
        index = end_frame
    return spans or [(0, waveform.size)]


def voiced_f0_hz(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32).reshape(1, -1))
    # detect_pitch_frequency median window needs ~30 frames at 10ms.
    min_samples = max(int(sample_rate * 0.4), 1)
    if waveform.shape[-1] < min_samples:
        return np.zeros(0, dtype=np.float32)
    try:
        freqs = AF.detect_pitch_frequency(waveform, sample_rate).detach().cpu().numpy().reshape(-1)
    except RuntimeError:
        return np.zeros(0, dtype=np.float32)
    voiced = freqs[(freqs >= _F0_MIN_HZ) & (freqs <= _F0_MAX_HZ)]
    return voiced.astype(np.float32, copy=False)


def mean_mfcc(audio: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> np.ndarray:
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32).reshape(1, -1))
    transform = AT.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs={"n_mels": 40})
    coeffs = transform(waveform).detach().cpu().numpy()
    return coeffs.mean(axis=-1).reshape(-1).astype(np.float32, copy=False)


def _mfcc_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return float("nan")
    delta = left.astype(np.float64) - right.astype(np.float64)
    return float(np.linalg.norm(delta))


def analyze_chunks(
    chunks: list[np.ndarray],
    sample_rate: int,
    starts_s: list[float] | None = None,
) -> TimbreReport:
    if not chunks:
        raise ValueError("No audio chunks to analyze.")

    features: list[tuple[float, float, np.ndarray]] = []
    for chunk in chunks:
        f0 = voiced_f0_hz(chunk, sample_rate)
        f0_mean = float(np.mean(f0)) if f0.size else float("nan")
        f0_std = float(np.std(f0)) if f0.size else float("nan")
        features.append((f0_mean, f0_std, mean_mfcc(chunk, sample_rate)))

    first_f0, _, first_mfcc = features[0]
    reports: list[ChunkTimbre] = []
    max_delta = 0.0
    cursor = 0.0
    for index, chunk in enumerate(chunks):
        f0_mean, f0_std, mfcc = features[index]
        start_s = starts_s[index] if starts_s is not None else cursor
        end_s = start_s + (len(chunk) / float(sample_rate) if sample_rate else 0.0)
        delta = abs(f0_mean - first_f0) if math.isfinite(f0_mean) and math.isfinite(first_f0) else float("nan")
        if math.isfinite(delta):
            max_delta = max(max_delta, delta)
        reports.append(
            ChunkTimbre(
                index=index,
                start_s=start_s,
                end_s=end_s,
                f0_mean_hz=f0_mean,
                f0_std_hz=f0_std,
                f0_delta_hz=0.0 if index == 0 else delta,
                mfcc_distance=0.0 if index == 0 else _mfcc_distance(mfcc, first_mfcc),
            )
        )
        cursor = end_s

    return TimbreReport(
        sample_rate=sample_rate,
        chunks=tuple(reports),
        max_f0_delta_hz=max_delta,
        within_lock_budget=max_delta <= LOCK_F0_BUDGET_HZ,
    )


def analyze_concatenated_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_silence_ms: float = _DEFAULT_MIN_SILENCE_MS,
    silence_rms: float = _DEFAULT_SILENCE_RMS,
) -> TimbreReport:
    spans = split_on_silence(
        audio,
        sample_rate,
        min_silence_ms=min_silence_ms,
        silence_rms=silence_rms,
    )
    chunks = [np.asarray(audio[start:end], dtype=np.float32) for start, end in spans]
    starts = [start / float(sample_rate) for start, _ in spans]
    return analyze_chunks(chunks, sample_rate, starts_s=starts)


def analyze_wav_file(
    path: str | Path,
    *,
    min_silence_ms: float = _DEFAULT_MIN_SILENCE_MS,
    silence_rms: float = _DEFAULT_SILENCE_RMS,
) -> TimbreReport:
    audio, sample_rate = load_mono_audio(path)
    return analyze_concatenated_audio(
        audio,
        sample_rate,
        min_silence_ms=min_silence_ms,
        silence_rms=silence_rms,
    )


def analyze_chunk_files(paths: list[str | Path]) -> TimbreReport:
    if not paths:
        raise ValueError("No chunk files provided.")
    chunks: list[np.ndarray] = []
    sample_rate = 0
    for path in paths:
        audio, rate = load_mono_audio(path)
        if not chunks:
            sample_rate = rate
        elif rate != sample_rate:
            raise ValueError(f"Sample rate mismatch in {path}: {rate} != {sample_rate}")
        chunks.append(audio)
    return analyze_chunks(chunks, sample_rate)


def format_report(report: TimbreReport, source: str = "") -> str:
    lines = []
    if source:
        lines.append(f"source: {source}")
    lines.append(f"sample_rate: {report.sample_rate}")
    lines.append(
        f"chunks: {len(report.chunks)}  max_f0_delta_hz: {report.max_f0_delta_hz:.2f}  "
        f"within_{LOCK_F0_BUDGET_HZ:.0f}hz: {report.within_lock_budget}"
    )
    lines.append(
        f"{'idx':>3}  {'start':>7}  {'end':>7}  {'f0_mean':>8}  {'f0_std':>7}  "
        f"{'df0':>7}  {'mfcc_d':>8}"
    )
    for chunk in report.chunks:
        lines.append(
            f"{chunk.index:3d}  {chunk.start_s:7.2f}  {chunk.end_s:7.2f}  "
            f"{chunk.f0_mean_hz:8.2f}  {chunk.f0_std_hz:7.2f}  "
            f"{chunk.f0_delta_hz:7.2f}  {chunk.mfcc_distance:8.3f}"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report per-chunk F0 / MFCC drift in TTS audio.")
    parser.add_argument("audio", nargs="?", help="Concatenated wav to split on silence")
    parser.add_argument("--chunks", nargs="+", help="Separate wav files, one per TTS chunk")
    parser.add_argument("--min-silence-ms", type=float, default=_DEFAULT_MIN_SILENCE_MS)
    parser.add_argument("--silence-rms", type=float, default=_DEFAULT_SILENCE_RMS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.chunks:
        report = analyze_chunk_files(args.chunks)
        source = ", ".join(str(path) for path in args.chunks)
    elif args.audio:
        report = analyze_wav_file(
            args.audio,
            min_silence_ms=args.min_silence_ms,
            silence_rms=args.silence_rms,
        )
        source = str(args.audio)
    else:
        raise SystemExit("Provide a wav path or --chunks a.wav b.wav ...")
    print(format_report(report, source=source))
    return 0 if report.within_lock_budget else 2


if __name__ == "__main__":
    raise SystemExit(main())
