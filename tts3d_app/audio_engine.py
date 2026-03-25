from __future__ import annotations

import math

import numpy as np
import scipy.signal

from tts3d_app.config import TARGET_HRIR_SAMPLE_RATE
from tts3d_app.hrir import HrirDataset


MODE_MONO = "mono"
MODE_STEREO = "stereo"
MODE_BEHIND_HEAD = "behind_head"
MODE_STATIC_HRIR = "static_hrir"
MODE_DYNAMIC_HRIR = "dynamic_hrir"

PATH_CLOCKWISE = "clockwise"
PATH_COUNTERCLOCKWISE = "counterclockwise"
PATH_SIDE_TO_SIDE = "side_to_side"


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[1] >= 1:
        return array.mean(axis=1, dtype=np.float32)
    return array.reshape(-1).astype(np.float32, copy=False)


def duplicate_stereo(audio_mono: np.ndarray) -> np.ndarray:
    return np.column_stack((audio_mono, audio_mono)).astype(np.float32, copy=False)


def normalize_audio(audio: np.ndarray, peak: float = 0.89) -> np.ndarray:
    max_value = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_value <= 0.0:
        return audio
    return (audio / max_value) * peak


def apply_behind_head_effect(audio_mono: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    left = audio_mono.copy()
    right = audio_mono.copy()

    notch_b, notch_a = scipy.signal.iirnotch(w0=4_000, Q=2.0, fs=sample_rate)
    left = scipy.signal.lfilter(notch_b, notch_a, left)
    right = scipy.signal.lfilter(notch_b, notch_a, right)

    delay_samples = max(int(sample_rate * 0.0006), 1)
    right = np.pad(right, (delay_samples, 0))[:-delay_samples]
    return np.column_stack((left, right)).astype(np.float32, copy=False), sample_rate


def apply_static_hrir(
    audio_mono: np.ndarray,
    dataset: HrirDataset,
    azimuth_deg: float,
    distance_m: float,
) -> np.ndarray:
    azimuth_index = dataset.closest_azimuth_index(azimuth_deg)
    distance_index = dataset.closest_distance_index(distance_m)
    impulse_left = dataset.left[azimuth_index, distance_index]
    impulse_right = dataset.right[azimuth_index, distance_index]

    out_left = scipy.signal.fftconvolve(audio_mono, impulse_left, mode="full")[: len(audio_mono)]
    out_right = scipy.signal.fftconvolve(audio_mono, impulse_right, mode="full")[: len(audio_mono)]
    return np.column_stack((out_left, out_right)).astype(np.float32, copy=False)


def resample_audio(audio_mono: np.ndarray, sample_rate: int, target_rate: int) -> tuple[np.ndarray, int]:
    if sample_rate == target_rate:
        return audio_mono, sample_rate

    target_samples = int(len(audio_mono) * target_rate / sample_rate)
    resampled = scipy.signal.resample(audio_mono, target_samples)
    return np.asarray(resampled, dtype=np.float32), target_rate


def resolve_dynamic_azimuth(
    path_type: str,
    time_position: float,
    cycle_time_s: float,
    start_azimuth_deg: float,
) -> float:
    if cycle_time_s <= 0:
        raise ValueError("cycle_time_s must be greater than 0.")

    if path_type == PATH_CLOCKWISE:
        return (start_azimuth_deg - (time_position / cycle_time_s) * 360) % 360
    if path_type == PATH_COUNTERCLOCKWISE:
        return (start_azimuth_deg + (time_position / cycle_time_s) * 360) % 360
    if path_type == PATH_SIDE_TO_SIDE:
        phase = math.cos(2 * math.pi * time_position / cycle_time_s)
        return (180 + phase * 90) % 360

    raise ValueError(f"Unknown dynamic path type: {path_type}")


def apply_dynamic_hrir(
    audio_mono: np.ndarray,
    sample_rate: int,
    dataset: HrirDataset,
    path_type: str,
    cycle_time_s: float,
    start_azimuth_deg: float,
    distance_m: float,
) -> tuple[np.ndarray, int]:
    audio_mono, sample_rate = resample_audio(audio_mono, sample_rate, TARGET_HRIR_SAMPLE_RATE)

    frame_length = max(int(sample_rate * 0.04), 1)
    hop_length = max(frame_length // 2, 1)
    window = np.hanning(frame_length).astype(np.float32, copy=False)

    impulse_length = len(dataset.left[0, 0])
    output_length = len(audio_mono) + impulse_length - 1
    out_left = np.zeros(output_length, dtype=np.float32)
    out_right = np.zeros(output_length, dtype=np.float32)

    distance_index = dataset.closest_distance_index(distance_m)
    impulse_pairs = list(zip(dataset.left[:, distance_index], dataset.right[:, distance_index]))

    for start in range(0, len(audio_mono), hop_length):
        chunk = audio_mono[start : start + frame_length]
        if len(chunk) < frame_length:
            chunk = np.pad(chunk, (0, frame_length - len(chunk)))

        chunk = np.asarray(chunk, dtype=np.float32) * window
        time_position = (start + hop_length) / sample_rate
        azimuth_deg = resolve_dynamic_azimuth(path_type, time_position, cycle_time_s, start_azimuth_deg)
        azimuth_index = dataset.closest_azimuth_index(azimuth_deg)

        impulse_left, impulse_right = impulse_pairs[azimuth_index]
        convolved_left = scipy.signal.fftconvolve(chunk, impulse_left)
        convolved_right = scipy.signal.fftconvolve(chunk, impulse_right)

        end_left = min(start + len(convolved_left), output_length)
        end_right = min(start + len(convolved_right), output_length)

        out_left[start:end_left] += convolved_left[: end_left - start].astype(np.float32, copy=False)
        out_right[start:end_right] += convolved_right[: end_right - start].astype(np.float32, copy=False)

    final_audio = np.column_stack((out_left, out_right))[: len(audio_mono)]
    return final_audio.astype(np.float32, copy=False), sample_rate
