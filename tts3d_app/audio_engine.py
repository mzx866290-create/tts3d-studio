from __future__ import annotations

import math

import numpy as np
import scipy.signal
import torch

try:
    import torchaudio.functional as torchaudio_functional
except Exception:  # pragma: no cover - optional runtime path
    torchaudio_functional = None

from tts3d_app.config import ENABLE_GPU_FFT_CONVOLUTION, TARGET_HRIR_SAMPLE_RATE, get_logger
from tts3d_app.hrir import (
    HrirDataset,
    PATH_CODE_CLOCKWISE,
    PATH_CODE_COUNTERCLOCKWISE,
    PATH_CODE_SIDE_TO_SIDE,
    resolve_dynamic_azimuth_for_plan,
)


LOGGER = get_logger("audio_engine")
MODE_MONO = "mono"


_logged_warnings: set[str] = set()


def _fft_convolve(signal: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    impulse = np.asarray(impulse, dtype=np.float32)

    if ENABLE_GPU_FFT_CONVOLUTION and torch.cuda.is_available() and torchaudio_functional is not None:
        try:
            device = torch.device("cuda")
            signal_tensor = torch.from_numpy(np.ascontiguousarray(signal)).to(device=device)
            impulse_tensor = torch.from_numpy(np.ascontiguousarray(impulse)).to(device=device)
            convolved = torchaudio_functional.fftconvolve(signal_tensor, impulse_tensor, mode="full")
            return convolved.detach().cpu().numpy().astype(np.float32, copy=False)
        except Exception as exc:  # pragma: no cover - depends on runtime CUDA stack
            warning_key = f"gpu_fft:{type(exc).__name__}"
            if warning_key not in _logged_warnings:
                LOGGER.warning("GPU fftconvolve unavailable, falling back to scipy.fftconvolve: %s", exc)
                _logged_warnings.add(warning_key)

    return np.asarray(scipy.signal.fftconvolve(signal, impulse, mode="full"), dtype=np.float32)
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


def apply_speed(audio: np.ndarray, sample_rate: int, factor: float) -> tuple[np.ndarray, int]:
    """Adjust audio playback speed.

    Args:
        audio: Audio samples.
        sample_rate: Sample rate in Hz.
        factor: Speed factor (1.0 = original, 2.0 = 2x speed, 0.5 = half speed).

    Returns:
        Tuple of (processed audio, original sample rate).
    """
    if abs(factor - 1.0) < 0.01:
        return audio, sample_rate

    if torchaudio_functional is None:
        return audio, sample_rate

    audio_tensor = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
    length_tensor = torch.tensor([audio_tensor.shape[-1]], dtype=torch.long)

    try:
        result, _ = torchaudio_functional.speed(
            audio_tensor,
            orig_freq=sample_rate,
            factor=factor,
            lengths=length_tensor,
        )
        return result.squeeze(0).numpy().astype(np.float32, copy=False), sample_rate
    except Exception:  # pragma: no cover - fallback on failure
        return audio, sample_rate


def apply_pitch_shift(audio: np.ndarray, sample_rate: int, semitones: float) -> tuple[np.ndarray, int]:
    """Shift pitch while preserving duration via torchaudio's phase-vocoder pitch_shift."""
    if abs(semitones) < 0.1:
        return audio, sample_rate

    if torchaudio_functional is None or not hasattr(torchaudio_functional, "pitch_shift"):
        LOGGER.warning("torchaudio pitch_shift is unavailable; leaving pitch unchanged")
        return audio, sample_rate

    from tts3d_app.config import ENABLE_GPU_PITCH_SHIFT

    waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    device = torch.device("cpu")
    if ENABLE_GPU_PITCH_SHIFT and torch.cuda.is_available():
        device = torch.device("cuda")

    try:
        shifted = torchaudio_functional.pitch_shift(
            waveform.to(device),
            sample_rate,
            n_steps=float(semitones),
        )
        return shifted.detach().cpu().squeeze(0).numpy().astype(np.float32, copy=False), sample_rate
    except Exception as exc:
        LOGGER.warning("pitch_shift failed, leaving pitch unchanged: %s", exc)
        return audio, sample_rate


def _path_type_to_code(path_type: str) -> int:
    if path_type == PATH_CLOCKWISE:
        return PATH_CODE_CLOCKWISE
    if path_type == PATH_COUNTERCLOCKWISE:
        return PATH_CODE_COUNTERCLOCKWISE
    if path_type == PATH_SIDE_TO_SIDE:
        return PATH_CODE_SIDE_TO_SIDE
    raise ValueError(f"Unknown dynamic path type: {path_type}")


def _fft_convolve_gpu(signal: torch.Tensor, impulse: torch.Tensor) -> torch.Tensor:
    """GPU batch FFT convolution.

    Args:
        signal: Signal tensor on GPU, shape (1, signal_len) or (signal_len,)
        impulse: Impulse tensor on GPU, shape (impulse_len,) or (1, impulse_len)

    Returns:
        Convolved result on GPU.
    """
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)
    if impulse.dim() == 1:
        impulse = impulse.unsqueeze(0)
    return torchaudio_functional.fftconvolve(signal, impulse, mode="full")


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

    use_gpu = ENABLE_GPU_FFT_CONVOLUTION and torch.cuda.is_available() and torchaudio_functional is not None

    if use_gpu:
        try:
            device = torch.device("cuda")
            signal_t = torch.from_numpy(np.ascontiguousarray(audio_mono)).unsqueeze(0).to(device)
            imp_l_t = torch.from_numpy(np.ascontiguousarray(impulse_left)).unsqueeze(0).to(device)
            imp_r_t = torch.from_numpy(np.ascontiguousarray(impulse_right)).unsqueeze(0).to(device)
            conv_l = _fft_convolve_gpu(signal_t, imp_l_t).squeeze(0).cpu().numpy()
            conv_r = _fft_convolve_gpu(signal_t, imp_r_t).squeeze(0).cpu().numpy()
            out_left = conv_l[: len(audio_mono)].astype(np.float32, copy=False)
            out_right = conv_r[: len(audio_mono)].astype(np.float32, copy=False)
            return np.column_stack((out_left, out_right)).astype(np.float32, copy=False)
        except Exception as exc:  # pragma: no cover - fallback to scipy
            warning_key = f"static_hrir_gpu:{type(exc).__name__}"
            if warning_key not in _logged_warnings:
                LOGGER.warning("Static HRIR GPU convolution failed, falling back: %s", exc)
                _logged_warnings.add(warning_key)

    out_left = _fft_convolve(audio_mono, impulse_left)[: len(audio_mono)]
    out_right = _fft_convolve(audio_mono, impulse_right)[: len(audio_mono)]
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
    path_code = _path_type_to_code(path_type)
    return resolve_dynamic_azimuth_for_plan(path_code, time_position, cycle_time_s, start_azimuth_deg)


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
    frame_starts, azimuth_indices = dataset.build_dynamic_frame_plan(
        audio_length=len(audio_mono),
        hop_length=hop_length,
        sample_rate=sample_rate,
        path_code=_path_type_to_code(path_type),
        cycle_time_s=cycle_time_s,
        start_azimuth_deg=start_azimuth_deg,
    )

    # Pre-load HRIR to GPU if available for faster per-frame convolution
    use_gpu = (
        ENABLE_GPU_FFT_CONVOLUTION
        and torch.cuda.is_available()
        and torchaudio_functional is not None
    )

    if use_gpu:
        device = torch.device("cuda")
        # Pre-upload all impulse responses to GPU once (avoid per-frame transfer)
        gpu_impulses_l = [torch.from_numpy(np.ascontiguousarray(imp[0])).unsqueeze(0).to(device) for imp in impulse_pairs]
        gpu_impulses_r = [torch.from_numpy(np.ascontiguousarray(imp[1])).unsqueeze(0).to(device) for imp in impulse_pairs]
    else:
        gpu_impulses_l = None
        gpu_impulses_r = None

    for start, azimuth_index in zip(frame_starts.tolist(), azimuth_indices.tolist()):
        chunk = audio_mono[start : start + frame_length]
        if len(chunk) < frame_length:
            chunk = np.pad(chunk, (0, frame_length - len(chunk)))

        chunk = np.asarray(chunk, dtype=np.float32) * window

        if use_gpu:
            try:
                chunk_t = torch.from_numpy(np.ascontiguousarray(chunk)).unsqueeze(0).to(device)
                conv_l_t = _fft_convolve_gpu(chunk_t, gpu_impulses_l[azimuth_index]).squeeze(0)
                conv_r_t = _fft_convolve_gpu(chunk_t, gpu_impulses_r[azimuth_index]).squeeze(0)
                convolved_left = conv_l_t.detach().cpu().numpy().astype(np.float32, copy=False)
                convolved_right = conv_r_t.detach().cpu().numpy().astype(np.float32, copy=False)
            except Exception:  # pragma: no cover - fallback to scipy
                convolved_left = _fft_convolve(chunk, impulse_pairs[azimuth_index][0])
                convolved_right = _fft_convolve(chunk, impulse_pairs[azimuth_index][1])
        else:
            convolved_left = _fft_convolve(chunk, impulse_pairs[azimuth_index][0])
            convolved_right = _fft_convolve(chunk, impulse_pairs[azimuth_index][1])

        end_left = min(start + len(convolved_left), output_length)
        end_right = min(start + len(convolved_right), output_length)

        out_left[start:end_left] += convolved_left[: end_left - start].astype(np.float32, copy=False)
        out_right[start:end_right] += convolved_right[: end_right - start].astype(np.float32, copy=False)

    final_audio = np.column_stack((out_left, out_right))[: len(audio_mono)]
    return final_audio.astype(np.float32, copy=False), sample_rate
