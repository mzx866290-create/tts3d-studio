from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tts3d_app.config import ENABLE_NUMBA_DYNAMIC_HRIR, get_logger


LOGGER = get_logger("hrir")
PATH_CODE_CLOCKWISE = 0
PATH_CODE_COUNTERCLOCKWISE = 1
PATH_CODE_SIDE_TO_SIDE = 2

try:
    if ENABLE_NUMBA_DYNAMIC_HRIR:
        from numba import njit
    else:  # pragma: no cover - environment controlled
        njit = None
except Exception:  # pragma: no cover - optional dependency
    njit = None


if njit is not None:
    @njit(cache=True)
    def _resolve_dynamic_azimuth_numba(
        path_code: int,
        time_position: float,
        cycle_time_s: float,
        start_azimuth_deg: float,
    ) -> float:
        if path_code == PATH_CODE_CLOCKWISE:
            return (start_azimuth_deg - (time_position / cycle_time_s) * 360.0) % 360.0
        if path_code == PATH_CODE_COUNTERCLOCKWISE:
            return (start_azimuth_deg + (time_position / cycle_time_s) * 360.0) % 360.0

        phase = math.cos(2.0 * math.pi * time_position / cycle_time_s)
        return (180.0 + phase * 90.0) % 360.0


    @njit(cache=True)
    def _build_dynamic_frame_plan_numba(
        audio_length: int,
        hop_length: int,
        sample_rate: int,
        path_code: int,
        cycle_time_s: float,
        start_azimuth_deg: float,
        azimuth_start: float,
        azimuth_step: float,
        azimuth_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        frame_count = (audio_length + hop_length - 1) // hop_length
        frame_starts = np.empty(frame_count, dtype=np.int64)
        azimuth_indices = np.empty(frame_count, dtype=np.int64)

        for frame_idx in range(frame_count):
            start = frame_idx * hop_length
            time_position = float(start + hop_length) / float(sample_rate)
            azimuth_deg = _resolve_dynamic_azimuth_numba(
                path_code,
                time_position,
                cycle_time_s,
                start_azimuth_deg,
            )
            azimuth_index = int(round((azimuth_deg - azimuth_start) / azimuth_step)) % azimuth_count
            frame_starts[frame_idx] = start
            azimuth_indices[frame_idx] = azimuth_index

        return frame_starts, azimuth_indices
else:
    _build_dynamic_frame_plan_numba = None


def resolve_dynamic_azimuth_for_plan(
    path_code: int,
    time_position: float,
    cycle_time_s: float,
    start_azimuth_deg: float,
) -> float:
    if _resolve_dynamic_azimuth_numba is not None:
        return _resolve_dynamic_azimuth_numba(path_code, time_position, cycle_time_s, start_azimuth_deg)
    if path_code == PATH_CODE_CLOCKWISE:
        return (start_azimuth_deg - (time_position / cycle_time_s) * 360.0) % 360.0
    if path_code == PATH_CODE_COUNTERCLOCKWISE:
        return (start_azimuth_deg + (time_position / cycle_time_s) * 360.0) % 360.0
    if path_code == PATH_CODE_SIDE_TO_SIDE:
        phase = math.cos(2.0 * math.pi * time_position / cycle_time_s)
        return (180.0 + phase * 90.0) % 360.0
    raise ValueError(f"Unknown dynamic path code: {path_code}")


@dataclass(frozen=True, slots=True)
class HrirDataset:
    left: np.ndarray
    right: np.ndarray
    azimuths: np.ndarray
    distances: np.ndarray
    _uniform_azimuth_step: float | None = field(init=False, default=None, repr=False)
    _azimuth_start: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.azimuths.size < 2:
            return

        diffs = np.diff(np.sort(self.azimuths.astype(np.float32)))
        if np.allclose(diffs, diffs[0], atol=1e-4):
            object.__setattr__(self, "_uniform_azimuth_step", float(diffs[0]))
            object.__setattr__(self, "_azimuth_start", float(np.min(self.azimuths)))

    def closest_azimuth_index(self, azimuth_deg: float) -> int:
        normalized = azimuth_deg % 360
        if self._uniform_azimuth_step:
            index = int(round((normalized - self._azimuth_start) / self._uniform_azimuth_step))
            return index % len(self.azimuths)

        wrapped_delta = (self.azimuths - normalized + 180) % 360 - 180
        return int(np.argmin(np.abs(wrapped_delta)))

    def closest_distance_index(self, distance_m: float) -> int:
        return int(np.argmin(np.abs(self.distances - distance_m)))

    def build_dynamic_frame_plan(
        self,
        audio_length: int,
        hop_length: int,
        sample_rate: int,
        path_code: int,
        cycle_time_s: float,
        start_azimuth_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if audio_length <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        if self._uniform_azimuth_step is not None and _build_dynamic_frame_plan_numba is not None:
            return _build_dynamic_frame_plan_numba(
                audio_length,
                hop_length,
                sample_rate,
                path_code,
                cycle_time_s,
                start_azimuth_deg,
                self._azimuth_start,
                self._uniform_azimuth_step,
                len(self.azimuths),
            )

        frame_starts = np.arange(0, audio_length, hop_length, dtype=np.int64)
        azimuth_indices = np.empty(frame_starts.shape[0], dtype=np.int64)
        for frame_idx, start in enumerate(frame_starts):
            time_position = float(start + hop_length) / float(sample_rate)
            azimuth_deg = resolve_dynamic_azimuth_for_plan(
                path_code,
                time_position,
                cycle_time_s,
                start_azimuth_deg,
            )
            azimuth_indices[frame_idx] = self.closest_azimuth_index(azimuth_deg)

        return frame_starts, azimuth_indices


def load_hrir_dataset(path: Path) -> HrirDataset | None:
    if not path.exists():
        LOGGER.warning("HRIR file not found: %s", path)
        return None

    try:
        data = np.load(path)
        dataset = HrirDataset(
            left=data["hrir_L"],
            right=data["hrir_R"],
            azimuths=data["azimuths"],
            distances=data["distances"],
        )
    except Exception as exc:
        LOGGER.exception("Failed to load HRIR data: %s", exc)
        return None

    LOGGER.info("HRIR data loaded: azimuths=%s distances=%s", dataset.azimuths.shape, dataset.distances.shape)
    return dataset
