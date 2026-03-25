from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tts3d_app.config import get_logger


LOGGER = get_logger("hrir")


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
