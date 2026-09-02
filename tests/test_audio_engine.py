from __future__ import annotations

import unittest

import numpy as np

from tts3d_app.audio_engine import (
    PATH_CLOCKWISE,
    PATH_SIDE_TO_SIDE,
    _fft_convolve,
    apply_behind_head_effect,
    apply_dynamic_hrir,
    apply_static_hrir,
    ensure_mono,
    normalize_audio,
    resolve_dynamic_azimuth,
)
from tts3d_app.hrir import HrirDataset, PATH_CODE_CLOCKWISE


def build_fake_hrir() -> HrirDataset:
    left = np.zeros((4, 2, 3), dtype=np.float32)
    right = np.zeros((4, 2, 3), dtype=np.float32)
    left[:, :, 0] = 1.0
    right[:, :, 0] = 1.0

    return HrirDataset(
        left=left,
        right=right,
        azimuths=np.array([0, 90, 180, 270], dtype=np.float32),
        distances=np.array([0.2, 0.5], dtype=np.float32),
    )


class AudioEngineTests(unittest.TestCase):
    def test_ensure_mono_averages_stereo(self) -> None:
        stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        mono = ensure_mono(stereo)
        np.testing.assert_allclose(mono, np.array([2.0, 3.0], dtype=np.float32))

    def test_normalize_audio_scales_peak(self) -> None:
        audio = np.array([0.5, -2.0, 1.0], dtype=np.float32)
        normalized = normalize_audio(audio)
        self.assertAlmostEqual(float(np.max(np.abs(normalized))), 0.89, places=6)

    def test_apply_behind_head_effect_returns_stereo(self) -> None:
        audio = np.zeros(128, dtype=np.float32)
        audio[0] = 1.0
        processed, sample_rate = apply_behind_head_effect(audio, 24_000)
        self.assertEqual(processed.shape, (128, 2))
        self.assertEqual(processed.dtype, np.float32)
        self.assertEqual(sample_rate, 24_000)

    def test_apply_static_hrir_uses_identity_impulse(self) -> None:
        dataset = build_fake_hrir()
        audio = np.array([1.0, -0.5, 0.25], dtype=np.float32)
        processed = apply_static_hrir(audio, dataset, azimuth_deg=87, distance_m=0.22)
        np.testing.assert_allclose(processed[:, 0], audio)
        np.testing.assert_allclose(processed[:, 1], audio)

    def test_fft_convolution_matches_time_domain_reference(self) -> None:
        audio = np.array([0.25, -0.5, 0.75, -1.0, 0.5], dtype=np.float32)
        impulse = np.array([0.4, -0.1, 0.05], dtype=np.float32)
        expected = np.convolve(audio, impulse, mode="full")
        actual = _fft_convolve(audio, impulse)
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_apply_dynamic_hrir_handles_last_chunk_overlap_safely(self) -> None:
        dataset = build_fake_hrir()
        audio = np.linspace(-1.0, 1.0, 1393, dtype=np.float32)
        processed, sample_rate = apply_dynamic_hrir(
            audio_mono=audio,
            sample_rate=44_100,
            dataset=dataset,
            path_type=PATH_CLOCKWISE,
            cycle_time_s=4.0,
            start_azimuth_deg=270,
            distance_m=0.2,
        )
        self.assertEqual(processed.shape, (1393, 2))
        self.assertEqual(sample_rate, 44_100)

    def test_dynamic_frame_plan_matches_python_resolver(self) -> None:
        dataset = build_fake_hrir()
        frame_starts, azimuth_indices = dataset.build_dynamic_frame_plan(
            audio_length=4000,
            hop_length=882,
            sample_rate=44_100,
            path_code=PATH_CODE_CLOCKWISE,
            cycle_time_s=4.0,
            start_azimuth_deg=270,
        )

        expected_indices = []
        for start in frame_starts.tolist():
            azimuth = resolve_dynamic_azimuth(
                path_type=PATH_CLOCKWISE,
                time_position=(start + 882) / 44_100,
                cycle_time_s=4.0,
                start_azimuth_deg=270,
            )
            expected_indices.append(dataset.closest_azimuth_index(azimuth))

        np.testing.assert_array_equal(azimuth_indices, np.asarray(expected_indices, dtype=np.int64))

    def test_uniform_azimuth_lookup_uses_expected_index(self) -> None:
        dataset = build_fake_hrir()
        self.assertEqual(dataset.closest_azimuth_index(87), 1)
        self.assertEqual(dataset.closest_azimuth_index(269), 3)

    def test_resolve_dynamic_azimuth_clockwise(self) -> None:
        azimuth = resolve_dynamic_azimuth(
            path_type=PATH_CLOCKWISE,
            time_position=1.0,
            cycle_time_s=4.0,
            start_azimuth_deg=270,
        )
        self.assertAlmostEqual(azimuth, 180.0)

    def test_resolve_dynamic_azimuth_side_to_side(self) -> None:
        azimuth = resolve_dynamic_azimuth(
            path_type=PATH_SIDE_TO_SIDE,
            time_position=2.0,
            cycle_time_s=4.0,
            start_azimuth_deg=270,
        )
        self.assertAlmostEqual(azimuth, 90.0)


if __name__ == "__main__":
    unittest.main()
