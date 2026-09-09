from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tts3d_app.timbre_report import (
    LOCK_F0_BUDGET_HZ,
    analyze_chunks,
    analyze_concatenated_audio,
    format_report,
    split_on_silence,
)


def _tone(freq_hz: float, seconds: float, sample_rate: int = 16_000) -> np.ndarray:
    times = np.arange(int(sample_rate * seconds), dtype=np.float32) / float(sample_rate)
    return (0.2 * np.sin(2.0 * np.pi * freq_hz * times)).astype(np.float32)


class TimbreReportTests(unittest.TestCase):
    def test_split_on_silence_finds_three_chunks(self) -> None:
        sample_rate = 16_000
        gap = np.zeros(int(sample_rate * 0.3), dtype=np.float32)
        audio = np.concatenate([_tone(150, 0.4, sample_rate), gap, _tone(150, 0.4, sample_rate), gap, _tone(180, 0.4, sample_rate)])
        spans = split_on_silence(audio, sample_rate, min_silence_ms=200)
        self.assertEqual(len(spans), 3)

    def test_stable_chunks_are_within_lock_budget(self) -> None:
        sample_rate = 16_000
        chunks = [_tone(150, 0.5, sample_rate), _tone(152, 0.5, sample_rate), _tone(148, 0.5, sample_rate)]
        report = analyze_chunks(chunks, sample_rate)
        self.assertLessEqual(report.max_f0_delta_hz, LOCK_F0_BUDGET_HZ)
        self.assertTrue(report.within_lock_budget)
        self.assertEqual(len(report.chunks), 3)
        self.assertIn("max_f0_delta_hz", format_report(report))

    def test_drifting_chunks_exceed_lock_budget(self) -> None:
        sample_rate = 16_000
        chunks = [_tone(150, 0.5, sample_rate), _tone(190, 0.5, sample_rate)]
        report = analyze_chunks(chunks, sample_rate)
        self.assertGreater(report.max_f0_delta_hz, LOCK_F0_BUDGET_HZ)
        self.assertFalse(report.within_lock_budget)

    def test_concatenated_file_roundtrip(self) -> None:
        sample_rate = 16_000
        gap = np.zeros(int(sample_rate * 0.3), dtype=np.float32)
        audio = np.concatenate([_tone(140, 0.4, sample_rate), gap, _tone(142, 0.4, sample_rate)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "concat.wav"
            sf.write(path, audio, sample_rate)
            from tts3d_app.timbre_report import analyze_wav_file

            report = analyze_wav_file(path)
            self.assertEqual(len(report.chunks), 2)
            self.assertTrue(report.within_lock_budget)

        concat_report = analyze_concatenated_audio(audio, sample_rate)
        self.assertEqual(len(concat_report.chunks), 2)


if __name__ == "__main__":
    unittest.main()
