from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from tts3d_app.voice_profile import (
    get_or_create_reference_clip,
    invalidate_reference_clip,
    load_reference_clip,
    resolve_calibration_text,
    voice_clip_id,
)


class VoiceProfileTests(unittest.TestCase):
    def create_dir(self) -> Path:
        parent = Path.cwd() / "outputs" / "test_runs"
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / f"voices_{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def test_clip_id_depends_on_prompt_seed_and_calibration_not_body_text(self) -> None:
        first = voice_clip_id("model-a", "calm voice", 7, "校准句。")
        second = voice_clip_id("model-a", "calm voice", 7, "校准句。")
        different_prompt = voice_clip_id("model-a", "other voice", 7, "校准句。")
        different_seed = voice_clip_id("model-a", "calm voice", 8, "校准句。")
        self.assertEqual(first, second)
        self.assertNotEqual(first, different_prompt)
        self.assertNotEqual(first, different_seed)

    def test_get_or_create_reuses_disk_cache(self) -> None:
        profile_dir = self.create_dir()
        calls = {"n": 0}

        def synthesize() -> tuple[np.ndarray, int]:
            calls["n"] += 1
            return np.array([0.1, -0.1, 0.2], dtype=np.float32), 24_000

        first = get_or_create_reference_clip(
            model_key="model-a",
            prompt="calm",
            seed=7,
            calibration_text="校准句。",
            synthesize=synthesize,
            profile_dir=profile_dir,
        )
        second = get_or_create_reference_clip(
            model_key="model-a",
            prompt="calm",
            seed=7,
            calibration_text="校准句。",
            synthesize=synthesize,
            profile_dir=profile_dir,
        )
        self.assertEqual(calls["n"], 1)
        self.assertEqual(first[2], second[2])
        np.testing.assert_array_almost_equal(first[0], second[0], decimal=3)
        loaded = load_reference_clip(first[2], profile_dir)
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded[0], first[0], decimal=3)

    def test_invalidate_forces_resynthesize(self) -> None:
        profile_dir = self.create_dir()
        calls = {"n": 0}

        def synthesize() -> tuple[np.ndarray, int]:
            calls["n"] += 1
            return np.array([0.3, -0.3], dtype=np.float32), 16_000

        _, _, clip_id = get_or_create_reference_clip(
            model_key="model-a",
            prompt="calm",
            seed=1,
            calibration_text="校准句。",
            synthesize=synthesize,
            profile_dir=profile_dir,
        )
        self.assertTrue(invalidate_reference_clip(clip_id, profile_dir))
        get_or_create_reference_clip(
            model_key="model-a",
            prompt="calm",
            seed=1,
            calibration_text="校准句。",
            synthesize=synthesize,
            profile_dir=profile_dir,
        )
        self.assertEqual(calls["n"], 2)

    def test_resolve_calibration_text_respects_max_chars(self) -> None:
        text = resolve_calibration_text("第一句。第二句很长很长很长。", max_chars=5)
        self.assertTrue(text)
        self.assertLessEqual(len(text), 5)
        self.assertEqual(resolve_calibration_text("你好。", max_chars=0), "")
