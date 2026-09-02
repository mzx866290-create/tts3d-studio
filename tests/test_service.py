from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import wave
from unittest import mock

from tts3d_app.audio_engine import MODE_MONO, MODE_STEREO, PATH_CLOCKWISE
from tts3d_app.service import GenerationRequest, TTSStudioService
from tts3d_app.tts_engines import ENGINE_QWEN3, ENGINE_QWEN3_BASE, TTSModelOption


class StubProvider:
    def __init__(self, engine_key: str, model_keys: list[str], sample_value: float) -> None:
        self.engine_key = engine_key
        self._model_keys = model_keys
        self.sample_value = sample_value
        self.load_calls: list[tuple[str, str]] = []
        self.generate_calls: list[tuple[str, dict[str, object]]] = []

    def model_options(self) -> tuple[TTSModelOption, ...]:
        return tuple(TTSModelOption(key=model_key, label=model_key) for model_key in self._model_keys)

    def resolve_model_key(self, model_key: str | None) -> str:
        if not model_key:
            return self._model_keys[0]
        if model_key not in self._model_keys:
            raise RuntimeError(f"Unsupported model: {model_key}")
        return model_key

    def load_model(self, model_key: str, device_name: str, torch_dtype) -> dict[str, str]:
        del torch_dtype
        self.load_calls.append((model_key, device_name))
        return {"engine": self.engine_key, "model_key": model_key}

    def generate_audio(
        self,
        model,
        *,
        text: str,
        voice_description: str,
        seed: int,
        reference_audio_path: str | None,
        reference_text: str,
    ) -> tuple[np.ndarray, int]:
        self.generate_calls.append(
            (
                model["model_key"],
                {
                    "text": text,
                    "voice_description": voice_description,
                    "seed": seed,
                    "reference_audio_path": reference_audio_path,
                    "reference_text": reference_text,
                },
            )
        )
        audio = np.array([self.sample_value, -self.sample_value, self.sample_value / 2], dtype=np.float32)
        return audio, 24_000


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._cuda_patcher = mock.patch("torch.cuda.is_available", return_value=False)
        self._cuda_patcher.start()

    def tearDown(self) -> None:
        self._cuda_patcher.stop()

    def create_workspace_dir(self) -> Path:
        parent = Path.cwd() / "outputs" / "test_runs"
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / f"run_{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def create_service(self) -> tuple[TTSStudioService, StubProvider, StubProvider, Path]:
        temp_dir = self.create_workspace_dir()
        qwen_provider = StubProvider(ENGINE_QWEN3, ["qwen-test-model"], 0.2)
        clone_provider = StubProvider(ENGINE_QWEN3_BASE, ["qwen-base-a", "qwen-base-b"], 0.4)
        service = TTSStudioService(
            model_name="qwen-test-model",
            output_dir=temp_dir,
            hrir_file=temp_dir / "missing_hrir.npz",
            providers={ENGINE_QWEN3: qwen_provider, ENGINE_QWEN3_BASE: clone_provider},
        )
        return service, qwen_provider, clone_provider, temp_dir

    def create_reference_audio(self, directory: Path, name: str = "ref.wav") -> Path:
        path = directory / name
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24_000)
            wav_file.writeframes(b"\x00\x00" * 24)
        return path

    def build_request(
        self,
        *,
        render_mode: str,
        tts_engine: str = ENGINE_QWEN3,
        tts_model_key: str = "",
        text: str = "same text",
        prompt: str = "A calm Chinese voice.",
        reference_audio_path: str | None = None,
        reference_text: str = "",
    ) -> GenerationRequest:
        return GenerationRequest(
            preset_key="",
            voice_description=prompt,
            text=text,
            seed=7,
            use_random_seed=False,
            render_mode=render_mode,
            static_azimuth_deg=270,
            static_distance_m=0.3,
            dynamic_path=PATH_CLOCKWISE,
            dynamic_cycle_time_s=8.0,
            dynamic_start_azimuth_deg=270,
            dynamic_distance_m=0.2,
            tts_engine=tts_engine,
            tts_model_key=tts_model_key,
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
        )

    def test_dry_audio_cache_reuses_output_for_same_engine_and_model(self) -> None:
        service, qwen_provider, _, _ = self.create_service()

        first = service.generate(self.build_request(render_mode=MODE_MONO, tts_model_key="qwen-test-model"))
        second = service.generate(self.build_request(render_mode=MODE_STEREO, tts_model_key="qwen-test-model"))

        self.assertEqual(qwen_provider.load_calls, [("qwen-test-model", "cpu")])
        self.assertEqual(len(qwen_provider.generate_calls), 1)
        self.assertIn("engine=qwen3", first.status)
        self.assertIn("model=qwen-test-model", first.status)
        self.assertIn("dry_cache=miss", first.status)
        self.assertIn("dry_cache=hit", second.status)
        self.assertTrue(Path(first.file_path).exists())
        self.assertTrue(Path(second.file_path).exists())

    def test_model_cache_and_dry_cache_invalidate_for_qwen_base_model_change(self) -> None:
        service, _, clone_provider, temp_dir = self.create_service()
        reference_audio = self.create_reference_audio(temp_dir)

        service.generate(
            self.build_request(
                render_mode=MODE_MONO,
                tts_engine=ENGINE_QWEN3_BASE,
                tts_model_key="qwen-base-a",
                prompt="",
                reference_audio_path=str(reference_audio),
                reference_text="reference text",
            )
        )
        service.generate(
            self.build_request(
                render_mode=MODE_STEREO,
                tts_engine=ENGINE_QWEN3_BASE,
                tts_model_key="qwen-base-b",
                prompt="",
                reference_audio_path=str(reference_audio),
                reference_text="reference text",
            )
        )

        self.assertEqual(
            clone_provider.load_calls,
            [("qwen-base-a", "cpu"), ("qwen-base-b", "cpu")],
        )
        self.assertEqual(len(clone_provider.generate_calls), 2)

    def test_dry_audio_cache_invalidates_when_reference_audio_changes(self) -> None:
        service, _, clone_provider, temp_dir = self.create_service()
        first_reference = self.create_reference_audio(temp_dir, "ref_a.wav")
        second_reference = self.create_reference_audio(temp_dir, "ref_b.wav")

        service.generate(
            self.build_request(
                render_mode=MODE_MONO,
                tts_engine=ENGINE_QWEN3_BASE,
                tts_model_key="qwen-base-a",
                prompt="",
                reference_audio_path=str(first_reference),
                reference_text="reference text",
            )
        )
        service.generate(
            self.build_request(
                render_mode=MODE_MONO,
                tts_engine=ENGINE_QWEN3_BASE,
                tts_model_key="qwen-base-a",
                prompt="",
                reference_audio_path=str(second_reference),
                reference_text="reference text",
            )
        )

        self.assertEqual(len(clone_provider.generate_calls), 2)

    def test_qwen_base_clone_requires_reference_audio_but_reference_text_is_optional(self) -> None:
        service, _, clone_provider, temp_dir = self.create_service()
        reference_audio = self.create_reference_audio(temp_dir, "ref_validation.wav")

        with self.assertRaisesRegex(RuntimeError, "reference audio"):
            service.generate(
                self.build_request(
                    render_mode=MODE_MONO,
                    tts_engine=ENGINE_QWEN3_BASE,
                    tts_model_key="qwen-base-a",
                    prompt="",
                    reference_text="reference text",
                )
            )

        service.generate(
            self.build_request(
                render_mode=MODE_MONO,
                tts_engine=ENGINE_QWEN3_BASE,
                tts_model_key="qwen-base-a",
                prompt="",
                reference_audio_path=str(reference_audio),
                reference_text="",
            )
        )
        self.assertEqual(clone_provider.generate_calls[-1][1]["reference_text"], "")

    def test_qwen_base_provider_uses_voice_clone_with_optional_reference_text(self) -> None:
        from tts3d_app.tts_engines import QwenBaseCloneProvider

        provider = QwenBaseCloneProvider()
        temp_dir = self.create_workspace_dir()
        reference_audio = self.create_reference_audio(temp_dir, "ref_clone.wav")
        calls: list[dict[str, object]] = []

        class FakeModel:
            def generate_voice_clone(self, **kwargs):
                calls.append(kwargs)
                return [np.array([0.25, -0.25], dtype=np.float32)], 24_000

        audio, sample_rate = provider.generate_audio(
            FakeModel(),
            text="clone me",
            voice_description="",
            seed=123,
            reference_audio_path=str(reference_audio),
            reference_text="",
        )
        provider.generate_audio(
            FakeModel(),
            text="clone me",
            voice_description="",
            seed=123,
            reference_audio_path=str(reference_audio),
            reference_text="reference text",
        )

        self.assertEqual(sample_rate, 24_000)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(calls[0]["ref_audio"], str(reference_audio))
        self.assertIsNone(calls[0]["ref_text"])
        self.assertTrue(calls[0]["x_vector_only_mode"])
        self.assertEqual(calls[1]["ref_text"], "reference text")
        self.assertFalse(calls[1]["x_vector_only_mode"])


if __name__ == "__main__":
    unittest.main()
