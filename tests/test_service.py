from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch
import wave
from unittest import mock

from tts3d_app.audio_engine import MODE_MONO, MODE_STEREO, PATH_CLOCKWISE
from tts3d_app.config import MAX_TTS_CHUNK_CHARS
from tts3d_app.service import GenerationCancelled, GenerationRequest, TTSStudioService
from tts3d_app.text_chunking import split_text_for_tts
from tts3d_app.tts_engines import (
    ENGINE_COSYVOICE2,
    ENGINE_QWEN3,
    ENGINE_QWEN3_BASE,
    TTSModelOption,
)
from tts3d_app.voice_profile import resolve_calibration_text, voice_clip_id


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
        instruct: str = "",
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
                    "instruct": instruct,
                },
            )
        )
        audio = np.array([self.sample_value, -self.sample_value, self.sample_value / 2], dtype=np.float32)
        return audio, 24_000

    def generate_cloned_audio(
        self,
        model,
        *,
        text: str,
        seed: int,
        ref_audio: str | tuple[np.ndarray, int],
        ref_text: str,
        instruct: str = "",
    ) -> tuple[np.ndarray, int]:
        if isinstance(ref_audio, tuple):
            reference_audio_path = "<memory>"
        else:
            reference_audio_path = ref_audio
        return self.generate_audio(
            model,
            text=text,
            voice_description="",
            seed=seed,
            reference_audio_path=reference_audio_path,
            reference_text=ref_text,
            instruct=instruct,
        )


class CosyVoiceStubProvider:
    """Clone-only stub mirroring the CosyVoice2 provider contract."""

    engine_key = ENGINE_COSYVOICE2

    def __init__(self) -> None:
        self.clone_calls: list[dict[str, object]] = []

    def model_options(self) -> tuple[TTSModelOption, ...]:
        return (TTSModelOption(key=self.engine_key, label="cosyvoice2"),)

    def resolve_model_key(self, model_key: str | None) -> str:
        return self.engine_key

    def load_model(self, model_key: str, device_name: str, torch_dtype) -> dict[str, str]:
        del model_key, device_name, torch_dtype
        return {"engine": self.engine_key}

    def generate_audio(self, model, **kwargs):
        raise AssertionError("cosyvoice2 is clone-only and must not be used as a primary engine")

    def generate_cloned_audio(
        self,
        model,
        *,
        text: str,
        seed: int,
        ref_audio: str | tuple[np.ndarray, int],
        ref_text: str,
        instruct: str = "",
    ) -> tuple[np.ndarray, int]:
        self.clone_calls.append(
            {
                "text": text,
                "seed": seed,
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "instruct": instruct,
            }
        )
        return np.array([0.3, -0.3, 0.15], dtype=np.float32), 24_000


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        # Force the resolver down the CPU path regardless of the host's real devices.
        self._cuda_patcher = mock.patch("torch.cuda.is_available", return_value=False)
        self._cuda_patcher.start()
        self._mps_patcher = mock.patch("torch.backends.mps.is_available", return_value=False)
        self._mps_patcher.start()

    def tearDown(self) -> None:
        self._mps_patcher.stop()
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
            providers={
                ENGINE_QWEN3: qwen_provider,
                ENGINE_QWEN3_BASE: clone_provider,
                ENGINE_COSYVOICE2: CosyVoiceStubProvider(),
            },
            voice_profile_dir=temp_dir / "voices",
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
        calibration_text: str = "",
        emotion_instruct: str = "",
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
            calibration_text=calibration_text,
            emotion_instruct=emotion_instruct,
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
        self.assertEqual(calls[0]["max_new_tokens"], 2048)
        self.assertIn("generator", calls[0])

        memory_calls: list[dict[str, object]] = []

        class MemoryModel:
            def generate_voice_clone(self, **kwargs):
                memory_calls.append(kwargs)
                return [np.array([0.1, -0.1], dtype=np.float32)], 24_000

        waveform = np.array([0.2, -0.2], dtype=np.float32)
        provider.generate_cloned_audio(
            MemoryModel(),
            text="lock me",
            seed=7,
            ref_audio=(waveform, 24_000),
            ref_text="first chunk",
        )
        self.assertEqual(memory_calls[0]["ref_audio"], (waveform, 24_000))
        self.assertEqual(memory_calls[0]["ref_text"], "first chunk")
        self.assertFalse(memory_calls[0]["x_vector_only_mode"])

    def test_voice_design_provider_forwards_seed_and_token_cap(self) -> None:
        from tts3d_app.tts_engines import QwenTTSProvider

        calls: list[dict[str, object]] = []

        class FakeModel:
            def generate_voice_design(self, **kwargs):
                calls.append(kwargs)
                return [np.array([0.1, -0.1], dtype=np.float32)], 24_000

        audio, sample_rate = QwenTTSProvider().generate_audio(
            FakeModel(),
            text="hello",
            voice_description="calm",
            seed=99,
            reference_audio_path=None,
            reference_text="",
        )
        self.assertEqual(sample_rate, 24_000)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(calls[0]["text"], "hello")
        self.assertEqual(calls[0]["instruct"], "calm")
        self.assertEqual(calls[0]["max_new_tokens"], 2048)
        self.assertIn("generator", calls[0])

    def test_long_text_is_split_into_multiple_tts_calls(self) -> None:
        service, qwen_provider, clone_provider, _ = self.create_service()
        long_text = "这是一句用来触发分块的测试。" * 40
        request = self.build_request(render_mode=MODE_MONO, text=long_text)

        result = service.generate(request)
        first_qwen_calls = len(qwen_provider.generate_calls)
        first_clone_calls = len(clone_provider.generate_calls)

        self.assertEqual(first_qwen_calls, 1)
        chunks = split_text_for_tts(long_text, max_chars=MAX_TTS_CHUNK_CHARS)
        calib = resolve_calibration_text()
        self.assertTrue(calib)
        self.assertEqual(qwen_provider.generate_calls[0][1]["text"], calib)
        self.assertNotEqual(calib, chunks[0].text)
        self.assertEqual(qwen_provider.generate_calls[0][1]["seed"], 7)
        self.assertEqual(len(clone_provider.generate_calls), len(chunks))
        self.assertTrue(all(call[1]["seed"] == 7 for call in clone_provider.generate_calls))
        self.assertTrue(all(call[1]["reference_text"] == calib for call in clone_provider.generate_calls))
        self.assertTrue(all(bool(call[1]["reference_text"]) for call in clone_provider.generate_calls))
        self.assertEqual(clone_provider.generate_calls[0][1]["reference_audio_path"], "<memory>")
        self.assertIn("chunks=", result.status)
        self.assertIn("speaker_lock=icl", result.status)
        self.assertNotIn("promote", result.status)
        self.assertIn("clip=", result.status)
        self.assertEqual(
            result.clip_id,
            voice_clip_id("qwen-test-model", "A calm Chinese voice.", 7, calib),
        )
        self.assertTrue(Path(result.file_path).exists())

        cached = service.generate(self.build_request(render_mode=MODE_STEREO, text=long_text))
        self.assertIn("dry_cache=hit", cached.status)
        self.assertEqual(len(qwen_provider.generate_calls), first_qwen_calls)
        self.assertEqual(len(clone_provider.generate_calls), first_clone_calls)

    def test_voice_design_reuses_calibration_clip_across_different_texts(self) -> None:
        service, qwen_provider, clone_provider, _ = self.create_service()
        first_text = "这是第一篇用来触发分块的测试。" * 40
        second_text = "这是另一篇完全不同的正文内容。" * 40
        first = service.generate(self.build_request(render_mode=MODE_MONO, text=first_text))
        qwen_after_first = len(qwen_provider.generate_calls)
        clone_after_first = len(clone_provider.generate_calls)
        second = service.generate(self.build_request(render_mode=MODE_MONO, text=second_text))
        self.assertEqual(len(qwen_provider.generate_calls), qwen_after_first)
        self.assertGreater(len(clone_provider.generate_calls), clone_after_first)
        self.assertEqual(first.clip_id, second.clip_id)
        self.assertTrue(first.clip_id)
        self.assertIn(first.clip_id, second.status)

    def test_short_text_uses_direct_voicedesign_without_clone(self) -> None:
        service, qwen_provider, clone_provider, _ = self.create_service()
        result = service.generate(self.build_request(render_mode=MODE_MONO, text="短文本。"))
        # Single-chunk short text goes straight to VoiceDesign: no calibration clip,
        # no clone call, the synth text IS the content text.
        self.assertEqual(len(qwen_provider.generate_calls), 1)
        self.assertEqual(qwen_provider.generate_calls[0][1]["text"], "短文本。")
        self.assertEqual(clone_provider.generate_calls, [])
        self.assertEqual(service.get_provider(ENGINE_COSYVOICE2).clone_calls, [])
        self.assertEqual(result.clip_id, "")
        self.assertIn("dry_cache=miss:direct", result.status)
        self.assertTrue(Path(result.file_path).exists())

    def test_short_text_direct_can_be_disabled_to_restore_lock(self) -> None:
        service, qwen_provider, clone_provider, _ = self.create_service()
        with mock.patch("tts3d_app.service.TTS_DIRECT_SINGLE_CHUNK", False):
            result = service.generate(self.build_request(render_mode=MODE_MONO, text="短文本。"))
        calib = resolve_calibration_text()
        self.assertEqual(qwen_provider.generate_calls[0][1]["text"], calib)
        self.assertEqual(len(clone_provider.generate_calls), 1)
        self.assertIn("speaker_lock=icl", result.status)
        self.assertTrue(result.clip_id)

    def test_short_text_with_emotion_instruct_still_locks(self) -> None:
        service, _, clone_provider, _ = self.create_service()
        service.generate(
            self.build_request(render_mode=MODE_MONO, text="短文本。", emotion_instruct="激动地说")
        )
        self.assertEqual(len(clone_provider.generate_calls), 0)
        self.assertEqual(len(service.get_provider(ENGINE_COSYVOICE2).clone_calls), 1)

    def test_preview_voice_reference_writes_clip_without_body_text(self) -> None:
        service, qwen_provider, clone_provider, _ = self.create_service()
        path, clip_id, seed, status = service.preview_voice_reference(
            voice_description="A calm Chinese voice.",
            seed=7,
            tts_engine=ENGINE_QWEN3,
            tts_model_key="qwen-test-model",
        )
        self.assertEqual(seed, 7)
        self.assertTrue(clip_id)
        self.assertTrue(Path(path).exists())
        self.assertIn("clip=", status)
        self.assertEqual(len(qwen_provider.generate_calls), 1)
        self.assertEqual(qwen_provider.generate_calls[0][1]["text"], resolve_calibration_text())
        self.assertEqual(clone_provider.generate_calls, [])
        again = service.preview_voice_reference(
            voice_description="A calm Chinese voice.",
            seed=7,
            tts_engine=ENGINE_QWEN3,
            tts_model_key="qwen-test-model",
        )
        self.assertEqual(again[1], clip_id)
        self.assertEqual(len(qwen_provider.generate_calls), 1)

    def test_request_level_calibration_text_overrides_global_default(self) -> None:
        service, qwen_provider, clone_provider, _ = self.create_service()
        custom_calib = "她在耳边轻声安慰：别怕，我一直都在。"
        self.assertNotEqual(custom_calib, resolve_calibration_text())

        result = service.generate(
            self.build_request(
                render_mode=MODE_MONO,
                text="这是一句用来触发分块锁定的较长测试文本。" * 40,
                calibration_text=custom_calib,
            )
        )

        self.assertEqual(qwen_provider.generate_calls[0][1]["text"], custom_calib)
        self.assertEqual(clone_provider.generate_calls[0][1]["reference_text"], custom_calib)
        self.assertEqual(
            result.clip_id,
            voice_clip_id("qwen-test-model", "A calm Chinese voice.", 7, custom_calib),
        )
        self.assertNotEqual(
            result.clip_id,
            voice_clip_id("qwen-test-model", "A calm Chinese voice.", 7, resolve_calibration_text()),
        )

    def test_request_level_calibration_text_partitions_dry_audio_cache(self) -> None:
        service, qwen_provider, _, _ = self.create_service()
        custom_calib = "低沉沙哑的嗓音在夜里响起，像大提琴的最低弦。"

        service.generate(self.build_request(render_mode=MODE_MONO, text="短文本。", calibration_text=custom_calib))
        service.generate(self.build_request(render_mode=MODE_MONO, text="短文本。"))
        # Different calibration text must not reuse the cached dry audio.
        self.assertEqual(len(qwen_provider.generate_calls), 2)

        service.generate(self.build_request(render_mode=MODE_MONO, text="短文本。", calibration_text=custom_calib))
        # Same calibration text hits the cache again.
        self.assertEqual(len(qwen_provider.generate_calls), 2)

    def test_preview_voice_reference_uses_request_calibration_text(self) -> None:
        service, qwen_provider, _, _ = self.create_service()
        custom_calib = "温柔的女声轻轻叹了口气：今晚真安静啊。"

        _, clip_id, _, _ = service.preview_voice_reference(
            voice_description="A calm Chinese voice.",
            seed=7,
            tts_engine=ENGINE_QWEN3,
            tts_model_key="qwen-test-model",
            calibration_text=custom_calib,
        )

        self.assertEqual(qwen_provider.generate_calls[0][1]["text"], custom_calib)
        self.assertEqual(
            clip_id,
            voice_clip_id("qwen-test-model", "A calm Chinese voice.", 7, custom_calib),
        )

    def test_emotion_instruct_routes_to_cosyvoice2_clone_engine(self) -> None:
        service, qwen_provider, clone_provider, _ = self.create_service()
        emotion = "用激动而悲伤的语气说，声音颤抖"

        result = service.generate(
            self.build_request(render_mode=MODE_MONO, text="短文本。", emotion_instruct=emotion)
        )

        cosy = service.get_provider(ENGINE_COSYVOICE2)
        self.assertEqual(len(cosy.clone_calls), 1)
        self.assertEqual(cosy.clone_calls[0]["text"], "短文本。")
        self.assertEqual(cosy.clone_calls[0]["instruct"], emotion)
        self.assertEqual(cosy.clone_calls[0]["ref_text"], resolve_calibration_text())
        self.assertIsInstance(cosy.clone_calls[0]["ref_audio"], tuple)
        # The in-family clone engine must stay idle when CosyVoice2 handles the lock chain.
        self.assertEqual(len(clone_provider.generate_calls), 0)
        # Voice identity is untouched by emotion: same clip_id as a plain request.
        self.assertEqual(
            result.clip_id,
            voice_clip_id("qwen-test-model", "A calm Chinese voice.", 7, resolve_calibration_text()),
        )
        self.assertTrue(Path(result.file_path).exists())

    def test_emotion_instruct_partitions_dry_audio_cache(self) -> None:
        service, _, _, _ = self.create_service()
        service.generate(self.build_request(render_mode=MODE_MONO, text="短文本。", emotion_instruct="平静叙述"))
        service.generate(self.build_request(render_mode=MODE_MONO, text="短文本。", emotion_instruct="激动哽咽"))
        cosy = service.get_provider(ENGINE_COSYVOICE2)
        # Different emotion instructs must not reuse the cached dry audio.
        self.assertEqual(len(cosy.clone_calls), 2)

        service.generate(self.build_request(render_mode=MODE_MONO, text="短文本。", emotion_instruct="平静叙述"))
        # Same emotion instruct hits the cache again.
        self.assertEqual(len(cosy.clone_calls), 2)

    def test_cosyvoice2_provider_formats_instruct_and_materializes_prompt_wav(self) -> None:
        import soundfile as sf

        from tts3d_app.tts_engines import CosyVoice2Provider

        class FakeWorker:
            def __init__(self) -> None:
                self.jobs: list[dict[str, object]] = []

            def synthesize(self, *, text, instruct_text, prompt_wav, out_wav, seed) -> int:
                self.jobs.append(
                    {
                        "text": text,
                        "instruct_text": instruct_text,
                        "prompt_wav": prompt_wav,
                        "seed": seed,
                    }
                )
                sf.write(out_wav, np.array([[0.1, -0.1, 0.05]], dtype=np.float32), 24_000)
                return 24_000

        workspace = self.create_workspace_dir()
        provider = CosyVoice2Provider(model_dir=workspace / "m", repo_dir=workspace / "r")
        worker = FakeWorker()
        audio, sample_rate = provider.generate_cloned_audio(
            worker,
            text="锁我",
            seed=7,
            ref_audio=(np.array([0.2, -0.2], dtype=np.float32), 24_000),
            ref_text="参考转写不参与 instruct2",
            instruct="用激动的语气说",
        )

        self.assertEqual(sample_rate, 24_000)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(len(worker.jobs), 1)
        self.assertEqual(worker.jobs[0]["text"], "锁我")
        self.assertEqual(worker.jobs[0]["seed"], 7)
        self.assertTrue(str(worker.jobs[0]["instruct_text"]).startswith("用激动的语气说"))
        self.assertTrue(str(worker.jobs[0]["instruct_text"]).endswith("<|endofprompt|>"))
        self.assertTrue(str(worker.jobs[0]["prompt_wav"]).endswith(".wav"))
        self.assertTrue(Path(str(worker.jobs[0]["prompt_wav"])).exists())

        # A missing <|endofprompt|> marker is appended exactly once.
        provider.generate_cloned_audio(
            worker,
            text="再来",
            seed=7,
            ref_audio=(np.array([0.2], dtype=np.float32), 24_000),
            ref_text="",
            instruct="平静<|endofprompt|>",
        )
        self.assertEqual(worker.jobs[1]["instruct_text"], "平静<|endofprompt|>")

    def test_cosyvoice2_provider_is_clone_only(self) -> None:
        from tts3d_app.tts_engines import CosyVoice2Provider

        provider = CosyVoice2Provider(model_dir=Path("/unused"), repo_dir=Path("/unused"))
        with self.assertRaisesRegex(RuntimeError, "clone-only"):
            provider.generate_audio(
                None,
                text="x",
                voice_description="",
                seed=1,
                reference_audio_path=None,
                reference_text="",
            )
        self.assertEqual(provider.resolve_model_key(""), ENGINE_COSYVOICE2)

    def test_seed_everything_seeds_mps_when_available(self) -> None:
        with (
            mock.patch("torch.backends.mps.is_available", return_value=True),
            mock.patch("torch.mps.manual_seed") as mps_seed,
        ):
            TTSStudioService.seed_everything(42)
        self.assertGreaterEqual(mps_seed.call_count, 1)
        mps_seed.assert_called_with(42)

    def test_seeded_multinomial_is_deterministic_for_same_seed(self) -> None:
        from tts3d_app.tts_engines import use_seeded_multinomial

        probs = torch.softmax(torch.arange(8, dtype=torch.float32), dim=0)

        def draw(seed: int) -> torch.Tensor:
            with use_seeded_multinomial(seed):
                return torch.stack([torch.multinomial(probs, 1) for _ in range(16)])

        self.assertTrue(torch.equal(draw(123), draw(123)))
        self.assertFalse(torch.equal(draw(123), draw(124)))

    def test_delete_audio_only_allows_files_inside_output_dir(self) -> None:
        service, _, _, temp_dir = self.create_service()
        inside = temp_dir / "keep.wav"
        inside.write_bytes(b"RIFF")
        outside = temp_dir.parent / f"outside_{temp_dir.name}.wav"
        outside.write_bytes(b"RIFF")
        self.addCleanup(lambda: outside.unlink(missing_ok=True))

        self.assertTrue(service.delete_audio(str(inside)))
        self.assertFalse(inside.exists())
        self.assertFalse(service.delete_audio(str(outside)))
        self.assertTrue(outside.exists())

    def test_cancel_stops_between_long_text_chunks(self) -> None:
        service, qwen_provider, clone_provider, _ = self.create_service()
        original = qwen_provider.generate_audio

        def wrapped(model, **kwargs):
            result = original(model, **kwargs)
            if len(qwen_provider.generate_calls) == 1:
                service.request_cancel()
            return result

        qwen_provider.generate_audio = wrapped  # type: ignore[method-assign]
        long_text = "这是一句用来触发分块的测试。" * 40
        with self.assertRaises(GenerationCancelled):
            service.generate(self.build_request(render_mode=MODE_MONO, text=long_text))
        self.assertEqual(len(qwen_provider.generate_calls), 1)
        self.assertEqual(clone_provider.generate_calls, [])

        short = service.generate(self.build_request(render_mode=MODE_MONO, text="短文本。"))
        self.assertTrue(Path(short.file_path).exists())
        self.assertEqual(len(qwen_provider.generate_calls), 2)
        # 取消恢复后的短文走单块直出，克隆引擎不应被调用
        self.assertEqual(clone_provider.generate_calls, [])

    def test_cancel_during_first_chunk_raises_without_writing_file(self) -> None:
        service, qwen_provider, _, temp_dir = self.create_service()

        def wrapped(model, **kwargs):
            del model, kwargs
            service.request_cancel()
            raise RuntimeError("Qwen3-TTS inference failed: cancelled")

        qwen_provider.generate_audio = wrapped  # type: ignore[method-assign]
        with self.assertRaises(GenerationCancelled):
            service.generate(self.build_request(render_mode=MODE_MONO, text="你好。"))
        self.assertEqual(list(temp_dir.glob("voice_*.wav")), [])

    def test_interruptible_talker_generate_raises_when_cancel_is_set(self) -> None:
        import threading

        from tts3d_app.tts_engines import interruptible_talker_generate

        class FakeTalker:
            def generate(self, **kwargs):
                criteria = kwargs["stopping_criteria"]
                dummy_ids = torch.zeros((1, 1), dtype=torch.long)
                dummy_scores = torch.zeros((1, 1))
                return bool(criteria(dummy_ids, dummy_scores))

        class FakeModel:
            def __init__(self) -> None:
                self.talker = FakeTalker()

        model = FakeModel()
        cancel_event = threading.Event()
        with interruptible_talker_generate(model, cancel_event):
            self.assertFalse(model.talker.generate())
            cancel_event.set()
            with self.assertRaises(GenerationCancelled):
                model.talker.generate()


if __name__ == "__main__":
    unittest.main()
