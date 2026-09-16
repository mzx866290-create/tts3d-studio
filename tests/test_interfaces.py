from __future__ import annotations

import io
import importlib
import inspect
import os
import shutil
import sys
import types
import unittest
import uuid
from pathlib import Path

import tts3d_app.cli as cli
from tts3d_app.cli import build_parser
from tts3d_app.config import HF_HOME, HF_HUB_CACHE, OUTPUT_DIR, VOICE_PROFILE_DIR, configure_runtime
from tts3d_app.presets import PRESETS
from tts3d_app.service import TTSStudioService
from tts3d_app.tts_engines import ENGINE_QWEN3, ENGINE_QWEN3_BASE, TTSModelOption
from tts3d_app.ui import create_demo, describe_tts_engine_ui_state, history_table_rows


class StubProvider:
    def __init__(self, engine_key: str, model_keys: list[str]) -> None:
        self.engine_key = engine_key
        self._model_keys = model_keys

    def model_options(self) -> tuple[TTSModelOption, ...]:
        return tuple(TTSModelOption(key=model_key, label=model_key) for model_key in self._model_keys)

    def resolve_model_key(self, model_key: str | None) -> str:
        return model_key or self._model_keys[0]

    def load_model(self, model_key: str, device_name: str, torch_dtype):
        del model_key, device_name, torch_dtype
        return object()

    def generate_audio(
        self,
        model,
        *,
        text: str,
        voice_description: str,
        seed: int,
        reference_audio_path: str | None,
        reference_text: str,
    ):
        del model, text, voice_description, seed, reference_audio_path, reference_text
        raise AssertionError("generate_audio should not be called in this test")


class FakeFastMCP:
    def __init__(self, name: str) -> None:
        self.name = name
        self.tools: dict[str, dict[str, object]] = {}
        self.resources: dict[str, object] = {}

    def tool(self, *, name: str, description: str):
        def decorator(fn):
            self.tools[name] = {"fn": fn, "description": description}
            return fn

        return decorator

    def resource(self, uri: str):
        def decorator(fn):
            self.resources[uri] = fn
            return fn

        return decorator

    def run(self) -> None:
        return None


class InterfacesTests(unittest.TestCase):
    def create_workspace_dir(self) -> Path:
        parent = Path.cwd() / "outputs" / "test_runs"
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / f"run_{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
        return path

    def create_service(self) -> TTSStudioService:
        temp_dir = self.create_workspace_dir()
        return TTSStudioService(
            model_name="qwen-test-model",
            output_dir=temp_dir,
            hrir_file=temp_dir / "missing_hrir.npz",
            providers={
                ENGINE_QWEN3: StubProvider(ENGINE_QWEN3, ["qwen-test-model"]),
                ENGINE_QWEN3_BASE: StubProvider(ENGINE_QWEN3_BASE, ["qwen-base-test-model"]),
            },
            voice_profile_dir=temp_dir / "voices",
        )

    def test_configure_runtime_sets_huggingface_cache_env_vars(self) -> None:
        configure_runtime()
        self.assertEqual(os.environ["HF_HOME"], str(HF_HOME))
        self.assertEqual(os.environ["HF_HUB_CACHE"], str(HF_HUB_CACHE))
        self.assertEqual(os.environ["HUGGINGFACE_HUB_CACHE"], str(HF_HUB_CACHE))
        self.assertEqual(os.environ["TRANSFORMERS_CACHE"], str(HF_HUB_CACHE))

    def test_describe_tts_engine_ui_state_switches_fields(self) -> None:
        service = self.create_service()

        qwen_state = describe_tts_engine_ui_state(service, ENGINE_QWEN3)
        self.assertEqual(qwen_state[0], [("qwen-test-model", "qwen-test-model")])
        self.assertEqual(qwen_state[1], "qwen-test-model")
        self.assertTrue(qwen_state[2])
        self.assertFalse(qwen_state[3])

        clone_state = describe_tts_engine_ui_state(service, ENGINE_QWEN3_BASE)
        self.assertEqual(clone_state[1], "qwen-base-test-model")
        self.assertFalse(clone_state[2])
        self.assertTrue(clone_state[3])
        self.assertEqual(
            clone_state[0],
            [("qwen-base-test-model", "qwen-base-test-model")],
        )

    def test_presets_carry_calibration_text_field(self) -> None:
        for name, preset in PRESETS.items():
            self.assertIn("calibration_text", preset, f"preset '{name}' missing calibration_text")

    def test_describe_generation_route_switches_by_text_and_emotion(self) -> None:
        from tts3d_app.ui import describe_generation_route

        self.assertIn("直出", describe_generation_route("短文本。", "", ENGINE_QWEN3))
        long_text = "这是一句用来触发分块的测试。" * 40
        self.assertIn("锁定", describe_generation_route(long_text, "", ENGINE_QWEN3))
        self.assertIn("情感克隆", describe_generation_route("短文本。", "用激动的语气说", ENGINE_QWEN3))
        self.assertIn("Base Clone", describe_generation_route("你好", "", ENGINE_QWEN3_BASE))
        with unittest.mock.patch("tts3d_app.ui.TTS_DIRECT_SINGLE_CHUNK", False):
            self.assertIn("锁定", describe_generation_route("短文本。", "", ENGINE_QWEN3))

    def test_create_demo_builds_without_error(self) -> None:
        demo = create_demo(self.create_service())
        self.assertIsNotNone(demo)

    def test_history_table_rows_use_list_of_lists(self) -> None:
        rows = history_table_rows(
            [
                {
                    "name": "voice_a.wav",
                    "path": "/tmp/outputs/voice_a.wav",
                    "size": 1024,
                    "modified": "2026-09-08 09:05:05",
                },
                {
                    "name": "voice_b.wav",
                    "path": "/tmp/outputs/voice_b.wav",
                    "size": 2048,
                    "modified": "2026-09-08 09:06:05",
                },
            ]
        )

        self.assertEqual(
            rows,
            [
                ["voice_a.wav", "2026-09-08 09:05:05", 1024],
                ["voice_b.wav", "2026-09-08 09:06:05", 2048],
            ],
        )

    def test_cli_parser_accepts_qwen_base_clone_smoke_test_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "smoke-test",
                "--tts-engine",
                "qwen3_base",
                "--tts-model",
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "--reference-audio",
                "sample.wav",
                "--reference-text",
                "hello world",
            ]
        )

        self.assertEqual(args.tts_engine, "qwen3_base")
        self.assertEqual(args.tts_model, "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        self.assertEqual(args.reference_audio, "sample.wav")
        self.assertEqual(args.reference_text, "hello world")

    def test_cli_run_parser_uses_auto_port_by_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run"])

        self.assertIsNone(args.server_port)

    def test_run_app_prints_resolved_local_url(self) -> None:
        fake_demo = unittest.mock.Mock()
        fake_demo.launch.return_value = (object(), "http://127.0.0.1:7861/", None)

        with (
            unittest.mock.patch.object(cli, "configure_runtime"),
            unittest.mock.patch.object(cli, "create_demo", return_value=fake_demo),
            unittest.mock.patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            exit_code = cli.run_app("127.0.0.1", None)

        self.assertEqual(exit_code, 0)
        self.assertEqual(fake_demo.launch.call_args.kwargs["server_name"], "127.0.0.1")
        self.assertIsNone(fake_demo.launch.call_args.kwargs["server_port"])
        self.assertTrue(fake_demo.launch.call_args.kwargs["prevent_thread_lock"])
        self.assertTrue(fake_demo.launch.call_args.kwargs["inbrowser"])
        self.assertEqual(
            fake_demo.launch.call_args.kwargs["allowed_paths"],
            [str(OUTPUT_DIR.resolve()), str(VOICE_PROFILE_DIR.resolve())],
        )
        fake_demo.block_thread.assert_called_once_with()
        self.assertIn("http://127.0.0.1:7861/", stdout.getvalue())

    def test_mcp_generate_tool_exposes_qwen_base_clone_parameters(self) -> None:
        fake_fastmcp_module = types.SimpleNamespace(FastMCP=FakeFastMCP)
        with unittest.mock.patch.dict(sys.modules, {"fastmcp": fake_fastmcp_module}):
            import tts3d_app.mcp_server as mcp_server

            mcp_server = importlib.reload(mcp_server)
            server = mcp_server.create_mcp_server(service=self.create_service())
            tool_function = server.tools["generate_3d_speech"]["fn"]
            parameters = inspect.signature(tool_function).parameters

            self.assertIn("tts_engine", parameters)
            self.assertIn("tts_model_key", parameters)
            self.assertIn("reference_audio_path", parameters)
            self.assertIn("reference_text", parameters)
            self.assertIn("speed_factor", parameters)
            self.assertIn("pitch_semitones", parameters)
            self.assertIn("calibration_text", parameters)
            self.assertIn("emotion_instruct", parameters)

            # Check batch_generate tool exists
            self.assertIn("batch_generate", server.tools)

            capabilities = server.resources["capabilities://tts"]()
            self.assertIn("Long text", capabilities)
            self.assertIn("MAX_TTS_CHUNK_CHARS", capabilities)

            # Check resources exist
            self.assertIn("presets://list", server.resources)
            self.assertIn("capabilities://tts", server.resources)


if __name__ == "__main__":
    unittest.main()
