from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from tts3d_app import config as config_module
from tts3d_app.service import TTSStudioService
from tts3d_app.tts_engines import ENGINE_QWEN3, ENGINE_QWEN3_BASE, TTSModelOption
from tts3d_app.webapp.server import create_app

import numpy as np


class FakeProvider:
    def __init__(self, engine_key: str, model_keys: list[str]) -> None:
        self.engine_key = engine_key
        self._model_keys = model_keys

    def model_options(self) -> tuple[TTSModelOption, ...]:
        return tuple(TTSModelOption(key=k, label=k) for k in self._model_keys)

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
        del model, voice_description, seed, reference_audio_path, reference_text
        samples = np.zeros(2400, dtype=np.float32)
        samples[-100:] = 0.5
        return samples, 24_000

    def generate_cloned_audio(
        self,
        model,
        *,
        text: str,
        seed: int,
        ref_audio: tuple[np.ndarray, int] | str,
        ref_text: str,
        instruct: str = "",
    ):
        del model, seed, ref_audio, ref_text
        assert text.strip(), "cloned chunk text should not be empty"
        samples = np.zeros(1800, dtype=np.float32)
        samples[-50:] = 0.3
        return samples, 24_000


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event = ""
        data = ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = line[len("data: "):]
        if event:
            events.append((event, json.loads(data) if data else {}))
    return events


class WebappTests(unittest.TestCase):
    def setUp(self) -> None:
        parent = Path.cwd() / "outputs" / "test_runs"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp_dir = parent / f"web_{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))

        self.service = TTSStudioService(
            model_name="qwen-test-model",
            output_dir=self.temp_dir,
            hrir_file=self.temp_dir / "missing_hrir.npz",
            providers={
                ENGINE_QWEN3: FakeProvider(ENGINE_QWEN3, ["qwen-test-model"]),
                ENGINE_QWEN3_BASE: FakeProvider(ENGINE_QWEN3_BASE, ["qwen-base-test-model"]),
            },
            voice_profile_dir=self.temp_dir / "voices",
        )
        self.client = TestClient(create_app(service=self.service))

    def test_meta_reports_engines_modes_and_hrir_availability(self) -> None:
        resp = self.client.get("/api/meta")
        self.assertEqual(resp.status_code, 200)
        meta = resp.json()
        engine_keys = [engine["key"] for engine in meta["engines"]]
        self.assertIn("qwen3", engine_keys)
        self.assertIn("qwen3_base", engine_keys)
        modes = {mode["key"]: mode["available"] for mode in meta["modes"]}
        self.assertTrue(modes["mono"])
        # 测试环境没有 HRIR 数据
        self.assertFalse(modes["static_hrir"])
        self.assertFalse(modes["dynamic_hrir"])
        self.assertEqual(meta["presets"][0]["key"], "默认")
        self.assertIn("text", meta["presets"][0])

    def test_route_hint_endpoint(self) -> None:
        resp = self.client.post(
            "/api/route-hint",
            json={"text": "短文本。", "emotion_instruct": "", "tts_engine": "qwen3"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("直出", resp.json()["text"])

        resp = self.client.post(
            "/api/route-hint",
            json={"text": "短文本。", "emotion_instruct": "用激动的语气说", "tts_engine": "qwen3"},
        )
        self.assertIn("情感克隆", resp.json()["text"])

        # 显式锁定音色：短文本也走锁定克隆
        resp = self.client.post(
            "/api/route-hint",
            json={"text": "短文本。", "emotion_instruct": "", "tts_engine": "qwen3", "lock_timbre": True},
        )
        self.assertIn("音色锁定", resp.json()["text"])

    def test_generate_with_lock_timbre_forces_clone_chain(self) -> None:
        """短文本 + lock_timbre：跳过直出，走参考音克隆（产生 clip 文件）。"""
        resp = self.client.post(
            "/api/generate",
            json={
                "text": "锁定音色测试。",
                "render_mode": "mono",
                "seed": 11,
                "lock_timbre": True,
            },
        )
        self.assertEqual(resp.status_code, 200)
        events = parse_sse(resp.text)
        stages = [payload.get("stage") for event, payload in events if event == "progress"]
        self.assertIn("voice_lock", stages)
        terminal = [payload for event, payload in events if event == "result"]
        self.assertEqual(len(terminal), 1)
        self.assertIn("speaker_lock=icl", terminal[0]["status"])
        clip_dir = self.temp_dir / "voices"
        clip_files = list(clip_dir.glob("*.wav"))
        self.assertGreaterEqual(len(clip_files), 1)

    def test_generate_sse_stream_emits_progress_and_result(self) -> None:
        resp = self.client.post(
            "/api/generate",
            json={
                "text": "你好，这是一次接口测试。",
                "voice_description": "a calm voice",
                "render_mode": "mono",
                "seed": 7,
            },
        )
        self.assertEqual(resp.status_code, 200)
        events = parse_sse(resp.text)
        stages = [payload.get("stage") for event, payload in events if event == "progress"]
        self.assertIn("model_loading", stages)
        self.assertIn("spatial", stages)
        terminal = [payload for event, payload in events if event == "result"]
        self.assertEqual(len(terminal), 1)
        payload = terminal[0]
        self.assertTrue(Path(payload["file_path"]).exists())
        self.assertEqual(payload["seed"], 7)
        self.assertTrue(payload["audio_url"].startswith("/media?path="))
        self.assertGreaterEqual(len(payload["history"]), 1)

    def test_generate_error_surfaces_via_error_event(self) -> None:
        # 无 HRIR 数据时，静态 HRIR 模式应产生错误事件而不是 500
        resp = self.client.post(
            "/api/generate",
            json={"text": "你好。", "render_mode": "static_hrir", "seed": 7},
        )
        events = parse_sse(resp.text)
        errors = [payload for event, payload in events if event == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("HRIR", errors[0]["message"])

    def test_history_list_and_delete(self) -> None:
        self.client.post(
            "/api/generate",
            json={"text": "历史记录测试。", "render_mode": "mono", "seed": 3},
        )
        items = self.client.get("/api/history").json()
        self.assertGreaterEqual(len(items), 1)
        target = items[0]["path"]
        resp = self.client.delete(f"/api/history?path={target}")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])

        # 目录穿越必须被拒绝（ok=False，且不抛 500）
        evil = self.client.delete("/api/history?path=/etc/hosts")
        self.assertEqual(evil.status_code, 200)
        self.assertFalse(evil.json()["ok"])

    def test_media_rejects_paths_outside_allowed_roots(self) -> None:
        resp = self.client.get("/media", params={"path": "/etc/passwd"})
        self.assertEqual(resp.status_code, 403)

    def test_favorites_roundtrip(self) -> None:
        fav_file = self.temp_dir / "favorites.json"
        preset_file = self.temp_dir / "presets.json"
        with (
            mock.patch.object(config_module, "FAVORITES_FILE", fav_file),
            mock.patch.object(config_module, "PRESETS_FILE", preset_file),
        ):
            resp = self.client.post(
                "/api/favorites",
                json={
                    "name": "测试声线",
                    "prompt": "a warm voice",
                    "text": "你好",
                    "seed": 42,
                    "mode_label": "虚拟脑后",
                },
            )
            self.assertEqual(resp.status_code, 200)
            favs = self.client.get("/api/favorites").json()
            self.assertEqual([f["name"] for f in favs], ["测试声线"])

            deleted = self.client.delete("/api/favorites", params={"name": "测试声线"})
            self.assertEqual(deleted.status_code, 200)
            self.assertEqual(self.client.get("/api/favorites").json(), [])

    def test_spa_fallback_returns_build_hint_when_not_built(self) -> None:
        from tts3d_app.webapp.server import STATIC_DIR

        if (STATIC_DIR / "index.html").exists():
            self.skipTest("前端已构建")
        resp = self.client.get("/")
        self.assertIn("npm run build", resp.text)


if __name__ == "__main__":
    unittest.main()
