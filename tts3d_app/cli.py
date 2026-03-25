from __future__ import annotations

import argparse
import shutil
import sys
from typing import Sequence

import gradio as gr
import torch

from tts3d_app.audio_engine import MODE_MONO, PATH_CLOCKWISE
from tts3d_app.config import DEFAULT_SERVER_NAME, HRIR_FILE, MODEL_NAME, OUTPUT_DIR, configure_runtime
from tts3d_app.service import GenerationRequest, TTSStudioService
from tts3d_app.ui import create_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TTS 3D Studio")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="启动 Gradio 应用")
    run_parser.add_argument("--server-name", default=DEFAULT_SERVER_NAME)
    run_parser.add_argument("--server-port", type=int, default=7860)

    subparsers.add_parser("doctor", help="检查 Python、Torch、CUDA、SoX 和资源状态")

    smoke_parser = subparsers.add_parser("smoke-test", help="执行一次最小语音生成测试")
    smoke_parser.add_argument("--text", default="你好，这是一次项目烟雾测试。")
    smoke_parser.add_argument("--prompt", default="A calm and clear Chinese voice.")
    smoke_parser.add_argument("--seed", type=int, default=42)

    return parser


def run_app(server_name: str, server_port: int) -> int:
    configure_runtime()
    demo = create_demo()
    demo.launch(server_name=server_name, server_port=server_port, theme=gr.themes.Soft())
    demo.block_thread()
    return 0


def run_doctor() -> int:
    configure_runtime()
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"SoX found: {shutil.which('sox') is not None}")
    print(f"Model name: {MODEL_NAME}")
    print(f"HRIR file exists: {HRIR_FILE.exists()}")
    print(f"Output dir: {OUTPUT_DIR}")
    return 0


def run_smoke_test(text: str, prompt: str, seed: int) -> int:
    configure_runtime()
    service = TTSStudioService()
    request = GenerationRequest(
        preset_key="默认",
        voice_description=prompt,
        text=text,
        seed=seed,
        use_random_seed=False,
        render_mode=MODE_MONO,
        static_azimuth_deg=270,
        static_distance_m=0.3,
        dynamic_path=PATH_CLOCKWISE,
        dynamic_cycle_time_s=8.0,
        dynamic_start_azimuth_deg=270,
        dynamic_distance_m=0.2,
    )
    result = service.generate(request)
    print(f"Smoke test ok: {result.file_path}")
    print(result.status)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "run"

    if command == "run":
        return run_app(args.server_name, args.server_port)
    if command == "doctor":
        return run_doctor()
    if command == "smoke-test":
        return run_smoke_test(args.text, args.prompt, args.seed)

    parser.error(f"Unknown command: {command}")
    return 2
