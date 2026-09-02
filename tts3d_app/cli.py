from __future__ import annotations

import argparse
import shutil
import sys
from typing import Sequence

import gradio as gr
import torch
from nicegui import ui

from tts3d_app.audio_engine import (
    MODE_BEHIND_HEAD,
    MODE_DYNAMIC_HRIR,
    MODE_MONO,
    MODE_STATIC_HRIR,
    MODE_STEREO,
    PATH_CLOCKWISE,
)
from tts3d_app.config import (
    ASR_AVAILABLE,
    DEFAULT_SERVER_NAME,
    FASTER_WHISPER_INSTALLED,
    FASTMCP_INSTALLED,
    FLASH_ATTN_INSTALLED,
    HF_HUB_CACHE,
    HRIR_FILE,
    MODEL_NAME,
    NUMBA_INSTALLED,
    OUTPUT_DIR,
    QWEN_BASE_MODEL_NAME,
    configure_runtime,
)
from tts3d_app.service import BatchRequest, GenerationRequest, TTSStudioService
from tts3d_app.tts_engines import DEFAULT_TTS_ENGINE, ENGINE_QWEN3, ENGINE_QWEN3_BASE
from tts3d_app.ui import create_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TTS 3D Studio")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="启动 Gradio 应用")
    run_parser.add_argument("--server-name", default=DEFAULT_SERVER_NAME)
    run_parser.add_argument("--server-port", type=int, default=None)

    nicegui_parser = subparsers.add_parser("nicegui", help="启动 NiceGUI 应用（现代 Web UI）")
    nicegui_parser.add_argument("--host", default="0.0.0.0")
    nicegui_parser.add_argument("--port", type=int, default=7860)

    subparsers.add_parser("doctor", help="检查 Python、Torch、CUDA、SoX 和模型依赖状态")

    smoke_parser = subparsers.add_parser("smoke-test", help="执行一次最小语音生成测试")
    smoke_parser.add_argument("--text", default="你好，这是一次项目烟雾测试。")
    smoke_parser.add_argument("--prompt", default="A calm and clear Chinese voice.")
    smoke_parser.add_argument("--seed", type=int, default=42)
    smoke_parser.add_argument("--tts-engine", choices=[ENGINE_QWEN3, ENGINE_QWEN3_BASE], default=DEFAULT_TTS_ENGINE)
    smoke_parser.add_argument("--tts-model", default="")
    smoke_parser.add_argument("--reference-audio", default="", help="Qwen3 Base Clone reference audio path")
    smoke_parser.add_argument("--reference-text", default="", help="Optional transcript for the reference audio")
    smoke_parser.add_argument("--speed", type=float, default=1.0)
    smoke_parser.add_argument("--pitch", type=float, default=0.0)

    batch_parser = subparsers.add_parser("batch-generate", help="批量生成多个变体")
    batch_parser.add_argument("--text", default="你好，这是一次批量生成测试。")
    batch_parser.add_argument("--prompt", default="A calm and clear Chinese voice.")
    batch_parser.add_argument("--tts-engine", choices=[ENGINE_QWEN3, ENGINE_QWEN3_BASE], default=DEFAULT_TTS_ENGINE)
    batch_parser.add_argument("--tts-model", default="")
    batch_parser.add_argument("--reference-audio", default="", help="Qwen3 Base Clone reference audio path")
    batch_parser.add_argument("--reference-text", default="", help="Optional transcript for the reference audio")
    batch_parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    batch_parser.add_argument(
        "--effect-types",
        nargs="+",
        default=[MODE_MONO, MODE_STEREO],
        choices=[MODE_MONO, MODE_STEREO, MODE_BEHIND_HEAD, MODE_STATIC_HRIR, MODE_DYNAMIC_HRIR],
    )
    batch_parser.add_argument(
        "--output-formats",
        nargs="+",
        default=["wav"],
        choices=["wav", "mp3", "ogg"],
    )
    batch_parser.add_argument("--speed", type=float, default=1.0)
    batch_parser.add_argument("--pitch", type=float, default=0.0)

    subparsers.add_parser("mcp", help="以 stdio 模式启动 MCP Server")
    return parser


def run_app(server_name: str, server_port: int | None) -> int:
    configure_runtime()
    demo = create_demo()

    # 宇宙深空主题 - Dark Cosmic Style
    theme = gr.themes.Default(
        primary_hue="purple",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=["Inter", "system-ui", "sans-serif"],
        radius_size="lg",
    )

    custom_css = """
    /* 全局深空背景 */
    body {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d1b2a 50%, #1a0a2e 100%) !important;
        min-height: 100vh;
    }

    /* 毛玻璃卡片效果 */
    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(90deg, #8b5cf6, #06b6d4, #d946ef);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px !important;
        text-shadow: 0 0 40px rgba(139, 92, 246, 0.5);
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 24px !important;
    }

    /* 玻璃态容器 */
    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* 霓虹按钮 */
    .generate-btn {
        width: 100%;
        font-weight: 700;
        letter-spacing: 1px;
        background: linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%) !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 16px 32px !important;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4), 0 0 40px rgba(6, 182, 212, 0.2);
        transition: all 0.3s ease !important;
        color: white !important;
    }
    .generate-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 30px rgba(139, 92, 246, 0.6), 0 0 60px rgba(6, 182, 212, 0.4) !important;
    }

    /* 警告框 */
    .warning-box {
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.3);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 12px 0;
        color: #fbbf24;
        font-size: 13px;
    }

    /* 输出卡片 - 极光效果 */
    .output-card {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(6, 182, 212, 0.15) 50%, rgba(217, 70, 239, 0.15) 100%) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    /* 输入框玻璃态 */
    input, textarea {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        color: #1e293b !important;
    }
    input::placeholder, textarea::placeholder {
        color: rgba(30, 41, 59, 0.5) !important;
    }
    input:focus, textarea:focus {
        border-color: rgba(139, 92, 246, 0.6) !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.4) !important;
    }

    /* 下拉框暗色 */
    .ss-label, .ss-placeholder {
        color: #e2e8f0 !important;
    }

    /* 滑块美化 */
    input[type="range"] {
        -webkit-appearance: none;
        height: 6px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
    }
    input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 20px;
        height: 20px;
        background: linear-gradient(135deg, #8b5cf6, #06b6d4);
        border-radius: 50%;
        cursor: pointer;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.5);
    }

    /* Accordion 暗色 */
    .accordion {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
    }

    /* 标签 */
    .tag {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(6, 182, 212, 0.2));
        border: 1px solid rgba(139, 92, 246, 0.3);
        color: #a78bfa;
        border-radius: 20px;
    }
    """

    _, local_url, _ = demo.launch(
        server_name=server_name,
        server_port=server_port,
        prevent_thread_lock=True,
        inbrowser=True,
        theme=theme,
        css=custom_css,
    )
    print(f"TTS 3D Studio is available at: {local_url}")
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
    print(f"flash-attn installed: {FLASH_ATTN_INSTALLED}")
    print(f"numba installed: {NUMBA_INSTALLED}")
    print(f"fastmcp installed: {FASTMCP_INSTALLED}")
    print(f"faster-whisper installed: {FASTER_WHISPER_INSTALLED}  (ASR auto-transcription available: {ASR_AVAILABLE})")
    print(f"Qwen VoiceDesign model name: {MODEL_NAME}")
    print(f"Qwen Base Clone model name: {QWEN_BASE_MODEL_NAME}")
    print(f"Hugging Face hub cache: {HF_HUB_CACHE}")
    hrir_exists = HRIR_FILE.exists()
    if hrir_exists:
        print(f"HRIR file: OK  ({HRIR_FILE})")
    else:
        print(f"HRIR file: MISSING — static/dynamic HRIR modes unavailable  (expected: {HRIR_FILE})")
    print(f"Output dir: {OUTPUT_DIR}")
    return 0


def run_smoke_test(
    text: str,
    prompt: str,
    seed: int,
    tts_engine: str,
    tts_model: str,
    reference_audio: str,
    reference_text: str,
    speed: float,
    pitch: float,
) -> int:
    configure_runtime()
    service = TTSStudioService()
    request = GenerationRequest(
        preset_key=service.default_preset_key(),
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
        tts_engine=tts_engine,
        tts_model_key=tts_model,
        reference_audio_path=reference_audio or None,
        reference_text=reference_text,
        speed_factor=speed,
        pitch_semitones=pitch,
    )
    result = service.generate(request)
    print(f"Smoke test ok: {result.file_path}")
    print(result.status)
    return 0


def run_mcp() -> int:
    configure_runtime()
    try:
        from tts3d_app.mcp_server import run_mcp_server
    except Exception as exc:
        print(f"MCP startup failed: {exc}", file=sys.stderr)
        return 1

    try:
        return run_mcp_server()
    except Exception as exc:
        print(f"MCP startup failed: {exc}", file=sys.stderr)
        return 1


def run_batch(
    text: str,
    prompt: str,
    tts_engine: str,
    tts_model: str,
    reference_audio: str,
    reference_text: str,
    seeds: list[int],
    effect_types: list[str],
    output_formats: list[str],
    speed: float,
    pitch: float,
) -> int:
    configure_runtime()
    service = TTSStudioService()
    request = BatchRequest(
        text=text,
        voice_description=prompt,
        tts_engine=tts_engine,
        tts_model_key=tts_model,
        reference_audio_path=reference_audio or None,
        reference_text=reference_text,
        seeds=seeds,
        effect_types=effect_types,
        output_formats=output_formats,
        speed_factor=speed,
        pitch_semitones=pitch,
    )
    result = service.generate_batch(request)
    print(f"Batch complete: {result.succeeded}/{result.total} succeeded, {result.failed} failed")
    for item in result.items:
        print(f"  [{item.effect_type}] seed={item.seed} format={item.format}: {item.file_path} - {item.status}")
    return 0 if result.failed == 0 else 1


def run_nicegui(host: str, port: int) -> int:
    configure_runtime()
    try:
        from tts3d_app.ui_nicegui import create_ui
        create_ui()
        ui.run(title="TTS 3D Studio", host=host, port=port, reload=False)
        return 0
    except ImportError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        print("请安装依赖: pip install nicegui", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "run"

    if command == "run":
        return run_app(args.server_name, args.server_port)
    if command == "nicegui":
        return run_nicegui(args.host, args.port)
    if command == "doctor":
        return run_doctor()
    if command == "smoke-test":
        return run_smoke_test(
            args.text,
            args.prompt,
            args.seed,
            args.tts_engine,
            args.tts_model,
            args.reference_audio,
            args.reference_text,
            args.speed,
            args.pitch,
        )
    if command == "batch-generate":
        return run_batch(
            args.text,
            args.prompt,
            args.tts_engine,
            args.tts_model,
            args.reference_audio,
            args.reference_text,
            args.seeds,
            args.effect_types,
            args.output_formats,
            args.speed,
            args.pitch,
        )
    if command == "mcp":
        return run_mcp()

    parser.error(f"Unknown command: {command}")
    return 2
