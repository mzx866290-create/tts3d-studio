from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tts3d_app.audio_engine import (
    MODE_BEHIND_HEAD,
    MODE_DYNAMIC_HRIR,
    MODE_MONO,
    MODE_STATIC_HRIR,
    MODE_STEREO,
    PATH_CLOCKWISE,
    PATH_COUNTERCLOCKWISE,
    PATH_SIDE_TO_SIDE,
)
from tts3d_app.presets import PRESETS
from tts3d_app.service import BatchRequest, BatchResult, TTSStudioService
from tts3d_app.tts_engines import DEFAULT_TTS_ENGINE, ENGINE_QWEN3, ENGINE_QWEN3_BASE


EffectType = Literal[
    "mono",
    "stereo",
    "behind_head",
    "static_hrir",
    "dynamic_hrir",
]
DynamicPath = Literal["clockwise", "counterclockwise", "side_to_side"]
TTSEngine = Literal["qwen3", "qwen3_base"]


@dataclass(frozen=True)
class GenerationResult:
    file_path: str
    seed: int
    status: str
    text: str


@dataclass(frozen=True)
class BatchItem:
    file_path: str
    seed: int
    effect_type: str
    format: str
    status: str


@dataclass(frozen=True)
class BatchGenerationResult:
    total: int
    succeeded: int
    failed: int
    items: list[BatchItem]


def create_mcp_server(service: TTSStudioService | None = None):
    try:
        from fastmcp import FastMCP
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "FastMCP is not installed. Install the 'fastmcp' package to use `python app.py mcp`."
        ) from exc

    service = service or TTSStudioService()
    mcp = FastMCP("TTS 3D Studio")

    # --- Resources ---
    @mcp.resource("presets://list")
    def list_presets() -> str:
        """List all available voice presets."""
        lines = []
        for key, preset in PRESETS.items():
            lines.append(f"## {key}")
            lines.append(f"- Text: {preset['text'][:50]}...")
            lines.append(f"- Prompt: {preset['prompt']}")
            lines.append("")
        return "\n".join(lines) if lines else "No presets available."

    @mcp.resource("capabilities://tts")
    def get_tts_capabilities() -> str:
        """Get TTS engine and model capabilities."""
        engines = []
        for label, key in service.get_tts_engine_choices():
            models = service.get_tts_model_choices(key)
            default = service.get_default_tts_model_key(key)
            model_list = ", ".join(m[1] for m in models)
            engines.append(f"- **{label}** ({key}): default={default}, models=[{model_list}]")
        return "\n".join([
            "# TTS 3D Studio Capabilities",
            "",
            "## TTS Engines",
            *engines,
            "",
            "## Effect Types",
            f"- mono: Single channel output",
            f"- stereo: Dual channel (left/right identical)",
            f"- behind_head: Virtual headphone effect (notch filter + delay)",
            f"- static_hrir: Head-related impulse response with fixed azimuth/distance",
            f"- dynamic_hrir: HRIR with moving source along path",
            "",
            "## Dynamic Paths",
            f"- clockwise: Source moves clockwise around listener",
            f"- counterclockwise: Source moves counterclockwise",
            f"- side_to_side: Source oscillates left to right",
            "",
            "## Speed/Pitch",
            "- speed_factor: 0.5 to 2.0 (1.0 = original speed)",
            "- pitch_semitones: -12 to +12 (0 = original pitch)",
            "",
            "## Output Formats",
            "- wav: Default, lossless",
            "- mp3: Compressed (requires pydub)",
            "- ogg: Compressed (requires pydub)",
            "",
            "## Long text",
            "- Inputs longer than MAX_TTS_CHUNK_CHARS (default 400) are split at sentence/paragraph boundaries.",
            "- Each chunk is generated separately (max_new_tokens default 2048) and concatenated with a short pause.",
            "- VoiceDesign synthesizes a fixed calibration sentence once per (model, prompt, seed), stores that clip, then clones every content chunk from it (ICL) so timbre stays consistent across texts.",
            "- This avoids the ~2.7 minute one-shot cap that otherwise silences or distorts the second half.",
        ])

    # --- Tools ---
    @mcp.tool(
        name="generate_3d_speech",
        description=(
            "Generate speech audio with TTS engine/model and spatial effects. "
            "For Qwen3-TTS Base Clone, provide reference_audio_path and optional reference_text. "
            "For Qwen3-TTS VoiceDesign, provide voice_description instead."
        ),
    )
    def generate_3d_speech(
        text: str,
        effect_type: EffectType = MODE_MONO,
        voice_description: str = "",
        seed: int = 42,
        static_azimuth_deg: float = 270.0,
        static_distance_m: float = 0.3,
        dynamic_path: DynamicPath = PATH_CLOCKWISE,
        dynamic_cycle_time_s: float = 8.0,
        dynamic_start_azimuth_deg: float = 270.0,
        dynamic_distance_m: float = 0.2,
        tts_engine: TTSEngine = DEFAULT_TTS_ENGINE,
        tts_model_key: str = "",
        reference_audio_path: str = "",
        reference_text: str = "",
        speed_factor: float = 1.0,
        pitch_semitones: float = 0.0,
    ) -> GenerationResult:
        """Generate a speech file and return its saved path plus generation metadata."""
        result = service.generate_3d_speech(
            text=text,
            effect_type=effect_type,
            voice_description=voice_description,
            seed=seed,
            static_azimuth_deg=static_azimuth_deg,
            static_distance_m=static_distance_m,
            dynamic_path=dynamic_path,
            dynamic_cycle_time_s=dynamic_cycle_time_s,
            dynamic_start_azimuth_deg=dynamic_start_azimuth_deg,
            dynamic_distance_m=dynamic_distance_m,
            tts_engine=tts_engine,
            tts_model_key=tts_model_key,
            reference_audio_path=reference_audio_path or None,
            reference_text=reference_text,
            speed_factor=speed_factor,
            pitch_semitones=pitch_semitones,
        )
        return GenerationResult(
            file_path=result.file_path,
            seed=result.seed,
            status=result.status,
            text=result.text,
        )

    @mcp.tool(
        name="batch_generate",
        description=(
            "Generate multiple speech variants in one call. "
            "Supports multiple seeds, effect types, and output formats. "
            "Returns all generated file paths and their status."
        ),
    )
    def batch_generate(
        text: str,
        voice_description: str = "",
        tts_engine: TTSEngine = DEFAULT_TTS_ENGINE,
        tts_model_key: str = "",
        reference_audio_path: str = "",
        reference_text: str = "",
        seeds: list[int] | None = None,
        effect_types: list[EffectType] | None = None,
        output_formats: list[str] | None = None,
        speed_factor: float = 1.0,
        pitch_semitones: float = 0.0,
    ) -> BatchGenerationResult:
        """Batch generate multiple speech variants."""
        batch_request = BatchRequest(
            text=text,
            voice_description=voice_description,
            tts_engine=tts_engine,
            tts_model_key=tts_model_key,
            reference_audio_path=reference_audio_path or None,
            reference_text=reference_text,
            seeds=seeds,
            effect_types=effect_types,
            output_formats=output_formats,
            speed_factor=speed_factor,
            pitch_semitones=pitch_semitones,
        )
        result = service.generate_batch(batch_request)
        return BatchGenerationResult(
            total=result.total,
            succeeded=result.succeeded,
            failed=result.failed,
            items=[
                BatchItem(
                    file_path=item.file_path,
                    seed=item.seed,
                    effect_type=item.effect_type,
                    format=item.format,
                    status=item.status,
                )
                for item in result.items
            ],
        )

    @mcp.tool(
        name="list_outputs",
        description="List recent audio files generated by this server.",
    )
    def list_outputs(limit: int = 20) -> list[str]:
        """Return paths to recent generated audio files."""
        output_dir = Path(service.output_dir)
        if not output_dir.exists():
            return []
        wav_files = sorted(output_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [str(f) for f in wav_files[:limit]]

    return mcp


def run_mcp_server() -> int:
    server = create_mcp_server()
    server.run()
    return 0


__all__ = [
    "create_mcp_server",
    "run_mcp_server",
    "GenerationResult",
    "BatchGenerationResult",
    "BatchItem",
    "MODE_MONO",
    "MODE_STEREO",
    "MODE_BEHIND_HEAD",
    "MODE_STATIC_HRIR",
    "MODE_DYNAMIC_HRIR",
    "PATH_CLOCKWISE",
    "PATH_COUNTERCLOCKWISE",
    "PATH_SIDE_TO_SIDE",
    "ENGINE_QWEN3",
    "ENGINE_QWEN3_BASE",
]
