from __future__ import annotations

import datetime as dt
import random
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf
import torch

from tts3d_app.audio_engine import (
    MODE_BEHIND_HEAD,
    MODE_DYNAMIC_HRIR,
    MODE_MONO,
    MODE_STATIC_HRIR,
    MODE_STEREO,
    PATH_CLOCKWISE,
    apply_behind_head_effect,
    apply_dynamic_hrir,
    apply_pitch_shift,
    apply_speed,
    apply_static_hrir,
    duplicate_stereo,
    ensure_mono,
    normalize_audio,
)
from tts3d_app.audio_export import DEFAULT_FORMAT, SUPPORTED_FORMATS, export_audio
from tts3d_app.config import (
    ASR_AVAILABLE,
    CLEAR_CUDA_CACHE_AFTER_GENERATE,
    DRY_AUDIO_CACHE_SIZE,
    HRIR_FILE,
    MODEL_NAME,
    OUTPUT_DIR,
    QWEN_MODEL_NAME,
    configure_runtime,
    get_logger,
)
from tts3d_app.hrir import HrirDataset, load_hrir_dataset
from tts3d_app.presets import PRESETS
from tts3d_app.tts_engines import (
    DEFAULT_TTS_ENGINE,
    ENGINE_CHOICES,
    ENGINE_QWEN3,
    ENGINE_QWEN3_BASE,
    TTSProvider,
    build_default_providers,
)


LOGGER = get_logger("service")


@dataclass(slots=True)
class GenerationRequest:
    preset_key: str
    voice_description: str
    text: str
    seed: int | None
    use_random_seed: bool
    render_mode: str
    static_azimuth_deg: float
    static_distance_m: float
    dynamic_path: str
    dynamic_cycle_time_s: float
    dynamic_start_azimuth_deg: float
    dynamic_distance_m: float
    tts_engine: str = DEFAULT_TTS_ENGINE
    tts_model_key: str = ""
    reference_audio_path: str | None = None
    reference_text: str = ""
    speed_factor: float = 1.0
    pitch_semitones: float = 0.0


@dataclass(slots=True)
class GenerationResult:
    file_path: str
    seed: int
    status: str
    text: str


@dataclass(slots=True)
class DryAudioCacheEntry:
    audio_mono: np.ndarray
    sample_rate: int


@dataclass(slots=True)
class BatchRequest:
    text: str
    voice_description: str
    tts_engine: str = DEFAULT_TTS_ENGINE
    tts_model_key: str = ""
    reference_audio_path: str | None = None
    reference_text: str = ""
    preset_key: str = ""
    seeds: list[int] | None = None
    effect_types: list[str] | None = None
    output_formats: list[str] | None = None
    static_azimuth_deg: float = 270
    static_distance_m: float = 0.3
    dynamic_path: str = PATH_CLOCKWISE
    dynamic_cycle_time_s: float = 8.0
    dynamic_start_azimuth_deg: float = 270
    dynamic_distance_m: float = 0.2
    speed_factor: float = 1.0
    pitch_semitones: float = 0.0


@dataclass(slots=True, frozen=True)
class BatchItemResult:
    file_path: str
    seed: int
    effect_type: str
    format: str
    status: str


@dataclass(slots=True)
class BatchResult:
    items: list[BatchItemResult]
    total: int
    succeeded: int
    failed: int


class TTSStudioService:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        output_dir: Path = OUTPUT_DIR,
        hrir_file: Path = HRIR_FILE,
        providers: Mapping[str, TTSProvider] | None = None,
    ) -> None:
        configure_runtime()
        self.qwen_model_name = model_name or QWEN_MODEL_NAME
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hrir_dataset = load_hrir_dataset(hrir_file)
        self._providers: dict[str, TTSProvider] = dict(providers or build_default_providers(self.qwen_model_name))
        self._models: dict[tuple[str, str], Any] = {}
        self._dry_audio_cache: OrderedDict[tuple[Any, ...], DryAudioCacheEntry] = OrderedDict()
        self._cuda_warmed = False

    def get_tts_engine_choices(self) -> list[tuple[str, str]]:
        return list(ENGINE_CHOICES)

    def get_tts_model_choices(self, engine_key: str) -> list[tuple[str, str]]:
        return [(option.label, option.key) for option in self.get_provider(engine_key).model_options()]

    def get_default_tts_model_key(self, engine_key: str) -> str:
        return self.get_provider(engine_key).model_options()[0].key

    def ensure_model_loaded(self, tts_engine: str, tts_model_key: str) -> Any:
        model_cache_key = (tts_engine, tts_model_key)
        cached_model = self._models.get(model_cache_key)
        if cached_model is not None:
            return cached_model

        provider = self.get_provider(tts_engine)
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = self.resolve_model_dtype()

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        model = provider.load_model(tts_model_key, device_name, torch_dtype)
        self._models[model_cache_key] = model
        LOGGER.info("Model loaded successfully for engine=%s model=%s", tts_engine, tts_model_key)
        return model

    def _ensure_cuda_warmup(self) -> None:
        """Warm up CUDA kernels on first inference to avoid JIT compilation overhead."""
        if self._cuda_warmed or not torch.cuda.is_available():
            return
        try:
            device = torch.device("cuda")
            dummy = torch.zeros(1024, device=device)
            _ = dummy + dummy
            torch.cuda.synchronize()
            self._cuda_warmed = True
            LOGGER.info("CUDA warmup completed")
        except Exception as exc:
            LOGGER.warning("CUDA warmup failed: %s", exc)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        tts_engine = self.resolve_tts_engine(request.tts_engine)
        tts_model_key = self.resolve_tts_model_key(tts_engine, request.tts_model_key)
        prompt = request.voice_description.strip()
        text = self.resolve_text(request.preset_key, request.text)
        reference_text = self._resolve_reference_text(request.reference_audio_path, request.reference_text, tts_engine)
        self.validate_request_inputs(tts_engine, text, request.reference_audio_path, reference_text)

        model = self.ensure_model_loaded(tts_engine, tts_model_key)
        self._ensure_cuda_warmup()
        seed = self.resolve_seed(request.seed, request.use_random_seed)
        self.seed_everything(seed)

        started_at = time.time()
        LOGGER.info(
            "Generating audio: engine=%s model=%s preset=%s mode=%s seed=%s",
            tts_engine,
            tts_model_key,
            request.preset_key,
            request.render_mode,
            seed,
        )

        try:
            audio_mono, sample_rate, cache_state = self.get_or_generate_dry_audio(
                model,
                tts_engine=tts_engine,
                tts_model_key=tts_model_key,
                text=text,
                voice_description=prompt,
                seed=seed,
                reference_audio_path=request.reference_audio_path,
                reference_text=reference_text,
            )

            audio_mono, sample_rate = apply_speed(audio_mono, sample_rate, request.speed_factor)
            audio_mono, sample_rate = apply_pitch_shift(audio_mono, sample_rate, request.pitch_semitones)

            final_audio, final_sample_rate, tag = self.render_audio(audio_mono, sample_rate, request)
            final_audio = normalize_audio(final_audio)

            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"voice_{timestamp}_{tts_engine}_{tag}.wav"
            output_path = self.output_dir / filename
            sf.write(output_path, final_audio, final_sample_rate)

            elapsed_s = time.time() - started_at
            speed_info = f" speed={request.speed_factor:.2f}" if request.speed_factor != 1.0 else ""
            pitch_info = f" pitch={request.pitch_semitones:+.0f}st" if request.pitch_semitones != 0 else ""
            status = (
                f"Completed in {elapsed_s:.2f}s | engine={tts_engine} | model={tts_model_key} | "
                f"mode={request.render_mode}{speed_info}{pitch_info} | dry_cache={cache_state}"
            )
            LOGGER.info("Audio saved to %s", output_path)
            return GenerationResult(
                file_path=str(output_path),
                seed=seed,
                status=status,
                text=text,
            )
        finally:
            if torch.cuda.is_available() and CLEAR_CUDA_CACHE_AFTER_GENERATE:
                torch.cuda.empty_cache()

    def generate_3d_speech(
        self,
        text: str,
        effect_type: str = MODE_MONO,
        voice_description: str = "",
        seed: int | None = 42,
        static_azimuth_deg: float = 270,
        static_distance_m: float = 0.3,
        dynamic_path: str = PATH_CLOCKWISE,
        dynamic_cycle_time_s: float = 8.0,
        dynamic_start_azimuth_deg: float = 270,
        dynamic_distance_m: float = 0.2,
        tts_engine: str = DEFAULT_TTS_ENGINE,
        tts_model_key: str = "",
        reference_audio_path: str | None = None,
        reference_text: str = "",
        speed_factor: float = 1.0,
        pitch_semitones: float = 0.0,
    ) -> GenerationResult:
        request = GenerationRequest(
            preset_key=self.default_preset_key(),
            voice_description=voice_description,
            text=text,
            seed=seed,
            use_random_seed=seed is None,
            render_mode=effect_type,
            static_azimuth_deg=static_azimuth_deg,
            static_distance_m=static_distance_m,
            dynamic_path=dynamic_path,
            dynamic_cycle_time_s=dynamic_cycle_time_s,
            dynamic_start_azimuth_deg=dynamic_start_azimuth_deg,
            dynamic_distance_m=dynamic_distance_m,
            tts_engine=tts_engine,
            tts_model_key=tts_model_key,
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
            speed_factor=speed_factor,
            pitch_semitones=pitch_semitones,
        )
        return self.generate(request)

    def get_or_generate_dry_audio(
        self,
        model: Any,
        *,
        tts_engine: str,
        tts_model_key: str,
        text: str,
        voice_description: str,
        seed: int,
        reference_audio_path: str | None,
        reference_text: str,
    ) -> tuple[np.ndarray, int, str]:
        cache_key = self.build_dry_audio_cache_key(
            tts_engine=tts_engine,
            tts_model_key=tts_model_key,
            text=text,
            voice_description=voice_description,
            seed=seed,
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
        )
        cached_entry = self._dry_audio_cache.get(cache_key)
        if cached_entry is not None:
            self._dry_audio_cache.move_to_end(cache_key)
            LOGGER.info("Dry audio cache hit for engine=%s model=%s", tts_engine, tts_model_key)
            return cached_entry.audio_mono.copy(), cached_entry.sample_rate, "hit"

        provider = self.get_provider(tts_engine)
        LOGGER.info("Dry audio cache miss for engine=%s model=%s", tts_engine, tts_model_key)
        raw_audio, sample_rate = provider.generate_audio(
            model,
            text=text,
            voice_description=voice_description,
            seed=seed,
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
        )
        audio_mono = ensure_mono(raw_audio)
        self.store_dry_audio(cache_key, audio_mono, sample_rate)
        return audio_mono, sample_rate, "miss"

    def store_dry_audio(
        self,
        cache_key: tuple[Any, ...],
        audio_mono: np.ndarray,
        sample_rate: int,
    ) -> None:
        if DRY_AUDIO_CACHE_SIZE <= 0:
            return

        self._dry_audio_cache[cache_key] = DryAudioCacheEntry(audio_mono.copy(), sample_rate)
        self._dry_audio_cache.move_to_end(cache_key)

        while len(self._dry_audio_cache) > DRY_AUDIO_CACHE_SIZE:
            self._dry_audio_cache.popitem(last=False)

    def render_audio(
        self,
        audio_mono: np.ndarray,
        sample_rate: int,
        request: GenerationRequest,
    ) -> tuple[np.ndarray, int, str]:
        if request.render_mode == MODE_MONO:
            return audio_mono, sample_rate, MODE_MONO

        if request.render_mode == MODE_STEREO:
            return duplicate_stereo(audio_mono), sample_rate, MODE_STEREO

        if request.render_mode == MODE_BEHIND_HEAD:
            stereo_audio, stereo_rate = apply_behind_head_effect(audio_mono, sample_rate)
            return stereo_audio, stereo_rate, MODE_BEHIND_HEAD

        if request.render_mode == MODE_STATIC_HRIR:
            dataset = self.require_hrir_dataset()
            stereo_audio = apply_static_hrir(
                audio_mono,
                dataset,
                request.static_azimuth_deg,
                request.static_distance_m,
            )
            return stereo_audio, sample_rate, f"static_{int(request.static_azimuth_deg)}"

        if request.render_mode == MODE_DYNAMIC_HRIR:
            dataset = self.require_hrir_dataset()
            stereo_audio, stereo_rate = apply_dynamic_hrir(
                audio_mono,
                sample_rate,
                dataset,
                request.dynamic_path,
                request.dynamic_cycle_time_s,
                request.dynamic_start_azimuth_deg,
                request.dynamic_distance_m,
            )
            return stereo_audio, stereo_rate, MODE_DYNAMIC_HRIR

        raise RuntimeError(f"Unknown output mode: {request.render_mode}")

    def require_hrir_dataset(self) -> HrirDataset:
        if self.hrir_dataset is None:
            raise RuntimeError("HRIR data is not loaded, so spatial audio modes are unavailable.")
        return self.hrir_dataset

    def get_provider(self, tts_engine: str) -> TTSProvider:
        provider = self._providers.get(tts_engine)
        if provider is None:
            raise RuntimeError(f"Unsupported TTS engine: {tts_engine}")
        return provider

    def resolve_tts_model_key(self, tts_engine: str, tts_model_key: str | None) -> str:
        provider = self.get_provider(tts_engine)
        return provider.resolve_model_key(tts_model_key)

    @staticmethod
    def resolve_tts_engine(tts_engine: str | None) -> str:
        if tts_engine in (None, ""):
            return DEFAULT_TTS_ENGINE
        if tts_engine not in {ENGINE_QWEN3, ENGINE_QWEN3_BASE}:
            raise RuntimeError(f"Unsupported TTS engine: {tts_engine}")
        return tts_engine

    @staticmethod
    def build_dry_audio_cache_key(
        *,
        tts_engine: str,
        tts_model_key: str,
        text: str,
        voice_description: str,
        seed: int,
        reference_audio_path: str | None,
        reference_text: str,
    ) -> tuple[Any, ...]:
        normalized_reference_audio = ""
        if reference_audio_path:
            normalized_reference_audio = str(Path(reference_audio_path).resolve(strict=False))

        return (
            tts_engine,
            tts_model_key,
            text,
            voice_description,
            seed,
            normalized_reference_audio,
            reference_text,
        )

    @staticmethod
    def _resolve_reference_text(
        reference_audio_path: str | None,
        reference_text: str,
        tts_engine: str,
    ) -> str:
        if tts_engine != ENGINE_QWEN3_BASE or reference_text.strip():
            return reference_text
        if not reference_audio_path or not Path(reference_audio_path).exists():
            return reference_text
        if not ASR_AVAILABLE:
            return reference_text

        from tts3d_app.asr import transcribe_audio

        LOGGER.info("Auto-transcribing reference audio: %s", reference_audio_path)
        try:
            transcribed = transcribe_audio(reference_audio_path)
            LOGGER.info("Auto-transcribed: %.80s", transcribed)
            return transcribed
        except Exception as exc:
            LOGGER.warning("Auto-transcription failed, proceeding without: %s", exc)
            return reference_text

    @staticmethod
    def validate_request_inputs(
        tts_engine: str,
        text: str,
        reference_audio_path: str | None,
        reference_text: str,
    ) -> None:
        if not text.strip():
            raise RuntimeError("Text cannot be empty.")

        if tts_engine != ENGINE_QWEN3_BASE:
            return

        if not reference_audio_path:
            raise RuntimeError("Qwen3-TTS Base Clone requires a reference audio file.")
        if not Path(reference_audio_path).exists():
            raise RuntimeError(f"Qwen3-TTS Base Clone reference audio not found: {reference_audio_path}")

    @staticmethod
    def resolve_seed(seed: int | None, use_random_seed: bool) -> int:
        if use_random_seed or seed is None:
            return random.randint(0, 2**32 - 1)
        return int(seed)

    @staticmethod
    def seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def resolve_text(preset_key: str, text: str) -> str:
        stripped = text.strip()
        if stripped:
            return stripped
        preset = PRESETS.get(preset_key)
        if preset:
            return preset["text"]
        return next(iter(PRESETS.values()))["text"]

    @staticmethod
    def default_preset_key() -> str:
        return next(iter(PRESETS))

    @staticmethod
    def resolve_model_dtype() -> torch.dtype:
        if not torch.cuda.is_available():
            return torch.float32
        # Check available GPU memory and choose dtype accordingly
        # Qwen3-TTS 1.7B needs ~6GB in float16, ~12GB in float32
        try:
            free_mem_bytes = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            free_mem_gb = free_mem_bytes / (1024**3)
            if free_mem_gb < 5:
                LOGGER.warning(
                    "Low GPU memory (%.1f GB free). Using float32 to avoid OOM. "
                    "Consider closing other GPU applications.",
                    free_mem_gb,
                )
                return torch.float32
        except Exception:
            pass
        return torch.float16

    def generate_batch(self, request: BatchRequest) -> BatchResult:
        seeds = request.seeds if request.seeds else [42]
        effect_types = request.effect_types if request.effect_types else [MODE_MONO]
        output_formats = request.output_formats if request.output_formats else [DEFAULT_FORMAT]

        items: list[BatchItemResult] = []
        succeeded = 0
        failed = 0

        for seed in seeds:
            tts_engine = self.resolve_tts_engine(request.tts_engine)
            tts_model_key = self.resolve_tts_model_key(tts_engine, request.tts_model_key)
            prompt = request.voice_description.strip()
            text = self.resolve_text(request.preset_key, request.text)
            reference_text = self._resolve_reference_text(request.reference_audio_path, request.reference_text, tts_engine)
            self.validate_request_inputs(tts_engine, text, request.reference_audio_path, reference_text)

            model = self.ensure_model_loaded(tts_engine, tts_model_key)
            self.seed_everything(seed)

            try:
                audio_mono, sample_rate, cache_state = self.get_or_generate_dry_audio(
                    model,
                    tts_engine=tts_engine,
                    tts_model_key=tts_model_key,
                    text=text,
                    voice_description=prompt,
                    seed=seed,
                    reference_audio_path=request.reference_audio_path,
                    reference_text=reference_text,
                )
                audio_mono, sample_rate = apply_speed(audio_mono, sample_rate, request.speed_factor)
                audio_mono, sample_rate = apply_pitch_shift(audio_mono, sample_rate, request.pitch_semitones)
            except Exception as exc:
                items.append(
                    BatchItemResult(
                        file_path="",
                        seed=seed,
                        effect_type=effect_types[0],
                        format=DEFAULT_FORMAT,
                        status=f"TTS generation error: {exc}",
                    )
                )
                failed += 1
                continue

            for effect_type in effect_types:
                request_render = GenerationRequest(
                    preset_key=request.preset_key or self.default_preset_key(),
                    voice_description=request.voice_description,
                    text=request.text,
                    seed=seed,
                    use_random_seed=False,
                    render_mode=effect_type,
                    static_azimuth_deg=request.static_azimuth_deg,
                    static_distance_m=request.static_distance_m,
                    dynamic_path=request.dynamic_path,
                    dynamic_cycle_time_s=request.dynamic_cycle_time_s,
                    dynamic_start_azimuth_deg=request.dynamic_start_azimuth_deg,
                    dynamic_distance_m=request.dynamic_distance_m,
                    tts_engine=request.tts_engine,
                    tts_model_key=request.tts_model_key,
                    reference_audio_path=request.reference_audio_path,
                    reference_text=request.reference_text,
                    speed_factor=request.speed_factor,
                    pitch_semitones=request.pitch_semitones,
                )
                try:
                    final_audio, final_sample_rate, tag = self.render_audio(audio_mono, sample_rate, request_render)
                    final_audio = normalize_audio(final_audio)

                    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"voice_{timestamp}_{tts_engine}_{tag}.wav"
                    output_path = self.output_dir / filename
                    sf.write(output_path, final_audio, final_sample_rate)

                    for fmt in output_formats:
                        if fmt == DEFAULT_FORMAT:
                            items.append(
                                BatchItemResult(
                                    file_path=str(output_path),
                                    seed=seed,
                                    effect_type=effect_type,
                                    format=fmt,
                                    status=f"Generated (dry_cache={cache_state})",
                                )
                            )
                            succeeded += 1
                        else:
                            try:
                                exported = export_audio(output_path, output_path.parent, [fmt])
                                if exported:
                                    items.append(
                                        BatchItemResult(
                                            file_path=str(exported[0]),
                                            seed=seed,
                                            effect_type=effect_type,
                                            format=fmt,
                                            status=f"Exported to {fmt}",
                                        )
                                    )
                                    succeeded += 1
                                else:
                                    items.append(
                                        BatchItemResult(
                                            file_path=str(output_path),
                                            seed=seed,
                                            effect_type=effect_type,
                                            format=fmt,
                                            status=f"Export to {fmt} failed",
                                        )
                                    )
                                    failed += 1
                            except Exception as exc:
                                items.append(
                                    BatchItemResult(
                                        file_path=str(output_path),
                                        seed=seed,
                                        effect_type=effect_type,
                                        format=fmt,
                                        status=f"Export error: {exc}",
                                    )
                                )
                                failed += 1
                except Exception as exc:
                    items.append(
                        BatchItemResult(
                            file_path="",
                            seed=seed,
                            effect_type=effect_type,
                            format=DEFAULT_FORMAT,
                            status=f"Render error: {exc}",
                        )
                    )
                    failed += 1

        total = len(items)
        return BatchResult(items=items, total=total, succeeded=succeeded, failed=failed)

    def list_generated_audios(self) -> list[dict[str, Any]]:
        """返回已生成的音频文件列表，按修改时间倒序"""
        audio_files = []
        for f in sorted(self.output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg"):
                stat = f.stat()
                audio_files.append({
                    "name": f.name,
                    "path": str(f),
                    "size": stat.st_size,
                    "modified": dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                })
        return audio_files

    def delete_audio(self, file_path: str) -> bool:
        """删除指定的音频文件"""
        p = Path(file_path)
        if not p.exists() or not p.is_file():
            return False
        try:
            p.unlink()
            LOGGER.info("Deleted audio file: %s", p)
            return True
        except Exception as exc:
            LOGGER.warning("Failed to delete %s: %s", p, exc)
            return False
