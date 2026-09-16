from __future__ import annotations

import datetime as dt
import random
import threading
import time
from collections import OrderedDict
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
    resample_audio,
)
from tts3d_app.audio_export import DEFAULT_FORMAT, SUPPORTED_FORMATS, export_audio
from tts3d_app.config import (
    ASR_AVAILABLE,
    CLEAR_CUDA_CACHE_AFTER_GENERATE,
    DRY_AUDIO_CACHE_SIZE,
    HRIR_FILE,
    MAX_TTS_CHUNK_CHARS,
    MODEL_NAME,
    OUTPUT_DIR,
    QWEN_MODEL_NAME,
    TTS_CHUNK_PARAGRAPH_PAUSE_MS,
    TTS_CHUNK_SENTENCE_PAUSE_MS,
    TTS_CLONE_ENGINE,
    TTS_DIRECT_SINGLE_CHUNK,
    VOICE_PROFILE_DIR,
    configure_runtime,
    get_logger,
)
from tts3d_app.hrir import HrirDataset, load_hrir_dataset
from tts3d_app.presets import PRESETS
from tts3d_app.text_chunking import concatenate_mono_chunks, split_text_for_tts
from tts3d_app.voice_profile import (
    get_or_create_reference_clip,
    reference_clip_wav_path,
    resolve_calibration_text,
    voice_clip_id,
)
from tts3d_app.tts_engines import (
    DEFAULT_TTS_ENGINE,
    ENGINE_CHOICES,
    ENGINE_COSYVOICE2,
    ENGINE_QWEN3,
    ENGINE_QWEN3_BASE,
    GenerationCancelled,
    TTSProvider,
    build_default_providers,
    interruptible_talker_generate,
)


LOGGER = get_logger("service")

ProgressCallback = Any


def _emit_progress(progress_callback: ProgressCallback | None, stage: str, **data: Any) -> None:
    """把生成进度推给 UI；回调异常绝不影响生成本身。"""
    if progress_callback is None:
        return
    try:
        progress_callback(stage, data)
    except Exception:
        LOGGER.debug("progress callback raised", exc_info=True)


def resolve_device_name() -> str:
    """Pick the best available compute device.

    Priority: CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    calibration_text: str = ""
    emotion_instruct: str = ""


@dataclass(slots=True)
class GenerationResult:
    file_path: str
    seed: int
    status: str
    text: str
    clip_id: str = ""
    clip_path: str = ""


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
    calibration_text: str = ""
    emotion_instruct: str = ""


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
        voice_profile_dir: Path | None = None,
    ) -> None:
        configure_runtime()
        self.qwen_model_name = model_name or QWEN_MODEL_NAME
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voice_profile_dir = voice_profile_dir or VOICE_PROFILE_DIR
        self.hrir_dataset = load_hrir_dataset(hrir_file)
        self._providers: dict[str, TTSProvider] = dict(providers or build_default_providers(self.qwen_model_name))
        self._models: dict[tuple[str, str], Any] = {}
        self._dry_audio_cache: OrderedDict[tuple[Any, ...], DryAudioCacheEntry] = OrderedDict()
        self._cuda_warmed = False
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()
        LOGGER.info("Cancel requested for in-flight audio generation")

    def clear_cancel(self) -> None:
        self._cancel_event.clear()

    def raise_if_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise GenerationCancelled()

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
        device_name = resolve_device_name()
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

    def generate(self, request: GenerationRequest, progress_callback: ProgressCallback | None = None) -> GenerationResult:
        self.clear_cancel()
        tts_engine = self.resolve_tts_engine(request.tts_engine)
        tts_model_key = self.resolve_tts_model_key(tts_engine, request.tts_model_key)
        prompt = request.voice_description.strip()
        text = self.resolve_text(request.preset_key, request.text)
        reference_text = self._resolve_reference_text(request.reference_audio_path, request.reference_text, tts_engine)
        self.validate_request_inputs(tts_engine, text, request.reference_audio_path, reference_text)

        _emit_progress(progress_callback, "model_loading", engine=tts_engine, model=tts_model_key)
        model = self.ensure_model_loaded(tts_engine, tts_model_key)
        self.raise_if_cancelled()
        self._ensure_cuda_warmup()
        self.raise_if_cancelled()
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
            audio_mono, sample_rate, cache_state, clip_id = self.get_or_generate_dry_audio(
                model,
                tts_engine=tts_engine,
                tts_model_key=tts_model_key,
                text=text,
                voice_description=prompt,
                seed=seed,
                reference_audio_path=request.reference_audio_path,
                reference_text=reference_text,
                calibration_text=request.calibration_text,
                emotion_instruct=request.emotion_instruct,
                progress_callback=progress_callback,
            )
            self.raise_if_cancelled()
            audio_mono, sample_rate = apply_speed(audio_mono, sample_rate, request.speed_factor)
            self.raise_if_cancelled()
            audio_mono, sample_rate = apply_pitch_shift(audio_mono, sample_rate, request.pitch_semitones)
            self.raise_if_cancelled()

            _emit_progress(progress_callback, "spatial", mode=request.render_mode)
            final_audio, final_sample_rate, tag = self.render_audio(audio_mono, sample_rate, request)
            self.raise_if_cancelled()
            final_audio = normalize_audio(final_audio)

            _emit_progress(progress_callback, "saving", mode=request.render_mode)
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"voice_{timestamp}_{tts_engine}_{tag}.wav"
            output_path = self.output_dir / filename
            sf.write(output_path, final_audio, final_sample_rate)

            elapsed_s = time.time() - started_at
            speed_info = f" speed={request.speed_factor:.2f}" if request.speed_factor != 1.0 else ""
            pitch_info = f" pitch={request.pitch_semitones:+.0f}st" if request.pitch_semitones != 0 else ""
            status = (
                f"Completed in {elapsed_s:.2f}s | seed={seed} | engine={tts_engine} | "
                f"model={tts_model_key} | mode={request.render_mode}{speed_info}{pitch_info} | "
                f"dry_cache={cache_state}"
            )
            LOGGER.info("Audio saved to %s", output_path)
            clip_path = str(reference_clip_wav_path(clip_id, self.voice_profile_dir)) if clip_id else ""
            return GenerationResult(
                file_path=str(output_path),
                seed=seed,
                status=status,
                text=text,
                clip_id=clip_id,
                clip_path=clip_path if clip_id and Path(clip_path).exists() else "",
            )
        except GenerationCancelled:
            LOGGER.info("Audio generation cancelled for engine=%s model=%s", tts_engine, tts_model_key)
            raise
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
        calibration_text: str = "",
        emotion_instruct: str = "",
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
            calibration_text=calibration_text,
            emotion_instruct=emotion_instruct,
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
        calibration_text: str = "",
        emotion_instruct: str = "",
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[np.ndarray, int, str, str]:
        # Request-level calibration text (情绪基调句) wins; empty falls back to the global default.
        lock_text = (
            resolve_calibration_text(calibration_text)
            if calibration_text.strip()
            else resolve_calibration_text()
        )
        emotion = emotion_instruct.strip()
        cache_key = self.build_dry_audio_cache_key(
            tts_engine=tts_engine,
            tts_model_key=tts_model_key,
            text=text,
            voice_description=voice_description,
            seed=seed,
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
            calibration_text=lock_text,
            emotion_instruct=emotion,
        )
        cached_entry = self._dry_audio_cache.get(cache_key)
        if cached_entry is not None:
            self.raise_if_cancelled()
            self._dry_audio_cache.move_to_end(cache_key)
            LOGGER.info("Dry audio cache hit for engine=%s model=%s", tts_engine, tts_model_key)
            cached_clip_id = ""
            if (
                tts_engine == ENGINE_QWEN3
                and lock_text
                and not self.uses_direct_single_chunk(tts_engine, text, emotion)
            ):
                cached_clip_id = voice_clip_id(
                    tts_model_key,
                    voice_description,
                    seed,
                    lock_text,
                )
            return cached_entry.audio_mono.copy(), cached_entry.sample_rate, "hit", cached_clip_id

        provider = self.get_provider(tts_engine)
        chunks = split_text_for_tts(
            text,
            max_chars=MAX_TTS_CHUNK_CHARS,
            sentence_pause_ms=TTS_CHUNK_SENTENCE_PAUSE_MS,
            paragraph_pause_ms=TTS_CHUNK_PARAGRAPH_PAUSE_MS,
        )
        if not chunks:
            raise RuntimeError("Text cannot be empty.")

        LOGGER.info(
            "Dry audio cache miss for engine=%s model=%s chunks=%s",
            tts_engine,
            tts_model_key,
            len(chunks),
        )
        audios: list[np.ndarray] = []
        pauses_ms: list[int] = []
        sample_rate = 24_000
        direct_single_chunk = self.uses_direct_single_chunk(tts_engine, text, emotion)
        speaker_lock_enabled = tts_engine == ENGINE_QWEN3 and bool(lock_text) and not direct_single_chunk
        used_speaker_lock = False
        clip_id = ""
        # VoiceDesign samples a new speaker from instruct+text on every call.
        # Synthesize a fixed calibration sentence once per (model, prompt, seed),
        # persist it, then clone every content chunk from that clip (ICL).
        lock_audio: np.ndarray | None = None
        lock_rate = 24_000
        if speaker_lock_enabled:
            def synthesize_lock() -> tuple[np.ndarray, int]:
                self.raise_if_cancelled()
                self.seed_everything(seed)
                with interruptible_talker_generate(model, self._cancel_event):
                    raw_lock, lock_rate_raw = provider.generate_audio(
                        model,
                        text=lock_text,
                        voice_description=voice_description,
                        seed=seed,
                        reference_audio_path=None,
                        reference_text="",
                    )
                self.raise_if_cancelled()
                return ensure_mono(raw_lock), int(lock_rate_raw)

            try:
                _emit_progress(progress_callback, "voice_lock")
                lock_audio, lock_rate, clip_id = get_or_create_reference_clip(
                    model_key=tts_model_key,
                    prompt=voice_description,
                    seed=seed,
                    calibration_text=lock_text,
                    synthesize=synthesize_lock,
                    profile_dir=self.voice_profile_dir,
                )
            except GenerationCancelled:
                raise
            except Exception:
                self.raise_if_cancelled()
                raise
            used_speaker_lock = lock_audio is not None and lock_audio.size > 0
            if not used_speaker_lock:
                speaker_lock_enabled = False

        for chunk_index, chunk in enumerate(chunks):
            self.raise_if_cancelled()
            self.seed_everything(seed)
            _emit_progress(
                progress_callback,
                "chunk_start",
                index=chunk_index + 1,
                total=len(chunks),
                chars=len(chunk.text),
            )
            try:
                if used_speaker_lock and lock_audio is not None:
                    raw_audio, chunk_rate = self._generate_locked_voice_chunk(
                        text=chunk.text,
                        seed=seed,
                        reference_audio=lock_audio,
                        reference_sample_rate=lock_rate,
                        reference_text=lock_text,
                        emotion_instruct=emotion,
                        progress_callback=progress_callback,
                    )
                else:
                    with interruptible_talker_generate(model, self._cancel_event):
                        raw_audio, chunk_rate = provider.generate_audio(
                            model,
                            text=chunk.text,
                            voice_description=voice_description,
                            seed=seed,
                            reference_audio_path=reference_audio_path,
                            reference_text=reference_text,
                        )
            except GenerationCancelled:
                raise
            except Exception:
                self.raise_if_cancelled()
                raise
            self.raise_if_cancelled()
            chunk_mono = ensure_mono(raw_audio)
            if chunk_index == 0:
                sample_rate = int(chunk_rate)
            elif int(chunk_rate) != sample_rate:
                chunk_mono, sample_rate = resample_audio(chunk_mono, int(chunk_rate), sample_rate)
            audios.append(chunk_mono)
            pauses_ms.append(chunk.pause_after_ms)
            _emit_progress(
                progress_callback,
                "chunk_done",
                index=chunk_index + 1,
                total=len(chunks),
                seconds=chunk_mono.size / float(sample_rate) if sample_rate else 0.0,
            )
            LOGGER.info(
                "Chunk %s/%s generated: %.1fs chars=%s",
                chunk_index + 1,
                len(chunks),
                chunk_mono.size / float(sample_rate) if sample_rate else 0.0,
                len(chunk.text),
            )

        audio_mono = concatenate_mono_chunks(audios, pauses_ms, sample_rate) if len(audios) > 1 else audios[0]
        if used_speaker_lock:
            cache_state = f"miss:chunks={len(chunks)},speaker_lock=icl,clip={clip_id}"
        elif direct_single_chunk:
            cache_state = "miss:direct"
        elif len(chunks) == 1:
            cache_state = "miss"
        else:
            cache_state = f"miss:chunks={len(chunks)}"
        self.store_dry_audio(cache_key, audio_mono, sample_rate)
        return audio_mono, sample_rate, cache_state, clip_id

    @staticmethod
    def uses_direct_single_chunk(tts_engine: str, text: str, emotion_instruct: str) -> bool:
        """单块短文本且无情感指令时直接 VoiceDesign 直出，跳过克隆链路。

        直出的声音由 (提示词, seed, 文本) 共同采样，换文本即换声；长文本或
        填了情感指令仍走音色锁定。TTS_DIRECT_SINGLE_CHUNK=0 可关闭此优化。
        """
        if not TTS_DIRECT_SINGLE_CHUNK or tts_engine != ENGINE_QWEN3 or emotion_instruct.strip():
            return False
        chunks = split_text_for_tts(
            text,
            max_chars=MAX_TTS_CHUNK_CHARS,
            sentence_pause_ms=TTS_CHUNK_SENTENCE_PAUSE_MS,
            paragraph_pause_ms=TTS_CHUNK_PARAGRAPH_PAUSE_MS,
        )
        return len(chunks) == 1

    def resolve_clone_engine(self, has_emotion_instruct: bool) -> str:
        """Pick the clone engine for the timbre-lock chain.

        TTS_CLONE_ENGINE: "auto" (CosyVoice2 when an emotion instruct is present,
        else in-family Qwen3 Base clone) or an explicit engine key.
        """
        configured = TTS_CLONE_ENGINE.strip()
        if configured in (ENGINE_QWEN3_BASE, ENGINE_COSYVOICE2):
            return configured
        return ENGINE_COSYVOICE2 if has_emotion_instruct else ENGINE_QWEN3_BASE

    def _generate_locked_voice_chunk(
        self,
        *,
        text: str,
        seed: int,
        reference_audio: np.ndarray,
        reference_sample_rate: int,
        reference_text: str,
        emotion_instruct: str = "",
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[np.ndarray, int]:
        clone_engine = self.resolve_clone_engine(bool(emotion_instruct.strip()))
        clone_provider = self.get_provider(clone_engine)
        clone_model_key = clone_provider.resolve_model_key("")
        _emit_progress(progress_callback, "model_loading", engine=clone_engine, model=clone_model_key)
        clone_model = self.ensure_model_loaded(clone_engine, clone_model_key)
        generate_cloned = getattr(clone_provider, "generate_cloned_audio", None)
        if generate_cloned is None:
            raise RuntimeError(f"Clone engine {clone_engine} cannot lock VoiceDesign speaker timbre.")
        with interruptible_talker_generate(clone_model, self._cancel_event):
            return generate_cloned(
                clone_model,
                text=text,
                seed=seed,
                ref_audio=(reference_audio, int(reference_sample_rate)),
                ref_text=reference_text,
                instruct=emotion_instruct.strip(),
            )

    def preview_voice_reference(
        self,
        *,
        voice_description: str,
        seed: int | None,
        tts_engine: str,
        tts_model_key: str,
        calibration_text: str = "",
        reroll: bool = False,
    ) -> tuple[str, str, int, str]:
        """Synthesize or reuse the VoiceDesign calibration clip for preview.

        Returns (clip_path, clip_id, seed, status).
        """
        self.clear_cancel()
        engine = self.resolve_tts_engine(tts_engine)
        if engine != ENGINE_QWEN3:
            resolved_seed = self.resolve_seed(seed, False)
            return "", "", resolved_seed, "当前引擎不使用 VoiceDesign 参考音"
        # Request-level calibration text (情绪基调句) wins; empty falls back to the global default.
        lock_text = (
            resolve_calibration_text(calibration_text)
            if calibration_text.strip()
            else resolve_calibration_text()
        )
        if not lock_text:
            resolved_seed = self.resolve_seed(seed, False)
            return "", "", resolved_seed, "音色锁定已关闭（TTS_SPEAKER_REF_MAX_CHARS=0）"

        resolved_seed = self.resolve_seed(None, True) if reroll else self.resolve_seed(seed, False)
        model_key = self.resolve_tts_model_key(engine, tts_model_key)
        prompt = voice_description.strip()
        model = self.ensure_model_loaded(engine, model_key)
        self.raise_if_cancelled()
        provider = self.get_provider(engine)

        def synthesize_lock() -> tuple[np.ndarray, int]:
            self.raise_if_cancelled()
            self.seed_everything(resolved_seed)
            with interruptible_talker_generate(model, self._cancel_event):
                raw_lock, lock_rate_raw = provider.generate_audio(
                    model,
                    text=lock_text,
                    voice_description=prompt,
                    seed=resolved_seed,
                    reference_audio_path=None,
                    reference_text="",
                )
            self.raise_if_cancelled()
            return ensure_mono(raw_lock), int(lock_rate_raw)

        try:
            _, _, clip_id = get_or_create_reference_clip(
                model_key=model_key,
                prompt=prompt,
                seed=resolved_seed,
                calibration_text=lock_text,
                synthesize=synthesize_lock,
                profile_dir=self.voice_profile_dir,
            )
        except GenerationCancelled:
            raise
        clip_path = reference_clip_wav_path(clip_id, self.voice_profile_dir)
        verb = "重新抽取" if reroll else "已准备"
        return str(clip_path), clip_id, resolved_seed, f"{verb}参考音 clip={clip_id} seed={resolved_seed}"

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
        calibration_text: str = "",
        emotion_instruct: str = "",
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
            calibration_text.strip(),
            emotion_instruct.strip(),
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
        return int(seed) % (2**32)

    @staticmethod
    def seed_everything(seed: int) -> None:
        seed = int(seed) % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            torch.mps.manual_seed(seed)

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
        device = resolve_device_name()
        if device == "cpu":
            return torch.float32
        if device == "mps":
            # Apple Silicon supports fp16/bf16. bf16 needs its own matmul support; fp16 is safest here.
            return torch.float16
        # CUDA: check available GPU memory and choose dtype accordingly
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

    def generate_batch(self, request: BatchRequest, progress_callback: ProgressCallback | None = None) -> BatchResult:
        self.clear_cancel()
        seeds = request.seeds if request.seeds else [42]
        effect_types = request.effect_types if request.effect_types else [MODE_MONO]
        output_formats = request.output_formats if request.output_formats else [DEFAULT_FORMAT]

        items: list[BatchItemResult] = []
        succeeded = 0
        failed = 0

        for seed in seeds:
            self.raise_if_cancelled()
            tts_engine = self.resolve_tts_engine(request.tts_engine)
            tts_model_key = self.resolve_tts_model_key(tts_engine, request.tts_model_key)
            prompt = request.voice_description.strip()
            text = self.resolve_text(request.preset_key, request.text)
            reference_text = self._resolve_reference_text(request.reference_audio_path, request.reference_text, tts_engine)
            self.validate_request_inputs(tts_engine, text, request.reference_audio_path, reference_text)

            model = self.ensure_model_loaded(tts_engine, tts_model_key)
            self.raise_if_cancelled()
            self.seed_everything(seed)
            _emit_progress(
                progress_callback,
                "batch_seed",
                seed=seed,
                seed_index=seeds.index(seed) + 1,
                seed_total=len(seeds),
            )

            try:
                audio_mono, sample_rate, cache_state, _clip_id = self.get_or_generate_dry_audio(
                    model,
                    tts_engine=tts_engine,
                    tts_model_key=tts_model_key,
                    text=text,
                    voice_description=prompt,
                    seed=seed,
                    reference_audio_path=request.reference_audio_path,
                    reference_text=reference_text,
                    calibration_text=request.calibration_text,
                    emotion_instruct=request.emotion_instruct,
                    progress_callback=progress_callback,
                )
                self.raise_if_cancelled()
                audio_mono, sample_rate = apply_speed(audio_mono, sample_rate, request.speed_factor)
                self.raise_if_cancelled()
                audio_mono, sample_rate = apply_pitch_shift(audio_mono, sample_rate, request.pitch_semitones)
            except GenerationCancelled:
                raise
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

            for effect_index, effect_type in enumerate(effect_types):
                self.raise_if_cancelled()
                _emit_progress(
                    progress_callback,
                    "batch_render",
                    seed=seed,
                    effect=effect_type,
                    index=len(items) + 1,
                )
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
                    calibration_text=request.calibration_text,
                    emotion_instruct=request.emotion_instruct,
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
                except GenerationCancelled:
                    raise
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
        if not self.output_dir.exists():
            return []
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
        """删除输出目录内的音频文件。拒绝目录穿越。"""
        try:
            target = Path(file_path).expanduser().resolve()
            output_root = self.output_dir.expanduser().resolve()
            target.relative_to(output_root)
        except (OSError, RuntimeError, ValueError):
            LOGGER.warning("Refused to delete path outside output dir: %s", file_path)
            return False
        if not target.exists() or not target.is_file():
            return False
        if target.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg"}:
            LOGGER.warning("Refused to delete non-audio file: %s", target)
            return False
        try:
            target.unlink()
            LOGGER.info("Deleted audio file: %s", target)
            return True
        except Exception as exc:
            LOGGER.warning("Failed to delete %s: %s", target, exc)
            return False
