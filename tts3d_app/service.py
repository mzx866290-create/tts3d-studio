from __future__ import annotations

import datetime as dt
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from tts3d_app.audio_engine import (
    MODE_BEHIND_HEAD,
    MODE_DYNAMIC_HRIR,
    MODE_MONO,
    MODE_STATIC_HRIR,
    MODE_STEREO,
    apply_behind_head_effect,
    apply_dynamic_hrir,
    apply_static_hrir,
    duplicate_stereo,
    ensure_mono,
    normalize_audio,
)
from tts3d_app.config import (
    CLEAR_CUDA_CACHE_AFTER_GENERATE,
    HRIR_FILE,
    MODEL_NAME,
    OUTPUT_DIR,
    configure_runtime,
    get_logger,
)
from tts3d_app.hrir import HrirDataset, load_hrir_dataset
from tts3d_app.presets import PRESETS


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


@dataclass(slots=True)
class GenerationResult:
    file_path: str
    seed: int
    status: str
    text: str


class TTSStudioService:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        output_dir: Path = OUTPUT_DIR,
        hrir_file: Path = HRIR_FILE,
    ) -> None:
        configure_runtime()
        self.model_name = model_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hrir_dataset = load_hrir_dataset(hrir_file)
        self._model: Any | None = None

    def ensure_model_loaded(self) -> Any:
        if self._model is not None:
            return self._model

        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        LOGGER.info("Loading model %s on %s", self.model_name, device_name)
        try:
            from qwen_tts import Qwen3TTSModel
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"Failed to import qwen_tts. Check dependencies and SoX: {exc}") from exc

        try:
            self._model = Qwen3TTSModel.from_pretrained(
                self.model_name,
                device_map=device_name,
                torch_dtype=torch_dtype,
            )
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"Model load failed: {exc}") from exc

        LOGGER.info("Model loaded successfully")
        return self._model

    def generate(self, request: GenerationRequest) -> GenerationResult:
        model = self.ensure_model_loaded()
        prompt = request.voice_description.strip()
        text = self.resolve_text(request.preset_key, request.text)
        seed = self.resolve_seed(request.seed, request.use_random_seed)
        self.seed_everything(seed)

        started_at = time.time()
        LOGGER.info("Generating audio: preset=%s mode=%s seed=%s", request.preset_key, request.render_mode, seed)

        try:
            with torch.inference_mode():
                wavs, sample_rate = self.call_generate_voice_design(
                    model,
                    text=text,
                    voice_description=prompt,
                )
            audio_mono = ensure_mono(self.extract_audio_data(wavs))
            final_audio, final_sample_rate, tag = self.render_audio(audio_mono, sample_rate, request)
            final_audio = normalize_audio(final_audio)

            filename = f"voice_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}.wav"
            output_path = self.output_dir / filename
            sf.write(output_path, final_audio, final_sample_rate)

            elapsed_s = time.time() - started_at
            status = (
                f"Completed in {elapsed_s:.2f}s | mode={request.render_mode} | "
                f"sample_rate={final_sample_rate}Hz"
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
    def call_generate_voice_design(model: Any, text: str, voice_description: str) -> tuple[Any, int]:
        try:
            return model.generate_voice_design(
                text=text,
                voice_description=voice_description,
                instruct="",
            )
        except TypeError:
            return model.generate_voice_design(
                text=text,
                voice_description=voice_description,
            )

    @staticmethod
    def extract_audio_data(wavs: Any) -> np.ndarray:
        first_item = wavs[0] if isinstance(wavs, list) else wavs

        if isinstance(first_item, np.ndarray):
            return first_item.astype(np.float32, copy=False)
        if hasattr(first_item, "detach"):
            return first_item.detach().cpu().float().numpy()
        if hasattr(first_item, "cpu"):
            return first_item.cpu().float().numpy()
        return np.asarray(first_item, dtype=np.float32)
