from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np
import torch

from tts3d_app.config import (
    ENABLE_TORCH_COMPILE,
    FLASH_ATTN_INSTALLED,
    HF_HUB_CACHE,
    QWEN_BASE_MODEL_NAME,
    QWEN_MODEL_NAME,
    TORCH_COMPILE_MODE,
    TTS_MAX_NEW_TOKENS,
    get_logger,
)


LOGGER = get_logger("tts_engines")
ENGINE_QWEN3 = "qwen3"
ENGINE_QWEN3_BASE = "qwen3_base"
DEFAULT_TTS_ENGINE = ENGINE_QWEN3
ENGINE_CHOICES = (
    ("Qwen3-TTS VoiceDesign", ENGINE_QWEN3),
    ("Qwen3-TTS Base Clone", ENGINE_QWEN3_BASE),
)


class GenerationCancelled(RuntimeError):
    """Raised when the user interrupts an in-flight TTS job."""

    def __init__(self, message: str = "Audio generation was cancelled.") -> None:
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class TTSModelOption:
    key: str
    label: str


class TTSProvider(Protocol):
    engine_key: str

    def model_options(self) -> tuple[TTSModelOption, ...]:
        ...

    def resolve_model_key(self, model_key: str | None) -> str:
        ...

    def load_model(self, model_key: str, device_name: str, torch_dtype: torch.dtype) -> Any:
        ...

    def generate_audio(
        self,
        model: Any,
        *,
        text: str,
        voice_description: str,
        seed: int,
        reference_audio_path: str | None,
        reference_text: str,
    ) -> tuple[np.ndarray, int]:
        ...


def build_tts_generate_kwargs(seed: int) -> dict[str, Any]:
    """Kwargs forwarded to Qwen generate_* so each call is seeded and token-capped."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**32))
    return {
        "max_new_tokens": TTS_MAX_NEW_TOKENS,
        "generator": generator,
    }


@contextmanager
def use_seeded_multinomial(seed: int) -> Iterator[None]:
    """Force sampling onto a CPU generator.

    Transformers' generate() calls ``torch.multinomial`` without a generator.
    On MPS that op ignores ``torch.manual_seed`` and is not reproducible.
    """
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(int(seed) % (2**32))
    original = torch.multinomial

    def _seeded_multinomial(
        input: torch.Tensor,
        num_samples: int,
        replacement: bool = False,
        *,
        generator: torch.Generator | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        active_generator = generator if generator is not None else cpu_generator
        if input.device.type == "cpu":
            return original(
                input,
                num_samples,
                replacement,
                generator=active_generator,
                out=out,
            )
        sampled = original(
            input.detach().to(device="cpu", dtype=torch.float32),
            num_samples,
            replacement,
            generator=active_generator,
        ).to(device=input.device)
        if out is not None:
            out.copy_(sampled)
            return out
        return sampled

    torch.multinomial = _seeded_multinomial  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.multinomial = original  # type: ignore[method-assign]


def _find_talker(model: Any) -> Any | None:
    for owner in (model, getattr(model, "model", None)):
        if owner is None:
            continue
        talker = getattr(owner, "talker", None)
        if talker is not None and callable(getattr(talker, "generate", None)):
            return talker
    return None


@contextmanager
def interruptible_talker_generate(model: Any, cancel_event: threading.Event) -> Iterator[None]:
    """Stop Qwen talker.generate at the next decode step when cancel_event is set."""
    talker = _find_talker(model)
    if talker is None:
        yield
        return

    try:
        from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
    except Exception:
        yield
        return

    class _CancelCriteria(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            del input_ids, scores, kwargs
            if cancel_event.is_set():
                raise GenerationCancelled()
            return False

    original = talker.generate

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        extra = _CancelCriteria()
        existing = kwargs.get("stopping_criteria")
        if existing is None:
            kwargs["stopping_criteria"] = StoppingCriteriaList([extra])
        elif isinstance(existing, StoppingCriteriaList):
            kwargs["stopping_criteria"] = StoppingCriteriaList([*existing, extra])
        else:
            kwargs["stopping_criteria"] = StoppingCriteriaList([existing, extra])
        return original(*args, **kwargs)

    talker.generate = wrapped
    try:
        yield
    finally:
        talker.generate = original


def invoke_qwen_generate(method: Any, *, seed: int, **kwargs: Any) -> tuple[Any, int]:
    call_kwargs = {**kwargs, **build_tts_generate_kwargs(seed)}
    with use_seeded_multinomial(seed):
        try:
            return method(**call_kwargs)
        except TypeError as exc:
            if "generator" not in str(exc).lower() and "unexpected" not in str(exc).lower():
                raise
            call_kwargs.pop("generator", None)
            return method(**call_kwargs)


def extract_audio_data(wavs: Any) -> np.ndarray:
    first_item = wavs[0] if isinstance(wavs, list) else wavs

    if isinstance(first_item, np.ndarray):
        return first_item.astype(np.float32, copy=False)
    if hasattr(first_item, "detach"):
        return first_item.detach().cpu().float().numpy()
    if hasattr(first_item, "cpu"):
        return first_item.cpu().float().numpy()
    return np.asarray(first_item, dtype=np.float32)


def get_model_options(
    engine_key: str,
    qwen_model_name: str = QWEN_MODEL_NAME,
    qwen_base_model_name: str = QWEN_BASE_MODEL_NAME,
) -> tuple[TTSModelOption, ...]:
    if engine_key == ENGINE_QWEN3:
        return (TTSModelOption(key=qwen_model_name, label=qwen_model_name),)
    if engine_key == ENGINE_QWEN3_BASE:
        return (TTSModelOption(key=qwen_base_model_name, label=qwen_base_model_name),)
    raise RuntimeError(f"Unsupported TTS engine: {engine_key}")


def get_default_model_key(
    engine_key: str,
    qwen_model_name: str = QWEN_MODEL_NAME,
    qwen_base_model_name: str = QWEN_BASE_MODEL_NAME,
) -> str:
    return get_model_options(engine_key, qwen_model_name, qwen_base_model_name)[0].key


def get_model_choice_tuples(
    engine_key: str,
    qwen_model_name: str = QWEN_MODEL_NAME,
    qwen_base_model_name: str = QWEN_BASE_MODEL_NAME,
) -> list[tuple[str, str]]:
    return [
        (option.label, option.key)
        for option in get_model_options(engine_key, qwen_model_name, qwen_base_model_name)
    ]


class QwenTTSProvider:
    engine_key = ENGINE_QWEN3

    def __init__(self, model_name: str = QWEN_MODEL_NAME) -> None:
        self.model_name = model_name

    def model_options(self) -> tuple[TTSModelOption, ...]:
        return get_model_options(self.engine_key, self.model_name)

    def resolve_model_key(self, model_key: str | None) -> str:
        if model_key in (None, "", self.model_name):
            return self.model_name
        raise RuntimeError(f"Unsupported Qwen3-TTS model: {model_key}")

    def load_model(self, model_key: str, device_name: str, torch_dtype: torch.dtype) -> Any:
        LOGGER.info(
            "Loading Qwen3-TTS model %s on %s with dtype=%s flash_attn=%s",
            model_key,
            device_name,
            torch_dtype,
            FLASH_ATTN_INSTALLED,
        )
        try:
            from qwen_tts import Qwen3TTSModel
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"Failed to import qwen_tts. Check dependencies and SoX: {exc}") from exc

        try:
            try:
                model = Qwen3TTSModel.from_pretrained(
                    model_key,
                    device_map=device_name,
                    torch_dtype=torch_dtype,
                    cache_dir=str(HF_HUB_CACHE),
                )
            except TypeError:
                model = Qwen3TTSModel.from_pretrained(
                    model_key,
                    device_map=device_name,
                    torch_dtype=torch_dtype,
                )
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"Model load failed: {exc}") from exc

        return self._maybe_compile_model(model)

    def generate_audio(
        self,
        model: Any,
        *,
        text: str,
        voice_description: str,
        seed: int,
        reference_audio_path: str | None,
        reference_text: str,
    ) -> tuple[np.ndarray, int]:
        del reference_audio_path, reference_text

        try:
            with torch.inference_mode():
                wavs, sample_rate = invoke_qwen_generate(
                    model.generate_voice_design,
                    seed=seed,
                    text=text,
                    instruct=voice_description,
                    language="Auto",
                    non_streaming_mode=True,
                )
        except GenerationCancelled:
            raise
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"Qwen3-TTS inference failed: {exc}") from exc

        return extract_audio_data(wavs), int(sample_rate)

    @staticmethod
    def _maybe_compile_model(model: Any) -> Any:
        if not ENABLE_TORCH_COMPILE or not hasattr(torch, "compile"):
            return model

        for attr_name in ("model", "tts_model", "llm", "backbone", "transformer"):
            candidate = getattr(model, attr_name, None)
            if not isinstance(candidate, torch.nn.Module):
                continue

            try:
                compiled_candidate = torch.compile(candidate, mode=TORCH_COMPILE_MODE)
                setattr(model, attr_name, compiled_candidate)
                LOGGER.info("Compiled Qwen submodule with torch.compile: %s", attr_name)
                return model
            except Exception as exc:  # pragma: no cover - runtime specific
                LOGGER.warning("torch.compile failed for %s: %s", attr_name, exc)

        LOGGER.info("Skipped torch.compile because no compatible Qwen submodule was found")
        return model


class QwenBaseCloneProvider(QwenTTSProvider):
    engine_key = ENGINE_QWEN3_BASE

    def __init__(self, model_name: str = QWEN_BASE_MODEL_NAME) -> None:
        super().__init__(model_name=model_name)

    def model_options(self) -> tuple[TTSModelOption, ...]:
        return (TTSModelOption(key=self.model_name, label=self.model_name),)

    def resolve_model_key(self, model_key: str | None) -> str:
        if model_key in (None, "", self.model_name):
            return self.model_name
        raise RuntimeError(f"Unsupported Qwen3-TTS Base model: {model_key}")

    def generate_audio(
        self,
        model: Any,
        *,
        text: str,
        voice_description: str,
        seed: int,
        reference_audio_path: str | None,
        reference_text: str,
    ) -> tuple[np.ndarray, int]:
        del voice_description

        text = text.strip()
        if not text:
            raise RuntimeError("Qwen3-TTS Base Clone requires non-empty text.")
        if not reference_audio_path:
            raise RuntimeError("Qwen3-TTS Base Clone requires a reference audio file.")

        reference_audio_file = Path(reference_audio_path)
        if not reference_audio_file.exists():
            raise RuntimeError(f"Qwen3-TTS Base Clone reference audio not found: {reference_audio_file}")

        return self.generate_cloned_audio(
            model,
            text=text,
            seed=seed,
            ref_audio=str(reference_audio_file),
            ref_text=reference_text,
        )

    def generate_cloned_audio(
        self,
        model: Any,
        *,
        text: str,
        seed: int,
        ref_audio: str | tuple[np.ndarray, int],
        ref_text: str,
    ) -> tuple[np.ndarray, int]:
        text = text.strip()
        if not text:
            raise RuntimeError("Qwen3-TTS Base Clone requires non-empty text.")

        normalized_reference_text = ref_text.strip()
        x_vector_only_mode = not bool(normalized_reference_text)

        try:
            with torch.inference_mode():
                wavs, sample_rate = invoke_qwen_generate(
                    model.generate_voice_clone,
                    seed=seed,
                    text=text,
                    language="Auto",
                    ref_audio=ref_audio,
                    ref_text=normalized_reference_text or None,
                    x_vector_only_mode=x_vector_only_mode,
                    non_streaming_mode=True,
                )
        except GenerationCancelled:
            raise
        except Exception as exc:  # pragma: no cover - depends on local runtime
            raise RuntimeError(f"Qwen3-TTS Base Clone inference failed: {exc}") from exc

        return extract_audio_data(wavs), int(sample_rate)


def build_default_providers(
    qwen_model_name: str = QWEN_MODEL_NAME,
    qwen_base_model_name: str = QWEN_BASE_MODEL_NAME,
) -> dict[str, TTSProvider]:
    return {
        ENGINE_QWEN3: QwenTTSProvider(model_name=qwen_model_name),
        ENGINE_QWEN3_BASE: QwenBaseCloneProvider(model_name=qwen_base_model_name),
    }
