from __future__ import annotations

import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np
import torch

from tts3d_app.config import (
    COSYVOICE_MODEL_DIR,
    COSYVOICE_REPO_DIR,
    ENABLE_TORCH_COMPILE,
    FLASH_ATTN_INSTALLED,
    HF_HUB_CACHE,
    QWEN_BASE_MODEL_NAME,
    QWEN_MODEL_NAME,
    TORCH_COMPILE_MODE,
    TTS_COSYVOICE_PYTHON,
    TTS_MAX_NEW_TOKENS,
    get_logger,
)


LOGGER = get_logger("tts_engines")
ENGINE_QWEN3 = "qwen3"
ENGINE_QWEN3_BASE = "qwen3_base"
ENGINE_COSYVOICE2 = "cosyvoice2"
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
        instruct: str = "",
    ) -> tuple[np.ndarray, int]:
        # Qwen3 Base clone has no emotion channel; instruct is accepted for
        # signature parity with the CosyVoice2 clone provider and dropped.
        del instruct
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


class CosyVoice2WorkerHandle:
    """CosyVoice2 长驻子进程封装（线程安全）。

    worker 通过 stdin/stdout 行式通信，回复行带 @@JSON@@ 前缀——
    库（如 modelscope）可能向 stdout 打印进度日志，前缀保证协议不被污染。
    """

    RESPONSE_PREFIX = "@@JSON@@"

    def __init__(self, python_bin: str, worker_script: Path, repo_dir: Path, model_dir: Path) -> None:
        import json
        import subprocess

        self._json = json
        self._lock = threading.Lock()
        if not Path(python_bin).exists():
            raise RuntimeError(
                f"CosyVoice2 python not found: {python_bin}. "
                "Set TTS_COSYVOICE_PYTHON (see tools/cosyvoice_experiment)."
            )
        self._proc = subprocess.Popen(
            [str(python_bin), str(worker_script), str(repo_dir), str(model_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self._readline()
        if not ready.get("ok"):
            raise RuntimeError(f"CosyVoice2 worker failed to start: {ready.get('error')}")
        self.sample_rate = int(ready["sample_rate"])
        self._job_id = 0

    def _readline(self) -> dict[str, Any]:
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("CosyVoice2 worker exited unexpectedly")
            line = line.strip()
            if not line.startswith(self.RESPONSE_PREFIX):
                continue  # 库的 stdout 噪音，跳过
            return self._json.loads(line[len(self.RESPONSE_PREFIX):])

    def synthesize(
        self,
        *,
        text: str,
        instruct_text: str,
        prompt_wav: str,
        out_wav: str,
        seed: int,
    ) -> int:
        with self._lock:
            self._job_id += 1
            job = {
                "id": self._job_id,
                "seed": int(seed),
                "text": text,
                "instruct_text": instruct_text,
                "prompt_wav": prompt_wav,
                "out_wav": out_wav,
            }
            self._proc.stdin.write(self._json.dumps(job) + "\n")
            self._proc.stdin.flush()
            response = self._readline()
            if not response.get("ok"):
                raise RuntimeError(f"CosyVoice2 inference failed: {response.get('error')}")
            return int(response["sample_rate"])


class CosyVoice2Provider:
    """Clone-only engine: reproduces the VoiceDesign clip timbre with an emotion instruct.

    Qwen3-TTS VoiceDesign still creates/auditions the voice (clip, seed, favorites);
    this provider only replaces the per-chunk clone step in the timbre-lock chain.
    Inference runs in a long-lived subprocess using the isolated CosyVoice2 venv,
    so none of its pinned dependencies leak into the app environment.
    """

    engine_key = ENGINE_COSYVOICE2

    def __init__(
        self,
        model_dir: Path = COSYVOICE_MODEL_DIR,
        repo_dir: Path = COSYVOICE_REPO_DIR,
        python_bin: str = TTS_COSYVOICE_PYTHON,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.repo_dir = Path(repo_dir)
        self.python_bin = python_bin
        self._temp_dir = Path(tempfile.gettempdir()) / "tts3d_cosyvoice_refs"

    def model_options(self) -> tuple[TTSModelOption, ...]:
        label = f"CosyVoice2 ({self.model_dir.name})"
        return (TTSModelOption(key=self.engine_key, label=label),)

    def resolve_model_key(self, model_key: str | None) -> str:
        if model_key in (None, "", self.engine_key):
            return self.engine_key
        raise RuntimeError(f"Unsupported CosyVoice2 model: {model_key}")

    def load_model(self, model_key: str, device_name: str, torch_dtype: torch.dtype) -> Any:
        del model_key, device_name, torch_dtype
        if not self.model_dir.exists():
            raise RuntimeError(
                f"CosyVoice2 model dir not found: {self.model_dir}. "
                "Set COSYVOICE_MODEL_DIR or download weights (see tools/cosyvoice_experiment)."
            )
        worker_script = Path(__file__).resolve().parent / "cosyvoice_worker.py"
        return CosyVoice2WorkerHandle(self.python_bin, worker_script, self.repo_dir, self.model_dir)

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
        del model, text, voice_description, seed, reference_audio_path, reference_text
        raise RuntimeError(
            "CosyVoice2 is a clone-only engine: create the voice with Qwen3-TTS VoiceDesign first."
        )

    def generate_cloned_audio(
        self,
        model: Any,
        *,
        text: str,
        seed: int,
        ref_audio: str | tuple[np.ndarray, int],
        ref_text: str,
        instruct: str = "",
    ) -> tuple[np.ndarray, int]:
        del ref_text  # instruct2 replaces the reference transcript slot with the style prompt
        text = text.strip()
        if not text:
            raise RuntimeError("CosyVoice2 requires non-empty text.")

        prompt_wav = self._ensure_prompt_wav_path(ref_audio)
        out_wav = self._new_temp_path(".wav")
        sample_rate = model.synthesize(
            text=text,
            instruct_text=self._format_instruct_text(instruct),
            prompt_wav=prompt_wav,
            out_wav=out_wav,
            seed=seed,
        )
        import soundfile as sf

        audio, file_sample_rate = sf.read(out_wav, dtype="float32", always_2d=True)
        mono = np.mean(audio, axis=1).astype(np.float32, copy=False)
        return mono, int(file_sample_rate) or int(sample_rate)

    def _ensure_prompt_wav_path(self, ref_audio: str | tuple[np.ndarray, int]) -> str:
        """The worker loads the prompt wav itself, so hand it a real file path."""
        if isinstance(ref_audio, (str, Path)):
            path = Path(ref_audio)
            if not path.exists():
                raise RuntimeError(f"CosyVoice2 reference audio not found: {path}")
            return str(path)

        waveform, sample_rate = ref_audio
        import soundfile as sf

        self._temp_dir.mkdir(parents=True, exist_ok=True)
        path = self._new_temp_path(".wav")
        sf.write(path, np.asarray(waveform, dtype=np.float32).reshape(-1), int(sample_rate), format="WAV")
        return path

    def _new_temp_path(self, suffix: str) -> str:
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(suffix=suffix, dir=self._temp_dir)
        os.close(fd)
        return path

    @staticmethod
    def _format_instruct_text(instruct: str) -> str:
        normalized = instruct.strip()
        if not normalized:
            normalized = "用自然平稳的语气说话"
        if not normalized.endswith("<|endofprompt|>"):
            normalized = f"{normalized}<|endofprompt|>"
        return normalized


def build_default_providers(
    qwen_model_name: str = QWEN_MODEL_NAME,
    qwen_base_model_name: str = QWEN_BASE_MODEL_NAME,
) -> dict[str, TTSProvider]:
    return {
        ENGINE_QWEN3: QwenTTSProvider(model_name=qwen_model_name),
        ENGINE_QWEN3_BASE: QwenBaseCloneProvider(model_name=qwen_base_model_name),
        # Clone-only engine; instantiating it is cheap, weights load lazily on first use.
        ENGINE_COSYVOICE2: CosyVoice2Provider(),
    }
