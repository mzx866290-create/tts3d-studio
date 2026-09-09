from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


_SENTENCE_RE = re.compile(
    r".+?(?:"
    r"[。！？!?]"
    r"|\.(?=\s|\n|$)"
    r"|[；;]"
    r"|$"
    r")",
    re.DOTALL,
)
_CLAUSE_RE = re.compile(r"(?<=[，,、])")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str
    pause_after_ms: int


def split_text_for_tts(
    text: str,
    max_chars: int = 400,
    sentence_pause_ms: int = 300,
    paragraph_pause_ms: int = 700,
) -> list[TextChunk]:
    """Split long TTS text at paragraph/sentence boundaries.

    Qwen3-TTS generates one shot up to max_new_tokens (default 2048, ~2.7 min).
    Packing sentences into modest chunks keeps each call under that cap.
    """
    stripped = text.strip()
    if not stripped:
        return []
    if max_chars <= 0 or len(stripped) <= max_chars:
        return [TextChunk(text=stripped, pause_after_ms=0)]

    units = _collect_units(stripped, max_chars)
    packed = _pack_units(units, max_chars, sentence_pause_ms, paragraph_pause_ms)
    if packed:
        last = packed[-1]
        packed[-1] = TextChunk(text=last.text, pause_after_ms=0)
    return packed


def take_reference_sentence(text: str, max_chars: int = 60) -> str:
    """Take the opening sentence(s) of text, packed under max_chars.

    Used as the VoiceDesign calibration transcript whose audio+text lock later
    Base Clone ICL calls. Overlong first sentences are split at clauses.
    """
    stripped = text.strip()
    if not stripped or max_chars <= 0:
        return ""
    if len(stripped) <= max_chars:
        return stripped

    packed: list[str] = []
    current_len = 0
    for sentence in _split_sentences(stripped):
        for piece in _split_overlong(sentence, max_chars):
            extra = len(piece)
            if packed and current_len + extra > max_chars:
                return _join_units(packed)
            packed.append(piece)
            current_len += extra
            if current_len >= max_chars:
                return _join_units(packed)
    return _join_units(packed)


def concatenate_mono_chunks(
    audios: list[np.ndarray],
    pauses_ms: list[int],
    sample_rate: int,
) -> np.ndarray:
    """Concatenate mono clips, inserting silence after each clip except the last."""
    if not audios:
        return np.zeros(0, dtype=np.float32)

    parts: list[np.ndarray] = []
    last_index = len(audios) - 1
    for index, audio in enumerate(audios):
        clip = np.asarray(audio, dtype=np.float32).reshape(-1)
        parts.append(clip)
        if index == last_index:
            continue
        pause_ms = pauses_ms[index] if index < len(pauses_ms) else 0
        pause_samples = int(sample_rate * max(pause_ms, 0) / 1000.0)
        if pause_samples > 0:
            parts.append(np.zeros(pause_samples, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32, copy=False)


def _collect_units(text: str, max_chars: int) -> list[tuple[str, bool]]:
    units: list[tuple[str, bool]] = []
    paragraphs = [part.strip() for part in _PARAGRAPH_RE.split(text) if part.strip()]
    if not paragraphs:
        paragraphs = [text]

    for paragraph in paragraphs:
        sentences = _split_sentences(paragraph)
        for sentence_index, sentence in enumerate(sentences):
            is_paragraph_end = sentence_index == len(sentences) - 1
            pieces = _split_overlong(sentence, max_chars)
            for piece_index, piece in enumerate(pieces):
                piece_is_para_end = is_paragraph_end and piece_index == len(pieces) - 1
                units.append((piece, piece_is_para_end))
    return units


def _pack_units(
    units: list[tuple[str, bool]],
    max_chars: int,
    sentence_pause_ms: int,
    paragraph_pause_ms: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    current: list[str] = []
    current_len = 0
    current_para_end = False

    def flush() -> None:
        nonlocal current, current_len, current_para_end
        if not current:
            return
        pause = paragraph_pause_ms if current_para_end else sentence_pause_ms
        chunks.append(TextChunk(text=_join_units(current), pause_after_ms=pause))
        current = []
        current_len = 0
        current_para_end = False

    for text, is_paragraph_end in units:
        extra = len(text)
        if current and current_len + extra > max_chars:
            flush()
        current.append(text)
        current_len += extra
        current_para_end = is_paragraph_end
        if current_len >= max_chars:
            flush()

    flush()
    return [chunk for chunk in chunks if chunk.text]


def _join_units(parts: list[str]) -> str:
    output = ""
    for part in parts:
        if output and part and (not output[-1].isspace()) and part[0].isascii() and part[0].isalnum():
            output += " "
        output += part
    return output.strip()


def _split_sentences(paragraph: str) -> list[str]:
    pieces = [match.group(0) for match in _SENTENCE_RE.finditer(paragraph)]
    sentences = [piece.strip() for piece in pieces if piece.strip()]
    return sentences or [paragraph]


def _split_overlong(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]

    clauses = [part for part in _CLAUSE_RE.split(sentence) if part]
    packed: list[str] = []
    buf = ""
    for clause in clauses:
        if buf and len(buf) + len(clause) > max_chars:
            packed.extend(_hard_wrap(buf, max_chars))
            buf = clause
        else:
            buf += clause
    if buf:
        packed.extend(_hard_wrap(buf, max_chars))
    return packed or _hard_wrap(sentence, max_chars)


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]
