"""UI 共享常量与文案：Gradio UI 与 Web UI 都从这里取，避免两处漂移。"""
from __future__ import annotations

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
from tts3d_app.config import (
    MAX_TTS_CHUNK_CHARS,
    TTS_CHUNK_PARAGRAPH_PAUSE_MS,
    TTS_CHUNK_SENTENCE_PAUSE_MS,
    TTS_CLONE_ENGINE,
    TTS_DIRECT_SINGLE_CHUNK,
    TTS_SPEAKER_REF_MAX_CHARS,
)
from tts3d_app.text_chunking import split_text_for_tts
from tts3d_app.tts_engines import ENGINE_QWEN3

MODE_LABEL_TO_KEY = {
    "单声道": MODE_MONO,
    "双声道": MODE_STEREO,
    "虚拟脑后": MODE_BEHIND_HEAD,
    "静态空间定位 (HRIR)": MODE_STATIC_HRIR,
    "动态空间环绕 (HRIR)": MODE_DYNAMIC_HRIR,
}
MODE_KEY_TO_LABEL = {value: key for key, value in MODE_LABEL_TO_KEY.items()}

PATH_LABEL_TO_KEY = {
    "顺时针绕头": PATH_CLOCKWISE,
    "逆时针绕头": PATH_COUNTERCLOCKWISE,
    "左右摆动": PATH_SIDE_TO_SIDE,
}
PATH_KEY_TO_LABEL = {value: key for key, value in PATH_LABEL_TO_KEY.items()}

EMOTION_PRESETS = (
    ("平静叙述", "用平静沉稳的语气叙述，语速适中"),
    ("温柔耳语", "用温柔轻缓的耳语说，气息贴近耳边"),
    ("激动哽咽", "用激动而哽咽的语气说，声音微微颤抖"),
    ("悬疑低语", "用低沉神秘的语气说，气息压低，语速偏慢"),
)


def describe_generation_route(text: str, emotion_instruct: str, tts_engine: str, lock_timbre: bool = False) -> str:
    """实时提示本次生成将走的路由：Base 克隆 / 单块直出 / 音色锁定 / 情感克隆。"""
    from tts3d_app.tts_engines import ENGINE_QWEN3_BASE

    if tts_engine != ENGINE_QWEN3:
        return "🎙️ Base Clone：用参考音频直接克隆，不走音色锁定"
    if not text.strip():
        return "⌨️ 输入文本后，此处会提示本次生成走哪条链路"
    if TTS_SPEAKER_REF_MAX_CHARS <= 0:
        return "🔓 音色锁定已关闭（TTS_SPEAKER_REF_MAX_CHARS=0）：每块独立 VoiceDesign，块间可能变声"

    chunk_count = len(
        split_text_for_tts(
            text.strip(),
            max_chars=MAX_TTS_CHUNK_CHARS,
            sentence_pause_ms=TTS_CHUNK_SENTENCE_PAUSE_MS,
            paragraph_pause_ms=TTS_CHUNK_PARAGRAPH_PAUSE_MS,
        )
    )
    emotion = emotion_instruct.strip()
    if emotion:
        if TTS_CLONE_ENGINE.strip() == "qwen3_base":
            return (
                f"⚠️ 当前 TTS_CLONE_ENGINE=qwen3_base 无情感通道，情感指令将被忽略；"
                f"将锁定音色分 {chunk_count} 块克隆"
            )
        return f"🎭 情感克隆：锁定音色 + CosyVoice2 情感演绎，共 {chunk_count} 块（首次需加载模型 1-2 分钟）"

    emotion_tip = (
        "💡 想更有感情？在下方「情感指令」填一句（如「用温柔哽咽的语气说」）或点快捷情绪按钮"
        if TTS_CLONE_ENGINE.strip() != "qwen3_base"
        else ""
    )
    if lock_timbre:
        return (
            f"🔒 音色锁定：用参考音的音色克隆本文（短文本也锁定），共 {chunk_count} 块。"
            f"抽卡试听听到什么声音，正文就是什么声音。{emotion_tip}"
        )
    if TTS_DIRECT_SINGLE_CHUNK and chunk_count == 1:
        return "⚡ 单块直出：不加载克隆引擎，速度最快。声音由提示词+seed+文本共同采样，换文本会换声（与参考音试听不是同一把声音；想用试听的音色请打开「用参考音音色生成本文」）"
    return f"🔒 音色锁定：校准句钉住音色，{chunk_count} 块逐块克隆后拼接，换稿不变声。{emotion_tip}"
