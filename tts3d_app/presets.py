from __future__ import annotations

DEFAULT_TEXT = "闭上眼睛，感受我的声音就在你的耳边轻声低语。"

# calibration_text 为该预设的「情绪基调句」（请求级音色校准句）。
# 留空则回退到全局 TTS_VOICE_CALIBRATION_TEXT；仅在 VoiceDesign 引擎生效。
PRESETS = {
    "默认": {
        "prompt": "",
        "text": DEFAULT_TEXT,
        "calibration_text": "",
    },
    "知性女声": {
        "prompt": "A warm, gentle female voice, very close to the ear, whispering.",
        "text": "闭上眼睛，感受我的声音就在你的耳边轻声低语。",
        "calibration_text": "她柔声说道，语气平静而笃定：别急，我们还有很多时间，慢慢来就好。",
    },
    "暗黑低语": {
        "prompt": "A dark, ominous voice with a threatening and breathy tone.",
        "text": "不要动，我就在你的周围，一直看着你。",
        "calibration_text": "他在黑暗中低声呢喃，气息压得很沉：嘘……别出声，我就在你身后。",
    },
    "ASMR 剪发": {
        "prompt": "ASMR, scissors snipping close to the ear.",
        "text": "Snip, snip, snip. Shk, shk.",
        "calibration_text": "她凑近耳边轻声细语：放松，今天就让我来照顾你，好吗？",
    },
}
