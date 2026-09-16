from __future__ import annotations

import random
from pathlib import Path

import gradio as gr

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
    ASR_AVAILABLE,
    MAX_TTS_CHUNK_CHARS,
    TTS_CHUNK_PARAGRAPH_PAUSE_MS,
    TTS_CHUNK_SENTENCE_PAUSE_MS,
    TTS_CLONE_ENGINE,
    TTS_DIRECT_SINGLE_CHUNK,
    TTS_SPEAKER_REF_MAX_CHARS,
    load_user_config,
    load_user_favorites,
    load_user_presets,
    save_user_favorites,
    save_user_presets,
    save_user_config,
)
from tts3d_app.presets import PRESETS
from tts3d_app.service import BatchRequest, GenerationCancelled, GenerationRequest, TTSStudioService
from tts3d_app.text_chunking import split_text_for_tts
from tts3d_app.tts_engines import DEFAULT_TTS_ENGINE, ENGINE_QWEN3
from tts3d_app.voice_profile import reference_clip_wav_path


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


def describe_generation_route(text: str, emotion_instruct: str, tts_engine: str) -> str:
    """实时提示本次生成将走的路由：Base 克隆 / 单块直出 / 音色锁定 / 情感克隆。"""
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

    if TTS_DIRECT_SINGLE_CHUNK and chunk_count == 1:
        return "⚡ 单块直出：不加载克隆引擎，速度最快。声音由提示词+seed+文本共同采样，换文本会换声（与参考音试听不是同一把声音；要锁音色请加长文本或填情感指令）"
    emotion_tip = (
        "💡 想更有感情？在下方「情感指令」填一句（如「用温柔哽咽的语气说」）或点快捷情绪按钮"
        if TTS_CLONE_ENGINE.strip() != "qwen3_base"
        else ""
    )
    return f"🔒 音色锁定：校准句钉住音色，{chunk_count} 块逐块克隆后拼接，换稿不变声。{emotion_tip}"


def describe_tts_engine_ui_state(
    service: TTSStudioService,
    engine_key: str,
) -> tuple[list[tuple[str, str]], str, bool, bool]:
    model_choices = service.get_tts_model_choices(engine_key)
    default_model_key = service.get_default_tts_model_key(engine_key)
    show_qwen_prompt = engine_key == ENGINE_QWEN3
    show_clone_inputs = not show_qwen_prompt
    return model_choices, default_model_key, show_qwen_prompt, show_clone_inputs


HISTORY_TABLE_HEADERS = ("文件名", "修改时间", "大小")
HISTORY_TABLE_COLUMNS = ("name", "modified", "size")


def load_audio_history(service: TTSStudioService) -> list[dict]:
    """加载历史音频列表"""
    return service.list_generated_audios()


def history_table_rows(files: list[dict]) -> list[list]:
    """返回二维列表，避免 Gradio DataFrame 把 dict 渲染成 [object Object]。"""
    return [[item[key] for key in HISTORY_TABLE_COLUMNS] for item in files]


def delete_audio_file(service: TTSStudioService, file_path: str) -> tuple[str, list[list]]:
    """删除音频文件"""
    success = service.delete_audio(file_path)
    rows = history_table_rows(service.list_generated_audios())
    if success:
        return "删除成功", rows
    return "删除失败", rows


def create_demo(service: TTSStudioService | None = None) -> gr.Blocks:
    service = service or TTSStudioService()
    has_hrir = service.hrir_dataset is not None

    mode_choices = ["单声道", "双声道", "虚拟脑后"]
    if has_hrir:
        mode_choices.extend(["静态空间定位 (HRIR)", "动态空间环绕 (HRIR)"])

    default_mode = "动态空间环绕 (HRIR)" if has_hrir else "虚拟脑后"
    default_preset_key = service.default_preset_key()
    default_engine_key = DEFAULT_TTS_ENGINE
    default_model_choices, default_model_key, _, _ = describe_tts_engine_ui_state(service, default_engine_key)
    saved_config = load_user_config()
    default_ref_audio = saved_config.get("last_reference_audio", "") if saved_config.get("last_reference_audio") and Path(saved_config["last_reference_audio"]).exists() else None

    def toggle_mode_panels(mode_label: str):
        mode_key = MODE_LABEL_TO_KEY[mode_label]
        return (
            gr.update(visible=mode_key == MODE_STATIC_HRIR),
            gr.update(visible=mode_key == MODE_DYNAMIC_HRIR),
        )

    def apply_preset(preset_key: str) -> tuple[str, str, str]:
        if preset_key in PRESETS:
            preset = PRESETS[preset_key]
            return preset["prompt"], preset["text"], preset.get("calibration_text", "")
        user_presets = load_user_presets()
        if preset_key in user_presets:
            preset = user_presets[preset_key]
            return preset["prompt"], preset["text"], preset.get("calibration_text", "")
        return "", "", ""

    def save_current_as_preset(
        preset_name: str,
        prompt: str,
        text: str,
        calibration_text: str,
    ) -> tuple[gr.update, str]:
        """保存当前 prompt、text 和情绪基调句为新预设"""
        if not preset_name or not preset_name.strip():
            return gr.update(), "请输入预设名称"
        name = preset_name.strip()
        if name in PRESETS:
            return gr.update(), f"预设名 '{name}' 与内置预设冲突，请换名"
        user_presets = load_user_presets()
        user_presets[name] = {"prompt": prompt, "text": text, "calibration_text": calibration_text}
        save_user_presets(user_presets)
        new_choices = list(PRESETS.keys()) + list(user_presets.keys())
        return gr.update(choices=new_choices, value=name), f"已保存为: {name}"

    def save_current_as_favorite(
        fav_name: str,
        prompt: str,
        text: str,
        calibration_text: str,
        emotion_instruct: str,
        seed: int | None,
        mode_label: str,
        speed_factor: float,
        pitch_semitones: float,
        clip_id: str,
    ) -> tuple[gr.update, str]:
        """把当前声线（prompt + 情绪基调句 + 情感指令 + seed + 参考音 clip + 空间效果）收藏保存"""
        if not fav_name or not fav_name.strip():
            return gr.update(), "请先填入收藏名称"
        name = fav_name.strip()
        favorites = load_user_favorites()
        favorites[name] = {
            "prompt": prompt,
            "text": text,
            "calibration_text": calibration_text,
            "emotion_instruct": emotion_instruct,
            "seed": int(seed) if seed is not None else None,
            "mode_label": mode_label,
            "speed_factor": float(speed_factor),
            "pitch_semitones": float(pitch_semitones),
            "clip_id": (clip_id or "").strip(),
        }
        save_user_favorites(favorites)
        new_choices = list(favorites.keys())
        clip_note = f" clip={clip_id}" if clip_id else ""
        return gr.update(choices=new_choices, value=name), f"已收藏声线: {name}{clip_note}"

    def apply_favorite(fav_name: str) -> tuple[str, str, str, str, int, str, float, float, bool, str, str | None, str]:
        """把收藏的声线回填到输入框，并取消随机 Seed，保证复现同一把声音。"""
        favorites = load_user_favorites()
        fav = favorites.get(fav_name)
        if not fav:
            fallback_mode = MODE_KEY_TO_LABEL.get(MODE_BEHIND_HEAD, "虚拟脑后")
            return "", "", "", "", 42, fallback_mode, 1.0, 0.0, True, f"未找到收藏: {fav_name}", None, ""
        prompt = fav.get("prompt", "")
        text = fav.get("text", "")
        calibration_text = str(fav.get("calibration_text") or "")
        emotion_instruct = str(fav.get("emotion_instruct") or "")
        seed = fav.get("seed")
        mode_label = fav.get("mode_label") or MODE_KEY_TO_LABEL.get(MODE_BEHIND_HEAD, "虚拟脑后")
        speed_factor = fav.get("speed_factor", 1.0)
        pitch_semitones = fav.get("pitch_semitones", 0.0)
        clip_id = str(fav.get("clip_id") or "")
        resolved_seed = int(seed) if seed is not None else 42
        clip_path = reference_clip_wav_path(clip_id, service.voice_profile_dir)
        clip_audio = str(clip_path) if clip_id and clip_path.exists() else None
        clip_note = f"，参考音 clip={clip_id}" if clip_id else ""
        emotion_note = "，含情感指令" if emotion_instruct else ""
        return (
            prompt,
            text,
            calibration_text,
            emotion_instruct,
            resolved_seed,
            mode_label,
            float(speed_factor),
            float(pitch_semitones),
            False,
            f"已应用声线「{fav_name}」，Seed 已固定为 {resolved_seed}（随机抽奖已关闭）{clip_note}{emotion_note}。换内容请改上方【文本】后点【🎵 生成音频】",
            clip_audio,
            clip_id,
        )

    def delete_favorite(fav_name: str) -> gr.update:
        """删除收藏的声线"""
        favorites = load_user_favorites()
        if fav_name in favorites:
            del favorites[fav_name]
            save_user_favorites(favorites)
        new_choices = list(favorites.keys())
        return gr.update(choices=new_choices, value=None)

    def reload_favorite_choices() -> gr.update:
        """从磁盘重读收藏列表。只更新选项，不改当前选中项，避免触发应用声线。"""
        names = list(load_user_favorites().keys())
        return gr.update(choices=names)

    def toggle_tts_engine(engine_key: str):
        model_choices, default_model_key, show_qwen_prompt, show_clone_inputs = describe_tts_engine_ui_state(
            service,
            engine_key,
        )
        return (
            gr.update(choices=model_choices, value=default_model_key),
            gr.update(visible=show_qwen_prompt),
            gr.update(visible=show_clone_inputs),
        )

    def _idle_generate_buttons() -> tuple[gr.update, gr.update]:
        return gr.update(interactive=True), gr.update(interactive=False)

    def _busy_generate_buttons() -> tuple[gr.update, gr.update]:
        return gr.update(interactive=False), gr.update(interactive=True)

    def on_generate_start() -> tuple[gr.update, gr.update, str]:
        return (*_busy_generate_buttons(), "正在生成…")

    def on_stop_generate() -> tuple[gr.update, gr.update, str]:
        service.request_cancel()
        return (*_idle_generate_buttons(), "正在停止…")

    def generate_audio(
        preset_key: str,
        tts_engine: str,
        tts_model_key: str,
        voice_description: str,
        reference_audio_path: str | None,
        reference_text: str,
        text: str,
        seed: int | float | None,
        use_random_seed: bool,
        mode_label: str,
        static_azimuth_deg: float,
        static_distance_m: float,
        dynamic_path_label: str,
        dynamic_cycle_time_s: float,
        dynamic_start_azimuth_deg: float,
        dynamic_distance_m: float,
        speed_factor: float,
        pitch_semitones: float,
        calibration_text: str,
        emotion_instruct: str,
    ) -> tuple[str | None, int | float | None, bool, str, gr.update, gr.update, str | None, str]:
        request = GenerationRequest(
            preset_key=preset_key,
            voice_description=voice_description,
            text=text,
            seed=int(seed) if seed is not None else None,
            use_random_seed=bool(use_random_seed),
            render_mode=MODE_LABEL_TO_KEY[mode_label],
            static_azimuth_deg=static_azimuth_deg,
            static_distance_m=static_distance_m,
            dynamic_path=PATH_LABEL_TO_KEY[dynamic_path_label],
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

        try:
            result = service.generate(request)
        except GenerationCancelled:
            return gr.update(), seed, bool(use_random_seed), "已中断生成", *_idle_generate_buttons(), None, ""
        except Exception as exc:
            return gr.update(), seed, bool(use_random_seed), f"生成失败: {exc}", *_idle_generate_buttons(), None, ""

        clip_audio = result.clip_path or None
        return (
            result.file_path,
            result.seed,
            False,
            result.status,
            *_idle_generate_buttons(),
            clip_audio,
            result.clip_id,
        )

    def preview_voice_clip(
        voice_description: str,
        seed: int | float | None,
        tts_engine: str,
        tts_model_key: str,
        calibration_text: str,
        reroll: bool,
    ) -> tuple[str | None, str, int, str]:
        try:
            clip_path, clip_id, resolved_seed, status = service.preview_voice_reference(
                voice_description=voice_description,
                seed=int(seed) if seed is not None else None,
                tts_engine=tts_engine,
                tts_model_key=tts_model_key,
                calibration_text=calibration_text,
                reroll=reroll,
            )
        except GenerationCancelled:
            return None, "", int(seed) if seed is not None else 42, "已中断参考音生成"
        except Exception as exc:
            fallback_seed = int(seed) if seed is not None else 42
            return None, "", fallback_seed, f"参考音生成失败: {exc}"
        return clip_path or None, clip_id, resolved_seed, status

    def preview_clip_from_ui(
        voice_description: str,
        seed: int | float | None,
        tts_engine: str,
        tts_model_key: str,
        calibration_text: str,
    ) -> tuple[str | None, str, int, str]:
        return preview_voice_clip(voice_description, seed, tts_engine, tts_model_key, calibration_text, False)

    def reroll_clip_from_ui(
        voice_description: str,
        seed: int | float | None,
        tts_engine: str,
        tts_model_key: str,
        calibration_text: str,
    ) -> tuple[str | None, str, int, str]:
        return preview_voice_clip(voice_description, seed, tts_engine, tts_model_key, calibration_text, True)

    def on_reference_audio_change(audio_path: str | None) -> str:
        """参考音频变更时保存路径到配置"""
        if audio_path and Path(audio_path).exists():
            config = load_user_config()
            config["last_reference_audio"] = audio_path
            save_user_config(config)
        return ""

    def auto_transcribe(audio_path: str | None) -> str:
        if not audio_path:
            raise gr.Error("请先上传参考音频")
        if not ASR_AVAILABLE:
            raise gr.Error("未安装 ASR 后端，请运行: pip install faster-whisper")
        try:
            from tts3d_app.asr import transcribe_audio
            return transcribe_audio(audio_path)
        except Exception as exc:
            raise gr.Error(f"识别失败: {exc}")

    def batch_generate_audio(
        tts_engine: str,
        tts_model_key: str,
        voice_description: str,
        reference_audio_path: str | None,
        reference_text: str,
        text: str,
        preset_key: str,
        static_azimuth_deg: float,
        static_distance_m: float,
        dynamic_path_label: str,
        dynamic_cycle_time_s: float,
        dynamic_start_azimuth_deg: float,
        dynamic_distance_m: float,
        speed_factor: float,
        pitch_semitones: float,
        calibration_text: str,
        emotion_instruct: str,
        batch_count: int,
        batch_effect_labels: list[str],
    ) -> tuple[str, list[str], gr.update, gr.update]:
        if not batch_effect_labels:
            return "请至少选择一个效果类型", [], *_idle_generate_buttons()
        effect_types = [MODE_LABEL_TO_KEY[label] for label in batch_effect_labels]
        seeds = [random.randint(0, 2**32 - 1) for _ in range(int(batch_count))]
        batch_request = BatchRequest(
            text=text,
            voice_description=voice_description,
            tts_engine=tts_engine,
            tts_model_key=tts_model_key,
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
            preset_key=preset_key,
            seeds=seeds,
            effect_types=effect_types,
            static_azimuth_deg=static_azimuth_deg,
            static_distance_m=static_distance_m,
            dynamic_path=PATH_LABEL_TO_KEY[dynamic_path_label],
            dynamic_cycle_time_s=dynamic_cycle_time_s,
            dynamic_start_azimuth_deg=dynamic_start_azimuth_deg,
            dynamic_distance_m=dynamic_distance_m,
            speed_factor=speed_factor,
            pitch_semitones=pitch_semitones,
            calibration_text=calibration_text,
            emotion_instruct=emotion_instruct,
        )
        try:
            result = service.generate_batch(batch_request)
        except GenerationCancelled:
            return "已中断批量生成", [], *_idle_generate_buttons()
        except Exception as exc:
            return f"批量生成失败: {exc}", [], *_idle_generate_buttons()

        status_lines = [f"完成 {result.succeeded}/{result.total}，失败 {result.failed}"]
        for item in result.items:
            status_lines.append(f"  [{item.effect_type}] seed={item.seed}: {item.status}")
        files = [item.file_path for item in result.items if item.file_path]
        return "\n".join(status_lines), files, *_idle_generate_buttons()

    with gr.Blocks(title="TTS 3D Studio") as demo:
        gr.Markdown("<p class='main-title'>TTS 3D Studio</p>")
        gr.Markdown("<p class='subtitle'>输入文本，选择 TTS 引擎与模型，叠加双声道、脑后或 HRIR 空间音频效果</p>")
        if not has_hrir:
            gr.HTML("<div class='warning-box'>⚠️ 提示：未检测到 HRIR 数据文件，当前只提供基础模式和虚拟脑后模式。</div>")

        with gr.Row():
            with gr.Column(scale=4):
                input_text = gr.Textbox(
                    label="文本",
                    value=PRESETS[default_preset_key]["text"],
                    lines=3,
                    info="短文本（单块）直接 VoiceDesign 直出：快、不加载克隆模型，但同提示词+seed 换文本会换声音。长文本自动分块并用校准句锁音色逐块克隆（换稿不变声）；填「情感指令」时始终走克隆链路。",
                )
                route_hint = gr.Markdown(
                    describe_generation_route(PRESETS[default_preset_key]["text"], "", default_engine_key)
                )

                with gr.Row():
                    all_presets = list(PRESETS.keys()) + list(load_user_presets().keys())
                    preset_dropdown = gr.Dropdown(
                        label="预设",
                        choices=all_presets,
                        value=default_preset_key,
                        scale=1,
                    )
                    preset_name_input = gr.Textbox(label="新预设名", placeholder="输入名称后点保存", scale=1)
                    save_preset_btn = gr.Button("保存预设", variant="secondary", size="sm", scale=1)
                    tts_engine_dropdown = gr.Dropdown(
                        label="TTS 引擎",
                        choices=service.get_tts_engine_choices(),
                        value=default_engine_key,
                        scale=1,
                    )
                    tts_model_dropdown = gr.Dropdown(
                        label="模型",
                        choices=default_model_choices,
                        value=default_model_key,
                        scale=2,
                    )

                with gr.Group(visible=True) as qwen_prompt_group:
                    prompt_input = gr.Textbox(
                        label="📝 TTS 提示词（英文）",
                        value=PRESETS[default_preset_key]["prompt"],
                        placeholder="填写英文提示词，描述声音特征",
                    )
                    calibration_text_input = gr.Textbox(
                        label="情绪基调句",
                        value=PRESETS[default_preset_key].get("calibration_text", ""),
                        placeholder="留空则使用全局校准句（TTS_VOICE_CALIBRATION_TEXT）",
                        info="用于音色锁定的校准句，决定参考音的语气与情绪；修改后音色会随之变化",
                    )
                    emotion_instruct_input = gr.Textbox(
                        label="情感指令（可选）",
                        value="",
                        placeholder="例如：用激动而悲伤的语气说，声音颤抖",
                        info="逐块控制演绎情感，同一音色可换情绪（需 CosyVoice2 克隆引擎，填写即自动启用；TTS_CLONE_ENGINE 可强制指定）",
                    )
                    with gr.Row():
                        for _label, _preset_text in EMOTION_PRESETS:
                            gr.Button(_label, size="sm", variant="secondary").click(
                                lambda t=_preset_text: t,
                                outputs=emotion_instruct_input,
                            ).then(
                                describe_generation_route,
                                inputs=[input_text, emotion_instruct_input, tts_engine_dropdown],
                                outputs=route_hint,
                            )
                    voice_clip_audio = gr.Audio(
                        label="参考音试听（与正文无关，满意后可收藏）",
                        type="filepath",
                        elem_classes="audio-player",
                    )
                    voice_clip_id = gr.Textbox(visible=False, value="")
                    with gr.Row():
                        preview_clip_btn = gr.Button("▶ 生成参考音", variant="secondary", scale=1)
                        reroll_clip_btn = gr.Button("🎲 重新抽一次参考音", variant="secondary", scale=1)
                with gr.Group(visible=False) as clone_reference_group:
                    reference_audio_input = gr.Audio(
                        label="Qwen3 Base 克隆参考音频",
                        type="filepath",
                        sources=["upload"],
                        value=default_ref_audio,
                        elem_classes="audio-player",
                    )
                    with gr.Row():
                        reference_text_input = gr.Textbox(
                            label="参考文本（可选）",
                            placeholder="填写参考音频对应文本可提升克隆质量；也可点击「自动识别」",
                            scale=4,
                        )
                        auto_transcribe_btn = gr.Button(
                            "🎙️ 自动识别",
                            variant="secondary",
                            scale=1,
                            interactive=ASR_AVAILABLE,
                        )

                with gr.Accordion("音频空间引擎", open=True):
                    mode_radio = gr.Radio(
                        choices=mode_choices,
                        value=default_mode,
                        label="输出模式",
                    )

                    with gr.Accordion("语速 / 音调  （在空间效果前处理）", open=False):
                        speed_slider = gr.Slider(0.5, 2.0, step=0.1, value=1.0, label="语速 (0.5x - 2x)")
                        pitch_slider = gr.Slider(-12, 12, step=1, value=0, label="音调 (半音, -12 ~ +12)")

                    with gr.Group(visible=False) as static_panel:
                        gr.Markdown("### 静态定位参数")
                        static_azimuth = gr.Slider(0, 360, step=5, value=270, label="方位角")
                        static_distance = gr.Slider(0.1, 1.0, step=0.1, value=0.3, label="距离（米）")

                    with gr.Group(visible=has_hrir and default_mode == "动态空间环绕 (HRIR)") as dynamic_panel:
                        gr.Markdown("### 动态轨迹参数")
                        dynamic_path = gr.Dropdown(
                            choices=list(PATH_LABEL_TO_KEY.keys()),
                            value=PATH_KEY_TO_LABEL[PATH_CLOCKWISE],
                            label="移动轨迹",
                        )
                        dynamic_cycle_time = gr.Slider(
                            2.0,
                            20.0,
                            step=0.5,
                            value=8.0,
                            label="环绕周期（秒）",
                        )
                        with gr.Row():
                            dynamic_start = gr.Slider(0, 360, step=5, value=270, label="起始角度")
                            dynamic_distance = gr.Slider(0.1, 1.0, step=0.1, value=0.2, label="距离（米）")

                seed_input = gr.Number(label="Seed", value=42, precision=0)
                use_random_seed = gr.Checkbox(
                    label="随机抽奖（勾选后点生成会换 seed；生成后自动关闭，方便用同一个 seed 再跑）",
                    value=False,
                    elem_id="random-seed-check",
                )

                with gr.Row():
                    generate_button = gr.Button("🎵 生成音频", variant="primary", size="lg", elem_classes="generate-btn", scale=3)
                    stop_button = gr.Button("⏹ 停止", variant="stop", size="lg", interactive=False, scale=1)

                with gr.Accordion("⭐ 声线收藏", open=False):
                    gr.Markdown("把抽中、喜欢的声线保存下来，下次一键复用（保存 prompt + seed + 参考音 + 空间效果）")
                    with gr.Row():
                        fav_name_input = gr.Textbox(label="收藏名称", placeholder="给这条声线起个名", scale=2)
                        save_favorite_btn = gr.Button("⭐ 收藏当前声线", variant="secondary", scale=1)
                    with gr.Row():
                        favorite_dropdown = gr.Dropdown(
                            label="已收藏声线",
                            choices=list(load_user_favorites().keys()),
                            scale=2,
                        )
                        apply_favorite_btn = gr.Button("🔄 应用", variant="secondary", scale=1)
                        refresh_favorite_btn = gr.Button("↻ 刷新列表", variant="secondary", scale=1)
                        delete_favorite_btn = gr.Button("🗑 删除", variant="stop", scale=1)

            with gr.Column(scale=3):
                with gr.Group(elem_classes="output-card"):
                    output_audio = gr.Audio(
                        label="🎧 生成结果",
                        type="filepath",
                        autoplay=True,
                        elem_id="generated-audio",
                        elem_classes="audio-player",
                    )
                    output_status = gr.Textbox(label="📋 状态")

        with gr.Accordion("📦 批量生成", open=False):
            gr.Markdown(
                "复用上方的文本、引擎、参考音频设置，生成多个随机 Seed 变体。"
                "每个 Seed × 每个效果类型产生一个文件。"
            )
            with gr.Row():
                batch_count = gr.Slider(1, 10, step=1, value=3, label="变体数量（随机 Seed）", scale=2)
                batch_effect_checkboxes = gr.CheckboxGroup(
                    label="效果类型",
                    choices=mode_choices,
                    value=[mode_choices[0]],
                    scale=3,
                )
            with gr.Row():
                batch_generate_btn = gr.Button("📦 开始批量生成", variant="secondary", scale=3)
                batch_stop_btn = gr.Button("⏹ 停止批量", variant="stop", interactive=False, scale=1)
            batch_status_output = gr.Textbox(label="批量状态", lines=5, interactive=False)
            batch_file_output = gr.Files(label="生成文件", interactive=False)

        with gr.Accordion("⚙️ 设置", open=False):
            gr.Markdown("高级配置参数")
            with gr.Row():
                dry_cache_info = gr.Markdown(
                    "**Dry Audio 缓存**: 相同文本切换空间效果时复用 TTS 干声。"
                    "缓存条目越多越占内存，默认 32 条。"
                )
            with gr.Row():
                output_dir_display = gr.Textbox(
                    label="输出目录",
                    value=str(service.output_dir),
                    interactive=False,
                    scale=2,
                )
                open_output_dir_btn = gr.Button("📂 打开", variant="secondary", scale=1)

        def on_open_output_dir():
            import os, platform
            path = str(service.output_dir)
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                os.system(f"open {path}")
            else:
                os.system(f"xdg-open {path}")
            return path

        open_output_dir_btn.click(
            on_open_output_dir,
            outputs=[output_dir_display],
        )

        mode_radio.change(toggle_mode_panels, inputs=mode_radio, outputs=[static_panel, dynamic_panel])
        preset_dropdown.change(apply_preset, inputs=preset_dropdown, outputs=[prompt_input, input_text, calibration_text_input]).then(
            describe_generation_route,
            inputs=[input_text, emotion_instruct_input, tts_engine_dropdown],
            outputs=route_hint,
        )
        save_preset_btn.click(
            save_current_as_preset,
            inputs=[preset_name_input, prompt_input, input_text, calibration_text_input],
            outputs=[preset_dropdown, output_status],
        )
        save_favorite_btn.click(
            save_current_as_favorite,
            inputs=[fav_name_input, prompt_input, input_text, calibration_text_input, emotion_instruct_input, seed_input, mode_radio, speed_slider, pitch_slider, voice_clip_id],
            outputs=[favorite_dropdown, output_status],
        )
        apply_favorite_btn.click(
            apply_favorite,
            inputs=[favorite_dropdown],
            outputs=[prompt_input, input_text, calibration_text_input, emotion_instruct_input, seed_input, mode_radio, speed_slider, pitch_slider, use_random_seed, output_status, voice_clip_audio, voice_clip_id],
        ).then(
            describe_generation_route,
            inputs=[input_text, emotion_instruct_input, tts_engine_dropdown],
            outputs=route_hint,
        )
        delete_favorite_btn.click(
            delete_favorite,
            inputs=[favorite_dropdown],
            outputs=[favorite_dropdown],
        )
        refresh_favorite_btn.click(
            reload_favorite_choices,
            outputs=[favorite_dropdown],
        )
        demo.load(
            reload_favorite_choices,
            outputs=[favorite_dropdown],
            show_progress="hidden",
            queue=False,
        )
        tts_engine_dropdown.change(
            toggle_tts_engine,
            inputs=tts_engine_dropdown,
            outputs=[tts_model_dropdown, qwen_prompt_group, clone_reference_group],
        ).then(
            describe_generation_route,
            inputs=[input_text, emotion_instruct_input, tts_engine_dropdown],
            outputs=route_hint,
        )
        input_text.change(
            describe_generation_route,
            inputs=[input_text, emotion_instruct_input, tts_engine_dropdown],
            outputs=route_hint,
        )
        emotion_instruct_input.change(
            describe_generation_route,
            inputs=[input_text, emotion_instruct_input, tts_engine_dropdown],
            outputs=route_hint,
        )
        auto_transcribe_btn.click(
            auto_transcribe,
            inputs=[reference_audio_input],
            outputs=[reference_text_input],
        )
        reference_audio_input.change(
            on_reference_audio_change,
            inputs=[reference_audio_input],
            outputs=[],
        )
        generate_event = generate_button.click(
            on_generate_start,
            outputs=[generate_button, stop_button, output_status],
        ).then(
            generate_audio,
            inputs=[
                preset_dropdown,
                tts_engine_dropdown,
                tts_model_dropdown,
                prompt_input,
                reference_audio_input,
                reference_text_input,
                input_text,
                seed_input,
                use_random_seed,
                mode_radio,
                static_azimuth,
                static_distance,
                dynamic_path,
                dynamic_cycle_time,
                dynamic_start,
                dynamic_distance,
                speed_slider,
                pitch_slider,
                calibration_text_input,
                emotion_instruct_input,
            ],
            outputs=[output_audio, seed_input, use_random_seed, output_status, generate_button, stop_button, voice_clip_audio, voice_clip_id],
            concurrency_id="tts-generate",
            concurrency_limit=1,
        )
        preview_clip_btn.click(
            preview_clip_from_ui,
            inputs=[prompt_input, seed_input, tts_engine_dropdown, tts_model_dropdown, calibration_text_input],
            outputs=[voice_clip_audio, voice_clip_id, seed_input, output_status],
            concurrency_id="tts-generate",
            concurrency_limit=1,
        )
        reroll_clip_btn.click(
            reroll_clip_from_ui,
            inputs=[prompt_input, seed_input, tts_engine_dropdown, tts_model_dropdown, calibration_text_input],
            outputs=[voice_clip_audio, voice_clip_id, seed_input, output_status],
            concurrency_id="tts-generate",
            concurrency_limit=1,
        )
        stop_button.click(
            on_stop_generate,
            outputs=[generate_button, stop_button, output_status],
            cancels=[generate_event],
            queue=False,
        )

        def on_batch_start() -> tuple[gr.update, gr.update, str]:
            return (*_busy_generate_buttons(), "正在批量生成…")

        batch_event = batch_generate_btn.click(
            on_batch_start,
            outputs=[batch_generate_btn, batch_stop_btn, batch_status_output],
        ).then(
            batch_generate_audio,
            inputs=[
                tts_engine_dropdown,
                tts_model_dropdown,
                prompt_input,
                reference_audio_input,
                reference_text_input,
                input_text,
                preset_dropdown,
                static_azimuth,
                static_distance,
                dynamic_path,
                dynamic_cycle_time,
                dynamic_start,
                dynamic_distance,
                speed_slider,
                pitch_slider,
                calibration_text_input,
                emotion_instruct_input,
                batch_count,
                batch_effect_checkboxes,
            ],
            outputs=[batch_status_output, batch_file_output, batch_generate_btn, batch_stop_btn],
            concurrency_id="tts-generate",
            concurrency_limit=1,
        )
        batch_stop_btn.click(
            on_stop_generate,
            outputs=[batch_generate_btn, batch_stop_btn, batch_status_output],
            cancels=[batch_event],
            queue=False,
        )

        with gr.Accordion("📂 历史记录", open=False):
            gr.Markdown("查看和删除已生成的音频文件")
            history_search = gr.Textbox(label="搜索文件名", placeholder="输入文件名过滤...", scale=3)
            history_refresh_btn = gr.Button("🔄 刷新", variant="secondary", scale=1)
            selected_file_path = gr.State("")
            history_table = gr.DataFrame(
                label="历史音频（点击行选择）",
                headers=list(HISTORY_TABLE_HEADERS),
                datatype=["str", "str", "number"],
                interactive=False,
                wrap=True,
                max_height=300,
            )
            with gr.Row():
                history_play_btn = gr.Button("▶ 播放", variant="primary", size="sm", scale=1)
                history_delete_btn = gr.Button("🗑️ 删除", variant="stop", size="sm", scale=1)
                history_msg = gr.Textbox(label="", interactive=False, scale=3)

            def filtered_history(search_term: str) -> list[dict]:
                files = service.list_generated_audios()
                if search_term:
                    files = [f for f in files if search_term.lower() in f["name"].lower()]
                return files

            def on_select_history(evt: gr.SelectData, search_term: str):
                files = filtered_history(search_term)
                index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                if 0 <= index < len(files):
                    return files[index]["path"]
                return ""

            def on_refresh_history(search_term: str):
                return history_table_rows(filtered_history(search_term))

            def on_play_file(file_path: str) -> tuple[str | None, str]:
                """播放音频"""
                if not file_path:
                    return None, "请先选择要播放的文件"
                src = Path(file_path).expanduser().resolve()
                try:
                    src.relative_to(service.output_dir.expanduser().resolve())
                except ValueError:
                    return None, "无效的文件路径"
                if not src.exists() or not src.is_file():
                    return None, f"文件不存在: {src.name}"
                return str(src), f"正在播放: {src.name}"

            history_table.select(
                on_select_history,
                inputs=[history_search],
                outputs=[selected_file_path],
            )
            history_search.change(
                on_refresh_history,
                inputs=[history_search],
                outputs=[history_table],
            )
            history_refresh_btn.click(
                on_refresh_history,
                inputs=[history_search],
                outputs=[history_table],
            )
            def on_delete_with_confirm(file_path: str) -> tuple[str, list[list]]:
                rows = history_table_rows(service.list_generated_audios())
                if not file_path:
                    return "请先选择要删除的文件", rows
                return f"CONFIRM:{file_path}", rows

            def on_confirm_delete(file_path: str) -> tuple[str, list[list], str, gr.update]:
                rows = history_table_rows(service.list_generated_audios())
                if not file_path:
                    return "请先选择要删除的文件", rows, "", gr.update(visible=False)
                if service.delete_audio(file_path):
                    return (
                        f"已删除: {Path(file_path).name}",
                        history_table_rows(service.list_generated_audios()),
                        "",
                        gr.update(visible=False),
                    )
                return "删除失败", rows, file_path, gr.update(visible=False)

            history_confirm_msg = gr.Textbox(label="确认删除", visible=False, interactive=False)
            with gr.Row(visible=False) as confirm_row:
                confirm_yes_btn = gr.Button("确认删除", variant="stop", size="sm")
                confirm_no_btn = gr.Button("取消", variant="secondary", size="sm")

            def on_show_confirm(msg: str) -> tuple[gr.update, str, list[list]]:
                rows = history_table_rows(service.list_generated_audios())
                if not msg.startswith("CONFIRM:"):
                    return gr.update(visible=False), "", rows
                file_path = msg[len("CONFIRM:"):]
                return gr.update(visible=True), f"确定要删除 {Path(file_path).name} 吗？", rows

            history_delete_btn.click(
                on_delete_with_confirm,
                inputs=[selected_file_path],
                outputs=[history_confirm_msg, history_table],
            )
            history_confirm_msg.change(
                on_show_confirm,
                inputs=[history_confirm_msg],
                outputs=[confirm_row, history_msg, history_table],
            )
            confirm_yes_btn.click(
                on_confirm_delete,
                inputs=[selected_file_path],
                outputs=[history_msg, history_table, selected_file_path, confirm_row],
            )
            confirm_no_btn.click(
                lambda: ("", history_table_rows(service.list_generated_audios()), gr.update(visible=False)),
                inputs=[],
                outputs=[history_msg, history_table, confirm_row],
            )
            history_play_btn.click(
                on_play_file,
                inputs=[selected_file_path],
                outputs=[output_audio, output_status],
            )

    return demo
