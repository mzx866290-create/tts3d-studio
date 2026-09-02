from __future__ import annotations

import asyncio
import random
from pathlib import Path

from nicegui import ui, events, app

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
    load_user_config,
    load_user_presets,
    save_user_presets,
    save_user_config,
)
from tts3d_app.presets import PRESETS
from tts3d_app.service import BatchRequest, GenerationRequest, TTSStudioService
from tts3d_app.tts_engines import DEFAULT_TTS_ENGINE, ENGINE_QWEN3
from tts3d_app.prompt_generator import generate_voice_prompt


MODE_LABEL_TO_KEY = {
    "单声道": MODE_MONO,
    "双声道": MODE_STEREO,
    "虚拟脑后": MODE_BEHIND_HEAD,
    "静态空间定位 (HRIR)": MODE_STATIC_HRIR,
    "动态空间环绕 (HRIR)": MODE_DYNAMIC_HRIR,
}
MODE_KEY_TO_LABEL = {v: k for k, v in MODE_LABEL_TO_KEY.items()}

PATH_LABEL_TO_KEY = {
    "顺时针绕头": PATH_CLOCKWISE,
    "逆时针绕头": PATH_COUNTERCLOCKWISE,
    "左右摆动": PATH_SIDE_TO_SIDE,
}
PATH_KEY_TO_LABEL = {v: k for k, v in PATH_LABEL_TO_KEY.items()}


def tuple_list_to_dict(choices: list[tuple[str, str]]) -> dict[str, str]:
    """将 [(标签, 值), ...] 转为 {标签: 值}"""
    return {label: value for label, value in choices}


def create_ui(service: TTSStudioService | None = None):
    service = service or TTSStudioService()
    has_hrir = service.hrir_dataset is not None

    mode_choices = ["单声道", "双声道", "虚拟脑后"]
    if has_hrir:
        mode_choices += ["静态空间定位 (HRIR)", "动态空间环绕 (HRIR)"]

    default_mode = "动态空间环绕 (HRIR)" if has_hrir else "虚拟脑后"
    default_preset = service.default_preset_key()
    default_engine = DEFAULT_TTS_ENGINE
    
    # 获取引擎选项（转为 dict）
    engine_choices_raw = service.get_tts_engine_choices()
    engine_options = tuple_list_to_dict(engine_choices_raw)
    
    # 获取模型选项（转为 dict）
    model_choices_raw, default_model_key, _, _ = describe_tts_engine_ui_state(service, default_engine)
    model_options = tuple_list_to_dict(model_choices_raw)
    
    saved_config = load_user_config()
    default_ref_audio = saved_config.get("last_reference_audio", "") if saved_config.get("last_reference_audio") and Path(saved_config["last_reference_audio"]).exists() else ""

    # ===== 全局样式 =====
    ui.dark_mode().enable()
    ui.add_head_html("""
    <style>
    body {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d1b2a 50%, #1a0a2e 100%) !important;
        min-height: 100vh;
    }
    .glass-panel {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    }
    .title-gradient {
        background: linear-gradient(90deg, #8b5cf6, #06b6d4, #d946ef);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 40px rgba(139, 92, 246, 0.5);
    }
    .neon-btn {
        background: linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%) !important;
        color: white !important;
        font-weight: bold !important;
        letter-spacing: 1px;
        border: none !important;
        border-radius: 50px !important;
        padding: 16px 32px !important;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4), 0 0 40px rgba(6, 182, 212, 0.2);
        transition: all 0.3s ease !important;
    }
    .neon-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 30px rgba(139, 92, 246, 0.6), 0 0 60px rgba(6, 182, 212, 0.4) !important;
    }
    .warning-box {
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.3);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 12px 0;
        color: #fbbf24;
        font-size: 13px;
    }
    .section-title {
        color: #a78bfa;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .output-card {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(6, 182, 212, 0.15) 50%, rgba(217, 70, 239, 0.15) 100%) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        padding: 24px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    }
    </style>
    """)

    # ===== 页面结构 =====
    with ui.column().classes("w-full min-h-screen p-6 items-center"):
        ui.html("<h1 class='text-3xl font-bold title-gradient text-center mt-4'>TTS 3D Studio</h1>")
        ui.label("输入文本，选择 TTS 引擎与模型，叠加双声道、脑后或 HRIR 空间音频效果").classes("text-center text-gray-400 text-sm mb-6")

        if not has_hrir:
            ui.html("<div class='warning-box'>⚠️ 提示：未检测到 HRIR 数据文件，当前只提供基础模式和虚拟脑后模式。</div>")

        with ui.row().classes("w-full max-w-6xl gap-6 items-start"):
            # ===== 左侧控制面板 =====
            with ui.column().classes("w-3/5 gap-4"):
                
                # --- 文本与预设 ---
                with ui.card().classes("glass-panel w-full"):
                    ui.label("📝 文本内容").classes("section-title")
                    text_input = ui.textarea(
                        value=PRESETS[default_preset]["text"],
                    ).props("outlined dark").classes("w-full")

                    with ui.row().classes("w-full gap-3 mt-3"):
                        all_presets = list(PRESETS.keys()) + list(load_user_presets().keys())
                        preset_select = ui.select(
                            options=all_presets,
                            label="预设",
                            value=default_preset,
                        ).classes("flex-1")
                        
                        with ui.column().classes("flex-1 gap-2"):
                            new_preset_input = ui.input(
                                label="新预设名",
                                placeholder="输入名称后点保存"
                            ).classes("w-full")
                            save_preset_btn = ui.button("💾 保存预设").classes("bg-purple-600 text-white px-3 py-2 rounded-lg w-full")

                    with ui.row().classes("w-full gap-3 mt-2"):
                        engine_select = ui.select(
                            options=engine_options,
                            label="TTS 引擎",
                            value=default_engine,
                        ).classes("flex-1")
                        model_select = ui.select(
                            options=model_options,
                            label="模型",
                            value=default_model_key,
                        ).classes("flex-[2]")

                # --- Qwen 提示词 ---
                qwen_card = ui.card().classes("glass-panel w-full")
                with qwen_card:
                    ui.label("🎤 声音描述（中文）").classes("section-title")
                    with ui.row().classes("w-full gap-2"):
                        voice_desc = ui.input(
                            placeholder="例如：温柔的客服女声、磁性的男声"
                        ).props("outlined dark").classes("flex-[3]")
                        gen_prompt_btn = ui.button("✨ AI 生成提示词").classes("bg-purple-600 text-white px-3 py-2 rounded-lg flex-1")

                    ui.label("📝 TTS 提示词").classes("section-title mt-2")
                    prompt_input = ui.textarea(
                        value=PRESETS[default_preset]["prompt"],
                        placeholder="填写英文提示词，描述声音特征"
                    ).props("outlined dark").classes("w-full")

                # --- Qwen3 Base clone reference audio ---
                clone_card = ui.card().classes("glass-panel w-full hidden")
                with clone_card:
                    ui.label("🎤 Qwen3 Base 克隆参考音频").classes("section-title")
                    ref_audio_input = ui.input(
                        label="参考音频路径",
                        value=default_ref_audio,
                        placeholder="上传或粘贴音频文件路径"
                    ).props("outlined dark").classes("w-full")
                    
                    with ui.row().classes("w-full gap-2 mt-2"):
                        ref_text_input = ui.textarea(
                            label="参考文本",
                            placeholder="填写参考音频对应的文本内容，或点击「自动识别」"
                        ).props("outlined dark").classes("flex-[4]")
                        auto_trans_btn = ui.button("🎙️ 自动识别").classes("bg-cyan-600 text-white px-3 py-2 rounded-lg flex-1")

                # --- 空间音频引擎 ---
                with ui.card().classes("glass-panel w-full"):
                    ui.label("🎧 空间音频引擎").classes("section-title")
                    mode_radio = ui.radio(
                        options=mode_choices,
                        value=default_mode,
                    ).classes("text-gray-200")

                    with ui.expansion("语速 / 音调  （在空间效果前处理）", value=False).classes("w-full mt-2"):
                        with ui.row().classes("w-full gap-6"):
                            with ui.column().classes("flex-1"):
                                ui.label("语速").classes("text-gray-400 text-xs mb-1")
                                speed_slider = ui.slider(min=0.5, max=2.0, step=0.1, value=1.0).classes("w-full")
                                speed_label = ui.label("1.0x").classes("text-cyan-300 text-sm text-center")
                            with ui.column().classes("flex-1"):
                                ui.label("音调").classes("text-gray-400 text-xs mb-1")
                                pitch_slider = ui.slider(min=-12, max=12, step=1, value=0).classes("w-full")
                                pitch_label = ui.label("0 半音").classes("text-cyan-300 text-sm text-center")

                    # 静态定位
                    static_card = ui.card().classes("w-full mt-2 hidden")
                    with static_card:
                        ui.label("📍 静态定位参数").classes("section-title")
                        with ui.row().classes("w-full gap-6"):
                            with ui.column().classes("flex-1"):
                                ui.label("方位角").classes("text-gray-400 text-xs mb-1")
                                static_azimuth = ui.slider(min=0, max=360, step=5, value=270).classes("w-full")
                                static_azimuth_label = ui.label("270°").classes("text-cyan-300 text-sm text-center")
                            with ui.column().classes("flex-1"):
                                ui.label("距离（米）").classes("text-gray-400 text-xs mb-1")
                                static_distance = ui.slider(min=0.1, max=1.0, step=0.1, value=0.3).classes("w-full")
                                static_distance_label = ui.label("0.3m").classes("text-cyan-300 text-sm text-center")

                    # 动态轨迹
                    dynamic_card = ui.card().classes("w-full mt-2")
                    with dynamic_card:
                        ui.label("🔄 动态轨迹参数").classes("section-title")
                        dynamic_path = ui.select(
                            options=list(PATH_LABEL_TO_KEY.keys()),
                            label="移动轨迹",
                            value=PATH_KEY_TO_LABEL[PATH_CLOCKWISE],
                        ).classes("w-full mb-2")
                        
                        with ui.row().classes("w-full gap-6"):
                            with ui.column().classes("flex-1"):
                                ui.label("环绕周期（秒）").classes("text-gray-400 text-xs mb-1")
                                dynamic_cycle = ui.slider(min=2.0, max=20.0, step=0.5, value=8.0).classes("w-full")
                                dynamic_cycle_label = ui.label("8.0s").classes("text-cyan-300 text-sm text-center")
                            with ui.column().classes("flex-1"):
                                ui.label("起始角度").classes("text-gray-400 text-xs mb-1")
                                dynamic_start = ui.slider(min=0, max=360, step=5, value=270).classes("w-full")
                                dynamic_start_label = ui.label("270°").classes("text-cyan-300 text-sm text-center")
                        
                        with ui.row().classes("w-full gap-6 mt-2"):
                            with ui.column().classes("flex-1"):
                                ui.label("距离（米）").classes("text-gray-400 text-xs mb-1")
                                dynamic_distance = ui.slider(min=0.1, max=1.0, step=0.1, value=0.2).classes("w-full")
                                dynamic_distance_label = ui.label("0.2m").classes("text-cyan-300 text-sm text-center")

                    with ui.row().classes("w-full gap-4 mt-3 items-center"):
                        seed_input = ui.number(label="Seed", value=42, precision=0).classes("flex-1")
                        use_random_seed = ui.checkbox("随机 Seed", value=True).classes("text-gray-300")

                generate_btn = ui.button("🎵 生成音频").classes("neon-btn w-full")

                # ===== 批量生成 =====
                with ui.expansion("📦 批量生成", value=False).classes("glass-panel w-full"):
                    ui.label("复用上方的文本、引擎、参考音频设置，生成多个随机 Seed 变体。每个 Seed × 每个效果类型产生一个文件。").classes("text-gray-400 text-sm mb-3")
                    
                    batch_count = ui.slider(min=1, max=10, step=1, value=3).classes("w-full")
                    ui.label("变体数量").classes("text-gray-400 text-xs -mt-2 mb-2")
                    
                    batch_effects = ui.select(
                        options=mode_choices,
                        label="效果类型（可多选）",
                        multiple=True,
                        value=[mode_choices[0]] if mode_choices else [],
                    ).classes("w-full mt-2")
                    
                    batch_btn = ui.button("📦 开始批量生成").classes("bg-cyan-600 text-white px-4 py-2 rounded-lg mt-2 w-full")
                    batch_status = ui.textarea(label="批量状态").props("outlined dark readonly").classes("w-full mt-2")

                # ===== 设置 =====
                with ui.expansion("⚙️ 设置", value=False).classes("glass-panel w-full"):
                    ui.label("高级配置参数").classes("section-title")
                    with ui.row().classes("w-full gap-3 items-center"):
                        output_dir_display = ui.input(
                            label="输出目录",
                            value=str(service.output_dir),
                        ).props("outlined dark readonly").classes("flex-[3]")
                        open_dir_btn = ui.button("📂 打开").classes("bg-gray-600 text-white px-3 py-2 rounded-lg flex-1")

            # ===== 右侧结果面板 =====
            with ui.column().classes("w-2/5 gap-4"):
                with ui.card().classes("output-card w-full"):
                    ui.label("🎧 生成结果").classes("text-lg font-semibold text-cyan-300 mb-3")
                    output_audio = ui.audio().classes("w-full mb-3").props("controls")
                    output_status = ui.textarea(label="状态").props("outlined dark readonly").classes("w-full")
                    output_seed = ui.number(label="Seed").props("outlined dark readonly").classes("w-full mt-2")

                # ===== 历史记录 =====
                with ui.expansion("📂 历史记录", value=False).classes("glass-panel w-full"):
                    ui.label("查看和删除已生成的音频文件").classes("text-gray-400 text-sm mb-2")
                    
                    with ui.row().classes("w-full gap-2 mb-2"):
                        history_search = ui.input(placeholder="搜索文件名").props("outlined dark").classes("flex-[3]")
                        refresh_btn = ui.button("🔄 刷新").classes("bg-gray-600 text-white px-3 py-2 rounded-lg flex-1")

                    history_table = ui.table(
                        columns=[
                            {"name": "name", "label": "文件名", "field": "name", "align": "left"},
                            {"name": "modified", "label": "修改时间", "field": "modified", "align": "left"},
                            {"name": "size", "label": "大小", "field": "size", "align": "right"},
                        ],
                        rows=[],
                        row_key="path",
                    ).classes("w-full")

                    with ui.row().classes("w-full gap-3 mt-2"):
                        play_selected_btn = ui.button("▶ 播放").classes("bg-purple-600 text-white px-3 py-2 rounded-lg flex-1")
                        delete_selected_btn = ui.button("🗑️ 删除").classes("bg-red-600 text-white px-3 py-2 rounded-lg flex-1")

                    history_msg = ui.label("").classes("text-sm text-gray-400 mt-2")

    # ===== 状态变量 =====
    selected_file_path = ui.state("")

    # ===== 事件处理 =====
    def update_slider_labels():
        speed_label.set_text(f"{speed_slider.value:.1f}x")
        pitch_label.set_text(f"{int(pitch_slider.value)} 半音")
        static_azimuth_label.set_text(f"{int(static_azimuth.value)}°")
        static_distance_label.set_text(f"{static_distance.value:.1f}m")
        dynamic_cycle_label.set_text(f"{dynamic_cycle.value:.1f}s")
        dynamic_start_label.set_text(f"{int(dynamic_start.value)}°")
        dynamic_distance_label.set_text(f"{dynamic_distance.value:.1f}m")

    # 绑定滑块
    speed_slider.on_value_change(update_slider_labels)
    pitch_slider.on_value_change(update_slider_labels)
    static_azimuth.on_value_change(update_slider_labels)
    static_distance.on_value_change(update_slider_labels)
    dynamic_cycle.on_value_change(update_slider_labels)
    dynamic_start.on_value_change(update_slider_labels)
    dynamic_distance.on_value_change(update_slider_labels)

    def toggle_mode_panels():
        mode_key = MODE_LABEL_TO_KEY[mode_radio.value]
        static_card.set_visibility(mode_key == MODE_STATIC_HRIR)
        dynamic_card.set_visibility(mode_key == MODE_DYNAMIC_HRIR)

    mode_radio.on_value_change(toggle_mode_panels)

    def apply_preset():
        key = preset_select.value
        if key in PRESETS:
            preset = PRESETS[key]
        else:
            user_presets = load_user_presets()
            if key in user_presets:
                preset = user_presets[key]
            else:
                return
        prompt_input.set_value(preset["prompt"])
        text_input.set_value(preset["text"])

    preset_select.on_value_change(apply_preset)

    def save_preset():
        name = new_preset_input.value.strip()
        if not name:
            output_status.set_value("请输入预设名称")
            return
        if name in PRESETS:
            output_status.set_value(f"预设名 '{name}' 与内置预设冲突")
            return
        user_presets = load_user_presets()
        user_presets[name] = {"prompt": prompt_input.value, "text": text_input.value}
        save_user_presets(user_presets)
        new_choices = list(PRESETS.keys()) + list(user_presets.keys())
        preset_select.set_options(new_choices)
        preset_select.set_value(name)
        output_status.set_value(f"已保存为: {name}")

    save_preset_btn.on_click(save_preset)

    def toggle_tts_engine():
        engine_key = engine_select.value
        model_choices_raw, default_model_key, show_qwen, show_clone = describe_tts_engine_ui_state(service, engine_key)
        model_options_dict = tuple_list_to_dict(model_choices_raw)
        model_select.set_options(model_options_dict)
        model_select.set_value(default_model_key)
        qwen_card.set_visibility(show_qwen)
        clone_card.set_visibility(show_clone)

    engine_select.on_value_change(toggle_tts_engine)

    def on_generate_prompt():
        desc = voice_desc.value.strip()
        if not desc:
            return
        try:
            prompt = generate_voice_prompt(desc)
            prompt_input.set_value(prompt)
        except Exception as exc:
            output_status.set_value(f"生成失败: {exc}")

    gen_prompt_btn.on_click(on_generate_prompt)

    def on_reference_change():
        audio_path = ref_audio_input.value
        if audio_path and Path(audio_path).exists():
            config = load_user_config()
            config["last_reference_audio"] = audio_path
            save_user_config(config)

    ref_audio_input.on_value_change(on_reference_change)

    def auto_transcribe():
        audio_path = ref_audio_input.value
        if not audio_path:
            output_status.set_value("请先上传参考音频")
            return
        if not ASR_AVAILABLE:
            output_status.set_value("未安装 ASR 后端，请运行: pip install faster-whisper")
            return
        try:
            from tts3d_app.asr import transcribe_audio
            text = transcribe_audio(audio_path)
            ref_text_input.set_value(text)
        except Exception as exc:
            output_status.set_value(f"识别失败: {exc}")

    auto_trans_btn.on_click(auto_transcribe)

    async def generate_audio():
        request = GenerationRequest(
            preset_key=preset_select.value,
            voice_description=prompt_input.value,
            text=text_input.value,
            seed=seed_input.value,
            use_random_seed=use_random_seed.value,
            render_mode=MODE_LABEL_TO_KEY[mode_radio.value],
            static_azimuth_deg=static_azimuth.value,
            static_distance_m=static_distance.value,
            dynamic_path=PATH_LABEL_TO_KEY[dynamic_path.value],
            dynamic_cycle_time_s=dynamic_cycle.value,
            dynamic_start_azimuth_deg=dynamic_start.value,
            dynamic_distance_m=dynamic_distance.value,
            tts_engine=engine_select.value,
            tts_model_key=model_select.value,
            reference_audio_path=ref_audio_input.value or None,
            reference_text=ref_text_input.value,
            speed_factor=speed_slider.value,
            pitch_semitones=pitch_slider.value,
        )

        try:
            generate_btn.set_text("⏳ 生成中...")
            generate_btn.props("disable")
            result = await asyncio.to_thread(service.generate, request)
            output_audio.set_source(result.file_path)
            output_seed.set_value(result.seed)
            output_status.set_value(result.status)
        except Exception as exc:
            output_status.set_value(f"生成失败: {exc}")
        finally:
            generate_btn.set_text("🎵 生成音频")
            generate_btn.props("disable=false")

    generate_btn.on_click(generate_audio)

    async def batch_generate():
        effects = batch_effects.value or []
        if not effects:
            batch_status.set_value("请至少选择一个效果类型")
            return

        effect_types = [MODE_LABEL_TO_KEY[label] for label in effects]
        seeds = [random.randint(0, 2**32 - 1) for _ in range(int(batch_count.value))]
        
        request = BatchRequest(
            text=text_input.value,
            voice_description=prompt_input.value,
            tts_engine=engine_select.value,
            tts_model_key=model_select.value,
            reference_audio_path=ref_audio_input.value or None,
            reference_text=ref_text_input.value,
            preset_key=preset_select.value,
            seeds=seeds,
            effect_types=effect_types,
            static_azimuth_deg=static_azimuth.value,
            static_distance_m=static_distance.value,
            dynamic_path=PATH_LABEL_TO_KEY[dynamic_path.value],
            dynamic_cycle_time_s=dynamic_cycle.value,
            dynamic_start_azimuth_deg=dynamic_start.value,
            dynamic_distance_m=dynamic_distance.value,
            speed_factor=speed_slider.value,
            pitch_semitones=pitch_slider.value,
        )

        try:
            batch_btn.set_text("⏳ 批量生成中...")
            batch_btn.props("disable")
            result = await asyncio.to_thread(service.generate_batch, request)
            lines = [f"完成 {result.succeeded}/{result.total}，失败 {result.failed}"]
            for item in result.items:
                lines.append(f"  [{item.effect_type}] seed={item.seed}: {item.status}")
            batch_status.set_value("\n".join(lines))
        except Exception as exc:
            batch_status.set_value(f"批量生成失败: {exc}")
        finally:
            batch_btn.set_text("📦 开始批量生成")
            batch_btn.props("disable=false")

    batch_btn.on_click(batch_generate)

    def open_output_dir():
        import os, platform
        path = str(service.output_dir)
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            os.system(f"open {path}")
        else:
            os.system(f"xdg-open {path}")

    open_dir_btn.on_click(open_output_dir)

    def refresh_history():
        files = service.list_generated_audios()
        search_term = history_search.value.lower() if history_search.value else ""
        if search_term:
            files = [f for f in files if search_term in f["name"].lower()]
        rows = []
        for f in files:
            rows.append({
                "name": f["name"],
                "modified": f["modified"],
                "size": f"{f['size']/1024:.1f} KB",
                "path": f["path"],
            })
        history_table.rows = rows

    refresh_btn.on_click(refresh_history)
    history_search.on_value_change(refresh_history)

    def on_select_history(e: events.GenericEventArguments):
        row = e.args
        if row and "path" in row:
            selected_file_path.value = row["path"]
            history_msg.set_text(f"已选择: {row['name']}")

    history_table.on_row_click(on_select_history)

    def play_selected():
        path = selected_file_path.value
        if not path:
            history_msg.set_text("请先选择文件")
            return
        output_audio.set_source(path)
        output_status.set_value(f"正在播放: {Path(path).name}")

    play_selected_btn.on_click(play_selected)

    def delete_selected():
        path = selected_file_path.value
        if not path:
            history_msg.set_text("请先选择要删除的文件")
            return
        
        with ui.dialog() as dialog, ui.card():
            ui.label(f"确定要删除 {Path(path).name} 吗？").classes("text-lg mb-4")
            with ui.row().classes("w-full gap-2 justify-end"):
                ui.button("取消", on_click=dialog.close).classes("bg-gray-600 text-white px-4 py-2 rounded-lg")
                
                def do_delete():
                    if service.delete_audio(path):
                        history_msg.set_text(f"已删除: {Path(path).name}")
                        refresh_history()
                    else:
                        history_msg.set_text("删除失败")
                    dialog.close()
                
                ui.button("确认删除", on_click=do_delete).classes("bg-red-600 text-white px-4 py-2 rounded-lg")
        
        dialog.open()

    delete_selected_btn.on_click(delete_selected)

    # 初始刷新历史
    refresh_history()


def describe_tts_engine_ui_state(
    service: TTSStudioService,
    engine_key: str,
) -> tuple[list[tuple[str, str]], str, bool, bool]:
    model_choices = service.get_tts_model_choices(engine_key)
    default_model_key = service.get_default_tts_model_key(engine_key)
    show_qwen_prompt = engine_key == ENGINE_QWEN3
    show_clone_inputs = not show_qwen_prompt
    return model_choices, default_model_key, show_qwen_prompt, show_clone_inputs
