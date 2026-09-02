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
from tts3d_app.config import ASR_AVAILABLE, load_user_config, load_user_presets, save_user_presets, save_user_config
from tts3d_app.presets import PRESETS
from tts3d_app.service import BatchRequest, GenerationRequest, TTSStudioService
from tts3d_app.tts_engines import DEFAULT_TTS_ENGINE, ENGINE_QWEN3
from tts3d_app.prompt_generator import generate_voice_prompt, get_generator


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


def describe_tts_engine_ui_state(
    service: TTSStudioService,
    engine_key: str,
) -> tuple[list[tuple[str, str]], str, bool, bool]:
    model_choices = service.get_tts_model_choices(engine_key)
    default_model_key = service.get_default_tts_model_key(engine_key)
    show_qwen_prompt = engine_key == ENGINE_QWEN3
    show_clone_inputs = not show_qwen_prompt
    return model_choices, default_model_key, show_qwen_prompt, show_clone_inputs


def load_audio_history(service: TTSStudioService) -> list[dict]:
    """加载历史音频列表"""
    return service.list_generated_audios()

def delete_audio_file(service: TTSStudioService, file_path: str) -> tuple[str, list[dict]]:
    """删除音频文件"""
    success = service.delete_audio(file_path)
    if success:
        return "删除成功", service.list_generated_audios()
    return "删除失败", service.list_generated_audios()

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

    def apply_preset(preset_key: str) -> tuple[str, str]:
        if preset_key in PRESETS:
            preset = PRESETS[preset_key]
            return preset["prompt"], preset["text"]
        user_presets = load_user_presets()
        if preset_key in user_presets:
            preset = user_presets[preset_key]
            return preset["prompt"], preset["text"]
        return "", ""

    def save_current_as_preset(
        preset_name: str,
        prompt: str,
        text: str,
    ) -> tuple[gr.update, str]:
        """保存当前 prompt 和 text 为新预设"""
        if not preset_name or not preset_name.strip():
            return gr.update(), "请输入预设名称"
        name = preset_name.strip()
        if name in PRESETS:
            return gr.update(), f"预设名 '{name}' 与内置预设冲突，请换名"
        user_presets = load_user_presets()
        user_presets[name] = {"prompt": prompt, "text": text}
        save_user_presets(user_presets)
        new_choices = list(PRESETS.keys()) + list(user_presets.keys())
        return gr.update(choices=new_choices, value=name), f"已保存为: {name}"

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

    def generate_audio(
        preset_key: str,
        tts_engine: str,
        tts_model_key: str,
        voice_description: str,
        reference_audio_path: str | None,
        reference_text: str,
        text: str,
        seed: int | None,
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
    ) -> tuple[str, int, str, str]:
        request = GenerationRequest(
            preset_key=preset_key,
            voice_description=voice_description,
            text=text,
            seed=seed,
            use_random_seed=use_random_seed,
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
        )

        try:
            result = service.generate(request)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

        return result.file_path, result.seed, result.status, result.text

    def on_generate_prompt(description: str) -> str:
        """根据中文描述生成英文提示词"""
        if not description or not description.strip():
            return ""
        try:
            prompt = generate_voice_prompt(description.strip())
            return prompt
        except Exception as exc:
            raise gr.Error(f"生成失败: {exc}")

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
        batch_count: int,
        batch_effect_labels: list[str],
    ) -> tuple[str, list[str]]:
        if not batch_effect_labels:
            raise gr.Error("请至少选择一个效果类型")
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
        )
        try:
            result = service.generate_batch(batch_request)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

        status_lines = [f"完成 {result.succeeded}/{result.total}，失败 {result.failed}"]
        for item in result.items:
            status_lines.append(f"  [{item.effect_type}] seed={item.seed}: {item.status}")
        files = [item.file_path for item in result.items if item.file_path]
        return "\n".join(status_lines), files

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
                )

                with gr.Row():
                    all_presets = list(PRESETS.keys()) + list(load_user_presets().keys())
                    preset_dropdown = gr.Dropdown(
                        label="预设",
                        choices=all_presets,
                        value=default_preset_key,
                        scale=1,
                    )
                    save_preset_btn = gr.Button("💾 保存", variant="secondary", size="sm", scale=1)
                    preset_name_input = gr.Textbox(label="新预设名", placeholder="输入名称后点保存", scale=1, visible=False)
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
                    with gr.Row():
                        voice_desc_input = gr.Textbox(
                            label="🎤 声音描述（中文）",
                            placeholder="例如：温柔的客服女声、磁性的男声",
                            scale=3,
                        )
                        generate_prompt_btn = gr.Button(
                            "✨ AI 生成提示词",
                            variant="secondary",
                            scale=1,
                        )
                    prompt_input = gr.Textbox(
                        label="📝 TTS 提示词",
                        value=PRESETS[default_preset_key]["prompt"],
                        placeholder="填写英文提示词，描述声音特征",
                    )

                    # 绑定 AI 生成按钮
                    generate_prompt_btn.click(
                        on_generate_prompt,
                        inputs=[voice_desc_input],
                        outputs=[prompt_input],
                    )
                with gr.Group(visible=False) as clone_reference_group:
                    reference_audio_input = gr.Audio(
                        label="Qwen3 Base 克隆参考音频",
                        type="filepath",
                        sources=["upload"],
                        value=default_ref_audio,
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

                    with gr.Row():
                        seed_input = gr.Number(label="Seed", value=42, precision=0)
                        use_random_seed = gr.Checkbox(label="随机 Seed", value=True)

                generate_button = gr.Button("🎵 生成音频", variant="primary", size="lg", elem_classes="generate-btn")

            with gr.Column(scale=3):
                with gr.Group(elem_classes="output-card"):
                    output_audio = gr.Audio(label="🎧 生成结果", type="filepath", autoplay=True)
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
            batch_generate_btn = gr.Button("📦 开始批量生成", variant="secondary")
            batch_status_output = gr.Textbox(label="批量状态", lines=5, interactive=False)
            batch_file_output = gr.Files(label="生成文件", interactive=False)

        with gr.Accordion("⚙️ 设置", open=False):
            gr.Markdown("高级配置参数")
            with gr.Row():
                dry_cache_info = gr.Markdown("**Dry Audio 缓存**: 用于缓存 TTS 原始输出，多效果渲染时复用。值越大越省显存。")
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
        preset_dropdown.change(apply_preset, inputs=preset_dropdown, outputs=[prompt_input, input_text])
        save_preset_btn.click(
            save_current_as_preset,
            inputs=[preset_name_input, prompt_input, input_text],
            outputs=[preset_dropdown, output_status],
        )
        tts_engine_dropdown.change(
            toggle_tts_engine,
            inputs=tts_engine_dropdown,
            outputs=[tts_model_dropdown, qwen_prompt_group, clone_reference_group],
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
        generate_button.click(
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
            ],
            outputs=[output_audio, seed_input, output_status],
        )
        batch_generate_btn.click(
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
                batch_count,
                batch_effect_checkboxes,
            ],
            outputs=[batch_status_output, batch_file_output],
        )

        with gr.Accordion("📂 历史记录", open=False):
            gr.Markdown("查看和删除已生成的音频文件")
            history_search = gr.Textbox(label="搜索文件名", placeholder="输入文件名过滤...", scale=3)
            history_refresh_btn = gr.Button("🔄 刷新", variant="secondary", scale=1)
            selected_file_path = gr.State("")
            history_table = gr.DataFrame(
                label="历史音频（点击行选择）",
                headers=["name", "modified", "size"],
                datatype=["str", "str", "number"],
                interactive=False,
                wrap=True,
                max_height=300,
            )
            with gr.Row():
                history_play_btn = gr.Button("▶ 播放", variant="primary", size="sm", scale=1)
                history_delete_btn = gr.Button("🗑️ 删除", variant="stop", size="sm", scale=1)
                history_msg = gr.Textbox(label="", interactive=False, scale=3)

            def on_select_history(evt: gr.SelectData, search_term: str):
                files = service.list_generated_audios()
                if search_term:
                    files = [f for f in files if search_term.lower() in f["name"].lower()]
                index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                if 0 <= index < len(files):
                    return files[index]["path"]
                return ""

            def on_refresh_history(search_term: str):
                files = service.list_generated_audios()
                if search_term:
                    files = [f for f in files if search_term.lower() in f["name"].lower()]
                return files

            def on_play_file(file_path: str) -> tuple[str, str]:
                """播放音频"""
                if not file_path:
                    return "", "请先选择要播放的文件"
                return file_path, f"正在播放: {Path(file_path).name}"

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
            def on_delete_with_confirm(file_path: str) -> tuple[str, list[dict], str]:
                if not file_path:
                    return "请先选择要删除的文件", service.list_generated_audios(), ""
                return f"CONFIRM:{file_path}", service.list_generated_audios(), ""

            def on_confirm_delete(file_path: str) -> tuple[str, list[dict]]:
                if not file_path:
                    return "请先选择要删除的文件", service.list_generated_audios()
                if service.delete_audio(file_path):
                    return f"已删除: {Path(file_path).name}", service.list_generated_audios()
                return "删除失败", service.list_generated_audios()

            history_confirm_msg = gr.Textbox(label="确认删除", visible=False, interactive=False)
            with gr.Row(visible=False) as confirm_row:
                confirm_yes_btn = gr.Button("确认删除", variant="stop", size="sm")
                confirm_no_btn = gr.Button("取消", variant="secondary", size="sm")

            def on_show_confirm(msg: str) -> tuple[gr.update, str, list[dict]]:
                if not msg.startswith("CONFIRM:"):
                    return gr.update(visible=False), "", service.list_generated_audios()
                file_path = msg[len("CONFIRM:"):]
                return gr.update(visible=True), f"确定要删除 {Path(file_path).name} 吗？", service.list_generated_audios()

            history_delete_btn.click(
                on_delete_with_confirm,
                inputs=[selected_file_path],
                outputs=[history_confirm_msg, history_table, selected_file_path],
            )
            history_confirm_msg.change(
                on_show_confirm,
                inputs=[history_confirm_msg],
                outputs=[confirm_row, history_msg, history_table],
            )
            confirm_yes_btn.click(
                on_confirm_delete,
                inputs=[selected_file_path],
                outputs=[history_msg, history_table],
            )
            confirm_no_btn.click(
                lambda: ("", service.list_generated_audios()),
                inputs=[],
                outputs=[history_msg, history_table],
            )
            history_play_btn.click(
                on_play_file,
                inputs=[selected_file_path],
                outputs=[output_audio, output_status],
            )

    return demo
