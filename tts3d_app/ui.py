from __future__ import annotations

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
from tts3d_app.presets import PRESETS
from tts3d_app.service import GenerationRequest, TTSStudioService


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


def create_demo(service: TTSStudioService | None = None) -> gr.Blocks:
    service = service or TTSStudioService()
    has_hrir = service.hrir_dataset is not None

    mode_choices = ["单声道", "双声道", "虚拟脑后"]
    if has_hrir:
        mode_choices.extend(["静态空间定位 (HRIR)", "动态空间环绕 (HRIR)"])

    default_mode = "动态空间环绕 (HRIR)" if has_hrir else "虚拟脑后"

    def toggle_panels(mode_label: str):
        mode_key = MODE_LABEL_TO_KEY[mode_label]
        return (
            gr.update(visible=mode_key == MODE_STATIC_HRIR),
            gr.update(visible=mode_key == MODE_DYNAMIC_HRIR),
        )

    def apply_preset(preset_key: str) -> tuple[str, str]:
        preset = PRESETS[preset_key]
        return preset["prompt"], preset["text"]

    def generate_audio(
        preset_key: str,
        voice_description: str,
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
        )

        try:
            result = service.generate(request)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

        return result.file_path, result.seed, result.status, result.text

    with gr.Blocks(title="TTS 3D Studio") as demo:
        gr.Markdown("## Qwen3-TTS 3D Studio")
        gr.Markdown("输入文本，生成语音，并叠加双声道、脑后或 HRIR 空间音频效果。")
        if not has_hrir:
            gr.Markdown("提示：未检测到 HRIR 数据文件，当前只提供基础模式和虚拟脑后模式。")

        with gr.Row():
            with gr.Column(scale=4):
                input_text = gr.Textbox(
                    label="文本",
                    value=PRESETS["默认"]["text"],
                    lines=3,
                )

                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        label="预设",
                        choices=list(PRESETS.keys()),
                        value="默认",
                        scale=1,
                    )
                    prompt_input = gr.Textbox(
                        label="声音描述",
                        value=PRESETS["默认"]["prompt"],
                        scale=2,
                    )

                with gr.Accordion("音频空间引擎", open=True):
                    mode_radio = gr.Radio(
                        choices=mode_choices,
                        value=default_mode,
                        label="输出模式",
                    )

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

                generate_button = gr.Button("生成音频", variant="primary", size="lg")

            with gr.Column(scale=3):
                output_audio = gr.Audio(label="生成结果", type="filepath", autoplay=True)
                output_status = gr.Textbox(label="状态")

        mode_radio.change(toggle_panels, inputs=mode_radio, outputs=[static_panel, dynamic_panel])
        preset_dropdown.change(apply_preset, inputs=preset_dropdown, outputs=[prompt_input, input_text])
        generate_button.click(
            generate_audio,
            inputs=[
                preset_dropdown,
                prompt_input,
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
            ],
            outputs=[output_audio, seed_input, output_status, input_text],
        )

    return demo
