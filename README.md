# TTS 3D Studio

TTS 3D Studio 是一个基于 `Qwen3-TTS` 的 3D 空间语音工作台，目前支持 VoiceDesign 文本描述生成与 Base Clone 参考音频克隆，并提供 Gradio UI、CLI 和 MCP Server 三种使用方式。

## 已实现能力

- Qwen3-TTS VoiceDesign：文本 + 声音描述生成
- Qwen3-TTS Base Clone：参考音频 + 可选参考文本 + 生成文本模式
- 单声道、双声道、虚拟脑后、静态 HRIR、动态 HRIR 输出模式
- 干声缓存：相同引擎、模型和输入条件下切换空间效果时复用干声结果
- Qwen3-TTS 在 CUDA 环境下优先使用 BF16/FP16，并尝试启用 `torch.compile`
- 动态 HRIR 轨迹预计算在可用时使用 Numba 加速
- MCP Server：支持通过 `python app.py mcp` 以 stdio 模式启动

## 模型缓存目录

项目会把 Hugging Face 模型缓存统一写入下面两个固定目录：

- `HF_HOME=E:\AI_Models\huggingface`
- `HF_HUB_CACHE=E:\AI_Models\huggingface\hub`

运行时会自动设置以下环境变量并创建目录：

- `HF_HOME`
- `HF_HUB_CACHE`
- `HUGGINGFACE_HUB_CACHE`
- `TRANSFORMERS_CACHE`

这意味着 Qwen3-TTS 下载的模型都会统一落到 `E:\AI_Models\huggingface\hub`。

## 安装

建议使用 Python 3.12+。

```bash
pip install -r requirements.txt
```

如需自动识别参考音频文本，可额外安装 `faster-whisper`。

## 启动方式

启动 Gradio 界面：

```bash
python app.py run
```

检查环境与依赖状态：

```bash
python app.py doctor
```

执行最小生成测试：

```bash
python app.py smoke-test
```

以 stdio 模式启动 MCP Server：

```bash
python app.py mcp
```

## UI 使用说明

页面会新增两组与 TTS 相关的控件：

- `TTS 引擎`：`Qwen3-TTS VoiceDesign`、`Qwen3-TTS Base Clone`
- `模型`：根据引擎自动切换

当选择 `Qwen3-TTS VoiceDesign` 时：

- 显示 `声音描述`
- 隐藏参考音频与参考文本输入

当选择 `Qwen3-TTS Base Clone` 时：

- 显示 `参考音频`
- 显示 `参考文本`，可留空；留空时使用 speaker embedding fallback，克隆质量通常低于填写准确参考文本
- `声音描述` 不参与生成

## CLI 参数

`smoke-test` 现在支持以下新增参数：

- `--tts-engine`: `qwen3` 或 `qwen3_base`
- `--tts-model`: 模型键
- `--reference-audio`: Qwen3 Base Clone 参考音频路径
- `--reference-text`: 参考音频对应文本，可选但推荐填写

Qwen3-TTS VoiceDesign 示例：

```bash
python app.py smoke-test --tts-engine qwen3 --tts-model "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
```

Qwen3-TTS Base Clone 示例：

```bash
python app.py smoke-test --tts-engine qwen3_base --tts-model "Qwen/Qwen3-TTS-12Hz-1.7B-Base" --reference-audio demo.wav --reference-text "Hello, this is the reference text."
```

## MCP 接入 Claude Desktop

将下面的配置加入 Claude Desktop 的 MCP 配置文件，并把路径替换为你的 `app.py` 绝对路径：

```json
{
  "mcpServers": {
    "tts-3d-studio": {
      "command": "python",
      "args": ["F:/PythonStudy/AI_Agent/AI_Work/app.py", "mcp"]
    }
  }
}
```

MCP 工具名为 `generate_3d_speech`，主要参数包括：

- `text`
- `effect_type`: `mono`, `stereo`, `behind_head`, `static_hrir`, `dynamic_hrir`
- `voice_description`
- `seed`
- `tts_engine`: `qwen3`, `qwen3_base`
- `tts_model_key`
- `reference_audio_path`
- `reference_text`
- `static_azimuth_deg`
- `static_distance_m`
- `dynamic_path`: `clockwise`, `counterclockwise`, `side_to_side`
- `dynamic_cycle_time_s`
- `dynamic_start_azimuth_deg`
- `dynamic_distance_m`

## 运行说明

- 如果缺少 `hrir_spatial_map.npz`，程序仍可运行，但只提供基础模式和虚拟脑后模式。
- `doctor` 会输出 CUDA、SoX、flash-attn、numba、fastmcp、ASR 等依赖状态。
- 如果本机支持 CUDA，Qwen3-TTS 会优先尝试 BF16 推理，并开启 TF32。
- Qwen3-TTS Base Clone 需要参考音频；参考文本可手动填写，也可在安装 ASR 后自动识别。

## 可调环境变量

- `QWEN_TTS_MODEL`: 指定当前 Qwen3-TTS 模型名
- `QWEN_TTS_BASE_MODEL`: 指定当前 Qwen3-TTS Base Clone 模型名
- `APP_SERVER_NAME`: 指定 Gradio 监听地址
- `TTS3D_OUTPUT_DIR`: 指定音频输出目录，默认为 `outputs`
- `CLEAR_CUDA_CACHE_AFTER_GENERATE=1`: 每次生成后清理 CUDA Cache
- `ENABLE_TORCH_COMPILE=0`: 关闭 `torch.compile`
- `TORCH_COMPILE_MODE`: 默认 `reduce-overhead`
- `DRY_AUDIO_CACHE_SIZE`: 干声缓存条目数，默认 `8`
- `ENABLE_GPU_FFT_CONVOLUTION=0`: 关闭 torchaudio GPU FFT 卷积路径
- `ENABLE_NUMBA_DYNAMIC_HRIR=0`: 关闭动态 HRIR 的 Numba 轨迹规划

## 测试

运行全部单元测试：

```bash
python -m unittest discover -s tests
```

当前测试覆盖了：

- FFT 卷积与时域卷积结果一致性
- 动态 HRIR 最后一帧重叠写入安全性
- 动态 HRIR 轨迹规划正确性
- 服务层按引擎和模型选择 provider
- 干声缓存会因引擎、模型和参考音频变化而失效
- Qwen3-TTS Base Clone 缺少参考音频时返回明确错误
- Gradio/CLI/MCP 暴露新的引擎和模型参数
- Hugging Face 缓存目录环境变量设置正确
