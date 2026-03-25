# TTS 3D Studio

这是一个基于 `Qwen3-TTS` 和 `Gradio` 的空间语音小项目。它支持把文本生成为语音，并叠加单声道、双声道、虚拟脑后、静态 HRIR 和动态 HRIR 等空间音频效果。

## 现在的项目结构

```text
.
|-- app.py
|-- pyproject.toml
|-- requirements.txt
|-- tts3d_app/
|   |-- audio_engine.py
|   |-- cli.py
|   |-- config.py
|   |-- hrir.py
|   |-- presets.py
|   |-- service.py
|   `-- ui.py
`-- tests/
    `-- test_audio_engine.py
```

## 运行方式

启动界面：

```bash
python app.py
```

检查环境：

```bash
python app.py doctor
```

做一次最小生成测试：

```bash
python app.py smoke-test
```

## 这次整理掉了什么

- 把 `check_gpu.py` 的环境检查合并进了统一 CLI。
- 把 `test_voice.py` 的最小生成验证合并进了统一 CLI。
- 删除旧版 `app01.py`，避免根目录同时存在两个主程序。
- 删除与当前 TTS 项目无关的 `MusicGen.py`。

## 验证命令

```bash
python -m compileall app.py tts3d_app tests
python -m unittest discover -s tests
```
