"""长驻 CosyVoice2 推理进程。

由 CosyVoice2Provider 通过独立 venv（TTS_COSYVOICE_PYTHON）启动，
避免把 CosyVoice 的依赖混入应用环境（应用 torch 与 CosyVoice 实验环境版本不同）。

协议：stdin 每行一个 JSON 任务，stdout 回复带 @@JSON@@ 前缀的行式 JSON
（库可能向 stdout 打印进度日志，父进程只解析带前缀的行）。
启动即回复 {"ok": true, "event": "ready", "sample_rate": N}；
任务 {"id": N, "seed": S, "text": T, "instruct_text": I, "prompt_wav": P, "out_wav": O}
回复 {"ok": true/false, "event": "done", "id": N, "sample_rate": N} 或 {"ok": false, "error": E}。
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

RESPONSE_PREFIX = "@@JSON@@"


def emit(payload: dict) -> None:
    print(RESPONSE_PREFIX + json.dumps(payload), flush=True)


def _ensure_repo_importable(repo_dir: Path) -> None:
    for entry in (str(repo_dir), str(repo_dir / "third_party" / "Matcha-TTS")):
        if Path(entry).exists() and entry not in sys.path:
            sys.path.insert(0, entry)
    # 训练工具链在 import 时加载 pyworld；纯推理不会调用，缺失时以 stub 顶替。
    try:
        import pyworld  # noqa: F401
    except ImportError:
        if "pyworld" not in sys.modules:
            stub = types.ModuleType("pyworld")
            # torch 等库会 hasattr() 探测模块属性，缺失属性必须按规范抛 AttributeError。
            stub.__file__ = None
            stub.__getattr__ = lambda name: (_ for _ in ()).throw(
                AttributeError(f"pyworld stub has no attribute {name!r} (training-only)")
            )  # type: ignore[method-assign]
            sys.modules["pyworld"] = stub


def main() -> None:
    repo_dir = Path(sys.argv[1])
    model_dir = sys.argv[2]

    _ensure_repo_importable(repo_dir)
    from cosyvoice.cli.cosyvoice import CosyVoice2
    import torch
    import torchaudio

    model = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)
    emit({"ok": True, "event": "ready", "sample_rate": int(model.sample_rate)})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        job = json.loads(line)
        try:
            torch.manual_seed(int(job["seed"]) % (2**32))
            parts = [
                chunk["tts_speech"]
                for chunk in model.inference_instruct2(
                    job["text"], job["instruct_text"], job["prompt_wav"], stream=False
                )
            ]
            if not parts:
                raise RuntimeError("CosyVoice2 returned no audio.")
            speech = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
            torchaudio.save(job["out_wav"], speech.cpu(), model.sample_rate)
            emit({"ok": True, "event": "done", "id": job["id"], "sample_rate": int(model.sample_rate)})
        except Exception as exc:  # noqa: BLE001 - 错误原样回传主进程
            emit({"ok": False, "event": "done", "id": job.get("id"), "error": str(exc)})


if __name__ == "__main__":
    main()
