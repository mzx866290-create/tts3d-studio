"""FastAPI Web UI 后端：REST + SSE 进度流，为 Vue 前端提供全部能力。

与 Gradio UI 共享同一个 TTSStudioService；生成类接口用 SSE 推送
model_loading / voice_lock / chunk / spatial / saving / result 等事件。
"""
from __future__ import annotations

import json
import platform
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

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
    OUTPUT_DIR,
    TTS_CHUNK_PARAGRAPH_PAUSE_MS,
    TTS_CHUNK_SENTENCE_PAUSE_MS,
    TTS_CLONE_ENGINE,
    TTS_DIRECT_SINGLE_CHUNK,
    USER_CONFIG_DIR,
    VOICE_PROFILE_DIR,
    load_user_favorites,
    load_user_presets,
    save_user_favorites,
    save_user_presets,
)
from tts3d_app.presets import PRESETS
from tts3d_app.service import (
    BatchRequest,
    GenerationCancelled,
    GenerationRequest,
    TTSStudioService,
)
from tts3d_app.tts_engines import DEFAULT_TTS_ENGINE
from tts3d_app.ux import EMOTION_PRESETS, describe_generation_route
from tts3d_app.voice_profile import reference_clip_wav_path

WEBAPP_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEBAPP_DIR / "static"
UPLOAD_DIR = USER_CONFIG_DIR / "uploads"
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg"}

MODE_DEFS = (
    (MODE_MONO, "单声道", True),
    (MODE_STEREO, "双声道", True),
    (MODE_BEHIND_HEAD, "虚拟脑后", True),
    (MODE_STATIC_HRIR, "静态空间定位", "hrir"),
    (MODE_DYNAMIC_HRIR, "动态空间环绕", "hrir"),
)
PATH_DEFS = (
    (PATH_CLOCKWISE, "顺时针绕头"),
    (PATH_COUNTERCLOCKWISE, "逆时针绕头"),
    (PATH_SIDE_TO_SIDE, "左右摆动"),
)


def _sse(event: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


class GenerateBody(BaseModel):
    text: str = ""
    voice_description: str = ""
    preset_key: str = ""
    seed: int | None = None
    use_random_seed: bool = False
    render_mode: str = MODE_MONO
    static_azimuth_deg: float = 270
    static_distance_m: float = 0.3
    dynamic_path: str = PATH_CLOCKWISE
    dynamic_cycle_time_s: float = 8.0
    dynamic_start_azimuth_deg: float = 270
    dynamic_distance_m: float = 0.2
    tts_engine: str = DEFAULT_TTS_ENGINE
    tts_model_key: str = ""
    reference_audio_path: str | None = None
    reference_text: str = ""
    speed_factor: float = 1.0
    pitch_semitones: float = 0.0
    calibration_text: str = ""
    emotion_instruct: str = ""


class BatchBody(GenerateBody):
    batch_count: int = 3
    effect_types: list[str] = [MODE_MONO]


class RouteHintBody(BaseModel):
    text: str = ""
    emotion_instruct: str = ""
    tts_engine: str = DEFAULT_TTS_ENGINE


class PreviewClipBody(BaseModel):
    voice_description: str = ""
    seed: int | None = None
    tts_engine: str = DEFAULT_TTS_ENGINE
    tts_model_key: str = ""
    calibration_text: str = ""
    reroll: bool = False


class FavoriteBody(BaseModel):
    name: str
    prompt: str = ""
    text: str = ""
    calibration_text: str = ""
    emotion_instruct: str = ""
    seed: int | None = None
    mode_label: str = ""
    speed_factor: float = 1.0
    pitch_semitones: float = 0.0
    clip_id: str = ""


class PresetBody(BaseModel):
    name: str
    prompt: str = ""
    text: str = ""
    calibration_text: str = ""


class TranscribeBody(BaseModel):
    path: str


class CancelBody(BaseModel):
    pass


def _file_to_url(path: Path | str) -> str:
    from urllib.parse import quote

    return "/media?path=" + quote(str(path), safe="")


def _resolve_media_path(raw: str) -> Path:
    """仅允许 outputs / 声线档案 / 上传目录内的文件，防目录穿越。"""
    if not raw:
        raise HTTPException(status_code=400, detail="缺少 path 参数")
    try:
        target = Path(raw).expanduser().resolve()
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="无效路径")
    roots = [OUTPUT_DIR.resolve(), VOICE_PROFILE_DIR.resolve(), UPLOAD_DIR.resolve()]
    if not any(target.is_relative_to(root) for root in roots):
        raise HTTPException(status_code=403, detail="路径不在允许的目录内")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return target


def create_app(service: TTSStudioService | None = None) -> FastAPI:
    service = service or TTSStudioService()
    app = FastAPI(title="TTS 3D Studio", docs_url=None, redoc_url=None)
    app.state.service = service
    # GPU 模型加载/推理不是线程安全且显存有限：同一时刻只跑一个生成/试听任务。
    app.state.job_lock = threading.Lock()

    def build_meta() -> dict[str, Any]:
        engines = []
        for label, key in service.get_tts_engine_choices():
            models = service.get_tts_model_choices(key)
            engines.append(
                {
                    "key": key,
                    "label": label,
                    "models": [{"key": m_key, "label": m_label} for m_label, m_key in models],
                    "default_model": service.get_default_tts_model_key(key),
                }
            )
        modes = []
        for mode_key, mode_label, availability in MODE_DEFS:
            if availability == "hrir":
                available = service.hrir_dataset is not None
            else:
                available = True
            modes.append({"key": mode_key, "label": mode_label, "available": available})
        presets = []
        for name, preset in {**PRESETS, **load_user_presets()}.items():
            presets.append(
                {
                    "key": name,
                    "prompt": preset.get("prompt", ""),
                    "text": preset.get("text", ""),
                    "calibration_text": preset.get("calibration_text", ""),
                }
            )
        favorites = []
        for name, fav in load_user_favorites().items():
            favorites.append({"name": name, **fav})
        return {
            "app": "TTS 3D Studio",
            "engines": engines,
            "modes": modes,
            "paths": [{"key": k, "label": v} for k, v in PATH_DEFS],
            "emotion_presets": [{"label": l, "text": t} for l, t in EMOTION_PRESETS],
            "presets": presets,
            "default_preset": service.default_preset_key(),
            "favorites": favorites,
            "asr_available": ASR_AVAILABLE,
            "has_hrir": service.hrir_dataset is not None,
            "output_dir": str(service.output_dir),
            "clone_engine": TTS_CLONE_ENGINE,
            "direct_single_chunk": TTS_DIRECT_SINGLE_CHUNK,
            "limits": {
                "max_chunk_chars": MAX_TTS_CHUNK_CHARS,
                "sentence_pause_ms": TTS_CHUNK_SENTENCE_PAUSE_MS,
                "paragraph_pause_ms": TTS_CHUNK_PARAGRAPH_PAUSE_MS,
            },
        }

    @app.get("/api/meta")
    def get_meta() -> dict[str, Any]:
        return build_meta()

    @app.post("/api/route-hint")
    def post_route_hint(body: RouteHintBody) -> dict[str, str]:
        return {"text": describe_generation_route(body.text, body.emotion_instruct, body.tts_engine)}

    def _sse_stream(job: Callable[[Callable[[str, dict], None]], Any]) -> StreamingResponse:
        """把一个会阻塞的生成任务包成 SSE 流：progress 事件 + 终态事件。"""

        def generator() -> Iterator[bytes]:
            lock: threading.Lock = app.state.job_lock
            if not lock.acquire(blocking=False):
                yield _sse("error", {"message": "已有生成任务在进行中，请等待完成或先停止。"})
                return
            started = time.time()
            try:
                # progress 回调在生成线程里执行，无法直接跨越 SSE 边界；
                # 用队列在生成线程与流线程之间转发。
                event_queue: "queue.Queue[tuple[str, dict]]" = queue.Queue()

                def queued_progress(stage: str, data: dict[str, Any]) -> None:
                    event_queue.put((stage, data))

                def run_job() -> None:
                    try:
                        result = job(queued_progress)
                    except GenerationCancelled:
                        event_queue.put(("__cancelled__", {}))
                    except Exception as exc:
                        event_queue.put(("__error__", {"message": f"{exc}"}))
                    else:
                        event_queue.put(("__done__", {"payload": result}))

                worker = threading.Thread(target=run_job, daemon=True)
                worker.start()
                terminal: tuple[str, dict] | None = None
                last_ping = time.time()
                while True:
                    try:
                        stage, data = event_queue.get(timeout=0.25)
                    except queue.Empty:
                        # 保活：每 2 秒发一个 ping，让前端确认连接还活着
                        if time.time() - last_ping >= 2.0:
                            last_ping = time.time()
                            yield _sse("ping", {"elapsed": round(time.time() - started, 1)})
                        if not worker.is_alive():
                            # worker 已退出但没有终态事件（理论上不会发生）
                            break
                        continue
                    if stage == "__done__":
                        terminal = ("result", data["payload"])
                    elif stage == "__cancelled__":
                        terminal = ("cancelled", {"message": "已中断生成"})
                    elif stage == "__error__":
                        terminal = ("error", data)
                    else:
                        yield _sse("progress", {"stage": stage, **data})
                        continue
                    break
                if terminal:
                    yield _sse(terminal[0], terminal[1])
            finally:
                lock.release()

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/generate")
    def post_generate(body: GenerateBody) -> StreamingResponse:
        request = GenerationRequest(
            preset_key=body.preset_key,
            voice_description=body.voice_description,
            text=body.text,
            seed=body.seed,
            use_random_seed=body.use_random_seed,
            render_mode=body.render_mode,
            static_azimuth_deg=body.static_azimuth_deg,
            static_distance_m=body.static_distance_m,
            dynamic_path=body.dynamic_path,
            dynamic_cycle_time_s=body.dynamic_cycle_time_s,
            dynamic_start_azimuth_deg=body.dynamic_start_azimuth_deg,
            dynamic_distance_m=body.dynamic_distance_m,
            tts_engine=body.tts_engine,
            tts_model_key=body.tts_model_key,
            reference_audio_path=body.reference_audio_path,
            reference_text=body.reference_text,
            speed_factor=body.speed_factor,
            pitch_semitones=body.pitch_semitones,
            calibration_text=body.calibration_text,
            emotion_instruct=body.emotion_instruct,
        )

        def job(progress: Callable[[str, dict], None]) -> dict[str, Any]:
            result = service.generate(request, progress_callback=progress)
            return {
                "file_path": result.file_path,
                "audio_url": _file_to_url(result.file_path),
                "seed": result.seed,
                "status": result.status,
                "clip_id": result.clip_id,
                "clip_path": result.clip_path,
                "clip_url": _file_to_url(result.clip_path) if result.clip_path else "",
                "history": service.list_generated_audios()[:50],
            }

        return _sse_stream(job)

    @app.post("/api/batch")
    def post_batch(body: BatchBody) -> StreamingResponse:
        if not body.effect_types:
            raise HTTPException(status_code=400, detail="请至少选择一个效果类型")
        import random as _random

        seeds = [_random.randint(0, 2**32 - 1) for _ in range(max(1, int(body.batch_count)))]
        batch_request = BatchRequest(
            text=body.text,
            voice_description=body.voice_description,
            tts_engine=body.tts_engine,
            tts_model_key=body.tts_model_key,
            reference_audio_path=body.reference_audio_path,
            reference_text=body.reference_text,
            preset_key=body.preset_key,
            seeds=seeds,
            effect_types=list(body.effect_types),
            static_azimuth_deg=body.static_azimuth_deg,
            static_distance_m=body.static_distance_m,
            dynamic_path=body.dynamic_path,
            dynamic_cycle_time_s=body.dynamic_cycle_time_s,
            dynamic_start_azimuth_deg=body.dynamic_start_azimuth_deg,
            dynamic_distance_m=body.dynamic_distance_m,
            speed_factor=body.speed_factor,
            pitch_semitones=body.pitch_semitones,
            calibration_text=body.calibration_text,
            emotion_instruct=body.emotion_instruct,
        )

        def job(progress: Callable[[str, dict], None]) -> dict[str, Any]:
            result = service.generate_batch(batch_request, progress_callback=progress)
            items = [
                {
                    "file_path": item.file_path,
                    "audio_url": _file_to_url(item.file_path) if item.file_path else "",
                    "seed": item.seed,
                    "effect_type": item.effect_type,
                    "format": item.format,
                    "status": item.status,
                }
                for item in result.items
            ]
            return {
                "total": result.total,
                "succeeded": result.succeeded,
                "failed": result.failed,
                "items": items,
                "history": service.list_generated_audios()[:50],
            }

        return _sse_stream(job)

    @app.post("/api/cancel")
    def post_cancel() -> dict[str, bool]:
        service.request_cancel()
        return {"ok": True}

    @app.post("/api/preview-clip")
    def post_preview_clip(body: PreviewClipBody) -> dict[str, Any]:
        lock: threading.Lock = app.state.job_lock
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="已有生成任务在进行中")
        try:
            clip_path, clip_id, seed, status = service.preview_voice_reference(
                voice_description=body.voice_description,
                seed=body.seed,
                tts_engine=body.tts_engine,
                tts_model_key=body.tts_model_key,
                calibration_text=body.calibration_text,
                reroll=body.reroll,
            )
        finally:
            lock.release()
        return {
            "clip_path": clip_path,
            "clip_url": _file_to_url(clip_path) if clip_path else "",
            "clip_id": clip_id,
            "seed": seed,
            "status": status,
        }

    @app.post("/api/reference/upload")
    async def post_reference_upload(file: UploadFile = File(...)) -> dict[str, str]:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename or "reference.wav").suffix.lower() or ".wav"
        target = UPLOAD_DIR / f"ref_{uuid.uuid4().hex[:8]}_{int(time.time())}{suffix}"
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="上传文件为空")
        target.write_bytes(data)
        return {"path": str(target), "url": _file_to_url(target), "name": target.name}

    @app.post("/api/reference/transcribe")
    def post_reference_transcribe(body: TranscribeBody) -> dict[str, str]:
        if not ASR_AVAILABLE:
            raise HTTPException(status_code=400, detail="未安装 ASR 后端，请运行: pip install faster-whisper")
        target = _resolve_media_path(body.path)
        from tts3d_app.asr import transcribe_audio

        try:
            text = transcribe_audio(str(target))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"识别失败: {exc}")
        return {"text": text}

    @app.get("/api/history")
    def get_history() -> list[dict[str, Any]]:
        items = service.list_generated_audios()
        for item in items:
            item["url"] = _file_to_url(item["path"])
        return items

    @app.delete("/api/history")
    def delete_history(path: str = Query(...)) -> dict[str, Any]:
        ok = service.delete_audio(path)
        return {"ok": ok, "history": service.list_generated_audios()}

    @app.get("/api/favorites")
    def get_favorites() -> list[dict[str, Any]]:
        return [{"name": name, **fav} for name, fav in load_user_favorites().items()]

    @app.post("/api/favorites")
    def post_favorites(body: FavoriteBody) -> dict[str, Any]:
        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="请先填入收藏名称")
        favorites = load_user_favorites()
        favorites[name] = {
            "prompt": body.prompt,
            "text": body.text,
            "calibration_text": body.calibration_text,
            "emotion_instruct": body.emotion_instruct,
            "seed": body.seed,
            "mode_label": body.mode_label,
            "speed_factor": body.speed_factor,
            "pitch_semitones": body.pitch_semitones,
            "clip_id": body.clip_id.strip(),
        }
        save_user_favorites(favorites)
        return {"ok": True, "name": name, "favorites": [{"name": n, **f} for n, f in favorites.items()]}

    @app.delete("/api/favorites")
    def delete_favorite(name: str = Query(...)) -> dict[str, Any]:
        favorites = load_user_favorites()
        favorites.pop(name, None)
        save_user_favorites(favorites)
        return {"ok": True, "favorites": [{"name": n, **f} for n, f in favorites.items()]}

    @app.post("/api/presets")
    def post_presets(body: PresetBody) -> dict[str, Any]:
        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="请输入预设名称")
        if name in PRESETS:
            raise HTTPException(status_code=400, detail=f"预设名 '{name}' 与内置预设冲突，请换名")
        user_presets = load_user_presets()
        user_presets[name] = {
            "prompt": body.prompt,
            "text": body.text,
            "calibration_text": body.calibration_text,
        }
        save_user_presets(user_presets)
        return {"ok": True, "name": name}

    @app.get("/api/clip-url")
    def get_clip_url(clip_id: str = Query(...)) -> dict[str, str]:
        path = reference_clip_wav_path(clip_id.strip())
        if not path.exists():
            raise HTTPException(status_code=404, detail="参考音不存在")
        return {"url": _file_to_url(path), "path": str(path)}

    @app.get("/media")
    def get_media(path: str = Query(...)) -> FileResponse:
        target = _resolve_media_path(path)
        return FileResponse(target)

    @app.post("/api/open-folder")
    def post_open_folder() -> dict[str, Any]:
        target = str(service.output_dir)
        try:
            if platform.system() == "Windows":
                subprocess.Popen(["explorer", target])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["xdg-open", target])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"打开目录失败: {exc}")
        return {"ok": True, "path": target}

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str) -> Any:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="未知接口")
        if full_path:
            candidate = (STATIC_DIR / full_path).resolve()
            if candidate.is_relative_to(STATIC_DIR.resolve()) and candidate.is_file():
                return FileResponse(candidate)
        index = STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(index)
        return HTMLResponse(
            "<h1>TTS 3D Studio Web UI</h1><p>前端尚未构建。请在项目根目录执行：</p>"
            "<pre>cd webui &amp;&amp; npm install &amp;&amp; npm run build</pre>"
            "<p>或使用 <code>python app.py run --ui gradio</code> 启动旧版界面。</p>",
            status_code=503,
        )

    return app


def run_web_app(server_name: str = "127.0.0.1", server_port: int | None = None) -> int:
    import threading
    import webbrowser

    import uvicorn

    from tts3d_app.config import configure_runtime, get_logger

    configure_runtime()
    logger = get_logger("webapp")
    app = create_app()

    resolved_port = server_port or 7860
    url = f"http://{server_name}:{resolved_port}/"

    def open_browser() -> None:
        time.sleep(1.2)
        webbrowser.open(url)

    if server_name in ("127.0.0.1", "localhost", "0.0.0.0"):
        threading.Thread(target=open_browser, daemon=True).start()

    logger.info("TTS 3D Studio Web UI: %s", url)
    print(f"TTS 3D Studio is available at: {url}")
    uvicorn.run(app, host=server_name, port=resolved_port, log_level="warning")
    return 0
