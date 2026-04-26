"""FastAPI: health, /chat, /chat/stream (SSE) — create_agent + local OpenAI-compatible Gemma (llama-server)."""

from __future__ import annotations

import json
import mimetypes
import os
import tempfile
import threading
import urllib.error
import urllib.request
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    ToolMessageChunk,
)
from pydantic import BaseModel, Field

from molmo_tool import (
    call_molmo_point,
    enrich_molmo_result_for_client,
    get_image_dimensions,
    molmo_service_reachable,
)
from state_graph import build_agent, prepare_for_model

GEMMA_HEALTH_URLS = [
    f"http://127.0.0.1:{os.environ.get('GEMMA_PORT', '8080')}/health",
    f"http://127.0.0.1:{os.environ.get('GEMMA_PORT', '8080')}/",
]


def _cors_origins() -> list[str]:
    o = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    for part in os.environ.get("CORS_EXTRA_ORIGINS", "").split(","):
        p = part.strip()
        if p:
            o.append(p)
    return o


app = FastAPI(title="LangGraph + local Gemma", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent: Any = None

# OAK-D captures pending consumption by /chat/stream-with-oak-capture-id.
# capture_id (stem of the JPEG) -> {"path": Path, "depth": np.ndarray, "w": int, "h": int}.
_OAK_CAPTURES: dict[str, dict[str, Any]] = {}
_OAK_CAPTURES_LOCK = threading.Lock()


def _oak_evict_capture(capture_id: str) -> None:
    """Pop a cached capture and unlink its JPEG. Safe to call with an unknown id."""
    with _OAK_CAPTURES_LOCK:
        entry = _OAK_CAPTURES.pop(capture_id, None)
    if entry is None:
        return
    path = entry.get("path")
    if isinstance(path, Path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def get_agent(
    uploaded_image_path: str | None = None,
    depth_lookup: Callable[[float, float], float | None] | None = None,
):
    global _agent
    if uploaded_image_path:
        return build_agent(uploaded_image_path, depth_lookup=depth_lookup)
    if _agent is None:
        _agent = build_agent()
    return _agent


class Msg(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Msg] = Field(min_length=1)


MOLMO_MAX_UPLOAD_BYTES = int(os.environ.get("MOLMO_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
MOLMO_ALLOWED_FILE_TYPES: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

_MOLMO_UPLOAD_DIR: Path | None = None


def _dir_writable(p: Path) -> bool:
    return os.access(p, os.W_OK)


def _default_repo_upload_dir() -> Path:
    return (Path(__file__).resolve().parent.parent / "run" / "molmo-uploads").resolve()


def _fallback_temp_upload_dir() -> Path:
    return (
        Path(tempfile.gettempdir())
        / f"pib-robocup-1-molmo-uploads-{os.getuid()}"
    ).resolve()


def _molmo_upload_dir() -> Path:
    global _MOLMO_UPLOAD_DIR
    if _MOLMO_UPLOAD_DIR is not None:
        return _MOLMO_UPLOAD_DIR
    raw = os.environ.get("MOLMO_UPLOAD_DIR", "").strip()
    if raw:
        p = Path(raw).expanduser().resolve()
    else:
        p = _default_repo_upload_dir()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        p = _fallback_temp_upload_dir()
        p.mkdir(parents=True, exist_ok=True)
    if not _dir_writable(p):
        p = _fallback_temp_upload_dir()
        p.mkdir(parents=True, exist_ok=True)
    _MOLMO_UPLOAD_DIR = p
    return p


def _gemma_reachable() -> bool:
    for url in GEMMA_HEALTH_URLS:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as r:
                if 200 <= r.getcode() < 500:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            continue
    return False


def _to_lc_messages(items: list[Msg]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in items:
        if m.role == "user":
            out.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            out.append(AIMessage(content=m.content))
        elif m.role == "system":
            out.append(SystemMessage(content=m.content))
        else:
            raise HTTPException(status_code=400, detail=f"Unknown role: {m.role!r}")
    return out


MOLMO_TOOL_NAMES = {"molmo_point_localize", "molmo_point_localize_uploaded"}


def _parse_molmo_tool_content(content: Any) -> dict[str, Any] | None:
    """Parse JSON or plain error string from `molmo_point_localize` tool return."""
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    s = content.strip()
    if s.startswith("{"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return {"error": s}
    return {"error": s}


def _molmo_from_tool_message(m: ToolMessage) -> dict[str, Any] | None:
    if m.name not in MOLMO_TOOL_NAMES:
        return None
    return _parse_molmo_tool_content(m.content)


def _is_molmo_tool_result_dict(p: dict[str, Any]) -> bool:
    """Distinguish molmo_point_localize JSON from other tools when `name` is missing."""
    if "points" in p and isinstance(p.get("points"), list):
        return True
    if "generated_text" in p:
        return True
    if "error" in p and len(p) == 1:
        return True
    return False


def _try_complete_molmo_json(s: str) -> dict[str, Any] | None:
    s = s.strip()
    if not s.startswith("{"):
        if s:
            return _parse_molmo_tool_content(s)
        return None
    try:
        json.loads(s)
    except json.JSONDecodeError:
        return None
    return _parse_molmo_tool_content(s)


def _molmo_payload_for_sse(
    p: dict[str, Any], upload_dims: tuple[int, int] | None
) -> dict[str, Any]:
    """For chat-with-image, force known image size + 0–1 points into every SSE `molmo_result` event."""
    return enrich_molmo_result_for_client(p, upload_dims)


def _collect_molmo_results(messages: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.name in MOLMO_TOOL_NAMES:
            p = _molmo_from_tool_message(m)
            if p is not None:
                out.append(p)
    return out


async def _read_upload_image_bytes(file: UploadFile) -> tuple[bytes, str]:
    body = b""
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        body += chunk
        if len(body) > MOLMO_MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="file too large")
    if not body:
        raise HTTPException(status_code=400, detail="empty file")
    ct = (file.content_type or "").split(";")[0].strip() or mimetypes.guess_type(
        file.filename or ""
    )[0]
    if not ct or ct not in MOLMO_ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="unsupported or missing image type; use jpeg, png, or webp",
        )
    return body, MOLMO_ALLOWED_FILE_TYPES[ct]


async def _save_uploaded_image(file: UploadFile) -> Path:
    body, ext = await _read_upload_image_bytes(file)
    dest_dir = _molmo_upload_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex}{ext}"
    dest.write_bytes(body)
    return dest


def _last_assistant_text(messages: list[Any]) -> str:
    """Last AIMessage with user-visible text, skipping tool-only assistant turns if present."""
    for m in reversed(messages):
        if not isinstance(m, AIMessage):
            continue
        if getattr(m, "tool_calls", None):
            continue
        c = m.content
        if isinstance(c, str) and c:
            return c
        if c is not None and c not in ("",):
            return str(c)
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content is not None:
            return m.content if isinstance(m.content, str) else str(m.content)
    raise ValueError("No assistant content in result")


@app.get("/health")
def health():
    return {
        "ok": True,
        "gemma": _gemma_reachable(),
        "molmo": molmo_service_reachable(),
    }


@app.post("/molmo/localize")
async def molmo_localize(
    file: UploadFile = File(...),
    prompt: str = Form(...),
):
    """Save an uploaded image to disk, run MolmoPoint, return points (and delete the temp file)."""
    p = (prompt or "").strip()
    if not p:
        raise HTTPException(status_code=400, detail="prompt is required")
    if len(p) > 4000:
        raise HTTPException(status_code=400, detail="prompt too long")

    dest = await _save_uploaded_image(file)
    try:
        out = call_molmo_point(str(dest), p)
    finally:
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass

    if isinstance(out, str):
        raise HTTPException(status_code=502, detail=out)

    payload: dict[str, Any] = {
        "ok": True,
        "points": out.get("points", []),
        "generated_text": out.get("generated_text", ""),
        "device": out.get("device", ""),
        "model_id": out.get("model_id", ""),
    }
    if out.get("image_width") is not None:
        payload["image_width"] = out["image_width"]
    if out.get("image_height") is not None:
        payload["image_height"] = out["image_height"]
    return payload


@app.post("/chat")
def chat_post(req: ChatRequest):
    try:
        lc = _to_lc_messages(req.messages)
    except HTTPException:
        raise
    msgs = prepare_for_model(lc)
    try:
        out = get_agent().invoke({"messages": msgs})
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Gemma/LangGraph error: {e!s}"
        ) from e
    if not isinstance(out, dict) or "messages" not in out:
        raise HTTPException(status_code=502, detail="Unexpected agent result shape")
    try:
        text = _last_assistant_text(out["messages"])
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    molmo_results = _collect_molmo_results(out["messages"])
    base: dict[str, Any] = {"message": {"role": "assistant", "content": text}}
    if molmo_results:
        base["molmo_results"] = molmo_results
    return base


def _token_text(chunk: Any) -> str | None:
    c = getattr(chunk, "content", None)
    if isinstance(c, str) and c:
        return c
    return None


def _tool_result_preview_for_ui(content: Any, max_len: int = 4000) -> str:
    if isinstance(content, str):
        s = content
    else:
        try:
            s = json.dumps(content, ensure_ascii=False)
        except (TypeError, ValueError):
            s = str(content)
    s = s.strip()
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _iter_ai_toolcall_sse_lines(chunk: Any) -> list[str]:
    """Convert AIMessageChunk tool-call streaming into UI-friendly JSON SSE lines."""
    if not isinstance(chunk, AIMessageChunk):
        return []
    tcs = getattr(chunk, "tool_call_chunks", None) or []
    if not tcs:
        return []
    out: list[str] = []
    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        tid = tc.get("id")
        name = tc.get("name")
        arg_fragment = tc.get("args")
        if name or arg_fragment is not None:
            out.append(
                json.dumps(
                    {
                        "tool_call": {
                            "id": tid,
                            "name": name,
                            "args": arg_fragment,
                        }
                    },
                    ensure_ascii=False,
                )
            )
    return out


async def _stream_body(req: ChatRequest) -> AsyncIterator[str]:
    async for line in _stream_body_with_image(req):
        yield line


async def _stream_body_with_image(
    req: ChatRequest,
    uploaded_image_path: str | None = None,
    depth_lookup: Callable[[float, float], float | None] | None = None,
) -> AsyncIterator[str]:
    try:
        lc = _to_lc_messages(req.messages)
    except HTTPException as e:
        err = json.dumps({"error": e.detail})
        yield f"data: {err}\n\n"
        return
    msgs = prepare_for_model(lc, uploaded_image_path)
    agent = get_agent(uploaded_image_path, depth_lookup=depth_lookup)
    upload_dims: tuple[int, int] | None = None
    if uploaded_image_path:
        upload_dims = get_image_dimensions(Path(uploaded_image_path))
    molmo_tmc_merged: dict[str, ToolMessageChunk] = {}
    molmo_sent: set[str] = set()
    try:
        async for item in agent.astream(
            {"messages": msgs},
            stream_mode="messages",
        ):
            chunk: Any
            if isinstance(item, tuple) and len(item) == 2:
                chunk, _ = item
            else:
                chunk = item
            for sse_line in _iter_ai_toolcall_sse_lines(chunk):
                yield f"data: {sse_line}\n\n"
            if isinstance(chunk, ToolMessageChunk):
                if chunk.name and chunk.name not in MOLMO_TOOL_NAMES:
                    pass
                else:
                    tid = chunk.tool_call_id
                    prev = molmo_tmc_merged.get(tid)
                    merged = (prev + chunk) if prev is not None else chunk
                    molmo_tmc_merged[tid] = merged
                    nm = merged.name
                    c = merged.content
                    if nm in MOLMO_TOOL_NAMES or (nm is None and isinstance(c, (str, dict))):
                        p: dict[str, Any] | None = None
                        if isinstance(c, dict) and (
                            nm in MOLMO_TOOL_NAMES or _is_molmo_tool_result_dict(c)
                        ):
                            p = c
                        elif isinstance(c, str):
                            p = _try_complete_molmo_json(c)
                        if (
                            p is not None
                            and (nm in MOLMO_TOOL_NAMES or _is_molmo_tool_result_dict(p))
                            and tid not in molmo_sent
                        ):
                            molmo_sent.add(tid)
                            p_out = _molmo_payload_for_sse(p, upload_dims)
                            line = json.dumps({"molmo_result": p_out})
                            yield f"data: {line}\n\n"
            if isinstance(chunk, ToolMessage) and not isinstance(
                chunk, ToolMessageChunk
            ):
                tid = chunk.tool_call_id
                if chunk.name and chunk.name not in MOLMO_TOOL_NAMES:
                    tr = {
                        "tool_result": {
                            "id": tid,
                            "name": chunk.name,
                            "content": _tool_result_preview_for_ui(chunk.content),
                        }
                    }
                    yield f"data: {json.dumps(tr, ensure_ascii=False)}\n\n"
                elif chunk.name in MOLMO_TOOL_NAMES:
                    if tid in molmo_sent:
                        pass
                    else:
                        molmo = _molmo_from_tool_message(chunk)
                        if molmo is not None:
                            molmo_sent.add(tid)
                            molmo_out = _molmo_payload_for_sse(molmo, upload_dims)
                            line = json.dumps({"molmo_result": molmo_out})
                            yield f"data: {line}\n\n"
            if isinstance(chunk, (ToolMessage, ToolMessageChunk)):
                continue
            text = _token_text(chunk)
            if text:
                line = json.dumps({"token": text})
                yield f"data: {line}\n\n"
    except Exception as e:
        err = json.dumps({"error": str(e)})
        yield f"data: {err}\n\n"
        return
    yield f"data: {json.dumps({'done': True})}\n\n"


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    return StreamingResponse(
        _stream_body(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/chat/stream-with-image")
async def chat_stream_with_image(
    file: UploadFile = File(...),
    messages_json: str = Form(...),
):
    try:
        parsed = json.loads(messages_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"messages_json must be valid JSON: {e!s}") from e
    if not isinstance(parsed, list):
        raise HTTPException(status_code=400, detail="messages_json must be a JSON array")
    try:
        req = ChatRequest(messages=[Msg.model_validate(m) for m in parsed])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid message payload: {e!s}") from e

    uploaded = await _save_uploaded_image(file)

    async def stream() -> AsyncIterator[str]:
        try:
            async for line in _stream_body_with_image(req, str(uploaded)):
                yield line
        finally:
            try:
                uploaded.unlink(missing_ok=True)
            except OSError:
                pass

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


class OakCaptureResponse(BaseModel):
    capture_id: str
    image_url: str
    width: int
    height: int


@app.post("/oak/capture", response_model=OakCaptureResponse)
async def oak_capture(discard_capture_id: str | None = None) -> OakCaptureResponse:
    """Grab one synchronized RGB+depth pair from the OAK-D and cache it for later use.

    Pass ``discard_capture_id`` (query string) to evict a previous capture in the same step
    (frontend Retake flow). Captures are consumed by ``/chat/stream-with-oak-capture-id``;
    until then, the JPEG can be fetched at the returned ``image_url``.
    """
    if discard_capture_id:
        _oak_evict_capture(discard_capture_id)

    try:
        from oak_camera import OakCamera, OakCameraError
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"OAK capture unavailable (depthai not importable): {e!s}",
        ) from e

    try:
        cam = OakCamera.get()
        rgb_path, depth, w, h = cam.capture(_molmo_upload_dir())
    except OakCameraError as e:
        raise HTTPException(status_code=503, detail=f"OAK capture failed: {e!s}") from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"OAK capture failed: {e!s}") from e

    capture_id = rgb_path.stem
    with _OAK_CAPTURES_LOCK:
        _OAK_CAPTURES[capture_id] = {"path": rgb_path, "depth": depth, "w": w, "h": h}
    return OakCaptureResponse(
        capture_id=capture_id,
        image_url=f"/oak/captures/{rgb_path.name}",
        width=w,
        height=h,
    )


@app.delete("/oak/capture/{capture_id}")
def oak_capture_delete(capture_id: str) -> dict[str, bool]:
    """Evict a cached OAK capture (used by the modal on cancel)."""
    _oak_evict_capture(capture_id)
    return {"ok": True}


class OakChatRequest(BaseModel):
    messages: list[Msg] = Field(min_length=1)
    capture_id: str = Field(min_length=1)


@app.post("/chat/stream-with-oak-capture-id")
async def chat_stream_with_oak_capture_id(req: OakChatRequest):
    """Stream the agent over a previously captured OAK-D frame.

    The frame and its aligned depth map must already be in the cache from a prior
    ``/oak/capture`` call. On stream completion (or error) the cache entry is evicted and
    the JPEG is unlinked.
    """
    with _OAK_CAPTURES_LOCK:
        entry = _OAK_CAPTURES.get(req.capture_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="unknown or expired capture_id")
    rgb_path: Path = entry["path"]
    depth = entry["depth"]

    try:
        from oak_camera import sample_depth
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"OAK capture unavailable (depthai not importable): {e!s}",
        ) from e

    def depth_lookup(x_norm: float, y_norm: float) -> float | None:
        return sample_depth(depth, x_norm, y_norm)

    chat_req = ChatRequest(messages=req.messages)

    async def stream() -> AsyncIterator[str]:
        try:
            async for line in _stream_body_with_image(
                chat_req, str(rgb_path), depth_lookup=depth_lookup
            ):
                yield line
        finally:
            _oak_evict_capture(req.capture_id)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/oak/captures/{name}")
def oak_capture_image(name: str):
    """Serve a cached OAK-D JPEG so the modal can preview it before the user sends."""
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="invalid name")
    upload_dir = _molmo_upload_dir().resolve()
    p = (upload_dir / name).resolve()
    try:
        p.relative_to(upload_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid name") from None
    if not p.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(p, media_type="image/jpeg")
