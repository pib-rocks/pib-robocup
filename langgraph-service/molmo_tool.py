"""LangChain tool: call local MolmoPoint `POST /point` (image_path + prompt -> points)."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import httpx
from langchain_core.tools import tool

MOLMO_BASE_URL = os.environ.get("MOLMO_BASE_URL", "http://127.0.0.1:8010").rstrip("/")
MOLMO_TIMEOUT = float(os.environ.get("MOLMO_TIMEOUT_SECONDS", "120"))


def _allowed_prefix() -> str | None:
    raw = os.environ.get("MOLMO_ALLOWED_PATH_PREFIX", "").strip()
    return raw or None


def _validate_image_path(image_path: str) -> Path | str:
    """Return resolved Path if valid, else an error string for the model."""
    try:
        p = Path(image_path).expanduser().resolve(strict=False)
    except (OSError, ValueError) as e:
        return f"Invalid image_path: {e!s}"

    # Symlink/traversal: resolution handles most cases; require absolute-looking input after expand
    prefix = _allowed_prefix()
    if prefix is not None:
        base = Path(prefix).expanduser().resolve(strict=False)
        pr = p.resolve(strict=False)
        try:
            pr.relative_to(base.resolve(strict=False))
        except ValueError:
            return (
                f"image_path must be under MOLMO_ALLOWED_PATH_PREFIX={prefix!r} "
                f"(got {pr!s})"
            )

    if not p.is_file():
        return f"Not a file or not found: {p}"

    return p


def _png_header_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w = int.from_bytes(data[16:20], "big")
    h = int.from_bytes(data[20:24], "big")
    if w > 0 and h > 0:
        return (w, h)
    return None


def _jpeg_sof_size(data: bytes) -> tuple[int, int] | None:
    """Width/height from first SOF0/SOF1/SOF2 segment (baseline / extended / progressive)."""
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return None
    i = 2
    n = len(data)
    while i + 3 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m in (0xD8,):
            i += 2
            continue
        if m in (0xD9, 0xDA):
            break
        seglen = int.from_bytes(data[i + 2 : i + 4], "big")
        if seglen < 2 or i + seglen > n:
            break
        if m in (0xC0, 0xC1, 0xC2) and seglen >= 8:
            h = int.from_bytes(data[i + 5 : i + 7], "big")
            w = int.from_bytes(data[i + 7 : i + 9], "big")
            if w > 0 and h > 0:
                return (w, h)
        i += 2 + seglen
    return None


def _dimensions_from_file_header(path: Path, max_read: int = 512_000) -> tuple[int, int] | None:
    try:
        n = min(path.stat().st_size, max_read)
        if n < 24:
            return None
        with path.open("rb") as f:
            head = f.read(n)
    except OSError:
        return None
    p = _png_header_size(head)
    if p:
        return p
    return _jpeg_sof_size(head)


def get_image_dimensions(path: Path) -> tuple[int, int] | None:
    """Read (width, height) for the uploaded image. PIL first; JPEG/PNG header parse if that fails."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.load()
            w, h = im.size
        if w > 0 and h > 0:
            return (int(w), int(h))
    except Exception:
        pass
    return _dimensions_from_file_header(path)


def enrich_molmo_result_for_client(
    d: dict, dims: tuple[int, int] | None
) -> dict:
    """Attach image size and normalize point coords to 0–1 for SSE/UI when we know pixel dimensions.

    Shallow-copies the dict and point objects so downstream mutations do not affect LangGraph state.
    Per-point ``depth_m`` (if attached upstream by the Molmo tool) is preserved.
    """
    if dims is None:
        return d
    wi, hi = dims
    out = dict(d)
    out["image_width"] = wi
    out["image_height"] = hi
    raw_pts = d.get("points")
    if not isinstance(raw_pts, list) or not raw_pts:
        return out
    new_pts: list[object] = []
    for p in raw_pts:
        if isinstance(p, dict):
            new_pts.append(dict(p))
        else:
            new_pts.append(p)
    _normalize_molmo_points_in_place(new_pts, wi, hi)
    out["points"] = new_pts
    return out


def _normalize_molmo_points_in_place(points: list[object], width: int, height: int) -> None:
    """MolmoPoint returns x,y in **pixel** space for the input image; UI expects 0..1.

    If either coordinate is > 1, treat as pixels and scale by (width, height). Purely
    normalized outputs (0..1) are left unchanged.
    """
    if width <= 0 or height <= 0:
        return
    w, h = float(width), float(height)
    for p in points:
        if not isinstance(p, dict):
            continue
        try:
            x = float(p.get("x", 0.0))
            y = float(p.get("y", 0.0))
        except (TypeError, ValueError):
            continue
        if x > 1.0 or y > 1.0:
            p["x"] = x / w
            p["y"] = y / h


def call_molmo_point(image_path: str, prompt: str) -> dict | str:
    """POST /point. Returns JSON dict on success, or error str."""
    v = _validate_image_path(image_path)
    if isinstance(v, str):
        return v

    body = {
        "image_path": str(v),
        "prompt": prompt,
    }
    try:
        with httpx.Client(timeout=MOLMO_TIMEOUT) as client:
            r = client.post(f"{MOLMO_BASE_URL}/point", json=body)
    except httpx.RequestError as e:
        return f"MolmoPoint unreachable ({MOLMO_BASE_URL}): {e!s}"

    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        return f"MolmoPoint error HTTP {r.status_code}: {detail!s}"

    try:
        out = r.json()
    except Exception as e:
        return f"Invalid JSON from MolmoPoint: {e!s}"
    if not isinstance(out, dict):
        return out
    if isinstance(out.get("points"), list) and out["points"]:
        dims = get_image_dimensions(v)
        if dims is not None:
            wi, hi = dims
            out["image_width"] = wi
            out["image_height"] = hi
            _normalize_molmo_points_in_place(out["points"], wi, hi)
    return out


def molmo_result_dict_for_json(out: dict) -> dict:
    """Payload embedded in tool return strings and optional API responses — keep fields in sync."""
    d: dict = {
        "points": out.get("points", []),
        "generated_text": out.get("generated_text", ""),
        "device": out.get("device", ""),
        "model_id": out.get("model_id", ""),
    }
    if "image_width" in out and out["image_width"] is not None:
        d["image_width"] = out["image_width"]
    if "image_height" in out and out["image_height"] is not None:
        d["image_height"] = out["image_height"]
    return d


def attach_depth_to_tool_payload(
    out: dict,
    depth_lookup: Callable[[float, float], float | None] | None,
) -> dict:
    """Mutate `out["points"]` to attach `depth_m` per point. Used inside the Molmo tool when an
    OAK capture's depth map is available, so the agent's tool message also carries distance.
    """
    if depth_lookup is None:
        return out
    pts = out.get("points")
    if not isinstance(pts, list):
        return out
    wi = out.get("image_width")
    hi = out.get("image_height")
    have_dims = isinstance(wi, int) and isinstance(hi, int) and wi > 0 and hi > 0
    for p in pts:
        if not isinstance(p, dict):
            continue
        try:
            xv = float(p.get("x", 0.0))
            yv = float(p.get("y", 0.0))
        except (TypeError, ValueError):
            p["depth_m"] = None
            continue
        # Points may still be in pixel space at this stage (call_molmo_point normalizes only when dims are known)
        xn = xv / wi if have_dims and xv > 1.0 else xv
        yn = yv / hi if have_dims and yv > 1.0 else yv
        try:
            dm = depth_lookup(xn, yn)
        except Exception:
            dm = None
        p["depth_m"] = round(dm, 2) if isinstance(dm, (int, float)) else None
    return out


@tool
def molmo_point_localize(image_path: str, prompt: str) -> str:
    """Run MolmoPoint on a host-local image file to locate objects from a text description.

    Use this when the user needs positions of objects in an image. The file must
    exist on the same machine as MolmoPoint; pass an absolute or resolvable path (e.g. a
    camera frame under /data). Returns JSON with a ``points`` list: object_id, image_index, x, y, distance.
    If MolmoPoint is not running or the path is not allowed, the string explains the error.
    """
    out = call_molmo_point(image_path, prompt)
    if isinstance(out, str):
        return out
    return json.dumps(molmo_result_dict_for_json(out), ensure_ascii=False)


def molmo_service_reachable() -> bool:
    """True if GET /health returns 200 (MolmoPoint model is ready to serve /point)."""
    try:
        with httpx.Client(timeout=2.0) as c:
            r = c.get(f"{MOLMO_BASE_URL}/health")
            return r.status_code == 200
    except httpx.RequestError:
        return False
