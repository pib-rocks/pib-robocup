"""Agent runtime via langchain.agents.create_agent (LangGraph under the hood; tools optional)."""

import json
import os
import re
from collections.abc import Callable, Sequence

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from molmo_tool import (
    attach_depth_to_tool_payload,
    call_molmo_point,
    molmo_point_localize,
    molmo_result_dict_for_json,
)
from system_prompt import SYSTEM_PROMPT


def get_llm() -> ChatOpenAI:
    base_url = os.environ.get("GEMMA_BASE_URL", "http://127.0.0.1:8080/v1")
    model = os.environ.get("GEMMA_OPENAI_MODEL", "gpt-3.5-turbo")
    api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.7,
    )


def _uploaded_image_system(uploaded_image_path: str) -> SystemMessage:
    return SystemMessage(
        content=(
            "A user image is attached for this turn and stored on the server at "
            f"{uploaded_image_path}. For Molmo (pointing / localization) you MUST use tool "
            "`molmo_point_localize_uploaded` only. Do not call any other tool that needs an image "
            "file path, and do not invent a path like `input_file_0.png` — the server already "
            "has the file. When needed, call `molmo_point_localize_uploaded` with a short prompt "
            "naming the object/region to locate. Do not invent coordinates or tool outputs."
        )
    )


def prepare_for_model(
    msgs: Sequence[BaseMessage], uploaded_image_path: str | None = None
) -> list[BaseMessage]:
    """Prepend a per-turn system message describing any uploaded image.

    The agent's persistent system prompt is set on ``create_agent`` via
    ``system_prompt=SYSTEM_PROMPT``; this function only adds the transient
    note about the uploaded file (when present).
    """
    out: list[BaseMessage] = []
    if uploaded_image_path:
        out.append(_uploaded_image_system(uploaded_image_path))
    out.extend(msgs)
    return out


def _normalize_molmo_upload_prompt(raw: str) -> str:
    """Turn user-style questions into MolmoPoint-friendly pointing phrasing.

    MolmoPoint is much more likely to emit point markup when the text asks to *point* at
    something, not a generic "where is …" question. The model may still pass "where is …";
    we fix that here so the HTTP call is always in a good shape.
    """
    s = (raw or "").strip()
    if not s:
        return "Point to the object the user is asking about in the image."

    low = s.lower()
    if low.startswith("point to ") or low.startswith("point at "):
        return s[0:1].upper() + s[1:] if s else s

    m = re.match(
        r"^where\s+is\s+(?:the\s+)?(.+?)\s*[\.\?!]*$", s, re.IGNORECASE | re.DOTALL
    )
    if m:
        target = m.group(1).strip()
        if target.lower().startswith("the "):
            return f"Point to {target}"
        return f"Point to the {target}"

    m = re.match(
        r"^where\s+are\s+(?:the\s+)?(.+?)\s*[\.\?!]*$", s, re.IGNORECASE | re.DOTALL
    )
    if m:
        target = m.group(1).strip()
        if target.lower().startswith("the "):
            return f"Point to {target}"
        return f"Point to the {target}"

    if not low.startswith("point "):
        if " " not in s and s.replace("-", "").isalnum():
            return f"Point to the {s}"
        return f"Point to {s}"
    return s


def _build_uploaded_image_tool(
    uploaded_image_path: str,
    depth_lookup: Callable[[float, float], float | None] | None = None,
):
    @tool("molmo_point_localize_uploaded")
    def molmo_point_localize_uploaded(prompt: str) -> str:
        """Run MolmoPoint on the image uploaded in this chat turn (server-side path; no `image_path` arg).

        **What to pass in `prompt`:** a short description of *what* to point at. The server
        rewrites common question forms into *pointing* phrasing before calling MolmoPoint.

        **Do not** rely on "where is …" style questions in this field. Prefer one of:
        - ``Point to the red cup``
        - ``Point to the left door handle``
        - ``the mug on the table``  (ok — will be turned into a ``Point to …`` string)

        **Avoid:** long chatty instructions; a noun phrase or a single ``Point to …`` line works best.

        When the image was captured from the OAK-D depth camera, each point in the result also
        carries a ``depth_m`` field giving the distance to the object in meters (or ``null`` for
        depth holes / out-of-range pixels).
        """
        effective = _normalize_molmo_upload_prompt(prompt)
        out = call_molmo_point(uploaded_image_path, effective)
        if isinstance(out, str):
            return out
        if depth_lookup is not None:
            attach_depth_to_tool_payload(out, depth_lookup)
        return json.dumps(molmo_result_dict_for_json(out), ensure_ascii=False)

    return molmo_point_localize_uploaded


def build_agent(
    uploaded_image_path: str | None = None,
    depth_lookup: Callable[[float, float], float | None] | None = None,
):
    """Return a compiled agent graph: Gemma + MolmoPoint localization tool (HTTP to :8010).

    If ``depth_lookup`` is provided (only meaningful with an uploaded image from the OAK-D),
    the Molmo tool annotates each returned point with a ``depth_m`` distance in meters.
    """
    if uploaded_image_path:
        tools = [_build_uploaded_image_tool(uploaded_image_path, depth_lookup=depth_lookup)]
    else:
        tools = [molmo_point_localize]
    return create_agent(
        get_llm(),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
