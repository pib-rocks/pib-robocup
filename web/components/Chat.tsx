"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { WebcamCaptureButton } from "@/components/WebcamCaptureButton";
import { OakCaptureButton } from "@/components/OakCaptureButton";
import { ImageWithPointOverlay, type MolmoPoint } from "@/components/ImageWithPointOverlay";

type Role = "user" | "assistant" | "system";

type Msg = {
  role: Role;
  content: string;
  imageUrl?: string;
  imageName?: string;
  /** Tool calls produced during this assistant turn (assistant messages only). */
  toolCalls?: ToolCallLogEntry[];
  /** MolmoPoint tool results produced during this assistant turn (assistant messages only). */
  molmoResults?: MolmoChatResult[];
  /** Blob URL of the image that backs this turn's overlay (assistant messages only). */
  contextImageUrl?: string;
};
type ChatPayloadMsg = { role: Role; content: string };

type MolmoChatResult = {
  points?: MolmoPoint[];
  generated_text?: string;
  device?: string;
  model_id?: string;
  error?: string;
  image_width?: number;
  image_height?: number;
};

type ToolCallLogEntry = {
  clientKey: string;
  id?: string;
  name?: string;
  args?: string;
  result?: string;
};

const API_BASE =
  process.env.NEXT_PUBLIC_LANGGRAPH_API_URL || "http://127.0.0.1:8008";

function formatPointCell(v: number) {
  if (Number.isNaN(v)) return "—";
  return v.toFixed(4);
}

function hasImageDims(m: MolmoChatResult) {
  const w = Number(m.image_width);
  const h = Number(m.image_height);
  return Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0;
}

/** Molmo may return 0–1 or pixels. Prefer API values; derive missing 0–1 or px from image size when present. */
function molmoPointCells(p: MolmoPoint, m: MolmoChatResult) {
  const rawX = Number(p.x);
  const rawY = Number(p.y);
  const looksPixel = rawX > 1 || rawY > 1;
  const dims = hasImageDims(m);
  const iw = Number(m.image_width);
  const ih = Number(m.image_height);

  if (!looksPixel) {
    const x01 = rawX;
    const y01 = rawY;
    return {
      x01,
      y01,
      xPx: dims ? x01 * iw : Number.NaN,
      yPx: dims ? y01 * ih : Number.NaN,
    };
  }
  if (dims) {
    return {
      x01: rawX / iw,
      y01: rawY / ih,
      xPx: rawX,
      yPx: rawY,
    };
  }
  return {
    x01: Number.NaN,
    y01: Number.NaN,
    xPx: rawX,
    yPx: rawY,
  };
}

export function Chat() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  /** capture_id of an OAK-D frame already cached on the backend, when imageFile came from there. */
  const [oakCaptureId, setOakCaptureId] = useState<string | null>(null);
  const [composerImagePreviewUrl, setComposerImagePreviewUrl] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  /** Indices of assistant messages whose Molmo side overlay is currently expanded. */
  const [overlayExpandedIdx, setOverlayExpandedIdx] = useState<Set<number>>(() => new Set());
  const listRef = useRef<HTMLDivElement>(null);
  const msgImageUrlsRef = useRef<string[]>([]);
  const nextToolKey = useRef(0);

  function toggleOverlay(idx: number) {
    setOverlayExpandedIdx((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }

  function pointsFromResults(rs: MolmoChatResult[] | undefined): MolmoPoint[] {
    const out: MolmoPoint[] = [];
    if (!rs) return out;
    for (const r of rs) {
      if (r.error) continue;
      for (const pt of r.points ?? []) out.push(pt);
    }
    return out;
  }

  /** Webcam, file picker, or "Remove image" — anything that's not an OAK capture clears the id. */
  const setNonOakImage = useCallback((file: File | null) => {
    setImageFile(file);
    setOakCaptureId(null);
  }, []);

  /** OAK button delivers a (File, capture_id) pair; both are tracked together. */
  const setOakImage = useCallback((file: File, captureId: string) => {
    setImageFile(file);
    setOakCaptureId(captureId);
  }, []);

  useEffect(() => {
    if (!imageFile) {
      setComposerImagePreviewUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return null;
      });
      return;
    }
    const u = URL.createObjectURL(imageFile);
    setComposerImagePreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return u;
    });
    return () => {
      URL.revokeObjectURL(u);
    };
  }, [imageFile]);

  useEffect(
    () => () => {
      for (const u of msgImageUrlsRef.current) URL.revokeObjectURL(u);
      if (composerImagePreviewUrl) URL.revokeObjectURL(composerImagePreviewUrl);
    },
    [composerImagePreviewUrl]
  );

  function scrollToBottom() {
    requestAnimationFrame(() => {
      listRef.current?.lastElementChild?.scrollIntoView({ behavior: "smooth" });
    });
  }

  async function consumeAgentStream(res: Response, assistantIdx: number) {
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || res.statusText);
    }

    const reader = res.body?.getReader();
    if (!reader) throw new Error("No response body");

    const updateAssistant = (fn: (m: Msg) => Msg) =>
      setMessages((prev) => {
        if (prev[assistantIdx]?.role !== "assistant") return prev;
        const next = [...prev];
        next[assistantIdx] = fn(next[assistantIdx]);
        return next;
      });

    const mergeToolCall = (
      list: ToolCallLogEntry[] | undefined,
      tc: { id?: string; name?: string; args?: string | null },
    ): ToolCallLogEntry[] => {
      const prev = list ?? [];
      if (tc.id) {
        const idx = prev.findIndex((e) => e.id === tc.id);
        if (idx >= 0) {
          const next = [...prev];
          const cur = next[idx]!;
          next[idx] = { ...cur, name: tc.name ?? cur.name, args: tc.args ?? cur.args };
          return next;
        }
        return [
          ...prev,
          { clientKey: `tc-${tc.id}`, id: tc.id, name: tc.name, args: tc.args ?? undefined },
        ];
      }
      const k = `tmp-${nextToolKey.current++}`;
      return [...prev, { clientKey: k, name: tc.name, args: tc.args ?? undefined }];
    };

    const mergeToolResult = (
      list: ToolCallLogEntry[] | undefined,
      tr: { id?: string; name?: string; content?: string },
    ): ToolCallLogEntry[] => {
      const prev = list ?? [];
      if (tr.id) {
        const idx = prev.findIndex((e) => e.id === tr.id);
        if (idx >= 0) {
          const next = [...prev];
          const cur = next[idx]!;
          next[idx] = { ...cur, name: tr.name ?? cur.name, result: tr.content ?? cur.result };
          return next;
        }
        return [
          ...prev,
          { clientKey: `tr-${tr.id}`, id: tr.id, name: tr.name, result: tr.content },
        ];
      }
      const k = `tr-tmp-${nextToolKey.current++}`;
      return [...prev, { clientKey: k, name: tr.name, result: tr.content }];
    };

    const dec = new TextDecoder();
    let acc = "";
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += dec.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (!payload) continue;
        let data: {
          token?: string;
          error?: string;
          done?: boolean;
          molmo_result?: MolmoChatResult;
          tool_call?: { id?: string; name?: string; args?: string | null };
          tool_result?: { id?: string; name?: string; content?: string };
        };
        try {
          data = JSON.parse(payload);
        } catch {
          continue;
        }
        if (data.error) throw new Error(data.error);
        if (data.molmo_result) {
          const mr = data.molmo_result;
          updateAssistant((m) => ({
            ...m,
            molmoResults: [...(m.molmoResults ?? []), mr],
          }));
          scrollToBottom();
          continue;
        }
        if (data.tool_call) {
          const tc = data.tool_call;
          updateAssistant((m) => ({ ...m, toolCalls: mergeToolCall(m.toolCalls, tc) }));
          scrollToBottom();
          continue;
        }
        if (data.tool_result) {
          const tr = data.tool_result;
          updateAssistant((m) => ({ ...m, toolCalls: mergeToolResult(m.toolCalls, tr) }));
          scrollToBottom();
          continue;
        }
        if (data.token) {
          acc += data.token;
          updateAssistant((m) => ({ ...m, content: acc }));
          scrollToBottom();
        }
      }
    }
  }

  function rollbackOnError(err: unknown) {
    setError(err instanceof Error ? err.message : "Request failed");
    setMessages((prev) => {
      if (prev.length < 2) return prev;
      if (prev[prev.length - 1].role === "assistant" && !prev[prev.length - 1].content) {
        return prev.slice(0, -1);
      }
      return prev;
    });
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if ((!text && !imageFile) || sending) return;

    setError(null);
    setInput("");

    const sentImage = imageFile;
    const sentOakCaptureId = oakCaptureId;
    setImageFile(null);
    setOakCaptureId(null);
    const sentImageUrl = sentImage ? URL.createObjectURL(sentImage) : undefined;
    if (sentImageUrl) {
      msgImageUrlsRef.current.push(sentImageUrl);
    }
    const userMsg: Msg = {
      role: "user",
      content: text || "Uploaded an image.",
      imageUrl: sentImageUrl,
      imageName: sentImage?.name,
    };
    const history: Msg[] = [...messages, userMsg];
    const historyPayload: ChatPayloadMsg[] = history.map(({ role, content }) => ({
      role,
      content,
    }));
    const assistantIdx = history.length;
    const assistantPlaceholder: Msg = {
      role: "assistant",
      content: "",
      contextImageUrl: sentImageUrl,
    };
    setMessages([...history, assistantPlaceholder]);
    setSending(true);
    scrollToBottom();

    try {
      let res: Response;
      if (sentOakCaptureId) {
        res = await fetch(`${API_BASE}/chat/stream-with-oak-capture-id`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: historyPayload,
            capture_id: sentOakCaptureId,
          }),
        });
        if (res.status === 404) {
          throw new Error(
            "OAK capture expired or not found on the backend. Take a new capture and try again."
          );
        }
      } else if (sentImage) {
        const fd = new FormData();
        fd.append("file", sentImage);
        fd.append("messages_json", JSON.stringify(historyPayload));
        res = await fetch(`${API_BASE}/chat/stream-with-image`, {
          method: "POST",
          body: fd,
        });
        if (res.status === 404) {
          throw new Error(
            "Image chat endpoint is unavailable on the running backend. Restart langgraph-service to load /chat/stream-with-image."
          );
        }
      } else {
        res = await fetch(`${API_BASE}/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: historyPayload }),
        });
      }
      await consumeAgentStream(res, assistantIdx);
    } catch (err) {
      rollbackOnError(err);
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="mx-auto flex h-[min(90vh,720px)] max-w-4xl flex-col gap-3 px-3 py-6">
      <header>
        <h1 className="text-xl font-semibold tracking-tight">Gemma 4 (local)</h1>
        <p className="text-sm text-foreground/70">
          LangGraph backend: <code className="text-xs opacity-80">{API_BASE}</code>
        </p>
      </header>

      <div
        ref={listRef}
        className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto rounded-lg border border-foreground/10 p-3"
      >
        {messages.length === 0 && !error && (
          <p className="text-sm text-foreground/60">Send a message to start.</p>
        )}
        {messages.map((m, i) => {
          if (m.role === "user") {
            return (
              <div
                key={i}
                className="ml-8 self-end rounded-lg bg-foreground/10 px-3 py-2"
              >
                <div className="text-xs font-medium text-foreground/50">user</div>
                {m.imageUrl && (
                  <div className="mb-2 space-y-1">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={m.imageUrl}
                      alt={m.imageName || "uploaded image"}
                      className="max-h-48 rounded border border-foreground/15"
                    />
                    {m.imageName && (
                      <div className="text-[11px] text-foreground/55">{m.imageName}</div>
                    )}
                  </div>
                )}
                <div className="whitespace-pre-wrap text-sm leading-relaxed">{m.content}</div>
              </div>
            );
          }
          const points = pointsFromResults(m.molmoResults);
          const canMolmoSideOverlay = m.contextImageUrl != null && points.length > 0;
          const showMolmoSideLayout = canMolmoSideOverlay && overlayExpandedIdx.has(i);
          return (
            <div
              key={i}
              className="w-full max-w-[100%] self-start rounded-lg bg-foreground/5 px-3 py-2"
            >
              <div className="text-xs font-medium text-foreground/50">assistant</div>
              {canMolmoSideOverlay && (
                <label className="mt-1.5 flex cursor-pointer items-center gap-2 text-xs text-foreground/70">
                  <input
                    type="checkbox"
                    className="size-3.5 rounded border-foreground/30 text-foreground accent-foreground"
                    checked={overlayExpandedIdx.has(i)}
                    onChange={() => toggleOverlay(i)}
                  />
                  <span>Show Molmo point map beside reply</span>
                </label>
              )}
              {showMolmoSideLayout ? (
                <div className="mt-1 flex min-w-0 flex-col gap-3 sm:flex-row sm:items-start sm:gap-4">
                  <div className="min-w-0 flex-1 whitespace-pre-wrap text-sm leading-relaxed">
                    {m.content}
                  </div>
                  <div className="min-w-0 w-full sm:w-auto sm:max-w-[45%] sm:flex-none">
                    <p className="mb-1.5 text-[11px] text-foreground/50">
                      Molmo detections (from tool result, not Gemma)
                    </p>
                    <ImageWithPointOverlay
                      imageUrl={m.contextImageUrl!}
                      points={points}
                      alt="User image with Molmo point overlay"
                    />
                  </div>
                </div>
              ) : (
                <div
                  className={
                    canMolmoSideOverlay
                      ? "mt-1 whitespace-pre-wrap text-sm leading-relaxed"
                      : "whitespace-pre-wrap text-sm leading-relaxed"
                  }
                >
                  {m.content}
                </div>
              )}
              {m.toolCalls && m.toolCalls.length > 0 && (
                <div className="mt-3 space-y-2 rounded-lg border border-foreground/15 bg-background/40 p-3">
                  <h3 className="text-xs font-medium text-foreground/80">Tool calls</h3>
                  <ul className="space-y-2 text-xs text-foreground/80">
                    {m.toolCalls.map((t) => (
                      <li key={t.clientKey} className="rounded border border-foreground/10 bg-background/60 p-2">
                        <div className="font-mono text-[11px] text-foreground/55">
                          {t.name ? <span className="text-foreground/80">{t.name}</span> : "(unnamed tool)"}
                          {t.id ? <span className="text-foreground/45"> · id {t.id}</span> : null}
                        </div>
                        {t.args ? (
                          <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap break-words text-foreground/75">
                            args: {t.args}
                          </pre>
                        ) : null}
                        {t.result ? (
                          <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap break-words text-foreground/75">
                            result: {t.result}
                          </pre>
                        ) : null}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {m.molmoResults && m.molmoResults.length > 0 && (
                <div className="mt-3 space-y-3 rounded-lg border border-foreground/15 bg-background/40 p-3">
                  <h3 className="text-xs font-medium text-foreground/80">
                    MolmoPoint (tool) — 0–1 in image space; not from Gemma
                  </h3>
                  {m.molmoResults.map((mr, k) => (
                    <div key={k} className="space-y-1.5 text-sm">
                      {mr.error && <p className="text-red-300">{mr.error}</p>}
                      {(mr.model_id || mr.device) && (
                        <p className="text-xs text-foreground/60">
                          {mr.model_id ? (
                            <>
                              <span className="text-foreground/80">Model:</span>{" "}
                              <code className="break-all">{mr.model_id}</code>
                              {mr.device ? " · " : null}
                            </>
                          ) : null}
                          {mr.device ? (
                            <>
                              <span className="text-foreground/80">Device:</span>{" "}
                              <code>{mr.device}</code>
                            </>
                          ) : null}
                        </p>
                      )}
                      {mr.points && mr.points.length > 0 && (
                        <div className="overflow-x-auto rounded border border-foreground/10">
                          <table className="w-full min-w-[18rem] text-left text-xs">
                            <thead>
                              <tr className="border-b border-foreground/10 text-foreground/50">
                                <th className="p-1.5 pr-2 font-medium">#</th>
                                <th className="p-1.5 pr-2 font-medium">object_id</th>
                                <th className="p-1.5 pr-2 font-medium">image</th>
                                <th className="p-1.5 pr-2 font-medium">x (0–1)</th>
                                <th className="p-1.5 pr-2 font-medium">y (0–1)</th>
                                <th className="p-1.5 pr-2 font-medium" title="Pixel x: from API in pixel space, or derived from 0–1 × width when image size is known">
                                  x (px)
                                </th>
                                <th className="p-1.5 pr-2 font-medium" title="Pixel y: from API in pixel space, or derived from 0–1 × height when image size is known">
                                  y (px)
                                </th>
                                <th className="p-1.5 font-medium" title="OAK-D depth at point (median over 7×7 ROI). — when not available or out-of-range.">
                                  distance (m)
                                </th>
                              </tr>
                            </thead>
                            <tbody>
                              {mr.points.map((p, j) => {
                                const c = molmoPointCells(p, mr);
                                return (
                                  <tr key={j} className="border-b border-foreground/5 last:border-0">
                                    <td className="p-1.5 pr-2 tabular-nums text-foreground/80">{j + 1}</td>
                                    <td className="p-1.5 pr-2 tabular-nums">{p.object_id}</td>
                                    <td className="p-1.5 pr-2 tabular-nums">{p.image_index}</td>
                                    <td className="p-1.5 pr-2 tabular-nums">{formatPointCell(c.x01)}</td>
                                    <td className="p-1.5 pr-2 tabular-nums">{formatPointCell(c.y01)}</td>
                                    <td className="p-1.5 pr-2 tabular-nums text-foreground/70">
                                      {Number.isNaN(c.xPx) ? "—" : c.xPx.toFixed(1)}
                                    </td>
                                    <td className="p-1.5 pr-2 tabular-nums text-foreground/70">
                                      {Number.isNaN(c.yPx) ? "—" : c.yPx.toFixed(1)}
                                    </td>
                                    <td className="p-1.5 tabular-nums text-foreground/80">
                                      {typeof p.depth_m === "number" ? p.depth_m.toFixed(2) : "—"}
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      )}
                      {mr.generated_text && (
                        <p className="whitespace-pre-wrap break-words text-xs text-foreground/70">
                          <span className="text-foreground/50">raw text: </span>
                          {mr.generated_text}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
        {error && (
          <div className="rounded border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
            {error}
          </div>
        )}
      </div>

      <form onSubmit={onSubmit} className="space-y-2">
        {composerImagePreviewUrl && (
          <div className="inline-flex items-start gap-2 rounded border border-foreground/15 bg-foreground/5 p-2">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={composerImagePreviewUrl}
              alt={imageFile?.name || "Selected image"}
              className="max-h-20 rounded border border-foreground/10"
            />
            <div className="space-y-1">
              <p className="text-xs text-foreground/70">{imageFile?.name || "selected image"}</p>
              <button
                type="button"
                className="rounded border border-foreground/20 px-2 py-1 text-xs text-foreground/80 hover:bg-foreground/5"
                onClick={() => setNonOakImage(null)}
                disabled={sending}
              >
                Remove image
              </button>
            </div>
          </div>
        )}
        <div className="flex flex-wrap items-center gap-2">
          <WebcamCaptureButton
            onCapture={setNonOakImage}
            disabled={sending}
          />
          <OakCaptureButton
            apiBase={API_BASE}
            onCapture={setOakImage}
            disabled={sending}
          />
          <label className="shrink-0 cursor-pointer text-sm text-foreground/55 underline decoration-foreground/25 underline-offset-2 hover:text-foreground/80">
            choose file
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
              onChange={(e) => setNonOakImage(e.target.files?.[0] ?? null)}
              disabled={sending}
            />
          </label>
        <input
          className="min-w-0 flex-1 rounded-md border border-foreground/15 bg-background px-3 py-2 text-sm outline-none ring-0 focus:border-foreground/30"
          placeholder="Message (optionally attach image)…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={sending}
        />
        <button
          type="submit"
          disabled={sending || (!input.trim() && !imageFile)}
          className="rounded-md bg-foreground px-4 py-2 text-sm font-medium text-background disabled:opacity-40"
        >
          {sending ? "…" : "Send"}
        </button>
        </div>
      </form>
    </div>
  );
}
