"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type OakCapture = {
  capture_id: string;
  image_url: string;
  width: number;
  height: number;
};

type Props = {
  apiBase: string;
  onCapture: (file: File, captureId: string) => void;
  disabled?: boolean;
};

export function OakCaptureButton({ apiBase, onCapture, disabled }: Props) {
  const [open, setOpen] = useState(false);
  const [capture, setCapture] = useState<OakCapture | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewBlob, setPreviewBlob] = useState<Blob | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const previewUrlRef = useRef<string | null>(null);
  useEffect(() => {
    previewUrlRef.current = previewUrl;
  }, [previewUrl]);

  const releasePreview = useCallback(() => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setPreviewUrl(null);
    setPreviewBlob(null);
  }, []);

  const close = useCallback(() => {
    setOpen(false);
    setErr(null);
    releasePreview();
    setCapture(null);
  }, [releasePreview]);

  const fetchCapture = useCallback(
    async (discardCaptureId?: string) => {
      setBusy(true);
      setErr(null);
      try {
        const url = new URL(`${apiBase}/oak/capture`);
        if (discardCaptureId) url.searchParams.set("discard_capture_id", discardCaptureId);
        const res = await fetch(url.toString(), { method: "POST" });
        if (!res.ok) {
          const detail = await res.text();
          throw new Error(detail || `OAK capture failed (HTTP ${res.status})`);
        }
        const data = (await res.json()) as OakCapture;
        const imgRes = await fetch(`${apiBase}${data.image_url}`);
        if (!imgRes.ok) throw new Error(`Could not load capture image (HTTP ${imgRes.status})`);
        const blob = await imgRes.blob();
        const url2 = URL.createObjectURL(blob);
        releasePreview();
        previewUrlRef.current = url2;
        setPreviewUrl(url2);
        setPreviewBlob(blob);
        setCapture(data);
      } catch (e) {
        setErr(e instanceof Error ? e.message : "OAK capture failed");
      } finally {
        setBusy(false);
      }
    },
    [apiBase, releasePreview]
  );

  // First capture on open
  useEffect(() => {
    if (!open) return;
    void fetchCapture();
  }, [open, fetchCapture]);

  // Cleanup blob URL on unmount
  useEffect(
    () => () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    },
    []
  );

  const cancel = useCallback(() => {
    if (capture) {
      void fetch(`${apiBase}/oak/capture/${capture.capture_id}`, { method: "DELETE" }).catch(
        () => {
          /* best-effort cleanup */
        }
      );
    }
    close();
  }, [apiBase, capture, close]);

  const useThis = useCallback(() => {
    if (!capture || !previewBlob) return;
    const file = new File([previewBlob], `${capture.capture_id}.jpg`, { type: "image/jpeg" });
    onCapture(file, capture.capture_id);
    setOpen(false);
    setErr(null);
    setCapture(null);
    // Don't release the blob URL — the parent now owns the File and may need to display it.
    previewUrlRef.current = null;
    setPreviewUrl(null);
    setPreviewBlob(null);
  }, [capture, previewBlob, onCapture]);

  return (
    <>
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen(true)}
        className="shrink-0 rounded-md border border-foreground/20 px-3 py-2 text-sm text-foreground/85 hover:bg-foreground/5 disabled:opacity-40"
        title="Capture from the OAK-D (RGB + depth)"
      >
        OAK-D
      </button>
      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          role="dialog"
          aria-modal="true"
          aria-label="Capture a photo from the OAK-D"
          onClick={cancel}
        >
          <div
            className="max-h-[min(90dvh,720px)] w-full max-w-lg overflow-hidden rounded-lg border border-foreground/20 bg-background p-3 shadow-lg"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="mb-2 text-sm font-medium text-foreground/90">OAK-D capture</h2>
            {err && (
              <p className="mb-2 rounded border border-red-500/30 bg-red-500/10 px-2 py-1.5 text-sm text-red-300">
                {err}
              </p>
            )}
            <div className="relative aspect-video w-full overflow-hidden rounded border border-foreground/15 bg-black/80">
              {previewUrl ? (
                /* eslint-disable-next-line @next/next/no-img-element */
                <img
                  src={previewUrl}
                  alt="OAK-D capture preview"
                  className="h-full w-full object-contain"
                />
              ) : (
                <div className="flex h-full w-full items-center justify-center text-sm text-foreground/55">
                  {busy ? "Capturing…" : err ? "Capture failed." : "…"}
                </div>
              )}
            </div>
            <div className="mt-3 flex flex-wrap items-center justify-end gap-2">
              <button
                type="button"
                onClick={cancel}
                className="rounded-md border border-foreground/20 px-3 py-2 text-sm text-foreground/85 hover:bg-foreground/5"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void fetchCapture(capture?.capture_id)}
                disabled={busy}
                className="rounded-md border border-foreground/20 px-3 py-2 text-sm text-foreground/85 hover:bg-foreground/5 disabled:opacity-40"
              >
                {busy ? "…" : "Retake"}
              </button>
              <button
                type="button"
                onClick={useThis}
                disabled={!capture || !previewBlob || busy}
                className="rounded-md bg-foreground px-3 py-2 text-sm font-medium text-background disabled:opacity-40"
              >
                Use this
              </button>
            </div>
            <p className="mt-2 text-[11px] text-foreground/50">
              Single still capture (RGB + aligned depth). Aim the camera, then take the shot — the
              depth map travels with the image to the agent.
            </p>
          </div>
        </div>
      )}
    </>
  );
}
