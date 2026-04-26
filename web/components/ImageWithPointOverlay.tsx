"use client";

import { useState } from "react";
import { normalizeMolmoXY } from "@/lib/molmoDisplay";

export type MolmoPoint = {
  object_id: number;
  image_index: number;
  x: number;
  y: number;
  /** Distance to the point in meters, when sourced from an OAK-D capture; null for depth holes. */
  depth_m?: number | null;
};

type Props = {
  imageUrl: string;
  points: MolmoPoint[];
  alt?: string;
};

const PALETTE = [
  "bg-rose-500",
  "bg-amber-500",
  "bg-emerald-500",
  "bg-sky-500",
  "bg-violet-500",
  "bg-fuchsia-500",
] as const;

function chipColorForObjectId(objectId: number): string {
  const i = ((objectId % PALETTE.length) + PALETTE.length) % PALETTE.length;
  return PALETTE[i]!;
}

/**
 * Renders the image and overlays markers. `x`/`y` are expected 0..1 in image space; if either
 * is &gt; 1, values are treated as **pixels** in the natural image and divided by the loaded
 * image size (for APIs that return pixel coords as MolmoPoint’s HTTP does).
 */
export function ImageWithPointOverlay({ imageUrl, points, alt = "Upload" }: Props) {
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });

  return (
    <div className="relative inline-block max-w-full align-top">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={imageUrl}
        alt={alt}
        className="max-h-[min(50vh,480px)] w-auto max-w-full rounded border border-foreground/15"
        onLoad={(e) => {
          const el = e.currentTarget;
          setImgSize({ w: el.naturalWidth, h: el.naturalHeight });
        }}
      />
      {points.map((pt, i) => {
        const { nx, ny } = normalizeMolmoXY(pt.x, pt.y, imgSize.w, imgSize.h);
        const displayNum = i + 1;
        const bg = chipColorForObjectId(pt.object_id);
        const depthSuffix =
          typeof pt.depth_m === "number"
            ? `, depth=${pt.depth_m.toFixed(2)} m`
            : pt.depth_m === null
              ? `, depth=— (out of range)`
              : "";
        const label = `Detection ${displayNum}: object_id=${pt.object_id}, image_index=${pt.image_index}, x=${nx.toFixed(4)}, y=${ny.toFixed(4)}${depthSuffix} (normalized 0-1 in image space)`;
        return (
          <div
            key={`${pt.object_id}-${pt.image_index}-${i}`}
            className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2"
            style={{
              left: `${nx * 100}%`,
              top: `${ny * 100}%`,
            }}
            title={label}
            role="img"
            aria-label={label}
          >
            <div className="flex flex-col items-center gap-0.5">
              <div
                className={`flex h-7 w-7 items-center justify-center rounded-full border-2 border-white text-xs font-bold text-white shadow-md ${bg}`}
              >
                {displayNum}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
