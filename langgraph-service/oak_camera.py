"""OAK-D Lite capture: aligned RGB + stereo depth, in-process via depthai."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

# Tunables
RGB_W, RGB_H = 1280, 720
MIN_DEPTH_M, MAX_DEPTH_M = 0.20, 19.0
SAMPLE_WINDOW = 7
WARMUP_FRAMES = 8


def _build_pipeline_and_queues():  # type: ignore[no-untyped-def]
    """Build the OAK-D pipeline (RGB + stereo depth aligned to RGB) and pre-create output queues.

    depthai 3.x dropped the ``XLinkOut`` node — queues are now created directly on output objects
    via ``output.createOutputQueue()`` before the pipeline is started.
    """
    import depthai as dai

    p = dai.Pipeline()

    # depthai 3.x: new `Camera` node supersedes `ColorCamera`. requestOutput() with BGR888i gives
    # us a host-friendly BGR ndarray via getCvFrame() with no manual format conversion.
    cam_rgb = p.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    rgb_out = cam_rgb.requestOutput((RGB_W, RGB_H), dai.ImgFrame.Type.BGR888i, fps=15)

    mono_l = p.create(dai.node.MonoCamera)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_l.setFps(15)

    mono_r = p.create(dai.node.MonoCamera)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_r.setFps(15)

    stereo = p.create(dai.node.StereoDepth)
    # depthai 3.x renamed HIGH_DENSITY -> DENSITY; favors fewer holes for object localization.
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    # Align depth output to the RGB camera so a normalized (x,y) on the JPEG indexes the depth map.
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    # Without an explicit output size, depthai aligns to the RGB sensor's native width (2104 on
    # the Lite IMX214), which is not a multiple of 16 — the stereo engine then errors out.
    stereo.setOutputSize(RGB_W, RGB_H)
    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    rgb_queue = rgb_out.createOutputQueue(maxSize=4, blocking=False)
    depth_queue = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
    return p, rgb_queue, depth_queue


class OakCameraError(RuntimeError):
    """Raised when the OAK-D cannot be opened or a capture fails."""


class OakCamera:
    """Lazy-singleton wrapper around a single dai.Device handle."""

    _instance: Optional["OakCamera"] = None
    _class_lock = threading.Lock()

    def __init__(self) -> None:
        self._pipeline = None  # type: ignore[assignment]
        self._rgb_q = None  # type: ignore[assignment]
        self._depth_q = None  # type: ignore[assignment]
        self._capture_lock = threading.Lock()
        self._open()

    @classmethod
    def get(cls) -> "OakCamera":
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._class_lock:
            if cls._instance is not None:
                try:
                    cls._instance.close()
                except Exception:
                    pass
                cls._instance = None

    def _open(self) -> None:
        try:
            import depthai as dai  # noqa: F401
        except ImportError as e:
            raise OakCameraError(
                "depthai is not installed; pip install depthai>=2.24"
            ) from e
        try:
            pipeline, rgb_q, depth_q = _build_pipeline_and_queues()
            pipeline.start()
            self._pipeline = pipeline
            self._rgb_q = rgb_q
            self._depth_q = depth_q
        except Exception as e:
            raise OakCameraError(f"could not open OAK-D device: {e!s}") from e

        # Warm-up: drain a few frames so AE/AWB and stereo settle before the first real capture.
        for _ in range(WARMUP_FRAMES):
            try:
                self._rgb_q.tryGet()
                self._depth_q.tryGet()
            except Exception:
                pass
            time.sleep(0.05)

    def capture(self, dest_dir: Path) -> tuple[Path, np.ndarray, int, int]:
        """Grab one synchronized RGB + depth pair.

        Returns (jpeg_path, depth_uint16_mm, width, height). Depth is aligned to RGB.
        """
        if self._pipeline is None or self._rgb_q is None or self._depth_q is None:
            raise OakCameraError("OAK-D pipeline is not running")

        with self._capture_lock:
            rgb_msg = self._rgb_q.get()
            depth_msg = self._depth_q.get()
            bgr = rgb_msg.getCvFrame()
            depth = depth_msg.getFrame()

        if bgr is None or depth is None:
            raise OakCameraError("empty frame from OAK-D")

        h, w = bgr.shape[:2]

        from PIL import Image

        dest_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = dest_dir / f"oak-{uuid.uuid4().hex}.jpg"
        Image.fromarray(bgr[..., ::-1]).save(rgb_path, "JPEG", quality=92)
        return rgb_path, depth, int(w), int(h)

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                if self._pipeline.isRunning():
                    self._pipeline.stop()
            finally:
                self._pipeline = None
                self._rgb_q = None
                self._depth_q = None


def sample_depth(
    depth: np.ndarray, x_norm: float, y_norm: float, window: int = SAMPLE_WINDOW
) -> Optional[float]:
    """Median of valid (non-zero, in-range) depth in a window×window ROI; meters; None if no valid samples."""
    if depth is None or depth.size == 0:
        return None
    h, w = depth.shape[:2]
    if w <= 0 or h <= 0:
        return None
    cx = int(round(max(0.0, min(1.0, x_norm)) * (w - 1)))
    cy = int(round(max(0.0, min(1.0, y_norm)) * (h - 1)))
    r = max(0, window // 2)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    roi = depth[y0:y1, x0:x1].astype(np.float32) / 1000.0
    valid = roi[(roi >= MIN_DEPTH_M) & (roi <= MAX_DEPTH_M)]
    if valid.size == 0:
        return None
    return float(np.median(valid))
