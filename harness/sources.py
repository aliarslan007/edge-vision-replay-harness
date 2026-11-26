"""Frame sources. Recorded video and the live camera expose the SAME method:

    read(media_ts) -> (image_bgr, actual_media_ts) or None when the stream has ended

For recorded sources media_ts selects the frame (nearest index at the native fps), so the
harness controls which frames are sampled. For the live camera, media_ts is ignored and the
newest captured frame is returned; its media_ts is the capture time relative to open().
"""
from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


class FrameSource(ABC):
    is_live = False
    native_fps: float = 30.0
    duration_s: Optional[float] = None

    def open(self) -> "FrameSource":
        return self

    @abstractmethod
    def read(self, media_ts: float) -> Optional[tuple[np.ndarray, float]]:
        ...

    def close(self):
        pass

    def describe(self) -> dict:
        return {"type": type(self).__name__, "native_fps": self.native_fps, "duration_s": self.duration_s}


class VideoFileSource(FrameSource):
    """A composed stream stored as a video file (e.g. stream_01.mp4). Decodes forward only."""

    def __init__(self, path: str, fps: Optional[float] = None):
        self.path = str(path)
        self._fps_override = fps

    def open(self):
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video {self.path}")
        self.native_fps = self._fps_override or self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.n_frames = n if n > 0 else None
        self.duration_s = n / self.native_fps if n > 0 else None
        self.pos = 0                        # index of the next frame the decoder will return
        return self

    def read(self, media_ts):
        idx = int(round(media_ts * self.native_fps))
        if self.n_frames is not None and idx >= self.n_frames:
            return None
        if idx < self.pos:                  # never seek backwards: reuse the next frame
            idx = self.pos
        while self.pos < idx:               # skip without converting to an image
            if not self.cap.grab():
                return None
            self.pos += 1
        ok, img = self.cap.read()
        if not ok:
            return None
        self.pos += 1
        return img, idx / self.native_fps

    def close(self):
        self.cap.release()

    def describe(self):
        return {**super().describe(), "path": self.path}


class ImageSequenceSource(FrameSource):
    """A stream given as images: a folder of frames, or a text file listing one image path per
    line (lets P1 compose streams from the original ChokePoint JPEGs without re-encoding)."""

    def __init__(self, path: str, fps: float = 30.0):
        self.path = Path(path)
        self.native_fps = fps

    def open(self):
        if self.path.is_dir():
            self.files = sorted(p for p in self.path.iterdir() if p.suffix.lower() in IMG_EXT)
        else:
            base = self.path.parent
            self.files = []
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    p = Path(line)
                    self.files.append(p if p.is_absolute() else base / p)
        if not self.files:
            raise FileNotFoundError(f"no frames found in {self.path}")
        self.duration_s = len(self.files) / self.native_fps
        return self

    def read(self, media_ts):
        idx = int(round(media_ts * self.native_fps))
        if idx >= len(self.files):
            return None
        img = cv2.imread(str(self.files[idx]))
        if img is None:
            raise IOError(f"cannot read frame {self.files[idx]}")
        return img, idx / self.native_fps

    def describe(self):
        return {**super().describe(), "path": str(self.path), "n_frames": len(self.files)}


# Low-latency FFmpeg options for network cameras (RTSP / HTTP-MJPEG): no read-ahead buffering, minimal probing.
# Measured by P4 on the Pi: transport latency 1.04 s -> 0.39 s. setdefault: an environment value set by the
# user still wins. Must be set before cv2.VideoCapture() opens the stream.
LOW_LATENCY_FFMPEG = "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0"


class CameraSource(FrameSource):
    """Live camera: USB index ("0") or a network URL (rtsp://..., http://.../video for MJPEG).

    A background thread keeps grabbing so read() always returns the NEWEST frame instead of a
    stale buffered one. Each frame keeps its capture time; media_ts = capture time - open time.
    """
    is_live = True

    def __init__(self, uri: str, width: Optional[int] = None, height: Optional[int] = None):
        self.uri = int(uri) if str(uri).isdigit() else uri
        self.size = (width, height)

    def open(self):
        self.ffmpeg_options = None
        if isinstance(self.uri, str):                     # network stream -> FFmpeg backend
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", LOW_LATENCY_FFMPEG)
            self.ffmpeg_options = os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
        self.cap = cv2.VideoCapture(self.uri)
        if not self.cap.isOpened():
            raise ConnectionError(f"cannot open camera {self.uri}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.size[0]:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        self.native_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.t_open = time.monotonic()
        self._latest, self._lock, self._stop = None, threading.Lock(), False
        self.n_captured = 0
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 5
        while self._latest is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if self._latest is None:
            raise ConnectionError("camera opened but delivered no frames within 5 s")
        return self

    def _grab_loop(self):
        while not self._stop:
            ok, img = self.cap.read()
            t = time.monotonic()
            if ok:
                with self._lock:
                    self._latest = (img, t - self.t_open)
                    self.n_captured += 1
            else:
                time.sleep(0.005)

    def read(self, media_ts):
        with self._lock:
            return self._latest

    def close(self):
        self._stop = True
        self._thread.join(timeout=2)
        self.cap.release()

    def describe(self):
        return {**super().describe(), "uri": str(self.uri), "ffmpeg_capture_options": self.ffmpeg_options}


def open_source(spec: str, fps: Optional[float] = None) -> FrameSource:
    """'camera:0', 'rtsp://...', 'http://...' -> camera; folder or .txt -> images; else video file."""
    if spec.startswith("camera:"):
        return CameraSource(spec.split(":", 1)[1]).open()
    if spec.startswith(("rtsp://", "http://", "https://")):
        return CameraSource(spec).open()
    p = Path(spec)
    if p.is_dir() or p.suffix.lower() == ".txt":
        return ImageSequenceSource(spec, fps or 30.0).open()
    return VideoFileSource(spec, fps).open()
