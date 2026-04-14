"""Thin video I/O helpers built on OpenCV.

Kept deliberately small — we only need frame iteration, sampling, and writing
mp4s with reasonable compression to mimic real mobile uploads.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class VideoMeta:
    path: Path
    width: int
    height: int
    fps: float
    n_frames: int


def probe(path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    meta = VideoMeta(
        path=Path(path),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)) or 0.0,
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return meta


def iter_frames(path: str | Path) -> Iterator[np.ndarray]:
    """Yield frames as BGR uint8 arrays."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def read_clip(
    path: str | Path,
    max_frames: int | None = None,
    stride: int = 1,
) -> np.ndarray:
    """Read frames into an array of shape (T, H, W, 3), BGR uint8."""
    frames: list[np.ndarray] = []
    for i, f in enumerate(iter_frames(path)):
        if i % stride != 0:
            continue
        frames.append(f)
        if max_frames is not None and len(frames) >= max_frames:
            break
    if not frames:
        raise IOError(f"No frames read from {path}")
    return np.stack(frames, axis=0)


def write_video(
    path: str | Path,
    frames: np.ndarray,
    fps: float,
    fourcc: str = "mp4v",
) -> None:
    """Write (T, H, W, 3) BGR uint8 frames to a video file."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected (T,H,W,3), got {frames.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h)
    )
    if not writer.isOpened():
        raise IOError(f"Cannot open writer for {path}")
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()


def list_videos(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS
    )
