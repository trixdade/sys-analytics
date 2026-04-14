"""Synthesize 'fake' selfie videos by applying time-varying affine transforms
to a single still image. This imitates the main spoofing attack we are
targeting: the attacker shows a photo inside an emulator and slightly moves /
zooms / rotates it to produce an mp4 that naive liveness models accept.

The trajectory for each parameter (angle, zoom, tx, ty, shear) is a sum of a
small number of sines with random amplitudes and phases, plus a small amount of
per-frame jitter. This yields smooth, realistic-looking "camera handling".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class AffineTrajectoryConfig:
    duration_s: float = 3.0
    fps: int = 24
    # Max absolute values of each parameter (trajectory amplitude is random
    # in [0, max]).
    max_angle_deg: float = 12.0
    max_zoom: float = 0.15          # zoom factor varies in [1 - x, 1 + x]
    max_translate_frac: float = 0.08  # fraction of image size
    max_shear: float = 0.03
    # Frame-level jitter (tiny, to mimic hand-holding — still fully affine)
    jitter_angle_deg: float = 0.3
    jitter_translate_frac: float = 0.002
    # Number of sine components summed into each trajectory
    n_harmonics: int = 2
    # Output crop: after warp we crop the central region to remove black
    # borders. Crop fraction shrinks width/height by this factor.
    crop_frac: float = 0.8


@dataclass
class FakeClipParams:
    """Realized trajectory for a single synthesized clip — kept for auditing."""
    angle_deg: np.ndarray
    zoom: np.ndarray
    tx_frac: np.ndarray
    ty_frac: np.ndarray
    shear: np.ndarray
    meta: dict = field(default_factory=dict)


def _sine_trajectory(
    n_frames: int,
    max_amp: float,
    n_harmonics: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """A smooth random trajectory in [-max_amp, +max_amp]."""
    if max_amp == 0 or n_harmonics == 0:
        return np.zeros(n_frames, dtype=np.float32)
    t = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
    out = np.zeros(n_frames, dtype=np.float32)
    for _ in range(n_harmonics):
        freq = rng.uniform(0.5, 2.5)        # cycles across the clip
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.3, 1.0)
        out += amp * np.sin(2.0 * np.pi * freq * t + phase)
    # Normalize to [-1, 1] then scale
    m = np.max(np.abs(out)) + 1e-8
    out = out / m
    # Random global amplitude, so some clips move a lot, some barely
    out *= max_amp * rng.uniform(0.2, 1.0)
    return out.astype(np.float32)


def sample_trajectory(
    cfg: AffineTrajectoryConfig,
    rng: np.random.Generator | None = None,
) -> FakeClipParams:
    rng = rng or np.random.default_rng()
    n_frames = int(round(cfg.duration_s * cfg.fps))

    angle = _sine_trajectory(n_frames, cfg.max_angle_deg, cfg.n_harmonics, rng)
    zoom_off = _sine_trajectory(n_frames, cfg.max_zoom, cfg.n_harmonics, rng)
    tx = _sine_trajectory(n_frames, cfg.max_translate_frac, cfg.n_harmonics, rng)
    ty = _sine_trajectory(n_frames, cfg.max_translate_frac, cfg.n_harmonics, rng)
    shear = _sine_trajectory(n_frames, cfg.max_shear, cfg.n_harmonics, rng)

    # Frame-level jitter — still a rigid-in-time affine but noisier
    if cfg.jitter_angle_deg > 0:
        angle = angle + rng.normal(0.0, cfg.jitter_angle_deg, n_frames).astype(
            np.float32
        )
    if cfg.jitter_translate_frac > 0:
        tx = tx + rng.normal(
            0.0, cfg.jitter_translate_frac, n_frames
        ).astype(np.float32)
        ty = ty + rng.normal(
            0.0, cfg.jitter_translate_frac, n_frames
        ).astype(np.float32)

    zoom = 1.0 + zoom_off

    return FakeClipParams(
        angle_deg=angle,
        zoom=zoom,
        tx_frac=tx,
        ty_frac=ty,
        shear=shear,
        meta={"n_frames": n_frames, "fps": cfg.fps},
    )


def _affine_matrix(
    w: int,
    h: int,
    angle_deg: float,
    zoom: float,
    tx_frac: float,
    ty_frac: float,
    shear: float,
) -> np.ndarray:
    """Build a 2x3 affine matrix centered on the image."""
    cx, cy = w / 2.0, h / 2.0
    # Rotation + scale around the center
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, zoom)
    # Add shear in x: multiply left by [[1, s, 0], [0, 1, 0]]
    if shear != 0.0:
        S = np.array([[1.0, shear, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        # Compose: out = S * [M; 0 0 1]
        M3 = np.vstack([M, [0.0, 0.0, 1.0]])
        M = (S @ M3).astype(np.float64)
    # Translation in fractions of image size
    M[0, 2] += tx_frac * w
    M[1, 2] += ty_frac * h
    return M


def render_fake_clip(
    image: np.ndarray,
    cfg: AffineTrajectoryConfig | None = None,
    rng: np.random.Generator | None = None,
    border_mode: int = cv2.BORDER_REFLECT_101,
) -> tuple[np.ndarray, FakeClipParams]:
    """Synthesize a fake clip from a single BGR image.

    Returns
    -------
    frames : (T, H', W', 3) uint8 BGR, center-cropped to hide black borders
    params : the trajectory used
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 BGR image, got {image.shape}")
    cfg = cfg or AffineTrajectoryConfig()
    rng = rng or np.random.default_rng()
    params = sample_trajectory(cfg, rng)

    h, w = image.shape[:2]
    ch = int(round(h * cfg.crop_frac))
    cw = int(round(w * cfg.crop_frac))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2

    frames = np.empty((len(params.angle_deg), ch, cw, 3), dtype=np.uint8)
    for i in range(len(params.angle_deg)):
        M = _affine_matrix(
            w,
            h,
            float(params.angle_deg[i]),
            float(params.zoom[i]),
            float(params.tx_frac[i]),
            float(params.ty_frac[i]),
            float(params.shear[i]),
        )
        warped = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=border_mode,
        )
        frames[i] = warped[y0 : y0 + ch, x0 : x0 + cw]
    return frames, params


def load_image(path: str | Path, max_side: int = 720) -> np.ndarray:
    """Load an image as BGR uint8, optionally downscaling so max(h,w) <= max_side."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    s = max(h, w)
    if s > max_side:
        scale = max_side / s
        img = cv2.resize(
            img,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return img
