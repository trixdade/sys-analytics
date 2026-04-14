"""Frame-to-frame affine residual estimation — the core signal of the baseline.

For a pair of consecutive frames (f0, f1) we estimate the best 2-D affine
transform A such that A(f0) ≈ f1, warp f0 under A, and compute residual
statistics against f1 over the inside-frame support (ignoring the border).

If the clip is a real video, residuals are large because human motion is
non-rigid (blinks, micro-expressions, parallax) and the sensor adds new noise
each frame. If the clip is a photo being panned/zoomed/rotated, A explains
almost everything and residuals collapse to near zero.

We try two estimators and keep the one with lower residual:

  1. `cv2.estimateAffinePartial2D` on ORB keypoint matches (4 DoF: rotation,
     uniform scale, translation). Robust to large motion when features match.
  2. `cv2.findTransformECC` on grayscale, init'd at identity (or previous A).
     Works well when keypoints fail (low-texture regions, motion blur).

Both are run on downscaled grayscale crops for speed — the CPU budget here is
dominated by these estimators, so we keep images small.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ----------------------------- config / data types -----------------------------


@dataclass
class PairResidual:
    """Per-pair measurements. All values are scalar floats in [0, 1] or px."""
    mae: float                    # L1 residual (0..1)
    rmse: float                   # L2 residual (0..1)
    ssim: float                   # SSIM between warped f0 and f1 (-1..1, usually ~1)
    residual_frac_hi: float       # fraction of pixels with |res|>5% of range
    translate_px: float           # magnitude of translation in A (px)
    rotation_deg: float           # rotation extracted from A (deg)
    scale: float                  # uniform scale extracted from A
    method: str                   # "orb" or "ecc" or "identity"
    ok: bool                      # whether estimation succeeded


# ----------------------------- small helpers -----------------------------


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def _downscale(gray: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return gray, 1.0
    scale = max_side / s
    new = cv2.resize(
        gray,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return new, scale


def _decompose_affine(M: np.ndarray) -> tuple[float, float, float]:
    """Return (translation_px, rotation_deg, uniform_scale) from a 2x3 matrix."""
    tx, ty = M[0, 2], M[1, 2]
    a, b = M[0, 0], M[0, 1]
    scale = float(np.sqrt(a * a + b * b))
    angle = float(np.degrees(np.arctan2(b, a)))
    return float(np.hypot(tx, ty)), angle, scale


# ----------------------------- estimators -----------------------------


_ORB = cv2.ORB_create(nfeatures=500, fastThreshold=7)
_BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def _estimate_orb(g0: np.ndarray, g1: np.ndarray) -> np.ndarray | None:
    kp0, d0 = _ORB.detectAndCompute(g0, None)
    kp1, d1 = _ORB.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(kp0) < 8 or len(kp1) < 8:
        return None
    matches = _BF.match(d0, d1)
    if len(matches) < 8:
        return None
    matches = sorted(matches, key=lambda m: m.distance)[:80]
    p0 = np.float32([kp0[m.queryIdx].pt for m in matches])
    p1 = np.float32([kp1[m.trainIdx].pt for m in matches])
    M, inliers = cv2.estimateAffinePartial2D(
        p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0
    )
    if M is None or inliers is None or int(inliers.sum()) < 8:
        return None
    return M


def _estimate_ecc(
    g0: np.ndarray, g1: np.ndarray, init: np.ndarray | None = None
) -> np.ndarray | None:
    warp = (
        init.astype(np.float32)
        if init is not None
        else np.eye(2, 3, dtype=np.float32)
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        1e-4,
    )
    try:
        _, M = cv2.findTransformECC(
            g0, g1, warp, motionType=cv2.MOTION_AFFINE, criteria=criteria
        )
        return M
    except cv2.error:
        return None


# ----------------------------- residual -----------------------------


def _residual_stats(
    warped: np.ndarray, target: np.ndarray, border_ignore_frac: float = 0.05
) -> tuple[float, float, float, float]:
    """Both arrays are uint8 grayscale, same shape."""
    h, w = warped.shape
    b = max(1, int(round(min(h, w) * border_ignore_frac)))
    a = warped[b:-b, b:-b].astype(np.float32) / 255.0
    t = target[b:-b, b:-b].astype(np.float32) / 255.0
    diff = a - t
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    frac_hi = float(np.mean(np.abs(diff) > 0.05))
    try:
        s = float(ssim(a, t, data_range=1.0))
    except Exception:
        s = 1.0 - rmse  # coarse fallback
    return mae, rmse, s, frac_hi


def estimate_pair(
    f0: np.ndarray,
    f1: np.ndarray,
    work_side: int = 256,
    prev_M: np.ndarray | None = None,
) -> PairResidual:
    """Estimate the affine relating f0->f1 and measure residual.

    Parameters
    ----------
    f0, f1 : BGR or grayscale frames, same shape
    work_side : downscale so that max(H,W) == work_side before estimation
    prev_M : optional warm-start for ECC (the M from the previous pair)
    """
    g0 = _to_gray(f0)
    g1 = _to_gray(f1)
    g0s, _ = _downscale(g0, work_side)
    g1s, _ = _downscale(g1, work_side)

    M = _estimate_orb(g0s, g1s)
    method = "orb" if M is not None else ""
    if M is None:
        M = _estimate_ecc(g0s, g1s, init=prev_M)
        method = "ecc" if M is not None else ""

    ok = M is not None
    if not ok:
        # Fall back to identity — residual will be high, which is fine: we
        # record the raw inter-frame difference as the signal.
        M = np.eye(2, 3, dtype=np.float32)
        method = "identity"

    warped = cv2.warpAffine(
        g0s, M, (g1s.shape[1], g1s.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    mae, rmse, s, frac_hi = _residual_stats(warped, g1s)
    trans_px, rot_deg, scale = _decompose_affine(M)
    return PairResidual(
        mae=mae,
        rmse=rmse,
        ssim=s,
        residual_frac_hi=frac_hi,
        translate_px=trans_px,
        rotation_deg=rot_deg,
        scale=scale,
        method=method,
        ok=ok,
    )


def clip_pair_residuals(
    frames: np.ndarray,
    work_side: int = 256,
    stride: int = 1,
) -> list[PairResidual]:
    """Compute PairResidual for all consecutive pairs (with optional stride)."""
    out: list[PairResidual] = []
    prev_M: np.ndarray | None = None
    for i in range(0, len(frames) - stride, stride):
        r = estimate_pair(frames[i], frames[i + stride], work_side=work_side)
        out.append(r)
    return out
