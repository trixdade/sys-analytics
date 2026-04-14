"""Clip-level features from per-pair residuals.

The idea: a single pair's residual is noisy; a clip (~70 pairs at 24 fps over
3 s) gives stable statistics. Each feature summarizes the *distribution* of
residuals across the clip, which is where the signal lives.

Output is a fixed-length feature vector per clip, suitable for LogReg / GBM.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

from .affine_residual import PairResidual


FEATURE_NAMES: tuple[str, ...] = (
    # central tendency of residual
    "mae_mean", "mae_median", "mae_p10", "mae_p90",
    "rmse_mean", "rmse_median",
    "ssim_mean", "ssim_median", "ssim_min",
    "frac_hi_mean", "frac_hi_p90",
    # motion parameters — magnitudes and variability
    "trans_px_mean", "trans_px_std",
    "rot_deg_abs_mean", "rot_deg_abs_max",
    "scale_dev_mean", "scale_dev_max",
    # reliability of estimator
    "ok_frac", "orb_frac",
    # temporal regularity — how smooth is the motion trajectory?
    # (second differences of tx-like proxies; large for jittery real video)
    "mae_diff_std", "ssim_diff_std",
    # length
    "n_pairs",
)


def _safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if len(x) > 1 else 0.0


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else 0.0


def clip_features(residuals: Sequence[PairResidual]) -> np.ndarray:
    if not residuals:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    mae = np.asarray([r.mae for r in residuals], dtype=np.float32)
    rmse = np.asarray([r.rmse for r in residuals], dtype=np.float32)
    s = np.asarray([r.ssim for r in residuals], dtype=np.float32)
    fh = np.asarray([r.residual_frac_hi for r in residuals], dtype=np.float32)
    tr = np.asarray([r.translate_px for r in residuals], dtype=np.float32)
    rot = np.asarray([abs(r.rotation_deg) for r in residuals], dtype=np.float32)
    scl = np.asarray([abs(r.scale - 1.0) for r in residuals], dtype=np.float32)
    ok = np.asarray([1.0 if r.ok else 0.0 for r in residuals], dtype=np.float32)
    orb = np.asarray(
        [1.0 if r.method == "orb" else 0.0 for r in residuals], dtype=np.float32
    )

    feats = {
        "mae_mean": float(mae.mean()),
        "mae_median": float(np.median(mae)),
        "mae_p10": _pct(mae, 10),
        "mae_p90": _pct(mae, 90),
        "rmse_mean": float(rmse.mean()),
        "rmse_median": float(np.median(rmse)),
        "ssim_mean": float(s.mean()),
        "ssim_median": float(np.median(s)),
        "ssim_min": float(s.min()),
        "frac_hi_mean": float(fh.mean()),
        "frac_hi_p90": _pct(fh, 90),
        "trans_px_mean": float(tr.mean()),
        "trans_px_std": _safe_std(tr),
        "rot_deg_abs_mean": float(rot.mean()),
        "rot_deg_abs_max": float(rot.max()),
        "scale_dev_mean": float(scl.mean()),
        "scale_dev_max": float(scl.max()),
        "ok_frac": float(ok.mean()),
        "orb_frac": float(orb.mean()),
        "mae_diff_std": _safe_std(np.diff(mae)),
        "ssim_diff_std": _safe_std(np.diff(s)),
        "n_pairs": float(len(residuals)),
    }
    return np.asarray(
        [feats[k] for k in FEATURE_NAMES], dtype=np.float32
    )
