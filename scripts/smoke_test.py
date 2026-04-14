"""End-to-end smoke test of the classical baseline.

Since the user does not have real videos yet, this script synthesizes both
sides procedurally:

  - source images: textured procedural images (shapes + gradients), enough
    feature content for ORB to work.
  - fake clips:    pure time-varying affine transforms (our generator).
  - "real" clips:  affine trajectory PLUS a small per-frame elastic warp +
                   additive sensor noise. The elastic warp is non-affine, so
                   frame pairs cannot be aligned by a single 2x3 matrix — this
                   mimics the non-rigid motion and sensor noise that real
                   selfie videos always carry.

If the baseline is working, fakes should have very low MAE / high SSIM after
affine alignment, and "reals" should have materially higher residuals.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from src.classical.affine_residual import clip_pair_residuals
from src.classical.features import FEATURE_NAMES, clip_features
from src.data.synthetic import (
    AffineTrajectoryConfig,
    render_fake_clip,
    sample_trajectory,
)
from src.data.video_io import write_video
from src.utils.seeds import seed_everything


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Procedural source images — not faces, but full of texture so ORB works well.
# ---------------------------------------------------------------------------


def make_source_image(seed: int, size: int = 320) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Gradient background — explicitly broadcast each channel to (H,W)
    yy, xx = np.meshgrid(
        np.linspace(0, 1, size, dtype=np.float32),
        np.linspace(0, 1, size, dtype=np.float32),
        indexing="ij",
    )
    r = 0.3 + 0.4 * xx + 0.3 * yy
    g = 0.2 + 0.5 * (1 - xx) + 0.0 * yy
    b = 0.1 + 0.0 * xx + 0.6 * yy
    base = np.stack([r, g, b], axis=-1)
    img = (np.clip(base, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()  # -> BGR
    # Random circles & rectangles for texture and ORB features
    for _ in range(40):
        c = tuple(int(v) for v in rng.integers(0, 256, size=3))
        p1 = tuple(int(v) for v in rng.integers(0, size, size=2))
        p2 = tuple(int(v) for v in rng.integers(0, size, size=2))
        if rng.random() < 0.5:
            cv2.rectangle(img, p1, p2, c, thickness=-1)
        else:
            r = int(rng.integers(5, 40))
            cv2.circle(img, p1, r, c, thickness=-1)
    # Crossing lines for corners
    for _ in range(20):
        p1 = tuple(int(v) for v in rng.integers(0, size, size=2))
        p2 = tuple(int(v) for v in rng.integers(0, size, size=2))
        c = tuple(int(v) for v in rng.integers(0, 256, size=3))
        cv2.line(img, p1, p2, c, thickness=int(rng.integers(1, 4)))
    return img


# ---------------------------------------------------------------------------
# "Real" clip synthesis: affine + elastic warp + noise.
# ---------------------------------------------------------------------------


def _elastic_displacement(
    h: int, w: int, amplitude_px: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth random displacement field of magnitude ~ amplitude_px."""
    # low-res noise, upsampled and Gaussian-blurred
    low = 16
    dx = rng.standard_normal((low, low)).astype(np.float32)
    dy = rng.standard_normal((low, low)).astype(np.float32)
    dx = cv2.resize(dx, (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy, (w, h), interpolation=cv2.INTER_CUBIC)
    dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=25.0)
    dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=25.0)
    # normalize to target amplitude
    def _norm(a):
        m = np.max(np.abs(a)) + 1e-8
        return (a / m) * amplitude_px
    return _norm(dx), _norm(dy)


def render_real_like_clip(
    image: np.ndarray,
    cfg: AffineTrajectoryConfig,
    rng: np.random.Generator,
    elastic_amplitude_px: float = 2.5,
    noise_std: float = 4.0,
) -> np.ndarray:
    """Affine trajectory + per-frame elastic warp + sensor noise."""
    params = sample_trajectory(cfg, rng)
    h, w = image.shape[:2]
    ch = int(round(h * cfg.crop_frac))
    cw = int(round(w * cfg.crop_frac))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2

    # Base grid for remap
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                        np.arange(h, dtype=np.float32))

    frames = np.empty((len(params.angle_deg), ch, cw, 3), dtype=np.uint8)
    for i in range(len(params.angle_deg)):
        # Affine step — reuse the synthetic module's matrix builder
        from src.data.synthetic import _affine_matrix
        M = _affine_matrix(
            w, h,
            float(params.angle_deg[i]),
            float(params.zoom[i]),
            float(params.tx_frac[i]),
            float(params.ty_frac[i]),
            float(params.shear[i]),
        )
        affine_warped = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        # Per-frame elastic (non-affine) displacement
        dx, dy = _elastic_displacement(h, w, elastic_amplitude_px, rng)
        map_x = (gx + dx).astype(np.float32)
        map_y = (gy + dy).astype(np.float32)
        elastic = cv2.remap(
            affine_warped, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        # Sensor noise — independent per frame
        noisy = elastic.astype(np.float32) + rng.normal(
            0.0, noise_std, elastic.shape
        ).astype(np.float32)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        frames[i] = noisy[y0 : y0 + ch, x0 : x0 + cw]
    return frames


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def clean_dir(p: Path) -> None:
    if p.exists():
        for child in p.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        p.mkdir(parents=True, exist_ok=True)


def main(
    n_sources: int = 8,
    n_fake: int = 16,
    n_real: int = 16,
    duration: float = 2.0,
    fps: int = 20,
    image_size: int = 320,
) -> int:
    seed_everything(0)
    rng = np.random.default_rng(0)

    src_dir = ROOT / "data" / "source_images"
    fake_dir = ROOT / "data" / "synthetic"
    real_dir = ROOT / "data" / "real" / "raw"
    feat_path = ROOT / "data" / "features" / "smoke.npz"

    clean_dir(src_dir)
    clean_dir(fake_dir)
    clean_dir(real_dir)
    feat_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Make procedural source images
    print(f"[1/5] creating {n_sources} source images -> {src_dir}")
    for i in range(n_sources):
        img = make_source_image(seed=100 + i, size=image_size)
        cv2.imwrite(str(src_dir / f"src_{i:03d}.png"), img)

    cfg = AffineTrajectoryConfig(duration_s=duration, fps=fps)

    # 2. Fakes
    print(f"[2/5] rendering {n_fake} fake clips -> {fake_dir}")
    for i in range(n_fake):
        img = cv2.imread(str(src_dir / f"src_{i % n_sources:03d}.png"))
        frames, _ = render_fake_clip(img, cfg=cfg, rng=rng)
        write_video(fake_dir / f"fake_{i:03d}.mp4", frames, fps=fps)

    # 3. Real-like
    print(f"[3/5] rendering {n_real} real-like clips -> {real_dir}")
    for i in range(n_real):
        img = cv2.imread(str(src_dir / f"src_{i % n_sources:03d}.png"))
        frames = render_real_like_clip(img, cfg, rng)
        write_video(real_dir / f"real_{i:03d}.mp4", frames, fps=fps)

    # 4. Feature extraction via the script
    print("[4/5] extracting features")
    rc = subprocess.call(
        [
            sys.executable, "-m", "scripts.extract_features",
            "--out", str(feat_path),
            "--max-frames", "60",
            "--work-side", "200",
        ],
        cwd=str(ROOT),
    )
    if rc != 0:
        print(f"extract_features failed with rc={rc}")
        return rc

    # Quick residual sanity printout
    data = np.load(feat_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    names = list(data["feature_names"])
    mae_idx = names.index("mae_mean")
    ssim_idx = names.index("ssim_mean")
    print("\n  mae_mean  (fake should be ~0, real should be noticeably higher):")
    print(f"    fake:  {X[y==1, mae_idx].mean():.4f} ± {X[y==1, mae_idx].std():.4f}")
    print(f"    real:  {X[y==0, mae_idx].mean():.4f} ± {X[y==0, mae_idx].std():.4f}")
    print("  ssim_mean (fake should be ~1, real should be lower):")
    print(f"    fake:  {X[y==1, ssim_idx].mean():.4f} ± {X[y==1, ssim_idx].std():.4f}")
    print(f"    real:  {X[y==0, ssim_idx].mean():.4f} ± {X[y==0, ssim_idx].std():.4f}")

    # 5. Train & evaluate
    print("\n[5/5] training classical baseline")
    rc = subprocess.call(
        [
            sys.executable, "-m", "scripts.train_classical",
            "--features", str(feat_path),
            "--test-size", "0.3",
        ],
        cwd=str(ROOT),
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
