"""Render a small set of illustrative fake-spoof clips + their source images
into samples/ so reviewers can see what the synthesizer produces without
running the pipeline themselves. Uses real face photos from
data/source_images/faces/ (fetched via scripts.download_faces).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data.synthetic import (
    AffineTrajectoryConfig,
    load_image,
    render_fake_clip,
)
from src.data.video_io import write_video
from src.utils.seeds import seed_everything


ROOT = Path(__file__).resolve().parents[1]

# Pick a couple of diverse source faces to showcase the generator on.
FEATURED = [
    "fr_obama.jpg",
    "dlib_tom_cruise.jpg",
    "fr_miranda.png",
]


def main() -> None:
    seed_everything(42)
    out = ROOT / "samples"
    out.mkdir(parents=True, exist_ok=True)

    faces_dir = ROOT / "data" / "source_images" / "faces"
    if not faces_dir.exists() or not any(faces_dir.iterdir()):
        raise SystemExit(
            f"No face images in {faces_dir}. "
            "Run: python -m scripts.download_faces"
        )

    cfg = AffineTrajectoryConfig(
        duration_s=3.0,
        fps=24,
        max_angle_deg=18.0,           # a bit more aggressive for visibility
        max_zoom=0.22,
        max_translate_frac=0.12,
        max_shear=0.04,
        n_harmonics=2,
        crop_frac=0.78,
    )
    rng = np.random.default_rng(42)

    for name in FEATURED:
        src = faces_dir / name
        if not src.exists():
            print(f"[skip] {src} not found")
            continue
        img = load_image(src, max_side=720)
        stem = src.stem
        cv2.imwrite(str(out / f"sample_{stem}_src.png"), img)
        frames, params = render_fake_clip(img, cfg=cfg, rng=rng)
        write_video(out / f"sample_{stem}_fake.mp4", frames, fps=cfg.fps)
        print(
            f"  {stem}: {len(frames)} frames, "
            f"max angle {float(np.max(np.abs(params.angle_deg))):.1f}°, "
            f"max zoom dev {float(np.max(np.abs(params.zoom - 1.0))):.2f}, "
            f"max |translate| "
            f"{float(np.max(np.abs(np.stack([params.tx_frac, params.ty_frac])))):.2f}"
        )
    print(f"\noutputs -> {out}")


if __name__ == "__main__":
    main()
