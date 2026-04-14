"""Render a single illustrative fake-spoof clip + its source image into
samples/ so reviewers can see what the synthesizer produces without running
the pipeline themselves.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data.synthetic import AffineTrajectoryConfig, render_fake_clip
from src.data.video_io import write_video
from src.utils.seeds import seed_everything
from scripts.smoke_test import make_source_image


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    seed_everything(42)
    out = ROOT / "samples"
    out.mkdir(parents=True, exist_ok=True)

    img = make_source_image(seed=42, size=480)
    cv2.imwrite(str(out / "sample_source.png"), img)

    cfg = AffineTrajectoryConfig(
        duration_s=3.0,
        fps=24,
        max_angle_deg=18.0,          # a bit more aggressive for visibility
        max_zoom=0.22,
        max_translate_frac=0.12,
        max_shear=0.04,
        n_harmonics=2,
        crop_frac=0.78,
    )
    rng = np.random.default_rng(42)
    frames, params = render_fake_clip(img, cfg=cfg, rng=rng)
    write_video(out / "sample_fake.mp4", frames, fps=cfg.fps)

    print(f"wrote {out / 'sample_source.png'}")
    print(f"wrote {out / 'sample_fake.mp4'}  ({len(frames)} frames)")
    print(
        f"  max angle: {float(np.max(np.abs(params.angle_deg))):.1f} deg, "
        f"max zoom dev: {float(np.max(np.abs(params.zoom - 1.0))):.2f}, "
        f"max |translate|: "
        f"{float(np.max(np.abs(np.stack([params.tx_frac, params.ty_frac])))):.2f} frac"
    )


if __name__ == "__main__":
    main()
