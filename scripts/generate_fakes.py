"""CLI: synthesize fake affine-spoof videos from still images.

Usage:
    python -m scripts.generate_fakes --n 50 --duration 3 --fps 24
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data.synthetic import (
    AffineTrajectoryConfig,
    load_image,
    render_fake_clip,
)
from src.data.video_io import write_video
from src.utils.seeds import seed_everything


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        type=Path,
        default=Path("data/source_images/faces"),
        help="Directory with source still images (searched recursively).",
    )
    p.add_argument("--out", type=Path, default=Path("data/synthetic"))
    p.add_argument("--n", type=int, default=50,
                   help="Total number of clips to generate")
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    images = list_images(args.src)
    if not images:
        raise SystemExit(
            f"No images found under {args.src}. Drop some .jpg/.png files there."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = AffineTrajectoryConfig(duration_s=args.duration, fps=args.fps)

    manifest: list[dict] = []
    for i in tqdm(range(args.n), desc="synthesizing fakes"):
        src_img_path = images[i % len(images)]
        img = load_image(src_img_path, max_side=args.max_side)
        frames, params = render_fake_clip(img, cfg=cfg, rng=rng)
        out_path = args.out / f"fake_{i:05d}.mp4"
        write_video(out_path, frames, fps=args.fps)
        manifest.append(
            {
                "path": str(out_path),
                "source_image": str(src_img_path),
                "n_frames": int(params.meta["n_frames"]),
                "fps": args.fps,
                "max_angle_deg": float(np.max(np.abs(params.angle_deg))),
                "max_zoom_dev": float(np.max(np.abs(params.zoom - 1.0))),
                "max_translate_frac": float(
                    np.max(np.abs(np.stack([params.tx_frac, params.ty_frac])))
                ),
            }
        )

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} clips -> {args.out}")


if __name__ == "__main__":
    main()
