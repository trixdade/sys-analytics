"""CLI: walk data/real/raw and data/synthetic, extract classical features,
and save them (together with labels) as a .npz file.

Label convention: 1 = fake (spoof), 0 = real.

Usage:
    python -m scripts.extract_features
    python -m scripts.extract_features --out data/features/v1.npz --max-frames 48
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.classical.affine_residual import clip_pair_residuals
from src.classical.features import FEATURE_NAMES, clip_features
from src.data.video_io import list_videos, read_clip


def extract_for_video(path: Path, max_frames: int, work_side: int, stride: int) -> np.ndarray:
    frames = read_clip(path, max_frames=max_frames)
    if stride > 1:
        frames = frames[::stride]
    residuals = clip_pair_residuals(frames, work_side=work_side, stride=1)
    return clip_features(residuals)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--real-dir", type=Path, default=Path("data/real/raw"))
    p.add_argument("--fake-dir", type=Path, default=Path("data/synthetic"))
    p.add_argument(
        "--out", type=Path, default=Path("data/features/baseline.npz")
    )
    p.add_argument("--max-frames", type=int, default=72,
                   help="Cap number of decoded frames per clip")
    p.add_argument("--work-side", type=int, default=256,
                   help="Downscale so max side == this before estimation")
    p.add_argument("--stride", type=int, default=1,
                   help="Use every k-th frame for residual pairs")
    args = p.parse_args()

    jobs: list[tuple[Path, int]] = []
    for v in list_videos(args.real_dir):
        jobs.append((v, 0))
    for v in list_videos(args.fake_dir):
        jobs.append((v, 1))

    if not jobs:
        raise SystemExit(
            "No videos found. Populate data/real/raw/ and data/synthetic/."
        )

    feats: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    for path, label in tqdm(jobs, desc="features"):
        try:
            f = extract_for_video(
                path, args.max_frames, args.work_side, args.stride
            )
        except Exception as e:
            print(f"[skip] {path}: {e}")
            continue
        feats.append(f)
        labels.append(label)
        paths.append(str(path))

    X = np.stack(feats, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        X=X,
        y=y,
        paths=np.asarray(paths),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    n_fake = int((y == 1).sum())
    n_real = int((y == 0).sum())
    print(
        f"saved {X.shape[0]} clips ({n_real} real / {n_fake} fake), "
        f"{X.shape[1]} features -> {args.out}"
    )


if __name__ == "__main__":
    main()
