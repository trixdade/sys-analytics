# sys-analytics — Affine-Spoof Liveness Detection

Detect whether a short (~3 s) selfie video is a genuine recording of a live person
or a static photo animated via time-varying affine transforms (zoom / rotate /
pan / shear) — a common attack against liveness checks in mobile apps.

## Core hypothesis

For a **fake** clip, every pair of consecutive frames is related by an (almost
exact) affine transform: `frame_{t+1} ≈ A_t · frame_t`. After estimating that
transform and warping one frame onto the other, the residual is near zero.

For a **real** clip, this is not true — there is non-rigid motion (blinks,
micro-expressions), parallax from 3D head geometry, sensor noise that changes
between frames, and lighting variation. The residual is strictly positive.

The first baseline is built entirely on this observation and uses no training
on video content — only a simple classifier on top of residual statistics.

## Roadmap (current scope: stages 0–2)

- [x] **0.** Project infrastructure
- [x] **1.** Data: synthetic fake generator + video I/O + dataset scan
- [x] **2.** Classical baseline: frame-to-frame affine residual + LR/GBM classifier
- [ ] 3. Learned classifier (2D + temporal pooling, 3D CNN)
- [ ] 4. Hybrid: residual maps as extra input channels
- [ ] 5. Stress tests (compression, noise, non-affine warps, low-motion reals)

## Layout

```
src/
  data/         # synthetic generator, video I/O, dataset scanning
  classical/    # affine residual estimation, clip-level feature extraction
  utils/        # seeding
scripts/        # CLI entry points
  download_faces.py     # fetch a small curated face dataset from OSS repos
  generate_fakes.py     # synthesize affine-spoof clips from still images
  extract_features.py   # compute classical features for all clips
  train_classical.py    # fit + evaluate LogReg / GBM classifier
  make_sample.py        # render a few illustrative clips into samples/
  smoke_test.py         # end-to-end sanity check
configs/        # YAML configs
data/
  real/raw/             # <-- drop real selfie videos here (.mp4/.mov/.avi)
  source_images/
    faces/              # real faces (populated by download_faces.py)
    procedural/         # procedural images (populated by smoke_test.py)
  synthetic/            # generated fake videos
  features/             # cached per-clip features + saved models
samples/                # illustrative generated fakes, committed to the repo
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Download a small face dataset (~15 images from OSS face-recognition repos)
python -m scripts.download_faces

# 2. Generate synthetic affine-spoof videos from them
python -m scripts.generate_fakes --n 50 --duration 3 --fps 24

# 3. Put real selfie videos into data/real/raw/

# 4. Extract classical features for all clips (both real and fake)
python -m scripts.extract_features

# 5. Train + evaluate the classical baseline
python -m scripts.train_classical
```

### End-to-end sanity check

Without supplying any real videos you can still exercise the full pipeline —
the smoke test synthesizes both sides by applying a non-affine elastic warp +
sensor noise on top of the affine trajectory for the "real" class:

```bash
python -m scripts.download_faces   # once
python -m scripts.smoke_test       # takes ~15 s on CPU
```

On this purely synthetic setup the classical baseline reaches ROC-AUC ≈ 1.0
with MAE/SSIM-based features — fake clips have residual `mae_mean ≈ 0.008`
after affine alignment vs `≈ 0.027` for the "real" class, confirming the core
hypothesis. On genuine real videos the gap will shrink; that is where
learned models and residual-map hybrids (stages 3–4) come in.
