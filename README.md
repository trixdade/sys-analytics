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
  utils/        # seeding, visualization
scripts/        # CLI entry points (generate, extract, train, eval)
configs/        # YAML configs
data/
  real/raw/         # <-- drop real selfie videos here (any .mp4/.mov/.avi)
  source_images/    # <-- drop single images here (used to synthesize fakes)
  synthetic/        # generated fake videos
  features/         # cached per-clip features
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Put source images into data/source_images/ (any jpg/png with faces)
# 2. Generate synthetic fake "videos" from them
python -m scripts.generate_fakes --n 50 --duration 3 --fps 24

# 3. Put real selfie videos into data/real/raw/
# 4. Extract classical features for all clips
python -m scripts.extract_features

# 5. Train + evaluate classical baseline
python -m scripts.train_classical
```
