# Step 1: Build the healthy-road memory bank (offline, one-time)

Run this on a laptop/desktop/cloud machine with a GPU -- **not** on the Pi.
It only needs to run once (or again if you add more healthy-road footage later).

## What it does

```
healthy road images
      |
      v
  SAM2 road mask   (segments road surface, drops sky/curb/vehicles)
      |
      v
  DINOv2 patch embeddings   (one feature vector per 14x14px patch, road patches only)
      |
      v
  coreset subsampling   (k-center greedy: keep a small, diverse subset)
      |
      v
  memory_bank/embeddings.npy + index.faiss + metadata.json
```

That memory bank is your definition of "what a healthy road looks like." Step 2
(inference on the Pi) will embed a live frame the same way and measure its
nearest-neighbor distance to this bank -- large distance = likely anomaly
(pothole, crack, patch, debris).

## Setup

```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git

# Download a SAM2 checkpoint (small/tiny is enough for masking, not full segmentation):
# https://github.com/facebookresearch/sam2#model-description
# Place it at the path set in config.SAM2_CHECKPOINT
```

## Dataset layout

```
data/healthy_roads/
  any/nested/folders/are/fine/
  img001.jpg
  img002.jpg
  ...
```

Only include images of genuinely healthy road surface. Any potholes, cracks, or
patches accidentally included here will get baked into the "normal" definition
and make the anomaly detector blind to that type of damage.

## Run

```bash
python build_memory_bank.py
```

Check `config.py` first -- in particular:
- `HEALTHY_ROADS_DIR` / `OUTPUT_DIR` paths
- `ROI_BOX_FRACTIONS` -- the default assumes the road occupies roughly the
  bottom 65% of the frame. Adjust to match your camera's mounting angle, or
  replace `RoadMasker.get_road_mask` with a fixed static mask if your camera
  doesn't move.
- `CORESET_RATIO` / `CORESET_MAX_POINTS` -- controls memory bank size. Bigger
  bank = better coverage of normal-road variation, but slower/heavier lookup
  on the Pi. 10-20k points is a reasonable starting budget for an 8GB Pi.

## Output

```
output/memory_bank/
  embeddings.npy    # (N, 384) float32 for dinov2_vits14 -- the coreset itself
  index.faiss        # FAISS flat-L2 index over embeddings.npy
  metadata.json       # what config was used, how many images/patches went in
```

Copy this whole `memory_bank/` folder to the Pi. It's what Step 2 (live
inference) will load and compare incoming frames against.

## Sanity-check before moving to Step 2

- Look at `metadata.json`: `n_skipped_images` should be small. A large number
  usually means the SAM2 ROI box isn't finding road in most frames -- fix
  `ROI_BOX_FRACTIONS` before proceeding.
- `n_coreset_points` should roughly match your `CORESET_RATIO` /
  `CORESET_MAX_POINTS` settings. If it's hitting the max cap every time,
  consider raising the cap or accepting the coverage tradeoff.
