# RoadSentinel — Part 1: DINOv2 Anomaly Detection Foundation

## What this does
Detects road anomalies (potholes, cracks, water patches) WITHOUT needing any
labeled pothole dataset. It only needs photos of healthy, undamaged road to
build a reference, then flags anything in a new photo that looks visually
different from that reference.

## How it works
1. `dinov2_features.py` — loads a frozen, pretrained DINOv2 model and extracts
   a dense grid of feature vectors per image (one vector per image patch).
2. `anomaly_detector.py` — builds a "memory bank" of patch features from your
   healthy-road images, then scores new images by how far each patch's
   features are from the nearest thing in that memory bank (k-NN distance).
   Higher distance = more anomalous.
3. `thresholding.py` — converts the raw anomaly heatmap into candidate
   bounding boxes, which will be fed into SAM2 in Part 2 for precise
   segmentation.
4. `test_part1_pipeline.py` — ties it all together: builds/loads the
   reference, scores a test image, and saves visual outputs.

## Setup
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Before running
1. Put a handful of clean/healthy road photos in `data/healthy_road/`
   (5-10 images is a reasonable starting point).
2. Put one or more road photos you want to test detection on in
   `data/test_road/`.

## Run
```bash
cd src
python test_part1_pipeline.py
```

Outputs land in `outputs/`:
- `<name>_heatmap.jpg` — the anomaly heatmap overlaid on the original photo
- `<name>_boxes.jpg` — candidate anomaly regions drawn as boxes

The reference memory bank is cached at `models/healthy_road_reference.npy`
after the first run, so you don't rebuild it every time — delete that file
if you want to rebuild it from a new/updated set of healthy-road images.

## Verified working
This pipeline was smoke-tested end-to-end with a synthetic road image
containing a fake dark "pothole" patch — the anomaly detector correctly
localized it, and the resulting bounding box landed precisely on the patch.
See `outputs/smoke_test_*.jpg` for that verification run.
(`_sandbox_smoke_test.py` is the test harness used for this — safe to
delete, it's not part of the real pipeline, it just avoids needing a
model download for a quick offline sanity check.)

## Next: Part 2
Feed these candidate boxes into SAM2 to get precise pixel-level pothole
masks instead of rough rectangles.
