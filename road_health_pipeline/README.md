# RoadSentinel — SAM2 + DINOv2 Road Health Pipeline

End-to-end prototype for detecting road surface anomalies (potholes and defects)
from aerial RGB imagery using DINOv2 feature extraction, a healthy-road memory bank,
and SAM2 segmentation.

```
RGB image → DINOv2 patch features → Healthy-road memory bank → Anomaly map
         → Candidate regions → SAM2 segmentation → Pothole mask
         → Area estimation → Depth interface → Severity interface → JSON
```

CARLA depth camera is used **only** as evaluation ground truth; the real pipeline
is RGB-only.

---

## Component Status

| Component | Status | Notes |
|---|---|---|
| DINOv2 feature extraction | **IMPLEMENTED** | ViT-S/14, patch tokens + CLS token |
| Healthy-road memory bank | **IMPLEMENTED** | FAISS inner-product index, coreset selection |
| Anomaly detection | **IMPLEMENTED** | kNN cosine distance, spatial heat-map |
| Candidate localisation | **IMPLEMENTED** | Connected components + heuristic scoring |
| SAM2 segmentation | **IMPLEMENTED** | Box prompt → mask + confidence |
| Area estimation | **IMPLEMENTED** | Requires altitude metadata |
| Depth estimation | **PLACEHOLDER** | NullDepthEstimator — requires metric RGB depth model |
| Severity scoring | **PLACEHOLDER** | Requires calibrated thresholds + real depth |
| GPS localisation | **PLACEHOLDER** | Requires real GNSS telemetry |
| Real-data validation | **NOT AVAILABLE** | Pending labelled dataset |

---

## GPU Requirements

| Resource | Minimum (tested) |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM) |
| CUDA | 12.x / 13.x |
| RAM | 16 GB recommended |

Memory bank building requires CUDA. Inference can run on CPU (slow).

---

## Installation

### 1. Python environment (GPU machine)

Use a CUDA-enabled PyTorch build matching your NVIDIA driver:

```bash
cd road_health_pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-gpu.txt
```

> **Note**: `requirements-gpu.txt` does **not** install a CUDA-enabled PyTorch
> automatically. Install the appropriate wheel first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Verify CUDA

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available; install a CUDA-enabled PyTorch build.")
PY
```

### 3. SAM2 installation

SAM2 is installed via pip as part of `requirements-gpu.txt`:

```bash
pip install sam2>=1.1.0
```

Verify:

```bash
python -c "from sam2.build_sam import build_sam2; print('SAM2 OK')"
```

### 4. SAM2 checkpoint download

The checkpoint is **not** committed to the repository (it is excluded by `.gitignore`).
Download it once:

```bash
wget -P road_health_pipeline/checkpoints/ \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

Expected path: `road_health_pipeline/checkpoints/sam2.1_hiera_small.pt` (~180 MB).

### 5. DINOv2 setup

DINOv2 is loaded automatically via `torch.hub` on first use:

```python
from inference.dinov2_embed import Dinov2Embedder
e = Dinov2Embedder.from_config(device="cuda")  # downloads ~330 MB on first run
```

The model is cached in `~/.cache/torch/hub/`.

### 6. Raspberry Pi

```bash
cd road_health_pipeline
python3 -m venv .venv-pi
source .venv-pi/bin/activate
pip install -r requirements-pi.txt
```

Pi code does **not** import SAM2, DINOv2, FAISS, or CUDA libraries.

---

## Configuration

All settings are in [`config.py`](config.py). Key parameters:

| Setting | Default | Environment variable |
|---|---|---|
| `device` | `"cuda"` | `ROADSENTINEL_DEVICE` |
| `camera_mode` | `"nadir"` | `ROADSENTINEL_CAMERA_MODE` |
| `dinov2_model_name` | `"dinov2_vits14"` | — |
| `sam2_checkpoint` | `checkpoints/sam2.1_hiera_small.pt` | — |
| `memory_bank_dir` | `output/memory_bank` | — |
| `anomaly_percentile` | `98.0` | — |
| `knn_k` | `5` | — |
| `pothole_confidence_threshold` | `0.55` | — |

Thresholds must be calibrated on a labelled validation set before deployment.

---

## DINOv2 Architecture Reference

| Property | Value |
|---|---|
| Model | `dinov2_vits14` (ViT-Small) |
| Feature dim | 384 |
| Patch size | 14 px |
| Input size | 518 px (37 × 37 = 1369 tokens) |
| Normalisation | ImageNet mean/std |
| Token layout | `x_norm_patchtokens` (1, 1369, 384) + `x_norm_clstoken` (1, 384) |

Token at grid `(r, c)` covers pixels `[r*14:(r+1)*14, c*14:(c+1)*14]` in the
resized 518×518 input. After upsampling the anomaly map back to original image
resolution, spatial correspondence is preserved by bilinear interpolation.

---

## Building the Memory Bank

Build from a directory of healthy-road images:

```bash
python memory_bank/build_memory_bank.py \
    --healthy-dir data/healthy_roads \
    --output-dir output/memory_bank \
    --device cuda
```

For CPU-only development (slow):

```bash
python memory_bank/build_memory_bank.py \
    --healthy-dir data/healthy_roads \
    --allow-cpu
```

Validate the result:

```bash
python memory_bank/validate_memory_bank.py
```

---

## Running Inference

Single image:

```bash
python inference/run_inference.py path/to/image.jpg \
    --device cuda \
    --memory-bank output/memory_bank \
    --output output/result.json
```

With telemetry metadata:

```bash
python inference/run_inference.py image.jpg \
    --metadata image_meta.json \
    --device cuda \
    --output output/result.json
```

Python API (model reuse across images):

```python
from inference.run_inference import load_pipeline, infer

pipeline = load_pipeline(device="cuda")
for path in image_list:
    result = infer(path, pipeline=pipeline)
```

Inference server (GPU machine):

```bash
python inference/server.py --host 0.0.0.0 --port 8000
```

---

## Running Tests

### Unit and integration tests

```bash
cd road_health_pipeline
source .venv/bin/activate

# All tests (DINOv2 tests require model; SAM2 real tests require checkpoint)
pytest tests/ -v --tb=short

# Only fast tests (no model loading)
pytest tests/ -v --tb=short -m "not slow"

# Explicit test classes
pytest tests/test_memory_bank.py tests/test_anomaly_detector.py -v
```

SAM2 checkpoint-dependent tests are automatically skipped when the checkpoint
file is absent. DINOv2 tests (`@pytest.mark.slow`) require the model to be
downloaded (~330 MB on first run).

### End-to-end test (no real dataset needed)

```bash
python tests/run_e2e.py --device cuda
```

Expected output directory: `output/e2e_test/`

---

## Expected Output

### JSON result schema

```json
{
  "image_path": "path/to/image.jpg",
  "timestamp": "2026-08-30T09:00:00Z",
  "frame_id": null,
  "telemetry": {},
  "image_shape": [720, 1280, 3],
  "anomaly_threshold": 0.42,
  "anomaly_score": 0.55,
  "potholes": [
    {
      "pothole_id": "20260830T090000Z-000",
      "timestamp": "2026-08-30T09:00:00Z",
      "latitude": null,
      "longitude": null,
      "altitude_m": null,
      "area_m2": null,
      "estimated_depth_m": null,
      "anomaly_score": 0.62,
      "pothole_confidence": 0.48,
      "severity_score": 0.29,
      "water_flag": false,
      "water_confidence": 0.0,
      "source_image": "path/to/image.jpg",
      "mask_area_px": 4200,
      "bbox_xyxy": [120, 80, 340, 250],
      "depth_source": "unavailable",
      "notes": ["Heuristic pothole candidate; generic anomaly detection is not a trained pothole classifier."]
    }
  ],
  "warnings": ["Metric RGB depth model was not provided; estimated_depth_m is null."]
}
```

### Output visualisations (from `run_e2e.py`)

| File | Description |
|---|---|
| `original.jpg` | Input image |
| `dinov2_anomaly_heatmap.jpg` | JET-colourised anomaly map overlaid on image |
| `candidate_regions.jpg` | Candidate bounding boxes + road mask tint |
| `sam2_mask.png` | Binary combined segmentation mask |
| `result.json` | Structured inference result |

---

## Scientific Limitations

1. **DINOv2 anomaly score ≠ pothole classifier.** High scores arise from road
   markings, shadows, repaired asphalt, stains, debris, cracks, and lighting
   changes — not only potholes.

2. **Memory bank quality depends on the healthy dataset.** The no-XML-damage
   filter is not a verified label of perfect health. Images used to build the
   bank should be reviewed.

3. **Heuristic confidence is not a trained score.** The formula weights
   (anomaly 0.40, shape 0.20, darkness 0.20, area 0.20) are heuristic starting
   points; do not report them as classifier accuracy.

4. **Depth is unavailable without a metric RGB depth model.** `estimated_depth_m`
   is `null` by default. The `NullDepthEstimator` is intentional and honest.

5. **Severity requires real depth and calibrated thresholds.** The current severity
   formula is a placeholder.

6. **GPS coordinates require real GNSS telemetry.** CARLA georeference is a
   simulation convenience; it is not real-world GNSS.

7. **No accuracy claims.** No detection accuracy, segmentation IoU, recall, or
   depth accuracy figures are reported because no labelled evaluation dataset
   has been processed on this branch.

---

## Git workflow

This code lives on branch `marion-sam2-dinov2`. Do **not** merge into `main`
without review.

```bash
git branch --show-current   # must be: marion-sam2-dinov2
git push -u origin marion-sam2-dinov2
```
