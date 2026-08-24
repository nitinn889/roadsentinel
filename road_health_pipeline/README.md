# RoadSentinel end-to-end prototype

This prototype separates the Raspberry Pi edge stage from GPU inference:

`RGB camera -> Pi quality/filtering/metadata/queue -> GPU SAM2 + DINOv2 + FAISS -> anomaly candidates -> SAM2 refinement -> area/GPS/water/severity -> JSON`

CARLA depth is used only as evaluation ground truth.

## What the supplied specification establishes

The source material says the RDD2020 preparation produced 6,472 images with no XML damage annotations, but it explicitly says these were **not** manually verified as perfectly healthy. It also says the previous Colab run failed at SAM2 initialization because the PyTorch build had no CUDA support. Therefore this project does not claim that a memory bank already exists; `build_memory_bank.py` must create and `validate_memory_bank.py` must validate it.

## Installation

### GPU machine

Use a CUDA-enabled PyTorch build appropriate for your NVIDIA driver. Do not install the CPU-only wheel. Then:

```bash
cd road_health_pipeline
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install -r requirements-gpu.txt
```

Install the official SAM 2 repository and its dependencies according to its current instructions. The code expects:

- `sam2/build_sam.py`
- `sam2/sam2_image_predictor.py`
- `configs/sam2.1/sam2.1_hiera_s.yaml`
- `checkpoints/sam2.1_hiera_small.pt`

A CUDA check must pass before building:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print('CUDA:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available; install a CUDA-enabled PyTorch build')
PY
```

### Raspberry Pi

```bash
cd road_health_pipeline
python3 -m venv .venv-pi
source .venv-pi/bin/activate
pip install -r requirements-pi.txt
```

The Pi code intentionally does not import SAM2, DINOv2, FAISS, or CUDA libraries.

### CARLA machine

Use the already-working CARLA Docker setup and the existing Python 3.10 environment. Install the CARLA Python client in that environment if not already present.

## Build and validate the memory bank

On the GPU machine:

```bash
python memory_bank/build_memory_bank.py --healthy-dir data/healthy_roads
python memory_bank/validate_memory_bank.py
```

The builder keeps the original DINOv2 ViT-S/14 design, but it never runs k-center on the full multi-million-point matrix. It random-presamples a bounded pool and performs blockwise farthest-point selection.

## Single-image smoke test

After a successful memory-bank build:

```bash
python inference/run_inference.py path/to/frame.png --device cuda --output output/smoke.json
```

Expected output is JSON with:

- image-level anomaly score
- candidate potholes
- bounding boxes and mask areas
- area in m² when altitude metadata is present
- GPS coordinates when metadata exists
- `estimated_depth_m: null` unless a metric RGB depth estimator is provided
- water flag and confidence from a heuristic
- severity score

## Inference API

Start on the GPU machine:

```bash
python inference/server.py --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

The Pi sends `image` and a `metadata_json` form field to `/infer`.

## CARLA simulation

The controller uses the road section in the supplied specification and a nadir camera. It prints the actual footprint and calculates the overlap implied by altitude, FOV, speed, and interval.

```bash
python carla_sim/drone_controller.py --altitude 30 --speed 8.33 --interval 3
```

Controls: `w=forward`, `s=reverse`, `d=stop`, `a/e=rotate`, `q=quit`.

Important: 8.33 m/s × 3 s is only ~25 m of travel; 70% overlap is **not** guaranteed until the camera footprint is calculated. Use the printed overlap to tune altitude, FOV, speed, or interval.

For synchronized RGB + CARLA ground-truth depth:

```bash
python carla_sim/rgb_depth_capture.py --out output/carla_sync
```

Depth should only be used in evaluation.

## Pi processing

To preprocess an existing directory of CARLA images as if they arrived at the Pi:

```bash
python pi_edge/edge_processor.py --input-dir output/carla_run/rgb --once
```

To continuously upload queued frames:

```bash
python pi_edge/uploader.py --url http://GPU_MACHINE_IP:8000/infer
```

The Pi module rejects blurry, badly exposed, and near-duplicate frames, resizes oversized frames, JPEG-compresses them, attaches telemetry, stores them locally, and retries transmission later.

## Depth

The real system is RGB-only. `NullDepthEstimator` is the default, intentionally returning no metric depth. This avoids fabricating a scientific result when no trained RGB metric-depth model was supplied.

To use a separately supplied metric depth model, wrap it with `ExternalMetricDepthEstimator` and pass a callable that maps `RGB -> HxW depth in metres`.

CARLA raw depth is decoded only for ground-truth evaluation:

```bash
python evaluation/depth_metrics.py predicted_depth.npy output/carla_sync/depth_gt/frame_00000000.png
```

Metrics: MAE, RMSE, and relative error.

## Scientific limitations

1. The FAISS memory bank represents normal/healthy-road appearance as defined by the no-XML-damage dataset filter; it is not a pothole classifier.
2. A high DINOv2 anomaly score can correspond to cracks, repairs, debris, shadows, or unusual texture.
3. The supplied prototype uses spatial grouping plus heuristic appearance/shape features and SAM2 box refinement to form pothole candidates. A trained pothole classifier/segmenter should replace or augment this stage for a research-grade detector.
4. RGB-only metric depth requires a trained/provided model or another validated geometric source. Without that, depth is reported as null.
5. The water flag is a heuristic and should be replaced with a trained classifier if it becomes a key evaluation metric.
6. The CARLA GPS conversion is a configurable local simulation georeference; it is not real-world GNSS.
