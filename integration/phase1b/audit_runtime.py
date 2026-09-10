#!/usr/bin/env python3
"""RoadSentinel Phase 1B: Runtime Benchmark Audit.

Benchmarks inference latency and throughput on available hardware:
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)
- Software: Python 3.14, PyTorch 2.13.0+cu130, CUDA 12.8, Ultralytics YOLOv8n

Benchmark Protocol:
- Batch size: 1 (real-time edge stream simulation)
- Fixed image resolutions: 512x512 for YOLO, 518x518 for DINOv2
- Warm-up iterations: 20 runs
- Timed repetitions: 100 timed runs
- Timing method: torch.cuda.Event(enable_timing=True) with strict cuda.synchronize()
- Reports: Mean, Median, Standard Deviation, p95, p99, Peak GPU Memory (MB)

Pipelines profiled:
1. YOLOv8n Alone
2. DINOv2 ViT-S/14 Domain Gate
3. Staged Pipeline (YOLO + DINO Gate + Reliability Scoring)

Outputs:
- artifacts/phase1b/runtime_benchmarks.csv
"""

from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("phase1b_runtime")

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts" / "phase1b"
YOLO_WEIGHTS = WORKSPACE_ROOT / "yolo" / "weights" / "best.pt"
REF_BANK_PATH = WORKSPACE_ROOT / "cross_domain" / "results" / "train_domain_reference_embeddings.npz"


def run_runtime_benchmark() -> None:
    log.info("Starting Runtime Benchmark Audit...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU runtime benchmarking.")

    device_name = torch.cuda.get_device_name(0)
    device_props = torch.cuda.get_device_properties(0)
    total_mem_mb = device_props.total_memory / (1024 ** 2)
    log.info("Hardware: %s | Total VRAM: %.1f MB", device_name, total_mem_mb)

    # 1. Load models
    log.info("Loading YOLOv8n from %s...", YOLO_WEIGHTS)
    yolo = YOLO(str(YOLO_WEIGHTS))
    # Dummy run to move to cuda
    dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
    _ = yolo(dummy_img, conf=0.25, verbose=False)

    log.info("Loading DINOv2 ViT-S/14...")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").cuda().eval()

    ref_data = np.load(REF_BANK_PATH)
    ref_embeds = torch.from_numpy(ref_data["embeds"]).cuda().float()
    ref_embeds = ref_embeds / torch.norm(ref_embeds, dim=1, keepdim=True)

    dummy_tensor_dino = torch.randn(1, 3, 518, 518, device="cuda")

    records: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Benchmark 1: YOLOv8n Alone
    # -------------------------------------------------------------------------
    log.info("Profiling YOLOv8n Alone (N=100 timed runs, 20 warm-up)...")
    torch.cuda.reset_peak_memory_stats()
    # Warmup
    for _ in range(20):
        _ = yolo(dummy_img, conf=0.25, verbose=False)
    torch.cuda.synchronize()

    yolo_times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(100):
        start_event.record()
        _ = yolo(dummy_img, conf=0.25, verbose=False)
        end_event.record()
        torch.cuda.synchronize()
        yolo_times.append(start_event.elapsed_time(end_event))

    peak_mem_yolo = torch.cuda.max_memory_allocated() / (1024 ** 2)
    yolo_times = np.array(yolo_times)
    records.append({
        "pipeline_tier": "YOLOv8n_Alone",
        "input_resolution": "512x512",
        "batch_size": 1,
        "warmup_runs": 20,
        "timed_runs": 100,
        "mean_latency_ms": round(float(np.mean(yolo_times)), 3),
        "median_latency_ms": round(float(np.median(yolo_times)), 3),
        "std_latency_ms": round(float(np.std(yolo_times)), 3),
        "p95_latency_ms": round(float(np.percentile(yolo_times, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(yolo_times, 99)), 3),
        "mean_throughput_fps": round(1000.0 / float(np.mean(yolo_times)), 2),
        "peak_gpu_memory_mb": round(peak_mem_yolo, 1),
        "device": device_name,
    })

    # -------------------------------------------------------------------------
    # Benchmark 2: DINOv2 Domain Gate Alone
    # -------------------------------------------------------------------------
    log.info("Profiling DINOv2 Domain Gate (N=100 timed runs, 20 warm-up)...")
    torch.cuda.reset_peak_memory_stats()
    # Warmup
    for _ in range(20):
        with torch.no_grad():
            emb = dino(dummy_tensor_dino)
            emb = emb / torch.norm(emb, dim=1, keepdim=True)
            sims = torch.mm(ref_embeds, emb.t())
            _ = torch.topk(sims, k=20, dim=0)
    torch.cuda.synchronize()

    dino_times = []
    for _ in range(100):
        start_event.record()
        with torch.no_grad():
            emb = dino(dummy_tensor_dino)
            emb = emb / torch.norm(emb, dim=1, keepdim=True)
            sims = torch.mm(ref_embeds, emb.t())
            _ = torch.topk(sims, k=20, dim=0)
        end_event.record()
        torch.cuda.synchronize()
        dino_times.append(start_event.elapsed_time(end_event))

    peak_mem_dino = torch.cuda.max_memory_allocated() / (1024 ** 2)
    dino_times = np.array(dino_times)
    records.append({
        "pipeline_tier": "DINOv2_Domain_Gate_Alone",
        "input_resolution": "518x518",
        "batch_size": 1,
        "warmup_runs": 20,
        "timed_runs": 100,
        "mean_latency_ms": round(float(np.mean(dino_times)), 3),
        "median_latency_ms": round(float(np.median(dino_times)), 3),
        "std_latency_ms": round(float(np.std(dino_times)), 3),
        "p95_latency_ms": round(float(np.percentile(dino_times, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(dino_times, 99)), 3),
        "mean_throughput_fps": round(1000.0 / float(np.mean(dino_times)), 2),
        "peak_gpu_memory_mb": round(peak_mem_dino, 1),
        "device": device_name,
    })

    # -------------------------------------------------------------------------
    # Benchmark 3: Staged Pipeline (YOLO + DINO Gate + Reliability Filter)
    # -------------------------------------------------------------------------
    log.info("Profiling Staged Pipeline (N=100 timed runs, 20 warm-up)...")
    torch.cuda.reset_peak_memory_stats()
    # Warmup
    for _ in range(20):
        _ = yolo(dummy_img, conf=0.25, verbose=False)
        with torch.no_grad():
            emb = dino(dummy_tensor_dino)
            emb = emb / torch.norm(emb, dim=1, keepdim=True)
            sims = torch.mm(ref_embeds, emb.t())
            _ = torch.topk(sims, k=20, dim=0)
    torch.cuda.synchronize()

    staged_times = []
    for _ in range(100):
        start_event.record()
        _ = yolo(dummy_img, conf=0.25, verbose=False)
        with torch.no_grad():
            emb = dino(dummy_tensor_dino)
            emb = emb / torch.norm(emb, dim=1, keepdim=True)
            sims = torch.mm(ref_embeds, emb.t())
            _ = torch.topk(sims, k=20, dim=0)
        end_event.record()
        torch.cuda.synchronize()
        staged_times.append(start_event.elapsed_time(end_event))

    peak_mem_staged = torch.cuda.max_memory_allocated() / (1024 ** 2)
    staged_times = np.array(staged_times)
    records.append({
        "pipeline_tier": "Staged_Pipeline_Gated",
        "input_resolution": "512x512 + 518x518",
        "batch_size": 1,
        "warmup_runs": 20,
        "timed_runs": 100,
        "mean_latency_ms": round(float(np.mean(staged_times)), 3),
        "median_latency_ms": round(float(np.median(staged_times)), 3),
        "std_latency_ms": round(float(np.std(staged_times)), 3),
        "p95_latency_ms": round(float(np.percentile(staged_times, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(staged_times, 99)), 3),
        "mean_throughput_fps": round(1000.0 / float(np.mean(staged_times)), 2),
        "peak_gpu_memory_mb": round(peak_mem_staged, 1),
        "device": device_name,
    })

    df_out = pd.DataFrame(records)
    out_csv = ARTIFACTS_DIR / "runtime_benchmarks.csv"
    df_out.to_csv(out_csv, index=False)
    log.info("Saved runtime benchmarks to %s", out_csv)
    for _, r in df_out.iterrows():
        log.info("  [%s] Mean: %.2f ms | Median: %.2f ms | p95: %.2f ms | Throughput: %.1f FPS | Peak VRAM: %.1f MB",
                 r["pipeline_tier"], r["mean_latency_ms"], r["median_latency_ms"], r["p95_latency_ms"],
                 r["mean_throughput_fps"], r["peak_gpu_memory_mb"])


if __name__ == "__main__":
    run_runtime_benchmark()
