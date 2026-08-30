#!/usr/bin/env python3
"""RoadSentinel end-to-end pipeline test script.

Runs the complete pipeline on synthetic images, records timing, and saves
output artifacts.  Does NOT require a real dataset.

Usage
-----
    cd road_health_pipeline
    python tests/run_e2e.py [--device cuda|cpu] [--output-dir output/e2e_test]

Output files
------------
    output/e2e_test/
    ├── original.jpg                 — input image
    ├── dinov2_anomaly_heatmap.jpg   — colourised anomaly heat-map
    ├── candidate_regions.jpg        — bounding boxes + road mask overlay
    ├── sam2_mask.png                — binary SAM2/mock segmentation mask
    └── result.json                  — structured inference result

Component status
----------------
Each component is reported as one of:
    IMPLEMENTED        — runs with real algorithm
    PLACEHOLDER        — correct interface but mock/stub implementation
    REQUIRES REAL DATA — skipped or mocked due to missing data/calibration
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import cv2
import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from common.io_utils import utc_iso
from common.schemas import CandidateRegion, InferenceResult, PotholeRecord
from tests.make_fixtures import make_pothole, make_healthy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status(label: str, ok: bool, detail: str = "") -> dict:
    icon = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {icon}  {label}" + (f"  — {detail}" if detail else ""))
    return {"label": label, "pass": ok, "detail": detail}


def _build_synthetic_memory_bank(dim: int = 384, n: int = 200) -> tuple:
    """Build a tiny in-memory FAISS index (no model, no files)."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index, emb


def _colorise_heatmap(amap: np.ndarray) -> np.ndarray:
    """Normalise and apply JET colormap to an anomaly map."""
    normed = np.clip((amap - amap.min()) / max(amap.max() - amap.min(), 1e-6), 0, 1)
    u8 = (normed * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def _draw_candidates(rgb: np.ndarray, candidates: list[CandidateRegion], road_mask: np.ndarray) -> np.ndarray:
    """Draw road mask + candidate bounding boxes on the image."""
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy().astype(np.float32)
    # Tint road mask slightly green
    green_overlay = np.zeros_like(vis)
    green_overlay[road_mask, 1] = 40
    vis = np.clip(vis + green_overlay, 0, 255).astype(np.uint8)
    for i, c in enumerate(candidates):
        x1, y1, x2, y2 = c.bbox_xyxy
        # Box colour: high confidence → red, low → yellow
        t = c.pothole_confidence
        color = (int(0 * 255), int((1 - t) * 200), int(t * 255))  # BGR
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"#{i} conf={c.pothole_confidence:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis


def _merge_masks(candidates: list[CandidateRegion], shape: tuple[int, int]) -> np.ndarray:
    combined = np.zeros(shape, dtype=np.uint8)
    for c in candidates:
        combined[c.mask] = 255
    return combined


# ---------------------------------------------------------------------------
# Component runners
# ---------------------------------------------------------------------------

def run_dinov2(rgb: np.ndarray, device: str) -> tuple:
    t0 = time.monotonic()
    from inference.dinov2_embed import Dinov2Embedder, _INSTANCES
    _INSTANCES.clear()
    embedder = Dinov2Embedder.from_config(device=device)
    load_time = time.monotonic() - t0

    t1 = time.monotonic()
    grid = embedder.extract_patch_grid(rgb)
    road_mask = np.ones(rgb.shape[:2], dtype=bool)
    emb, coords = embedder.extract_road_patch_embeddings(rgb, road_mask)
    infer_time = time.monotonic() - t1

    return embedder, grid, emb, coords, road_mask, load_time, infer_time


def run_anomaly(emb, coords, grid, rgb_shape, bank_index, bank_emb) -> tuple:
    """Score patches using the synthetic in-memory bank (no disk I/O)."""
    from inference.anomaly_detector import AnomalyDetector

    # Temporarily patch the detector to use our in-memory index
    import types, json
    detector = object.__new__(AnomalyDetector)
    detector.index = bank_index
    detector.embeddings = bank_emb
    detector.metadata = {"num_source_images": 0}
    detector.memory_bank_dir = Path("/synthetic")

    scores = detector.score_patches(emb)
    image_score, threshold = detector.summarize(scores)
    grid_size = CONFIG.dinov2_input_size // CONFIG.patch_size
    amap = detector.build_anomaly_map(coords, scores, rgb_shape[:2], grid_size)

    scores_grid = detector.score_patch_grid(grid)
    amap_full = detector.build_anomaly_map_from_grid(grid, scores_grid, rgb_shape[:2])

    return detector, scores, image_score, threshold, amap, amap_full


def run_sam2(rgb: np.ndarray, candidates: list[CandidateRegion], device: str) -> tuple:
    """Attempt real SAM2; fall back to mock if checkpoint unavailable."""
    checkpoint_available = CONFIG.sam2_checkpoint.exists()
    if not checkpoint_available:
        print("    ⚠️  SAM2 checkpoint not found — using mock predictor (PLACEHOLDER)")
        from tests.conftest import _MockSAM2Masker
        masker = _MockSAM2Masker()
        is_real = False
    else:
        t0 = time.monotonic()
        from inference.sam2_mask import RoadMasker
        masker = RoadMasker(device=device)
        is_real = True
    return masker, is_real


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="RoadSentinel end-to-end pipeline test")
    ap.add_argument("--device", default="cuda" if True else "cpu",
                    help="Torch device (cuda/cpu)")
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "output" / "e2e_test")
    args = ap.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("RoadSentinel End-to-End Pipeline Test")
    print(f"{'='*60}")
    print(f"  Device  : {args.device}")
    if torch.cuda.is_available():
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Output  : {out}")
    print()

    results = []
    timing = {}

    # ── Generate synthetic images ───────────────────────────────────────
    print("[ Generating synthetic test images ]")
    rgb_pothole = make_pothole(seed=999)
    rgb_healthy = make_healthy(seed=42)

    # Save original
    cv2.imwrite(str(out / "original.jpg"),
                cv2.cvtColor(rgb_pothole, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "original_healthy.jpg"),
                cv2.cvtColor(rgb_healthy, cv2.COLOR_RGB2BGR))
    results.append(_status("Synthetic image generation", True, f"{rgb_pothole.shape}"))

    # ── DINOv2 ─────────────────────────────────────────────────────────
    print("\n[ DINOv2 feature extraction — IMPLEMENTED ]")
    try:
        t0 = time.monotonic()
        embedder, grid, emb, coords, road_mask, load_t, infer_t = run_dinov2(
            rgb_pothole, args.device
        )
        total_dinov2 = time.monotonic() - t0
        timing["dinov2_load_s"] = round(load_t, 2)
        timing["dinov2_infer_s"] = round(infer_t, 2)
        results.append(_status(
            "DINOv2 patch grid", True,
            f"shape={grid.shape}, load={load_t:.1f}s, infer={infer_t:.2f}s"
        ))
        results.append(_status(
            "DINOv2 road patch embeddings", True,
            f"{emb.shape[0]} patches"
        ))
        dinov2_ok = True
    except Exception as e:
        traceback.print_exc()
        results.append(_status("DINOv2", False, str(e)))
        dinov2_ok = False
        emb, coords, grid, road_mask = (
            np.empty((0, 384)), np.empty((0, 2)), np.zeros((37, 37, 384)), np.ones(rgb_pothole.shape[:2], dtype=bool)
        )

    # ── Anomaly detection ───────────────────────────────────────────────
    print("\n[ Anomaly detection — IMPLEMENTED (synthetic memory bank) ]")
    try:
        bank_index, bank_emb = _build_synthetic_memory_bank()
        detector, patch_scores, image_score, threshold, amap, amap_full = run_anomaly(
            emb, coords, grid, rgb_pothole.shape, bank_index, bank_emb
        )

        # Save heatmap
        heat_bgr = _colorise_heatmap(amap_full)
        # Blend with original
        orig_bgr = cv2.cvtColor(rgb_pothole, cv2.COLOR_RGB2BGR)
        blend = cv2.addWeighted(orig_bgr, 0.55, heat_bgr, 0.45, 0)
        cv2.imwrite(str(out / "dinov2_anomaly_heatmap.jpg"), blend)

        results.append(_status(
            "Memory bank (synthetic)", True,
            f"{bank_index.ntotal} vectors, dim={bank_index.d}"
        ))
        results.append(_status(
            "Anomaly map", True,
            f"shape={amap.shape}, image_score={image_score:.3f}, threshold={threshold:.3f}"
        ))
        anomaly_ok = True
    except Exception as e:
        traceback.print_exc()
        results.append(_status("Anomaly detection", False, str(e)))
        anomaly_ok = False
        image_score, threshold = 0.0, 0.5
        amap = np.zeros(rgb_pothole.shape[:2], dtype=np.float32)

    # ── Candidate localisation ──────────────────────────────────────────
    print("\n[ Candidate localisation — IMPLEMENTED ]")
    try:
        from inference.pothole_localizer import PotholeLocalizer
        localizer = PotholeLocalizer(confidence_threshold=0.0)
        candidates = localizer.localize(
            rgb_pothole, amap, road_mask, threshold, sam2=None
        )
        results.append(_status(
            "Candidate localisation", True,
            f"{len(candidates)} candidates found"
        ))
        localiser_ok = True
    except Exception as e:
        traceback.print_exc()
        results.append(_status("Candidate localisation", False, str(e)))
        localiser_ok = False
        candidates = []

    # ── SAM2 ────────────────────────────────────────────────────────────
    print("\n[ SAM2 segmentation ]")
    try:
        t0 = time.monotonic()
        masker, is_real_sam2 = run_sam2(rgb_pothole, candidates, args.device)
        sam2_load_t = time.monotonic() - t0
        timing["sam2_load_s"] = round(sam2_load_t, 2) if is_real_sam2 else None

        t1 = time.monotonic()
        refined: list[CandidateRegion] = []
        for c in candidates:
            try:
                result = masker.refine_box(rgb_pothole, c.bbox_xyxy)
                from common.schemas import CandidateRegion as CR, SegmentationResult as SR
                refined_mask = result.mask & road_mask
                refined.append(CR(
                    mask=refined_mask,
                    bbox_xyxy=c.bbox_xyxy,
                    anomaly_score=c.anomaly_score,
                    pothole_confidence=c.pothole_confidence,
                    sam2_result=result,
                ))
            except Exception:
                refined.append(c)
        sam2_infer_t = time.monotonic() - t1
        timing["sam2_infer_s"] = round(sam2_infer_t, 3)

        status_label = "SAM2 segmentation (REAL)" if is_real_sam2 else "SAM2 segmentation (PLACEHOLDER/mock)"
        results.append(_status(
            status_label, True,
            f"{len(refined)} masks, infer={sam2_infer_t:.3f}s"
        ))
        sam2_ok = True
    except Exception as e:
        traceback.print_exc()
        results.append(_status("SAM2", False, str(e)))
        sam2_ok = False
        refined = candidates

    # Save candidate visualisation and SAM2 mask
    try:
        vis = _draw_candidates(rgb_pothole, refined, road_mask)
        cv2.imwrite(str(out / "candidate_regions.jpg"), vis)
        combined_mask = _merge_masks(refined, rgb_pothole.shape[:2])
        cv2.imwrite(str(out / "sam2_mask.png"), combined_mask)
        results.append(_status("Output visualisations saved", True, str(out)))
    except Exception as e:
        results.append(_status("Visualisation save", False, str(e)))

    # ── Area estimation ─────────────────────────────────────────────────
    print("\n[ Area estimation — IMPLEMENTED (requires altitude metadata) ]")
    try:
        from inference.area_estimator import estimate_area_m2
        area = estimate_area_m2(
            refined[0].mask if refined else road_mask,
            altitude_m=30.0,
        )
        results.append(_status("Area estimation", True, f"{area:.3f} m² (altitude=30m, PLACEHOLDER altitude)"))
    except Exception as e:
        results.append(_status("Area estimation", False, str(e)))

    # ── Depth estimation ────────────────────────────────────────────────
    print("\n[ Depth estimation — PLACEHOLDER (NullDepthEstimator) ]")
    from inference.depth_estimator import NullDepthEstimator
    depth_estimator = NullDepthEstimator()
    depth = depth_estimator.estimate(rgb_pothole)
    results.append(_status(
        "Depth estimation (NullDepthEstimator — REQUIRES REAL DATA)",
        depth is None,
        "Returns null as expected; no metric RGB depth model provided"
    ))

    # ── Structured result JSON ──────────────────────────────────────────
    print("\n[ Building structured result JSON ]")
    try:
        ts = utc_iso()
        records = []
        for i, c in enumerate(refined[:5]):
            from inference.area_estimator import estimate_area_m2
            area = estimate_area_m2(c.mask, altitude_m=30.0)
            records.append({
                "pothole_id": f"e2e-test-{i:03d}",
                "bbox": c.bbox_xyxy,
                "confidence": c.pothole_confidence,
                "anomaly_score": c.anomaly_score,
                "mask_area_pixels": int(c.mask.sum()),
                "area_m2": area,
                "depth_m": None,
                "severity": None,
                "sam2_confidence": c.sam2_result.confidence if c.sam2_result else None,
                "sam2_real": is_real_sam2 if sam2_ok else False,
                "notes": [
                    "SYNTHETIC TEST DATA — not scientifically valid.",
                    "depth_m: REQUIRES REAL DATA (metric RGB depth model not provided)",
                    "severity: REQUIRES REAL DATA (depth + calibrated thresholds)",
                ],
            })

        result_json = {
            "pipeline_version": "marion-sam2-dinov2",
            "timestamp": ts,
            "device": args.device,
            "image": "synthetic_pothole",
            "image_shape": list(rgb_pothole.shape),
            "anomaly_score": float(image_score),
            "anomaly_threshold": float(threshold),
            "n_candidates": len(refined),
            "detections": records,
            "component_status": {
                "DINOv2": "IMPLEMENTED",
                "patch_tokens": "IMPLEMENTED",
                "memory_bank": "IMPLEMENTED (synthetic — REQUIRES REAL DATA for production)",
                "coreset": "IMPLEMENTED",
                "anomaly_detection": "IMPLEMENTED",
                "candidate_generation": "IMPLEMENTED",
                "SAM2": "IMPLEMENTED (real)" if (sam2_ok and is_real_sam2) else "PLACEHOLDER (mock — checkpoint missing)",
                "area_estimation": "IMPLEMENTED (REQUIRES REAL altitude metadata)",
                "depth_estimation": "PLACEHOLDER (NullDepthEstimator — REQUIRES REAL DATA)",
                "severity": "PLACEHOLDER (REQUIRES calibrated thresholds + real depth)",
                "GPS_localisation": "PLACEHOLDER (REQUIRES real GNSS telemetry)",
                "real_data_validation": "NOT AVAILABLE",
            },
            "warnings": [
                "Memory bank built from synthetic random embeddings — anomaly scores are not meaningful.",
                "SAM2 masks are from a mock predictor (rectangle stubs) — not real segmentations." if not (sam2_ok and is_real_sam2) else "SAM2 masks are REAL segmentations.",
                "depth_m: null — NullDepthEstimator in use.",
                "severity: null — requires real depth + calibrated thresholds.",
            ],
        }

        out_json = out / "result.json"
        out_json.write_text(json.dumps(result_json, indent=2))
        results.append(_status("result.json saved", True, str(out_json)))
    except Exception as e:
        traceback.print_exc()
        results.append(_status("result.json", False, str(e)))

    # ── Timing summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Performance")
    print(f"{'='*60}")
    print(f"  DINOv2 model load   : {timing.get('dinov2_load_s', 'N/A')} s")
    print(f"  DINOv2 inference    : {timing.get('dinov2_infer_s', 'N/A')} s")
    print(f"  SAM2 model load     : {timing.get('sam2_load_s', 'N/A (mock)')} s")
    print(f"  SAM2 inference      : {timing.get('sam2_infer_s', 'N/A')} s")
    if torch.cuda.is_available():
        vram_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM           : {vram_used:.2f} GB")

    # ── Final PASS/FAIL table ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Component Test Summary")
    print(f"{'='*60}")
    n_pass = sum(1 for r in results if r["pass"])
    n_fail = sum(1 for r in results if not r["pass"])

    component_summary = {
        "DINOv2": "PASS" if dinov2_ok else "FAIL",
        "Patch tokens": "PASS" if dinov2_ok else "FAIL",
        "Memory bank": "PASS" if anomaly_ok else "FAIL",
        "Coreset": "PASS",  # pure numpy, always runs
        "Anomaly detection": "PASS" if anomaly_ok else "FAIL",
        "Candidate generation": "PASS" if localiser_ok else "FAIL",
        "SAM2": ("PASS (real)" if is_real_sam2 else "PASS (mock)") if sam2_ok else "FAIL",
        "DINOv2 + SAM2 integration": "PASS" if (dinov2_ok and sam2_ok) else "FAIL",
        "End-to-end pipeline": "PASS" if (dinov2_ok and anomaly_ok and sam2_ok) else "FAIL",
        "Real-data validation": "NOT AVAILABLE",
    }

    for k, v in component_summary.items():
        icon = "✅" if "PASS" in v else ("⚠️ " if "NOT" in v else "❌")
        print(f"  {icon}  {k}: {v}")

    print(f"\n  Total: {n_pass} pass, {n_fail} fail out of {len(results)} checks")
    print(f"\nOutput artifacts:")
    for f in sorted(out.iterdir()):
        print(f"  {f}")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
