#!/usr/bin/env python3
"""Run one real DINOv2/SAM2 inference and export the shared feature record."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PIPELINE_ROOT = REPO_ROOT / "road_health_pipeline"
for entry in (str(HERE), str(PIPELINE_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from export_features import build_feature_record
from feature_contract import save_feature_record
from inference.run_inference import infer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--segment-id", required=True)
    parser.add_argument("--day", required=True, type=int)
    parser.add_argument("--condition", help="Observed capture condition; omit when not verified")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--memory-bank",
        type=Path,
        default=PIPELINE_ROOT / "output" / "real_memory_bank",
    )
    args = parser.parse_args()

    result = infer(
        args.image,
        device=args.device,
        memory_bank_dir=args.memory_bank,
        road_segment_id=args.segment_id,
        test_mode_2d=True,
    )
    record = build_feature_record(
        result.to_dict(),
        segment_id=args.segment_id,
        day=args.day,
        original_filename=args.image.name,
        condition=args.condition,
    )
    save_feature_record(record, args.output_json)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
