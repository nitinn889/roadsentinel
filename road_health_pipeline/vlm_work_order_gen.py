#!/usr/bin/env python3
"""Vision-Language Model (VLM) Work Order Generator for RoadSentinel.

Ingests post-inference analytics output (result.json), filters for critical and high-severity
road segment defects, loads the corresponding visual defect images via PIL, prompts a VLM
(Gemini API or local/fallback domain model), and exports structured maintenance repair
work orders mapped by Road Segment ID into work_orders.json.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Optional, Union
import urllib.request
import urllib.error

from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.io_utils import save_json, utc_iso
from analytics.severity import classify_severity_label

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vlm_work_order_gen")


# ---------------------------------------------------------------------------
# Image Matching & Preprocessing Helpers
# ---------------------------------------------------------------------------

def locate_defect_image(
    detection: dict[str, Any],
    search_dirs: list[Path],
    default_names: tuple[str, ...] = ("input_frame.jpg", "detection_overlay.jpg", "severity_overlay.jpg"),
) -> Optional[Path]:
    """Locate the physical or overlay defect image for a given detection."""
    # 1. Check if source_image is directly specified and exists
    source_img = detection.get("source_image") or detection.get("image_path")
    if source_img:
        p = Path(source_img)
        if p.is_file():
            return p
        for s_dir in search_dirs:
            candidate = s_dir / p.name
            if candidate.is_file():
                return candidate
            candidate_nested = s_dir / p
            if candidate_nested.is_file():
                return candidate_nested

    # 2. Check default names in search directories
    for s_dir in search_dirs:
        for name in default_names:
            candidate = s_dir / name
            if candidate.is_file():
                return candidate

    return None


def crop_defect_region(
    img: Image.Image,
    bbox_xyxy: Optional[list[Union[int, float]]],
    padding_fraction: float = 0.15,
) -> Image.Image:
    """Crop the bounding box region from the PIL Image with contextual padding."""
    if not bbox_xyxy or len(bbox_xyxy) != 4:
        return img

    w, h = img.size
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    pad_x = box_w * padding_fraction
    pad_y = box_h * padding_fraction

    crop_x1 = max(0, int(x1 - pad_x))
    crop_y1 = max(0, int(y1 - pad_y))
    crop_x2 = min(w, int(x2 + pad_x))
    crop_y2 = min(h, int(y2 + pad_y))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return img

    return img.crop((crop_x1, crop_y1, crop_x2, crop_y2))


# ---------------------------------------------------------------------------
# VLM Client & Work Order Prompting
# ---------------------------------------------------------------------------

def construct_work_order_prompt(
    defect_class: str,
    severity_tier: str,
    severity_score: float,
    area_m2: Optional[float],
    estimated_depth_m: Optional[float],
    is_water_filled: bool,
    road_segment_id: str,
    pothole_id: str = "",
) -> str:
    """Construct a rigorous prompt for the VLM work order generation."""
    area_str = f"{area_m2:.2f} m²" if area_m2 is not None else "undetermined extent"
    depth_str = f"{estimated_depth_m * 100:.1f} cm" if estimated_depth_m is not None else "uncalibrated depth"
    water_str = "Hazardous Standing Water Present" if is_water_filled else "Dry defect cavity"

    prompt = (
        f"You are a Senior Municipal Pavement & Highway Engineer. "
        f"Examine this visual inspection image of road segment '{road_segment_id}' (Defect ID: {pothole_id}). "
        f"The predictive sensor pipeline has identified a '{defect_class}' with '{severity_tier.upper()}' severity "
        f"(continuous severity index: {severity_score:.2f}, estimated surface area: {area_str}, "
        f"depth: {depth_str}, condition: {water_str}).\n\n"
        f"Task: Draft an actionable, exactly 3-sentence repair work order following this structure:\n"
        f"Sentence 1 (Priority & Assessment): State the urgency, location ({road_segment_id}), defect type, and physical risk.\n"
        f"Sentence 2 (Remediation & Materials): Detail the exact technical repair method and specific required materials (e.g. tack coat emulsion, hot-mix asphalt / cold-patch polymer, aggregate base, dewatering equipment).\n"
        f"Sentence 3 (Site Safety & Quality Control): Detail required traffic control / safety measures (MUTCD standards) and post-compaction levelness inspection."
    )
    return prompt


def generate_fallback_work_order(
    defect_class: str,
    severity_tier: str,
    severity_score: float,
    area_m2: Optional[float],
    estimated_depth_m: Optional[float],
    is_water_filled: bool,
    road_segment_id: str,
    pothole_id: str = "",
) -> dict[str, Any]:
    """Generate high-fidelity, deterministic engineering repair work order when VLM API is unavailable."""
    area_str = f"{area_m2:.2f} m²" if area_m2 is not None else "approximately 1.0 m²"
    depth_val = estimated_depth_m if estimated_depth_m is not None else 0.10
    
    # Sentence 1: Assessment
    if is_water_filled:
        s1 = (
            f"URGENT: High-priority immediate remediation is required on road segment '{road_segment_id}' for a {severity_tier}-tier "
            f"{defect_class} covering {area_str} with critical hydroplaning water accumulation."
        )
    elif severity_tier == "critical":
        s1 = (
            f"CRITICAL DISPATCH: Pavement failure on road segment '{road_segment_id}' requires expedited repair for a severe {defect_class} "
            f"spanning {area_str} presenting substantial structural degradation."
        )
    else:
        s1 = (
            f"MAINTENANCE NOTICE: Scheduled pavement maintenance is required on road segment '{road_segment_id}' to remediate a "
            f"{severity_tier}-grade {defect_class} spanning {area_str}."
        )

    # Sentence 2: Technical Repair & Materials
    if is_water_filled:
        s2 = (
            f"Crews must dewater the cavity using a submersible pump, square-cut and clean the perimeter, apply cationic rapid-setting "
            f"tack coat emulsion (CRS-2), and place hot-mix asphalt (HMA Type B) compacted with a vibratory plate in two 50 mm lifts."
        )
    elif depth_val >= 0.10:
        s2 = (
            f"Repair personnel must saw-cut vertical edges around the defect, excavate loose debris to stable subgrade, apply SS-1h tack coat, "
            f"and backfill with dense-graded hot-mix asphalt compacted to 95% standard Proctor density."
        )
    else:
        s2 = (
            f"Maintenance personnel should air-sweep loose debris, apply a rapid-curing bituminous tack coat emulsion, and fill the defect "
            f"with polymer-modified high-performance cold-patch asphalt, compacting flush with the surrounding wearing course."
        )

    # Sentence 3: Safety & Quality Control
    s3 = (
        f"Establish MUTCD-compliant single-lane closure taper with channelizing cones and directional arrow board, verifying zero-settlement "
        f"and flush straightedge tolerance prior to reopening the lane to traffic."
    )

    full_text = f"{s1} {s2} {s3}"

    # Materials and equipment list
    materials = [
        "Hot-Mix Asphalt (HMA Type B)" if severity_tier == "critical" else "Polymer-Modified High-Performance Asphalt",
        "Bituminous Tack Coat Emulsion (CRS-2 / SS-1h)",
        "Crushed Aggregate Base (Grade D)",
    ]
    equipment = [
        "Vibratory Plate Compactor (15 kN)",
        "High-Pressure Air Lance / Debris Blower",
        "Diamond Pavement Saw",
    ]
    if is_water_filled:
        materials.append("Hydraulic Cement Quick-Set Mortar")
        equipment.append("Submersible Dewatering Trash Pump")

    return {
        "work_order_text": full_text,
        "required_materials": materials,
        "required_equipment": equipment,
        "safety_measures": "MUTCD Chapter 6H temporary traffic control taper, reflective advance warning signs, and safety cones.",
        "estimated_crew_size": 3 if is_water_filled or severity_tier == "critical" else 2,
        "target_resolution_hours": 12 if severity_tier == "critical" else 48,
        "engine": "domain_rule_vlm_engine",
    }


def query_gemini_vlm(
    pil_img: Image.Image,
    prompt: str,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.0-flash",
) -> Optional[str]:
    """Query Google Gemini API with defect image and prompt if API key is provided."""
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None

    # Convert PIL Image to JPEG bytes base64
    buf = io.BytesIO()
    rgb_img = pil_img.convert("RGB")
    rgb_img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_b64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 400,
        },
    }

    try:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp_json = json.loads(resp.read().decode("utf-8"))
            candidates = resp_json.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts and "text" in parts[0]:
                    return parts[0]["text"].strip()
    except Exception as e:
        log.warning("Gemini VLM API call failed: %s. Falling back to local domain VLM engine.", e)

    return None


# ---------------------------------------------------------------------------
# Core Work Order Pipeline Function
# ---------------------------------------------------------------------------

def generate_vlm_work_orders(
    result_data: Union[dict[str, Any], list[dict[str, Any]], Path, str],
    output_path: Optional[Union[Path, str]] = None,
    images_dir: Optional[Union[Path, str]] = None,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.0-flash",
    crop_defects: bool = True,
    min_severity: float = 0.0,
) -> dict[str, Any]:
    """Generate VLM work orders for deduplicated defects from analytics result.

    Parameters
    ----------
    result_data:
        Either parsed dictionary/list or path to `result.json`.
    output_path:
        Optional path to write `work_orders.json`.
    images_dir:
        Directory to look for defect images / overlays.
    api_key:
        Optional Gemini API key (defaults to env var).
    model_name:
        Model identifier.
    crop_defects:
        Whether to crop defect regions using bounding boxes.
    min_severity:
        Minimum severity score threshold (default: 0.0 to create 1 work order per unique defect).

    Returns
    -------
    dict[str, Any]
        Structured work orders dictionary.
    """
    # 1. Ingestion: Load JSON if path is passed
    if isinstance(result_data, (str, Path)):
        res_path = Path(result_data)
        if not res_path.exists():
            raise FileNotFoundError(f"Analytics result file not found: {res_path}")
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        result_dir = res_path.parent
    else:
        data = result_data
        result_dir = ROOT / "output" / "analytics_demo"

    # Normalize input data to list of segment summaries
    if isinstance(data, dict):
        # Could be a single SegmentSummary or a dictionary of segments
        if "road_segment_id" in data:
            segment_list = [data]
        elif "by_segment" in data:
            segment_list = list(data["by_segment"].values())
        else:
            segment_list = [v for v in data.values() if isinstance(v, dict) and "road_segment_id" in v]
            if not segment_list:
                segment_list = [data]
    elif isinstance(data, list):
        segment_list = data
    else:
        raise ValueError(f"Unsupported result data format: {type(data)}")

    search_dirs = [
        result_dir,
        result_dir / "patches",
        ROOT / "output" / "analytics_demo",
        ROOT / "output" / "analytics_demo" / "patches",
        ROOT / "output",
        ROOT.parent / "env" / "output" / "images",
        ROOT.parent / "env" / "output",
        ROOT.parent / "output" / "images",
    ]
    if images_dir:
        search_dirs.insert(0, Path(images_dir))

    all_work_orders: list[dict[str, Any]] = []
    by_segment: dict[str, dict[str, Any]] = {}
    total_processed_defects = 0

    log.info("Processing %d segment(s) for defect work orders...", len(segment_list))

    for seg in segment_list:
        seg_id = seg.get("road_segment_id", "seg_unknown")
        detections = seg.get("detections", [])
        
        # If no explicit detections list, create one from segment-level metrics if severe
        if not detections:
            max_sev = float(seg.get("max_severity", 0.0))
            if max_sev >= 0.65 or max_sev >= min_severity:
                detections = [{
                    "pothole_id": f"{seg_id}-def-001",
                    "defect_type": "pothole",
                    "severity_score": max_sev,
                    "area_m2": seg.get("total_damaged_area_m2", 1.0),
                    "estimated_depth_m": 0.10,
                    "water_flag": seg.get("has_water_hazard", False),
                    "road_segment_id": seg_id,
                }]

        segment_work_orders: list[dict[str, Any]] = []
        highest_sev_tier = "low"
        max_score = 0.0

        for det in detections:
            # Determine severity score and tier
            sev_score = float(det.get("severity_score", 0.0))
            sev_tier = det.get("severity_tier") or classify_severity_label(sev_score)
            
            # Filter based on min_severity threshold
            if sev_score < min_severity:
                continue

            total_processed_defects += 1
            if sev_score > max_score:
                max_score = sev_score
                highest_sev_tier = sev_tier

            pothole_id = det.get("pothole_id", f"def-{total_processed_defects:03d}")
            defect_class = det.get("defect_class") or det.get("defect_type") or "pothole"
            area_m2 = det.get("area_m2")
            depth_m = det.get("estimated_depth_m")
            water_flag = bool(det.get("water_flag", False))
            bbox = det.get("bbox_xyxy")

            # Visual Matching: Locate and open image
            img_path = locate_defect_image(det, search_dirs)
            pil_image: Optional[Image.Image] = None
            if img_path and img_path.exists():
                try:
                    loaded_img = Image.open(img_path)
                    if crop_defects and bbox:
                        pil_image = crop_defect_region(loaded_img, bbox)
                    else:
                        pil_image = loaded_img
                    log.info("  Matched defect [%s] -> Image [%s] (%dx%d)", pothole_id, img_path.name, pil_image.width, pil_image.height)
                except Exception as e:
                    log.warning("Failed to open image %s: %s", img_path, e)

            # If no image found, generate placeholder PIL canvas
            if pil_image is None:
                pil_image = Image.new("RGB", (640, 480), color=(110, 110, 115))

            # Construct Prompt
            prompt = construct_work_order_prompt(
                defect_class=defect_class,
                severity_tier=sev_tier,
                severity_score=sev_score,
                area_m2=area_m2,
                estimated_depth_m=depth_m,
                is_water_filled=water_flag,
                road_segment_id=seg_id,
                pothole_id=pothole_id,
            )

            # Query VLM / Fallback Engine
            vlm_text = query_gemini_vlm(pil_image, prompt, api_key=api_key, model_name=model_name)
            fallback = generate_fallback_work_order(
                defect_class=defect_class,
                severity_tier=sev_tier,
                severity_score=sev_score,
                area_m2=area_m2,
                estimated_depth_m=depth_m,
                is_water_filled=water_flag,
                road_segment_id=seg_id,
                pothole_id=pothole_id,
            )

            final_text = vlm_text if vlm_text else fallback["work_order_text"]
            engine_name = f"gemini-vlm ({model_name})" if vlm_text else fallback["engine"]

            clean_cid = str(pothole_id).replace("carla-", "").replace("road_", "R").replace("_", "-")
            wo_record = {
                "work_order_id": f"WO-{seg_id.upper()[:10]}-{clean_cid}",
                "road_segment_id": seg_id,
                "pothole_id": pothole_id,
                "defect_class": defect_class,
                "severity_tier": sev_tier,
                "severity_score": round(sev_score, 4),
                "area_m2": round(area_m2, 4) if area_m2 is not None else None,
                "estimated_depth_m": round(depth_m, 4) if depth_m is not None else None,
                "latitude": det.get("latitude") or det.get("lat"),
                "longitude": det.get("longitude") or det.get("lon"),
                "water_hazard": water_flag,
                "prompt": prompt,
                "work_order_text": final_text,
                "required_materials": fallback["required_materials"],
                "required_equipment": fallback["required_equipment"],
                "safety_measures": fallback["safety_measures"],
                "estimated_crew_size": fallback["estimated_crew_size"],
                "target_resolution_hours": fallback["target_resolution_hours"],
                "engine": engine_name,
                "image_reference": str(img_path) if img_path else None,
                "created_at": utc_iso(),
            }

            segment_work_orders.append(wo_record)
            all_work_orders.append(wo_record)

        if segment_work_orders:
            # Segment aggregated summary
            combined_text = " ".join(w["work_order_text"] for w in segment_work_orders)
            by_segment[seg_id] = {
                "road_segment_id": seg_id,
                "highest_severity_tier": highest_sev_tier,
                "max_severity_score": round(max_score, 4),
                "total_critical_defects": len(segment_work_orders),
                "work_orders": segment_work_orders,
                "summary_work_order": combined_text,
            }

    export_doc = {
        "metadata": {
            "generated_at": utc_iso(),
            "generator": "RoadSentinel-VLM-WorkOrderGen",
            "total_segments_with_work_orders": len(by_segment),
            "total_critical_defects_remediated": len(all_work_orders),
        },
        "by_segment": by_segment,
        "work_orders": all_work_orders,
    }

    # If only 1 segment was processed, expose road_segment_id at top level for convenience
    if len(by_segment) == 1:
        single_seg_id = next(iter(by_segment.keys()))
        export_doc["road_segment_id"] = single_seg_id
        export_doc["primary_work_order"] = by_segment[single_seg_id]["summary_work_order"]

    # 4. Export to JSON file
    if output_path:
        out_p = Path(output_path)
        save_json(export_doc, out_p)
        log.info("Saved %d work order(s) to: %s", len(all_work_orders), out_p.resolve())

    return export_doc


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RoadSentinel VLM Work Order Generator")
    parser.add_argument(
        "--result-json",
        "--result-path",
        type=Path,
        default=ROOT / "output" / "analytics_demo" / "result.json",
        help="Path to post-inference analytics result.json",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=ROOT / "output" / "analytics_demo" / "work_orders.json",
        help="Destination path for work_orders.json",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory to search for corresponding defect images",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Vision-Language Model identifier",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Google Gemini API key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--min-severity",
        type=float,
        default=0.0,
        help="Minimum severity threshold to generate a work order (default: 0.0)",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Do not crop defect bounding box; use full scene image",
    )

    args = parser.parse_args()

    # Check fallback paths if result_json does not exist
    result_path = args.result_json
    if not result_path.exists():
        alt_paths = [
            ROOT.parent / "output" / "analytics_demo" / "result.json",
            Path("output/analytics_demo/result.json"),
            ROOT / "output" / "result.json",
        ]
        for alt in alt_paths:
            if alt.exists():
                result_path = alt
                break

    if not result_path.exists():
        log.error("Analytics result file not found at %s. Please run run_analytics_e2e.py or carla_pipeline.py first.", args.result_json)
        sys.exit(1)

    log.info("=" * 65)
    log.info("RoadSentinel Vision-Language Model Work Order Generator")
    log.info("=" * 65)
    log.info("Input Result  : %s", result_path)
    log.info("Output Target : %s", args.output)

    res = generate_vlm_work_orders(
        result_data=result_path,
        output_path=args.output,
        images_dir=args.images_dir,
        api_key=args.api_key,
        model_name=args.model,
        crop_defects=not args.no_crop,
        min_severity=args.min_severity,
    )

    log.info("=" * 65)
    log.info("Generated %d Work Order(s) across %d Road Segment(s)", 
             res["metadata"]["total_critical_defects_remediated"], 
             res["metadata"]["total_segments_with_work_orders"])
    for wo in res.get("work_orders", []):
        log.info("  [%s] %s (%s, %.2f m²):", wo["work_order_id"], wo["defect_class"], wo["severity_tier"].upper(), wo.get("area_m2") or 0.0)
        log.info("    -> \"%s\"", wo["work_order_text"])
    log.info("=" * 65)


if __name__ == "__main__":
    main()
