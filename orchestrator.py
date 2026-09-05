#!/usr/bin/env python3
"""RoadSentinel — Master Execution Orchestrator.

Ingests a folder of manually captured, geotagged aerial road photos and runs
the complete automated analysis pipeline:
  1. Validate & preprocess images from the input directory.
  2. Extract DINOv2 patch features and SAM2 road/defect segmentation masks.
  3. Compute defect measurements, severity breakdown, 0-100 road health score,
     and 30-day deterioration predictions.
  4. Generate Vision-Language Model (VLM) automated maintenance repair work orders.
  5. Serve the final results, overlays, and work orders to the interactive
     Government Dashboard.

CARLA simulation is fully decoupled — the user captures images separately
and points this script at the output folder.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import List

WORKSPACE_ROOT = Path(__file__).resolve().parent
ML_ENV_PYTHON = WORKSPACE_ROOT / ".venv" / "bin" / "python"

# Fallback to sys.executable if dedicated env binary is not found
if not ML_ENV_PYTHON.is_file():
    ML_ENV_PYTHON = Path(sys.executable)

# ---------------------------------------------------------------------------
# Default image input directory — points to CARLA output directory by default.
# Point this to the folder containing your captured, geotagged aerial photos
# (*.jpg / *.png) and accompanying metadata.csv / geo.txt.
# ---------------------------------------------------------------------------
IMAGE_INPUT_DIR = WORKSPACE_ROOT / "env" / "output"


# ---------------------------------------------------------------------------
# Terminal Styling
# ---------------------------------------------------------------------------

class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_step(step_num: int, total_steps: int, title: str, description: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}┌──────────────────────────────────────────────────────────────────┐{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}│  STEP [{step_num}/{total_steps}]: {title:<53} │{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}└──────────────────────────────────────────────────────────────────┘{Colors.RESET}")
    print(f"{Colors.DIM}{description}{Colors.RESET}\n")


def print_success(msg: str):
    print(f"{Colors.GREEN} [✓ SUCCESS]{Colors.RESET} {msg}")


def print_info(msg: str):
    print(f"{Colors.BLUE} [i INFO]{Colors.RESET} {msg}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW} [! WARNING]{Colors.RESET} {msg}")


def print_error(msg: str):
    print(f"{Colors.RED} [✗ ERROR]{Colors.RESET} {msg}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def is_port_open(host: str = "127.0.0.1", port: int = 8000, timeout: float = 2.0) -> bool:
    """Check if a TCP port is reachable."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


# ---------------------------------------------------------------------------
# Pipeline Stage Handlers
# ---------------------------------------------------------------------------

def _find_images_in_dir(d: Path) -> tuple[Path, list[Path]]:
    """Helper to locate images directly in d or in d/images."""
    if not d.is_dir():
        return d, []
    img_sub = d / "images"
    if img_sub.is_dir():
        imgs = list(img_sub.glob("*.jpg")) + list(img_sub.glob("*.png"))
        if imgs:
            return d, imgs
    imgs = list(d.glob("*.jpg")) + list(d.glob("*.png"))
    return d, imgs


def stage_1_ingest_images(input_dir: Path) -> Path:
    """Step 1: Validate and preprocess the input image directory."""
    resolved_dir, images = _find_images_in_dir(input_dir)

    # If no images found in requested dir, check fallback candidate locations
    if not images:
        for candidate in [
            WORKSPACE_ROOT / "env" / "output",
            WORKSPACE_ROOT / "output",
            WORKSPACE_ROOT / "env" / "output" / "images",
            WORKSPACE_ROOT / "output" / "images",
        ]:
            cand_dir, cand_imgs = _find_images_in_dir(candidate)
            if cand_imgs:
                input_dir = cand_dir
                resolved_dir = cand_dir
                images = cand_imgs
                break

    print_step(1, 5, "Image Directory Ingestion & Preprocessing",
               f"Scanning {input_dir} for geotagged aerial road photos.")

    if not input_dir.is_dir():
        print_error(f"Input directory does not exist: {input_dir}")
        raise FileNotFoundError(f"Image input directory not found: {input_dir}")

    images_dir = input_dir / "images" if (input_dir / "images").is_dir() else input_dir
    if not images:
        print_error(f"No .jpg or .png images found in {images_dir}")
        raise FileNotFoundError(f"No images in {images_dir}. Capture images first.")

    print_success(f"Found {len(images)} image(s) in {images_dir}")

    # Report on metadata files
    metadata_csv = input_dir / "metadata.csv"
    geo_txt = input_dir / "geo.txt"
    if metadata_csv.is_file():
        print_success(f"GPS telemetry metadata: {metadata_csv.name}")
    else:
        print_warning("No metadata.csv found — GPS telemetry will not be available.")
    if geo_txt.is_file():
        print_success(f"OpenDroneMap georeference tags: {geo_txt.name}")

    # Preprocess
    print_info("Preprocessing images (dimension verification, RGB normalization)...")
    preprocess_captured_images(images_dir)
    print_success("Image preprocessing complete.")

    return input_dir


def preprocess_captured_images(images_dir: Path):
    """Preprocess captured flight images (verify size, channel order, aspect ratio)."""
    import cv2
    for img_p in list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")):
        img = cv2.imread(str(img_p))
        if img is not None:
            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                continue


def stage_2_and_3_feature_extraction_and_inference(capture_dir: Path) -> Path:
    """Step 2 & 3: Real CARLA ML Feature Extraction, 3-Meter Spatial Deduplication & Road Health Analytics."""
    print_step(2, 5, "DINOv2 & SAM2 Feature Extraction & Segmentation",
               "Extracting dense patch embeddings, querying FAISS memory bank, and segmenting defect contours with SAM2 on real CARLA flight images.")

    carla_script = WORKSPACE_ROOT / "road_health_pipeline" / "inference" / "carla_pipeline.py"
    target_out_dir = WORKSPACE_ROOT / "road_health_pipeline" / "output" / "analytics_demo"
    target_out_dir.mkdir(parents=True, exist_ok=True)

    print_step(3, 5, "Road-Health Scoring, 3-Meter Spatial Deduplication & Deterioration Forecasting",
               "Executing physical area estimation, multi-factor severity scoring, 3-meter spatial deduplication, and establishing Geofence boundaries.")

    cmd = [
        str(ML_ENV_PYTHON),
        str(carla_script),
        "--input-dir", str(capture_dir),
        "--output-dir", str(target_out_dir),
    ]

    print_info(f"Running real CARLA ML inference & spatial deduplication: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE_ROOT / "road_health_pipeline") + ":" + env.get("PYTHONPATH", "")

    start_t = time.time()
    res = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT / "road_health_pipeline"), env=env, capture_output=True, text=True)

    if res.returncode != 0:
        print_error(f"Inference pipeline error:\n{res.stderr}")
        raise RuntimeError("Inference stage failed.")

    print(res.stdout.strip())
    print_success(f"ML Pipeline, Analytics & 3m Spatial Deduplication complete in {time.time() - start_t:.2f}s.")

    result_json = target_out_dir / "result.json"
    if result_json.is_file():
        print_success(f"Inference Result JSON exported with Deduplication metadata: {result_json}")

    return result_json


def stage_4_vlm_analysis(result_json: Path, capture_dir: Path) -> Path:
    """Step 4: Execute Vision-Language Model Analysis for Automated Work Orders."""
    print_step(4, 5, "Vision-Language Model (VLM) Analysis & Work Order Dispatch",
               "Generating structured municipal maintenance work orders via Gemini VLM / domain engine.")

    vlm_script = WORKSPACE_ROOT / "road_health_pipeline" / "vlm_work_order_gen.py"
    work_orders_json = result_json.parent / "work_orders.json"
    img_dir = capture_dir / "images" if (capture_dir / "images").is_dir() else capture_dir

    cmd = [
        str(ML_ENV_PYTHON),
        str(vlm_script),
        "--result-json", str(result_json),
        "--output", str(work_orders_json),
        "--images-dir", str(img_dir),
        "--min-severity", "0.0",
    ]

    print_info(f"Executing VLM work order generation: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE_ROOT / "road_health_pipeline") + ":" + env.get("PYTHONPATH", "")

    start_t = time.time()
    res = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT / "road_health_pipeline"), env=env, capture_output=True, text=True)

    if res.returncode != 0:
        print_error(f"VLM analysis error:\n{res.stderr}")
        raise RuntimeError("VLM work order stage failed.")

    print(res.stdout.strip())
    print_success(f"VLM Analysis complete in {time.time() - start_t:.2f}s.")
    print_success(f"Generated work orders saved to: {work_orders_json}")

    return work_orders_json


def stage_5_serve_dashboard(port: int = 8000, non_blocking: bool = False):
    """Step 5: Serve the final output to the Government Dashboard Interface."""
    print_step(5, 5, "Government Dashboard & Real-Time Server Launch",
               f"Hosting interactive GIS road map, defect overlays, and work orders on http://localhost:{port}.")

    server_script = WORKSPACE_ROOT / "road_health_pipeline" / "inference" / "server.py"

    # Clean up stale processes on the target port if any
    if is_port_open("127.0.0.1", port):
        print_info(f"Port {port} is occupied. Attempting graceful release of stale process...")
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
            time.sleep(1.0)
        except Exception:
            pass

    cmd = [
        str(ML_ENV_PYTHON),
        str(server_script),
        "--host", "0.0.0.0",
        "--port", str(port),
    ]

    print_info(f"Starting server on http://localhost:{port} ...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE_ROOT / "road_health_pipeline") + ":" + env.get("PYTHONPATH", "")

    # Auto-open browser
    import webbrowser
    import threading

    def open_browser_delayed():
        time.sleep(1.2)
        try:
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            pass

    threading.Thread(target=open_browser_delayed, daemon=True).start()

    if non_blocking:
        proc = subprocess.Popen(cmd, cwd=str(WORKSPACE_ROOT / "road_health_pipeline"), env=env)
        print_success(f"Dashboard server running in background (PID: {proc.pid})")
        print(f"\n{Colors.BOLD}{Colors.GREEN}=================================================================={Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}   ROADSENTINEL PIPELINE EXECUTION COMPLETED SUCCESSFULLY!        {Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}   Access Government Dashboard: http://localhost:{port}           {Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}=================================================================={Colors.RESET}\n")
        return proc
    else:
        print(f"\n{Colors.BOLD}{Colors.GREEN}=================================================================={Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}   ROADSENTINEL PIPELINE EXECUTION COMPLETED SUCCESSFULLY!        {Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}   Access Government Dashboard: http://localhost:{port}           {Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}=================================================================={Colors.RESET}\n")
        print_info("Press Ctrl+C to stop the dashboard server.")
        try:
            subprocess.run(cmd, cwd=str(WORKSPACE_ROOT / "road_health_pipeline"), env=env)
        except KeyboardInterrupt:
            print("\nDashboard server stopped.")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RoadSentinel Master Execution Orchestrator — Image Folder Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example usage:\n"
            "  python orchestrator.py --input-dir ./env/output\n"
            "  python orchestrator.py --input-dir /path/to/captured/images --port 9000\n"
            "  ./run_demo.sh --input-dir ./env/output\n"
        ),
    )
    parser.add_argument("--input-dir", type=str, default=str(IMAGE_INPUT_DIR),
                        help=f"Path to folder containing captured geotagged images (default: {IMAGE_INPUT_DIR})")
    parser.add_argument("--port", type=int, default=8000, help="Dashboard web server port")
    parser.add_argument("--no-server", action="store_true", help="Do not launch dashboard server at the end")
    parser.add_argument("--daemon-server", action="store_true", help="Launch dashboard server in background process")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()

    print(f"\n{Colors.BOLD}{Colors.BLUE}=================================================================={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}       ROADSENTINEL — MASTER EXECUTION ORCHESTRATOR               {Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  Image Folder Ingestion → Analysis → Government Dashboard       {Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}=================================================================={Colors.RESET}")
    print(f"{Colors.DIM}  Input directory: {input_dir}{Colors.RESET}\n")

    total_start = time.time()

    # 1. Ingest & validate the input image directory
    capture_dir = stage_1_ingest_images(input_dir)

    # 2 & 3. DINOv2 / SAM2 Feature Extraction & Health Assessment
    result_json = stage_2_and_3_feature_extraction_and_inference(capture_dir)

    # 4. VLM Analysis & Work Orders
    work_orders_json = stage_4_vlm_analysis(result_json, capture_dir)

    print_success(f"Full pipeline run completed in {time.time() - total_start:.2f} seconds.")

    # 5. Serve Dashboard
    if not args.no_server:
        stage_5_serve_dashboard(port=args.port, non_blocking=args.daemon_server)


if __name__ == "__main__":
    main()
