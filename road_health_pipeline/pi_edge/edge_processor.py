from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import hashlib
import json
import logging
import threading
import time

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from config import CONFIG
from common.io_utils import find_images, load_json, save_json, utc_iso

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pi_edge")


def quality_metrics(bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "mean_brightness": float(gray.mean()),
        "width": int(bgr.shape[1]),
        "height": int(bgr.shape[0]),
    }


def perceptual_hash(bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (16, 16), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).astype(np.uint8).flatten()


def hamming(a, b) -> int:
    return int(np.count_nonzero(a != b))


def preprocess(bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    metrics = quality_metrics(bgr)
    if metrics["sharpness"] < CONFIG.edge_sharpness_min:
        raise ValueError(f"blurred frame: sharpness={metrics['sharpness']:.1f}")
    if not CONFIG.edge_brightness_min <= metrics["mean_brightness"] <= CONFIG.edge_brightness_max:
        raise ValueError(f"bad exposure: brightness={metrics['mean_brightness']:.1f}")

    h, w = bgr.shape[:2]
    scale = min(1.0, CONFIG.edge_max_image_side / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return bgr, metrics


class EdgeProcessor:
    def __init__(self, queue_dir: Path = CONFIG.edge_queue_dir, uploader_url: str = CONFIG.uploader_url):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.uploader_url = uploader_url
        self._recent_hashes: deque[np.ndarray] = deque(maxlen=10)
        self.stop_event = threading.Event()

    def process_frame(self, bgr: np.ndarray, metadata: dict) -> Path | None:
        try:
            out_img, metrics = preprocess(bgr)
        except ValueError as exc:
            log.warning("Frame rejected: %s", exc)
            return None
        ph = perceptual_hash(out_img)
        if any(hamming(ph, prev) <= CONFIG.edge_duplicate_phash_distance for prev in self._recent_hashes):
            log.info("Frame rejected as duplicate")
            return None
        self._recent_hashes.append(ph)
        frame_id = metadata.get("frame_id", int(time.time() * 1000))
        stem = f"frame_{frame_id:08d}"
        image_path = self.queue_dir / f"{stem}.jpg"
        meta_path = self.queue_dir / f"{stem}.json"
        metadata = dict(metadata)
        metadata["timestamp"] = metadata.get("timestamp", utc_iso())
        metadata["edge_quality"] = metrics
        ok, encoded = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG.edge_jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        image_path.write_bytes(encoded.tobytes())
        save_json(metadata, meta_path)
        return image_path

    def flush(self, once: bool = False) -> None:
        while not self.stop_event.is_set():
            for image_path in sorted(self.queue_dir.glob("*.jpg")):
                meta_path = image_path.with_suffix(".json")
                try:
                    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                    with image_path.open("rb") as f:
                        r = requests.post(
                            self.uploader_url,
                            files={"image": (image_path.name, f, "image/jpeg")},
                            data={"metadata_json": json.dumps(metadata)},
                            timeout=60,
                        )
                    if r.ok:
                        result_path = image_path.with_suffix(".result.json")
                        result_path.write_text(r.text, encoding="utf-8")
                        image_path.unlink(missing_ok=True)
                        meta_path.unlink(missing_ok=True)
                        log.info("Uploaded %s", image_path.name)
                    else:
                        log.warning("Upload failed HTTP %s: %s", r.status_code, r.text[:200])
                except Exception as exc:
                    log.warning("Upload unavailable: %s", exc)
            if once:
                break
            time.sleep(2.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=None)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--uploader-url", default=CONFIG.uploader_url)
    args = ap.parse_args()
    edge = EdgeProcessor(uploader_url=args.uploader_url)
    if args.input_dir:
        for image_path in find_images(args.input_dir):
            meta_path = image_path.with_suffix(".json")
            metadata = load_json(meta_path) if meta_path.exists() else {"source_image": str(image_path)}
            bgr = cv2.imread(str(image_path))
            if bgr is not None:
                edge.process_frame(bgr, metadata)
    edge.flush(once=args.once)


if __name__ == "__main__":
    main()
