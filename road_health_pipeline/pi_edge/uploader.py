from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG
from pi_edge.edge_processor import EdgeProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path, default=CONFIG.edge_queue_dir)
    ap.add_argument("--url", default=CONFIG.uploader_url)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    EdgeProcessor(queue_dir=args.queue, uploader_url=args.url).flush(once=args.once)


if __name__ == "__main__":
    main()
