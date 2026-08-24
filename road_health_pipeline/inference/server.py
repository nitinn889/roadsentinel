from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import json
import sys

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import CONFIG
from inference.run_inference import infer

app = FastAPI(title="RoadSentinel Inference API", version="0.1")


@app.get("/health")
def health():
    return {"status": "ok", "service": "roadsentinel-inference"}


@app.post("/infer")
async def infer_endpoint(image: UploadFile = File(...), metadata_json: str = Form("{}")):
    suffix = Path(image.filename or "frame.jpg").suffix or ".jpg"
    with tempfile.TemporaryDirectory() as td:
        image_path = Path(td) / f"input{suffix}"
        image_path.write_bytes(await image.read())
        meta_path = Path(td) / "metadata.json"
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"Invalid metadata_json: {exc}")
        meta_path.write_text(json.dumps(metadata), encoding="utf-8")
        try:
            result = infer(image_path, meta_path, CONFIG.device, CONFIG.memory_bank_dir)
        except Exception as exc:
            raise HTTPException(500, str(exc))
        return JSONResponse(result.to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
