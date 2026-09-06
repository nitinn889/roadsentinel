"""
Central config for the offline memory-bank build.
Edit these paths/values, then run build_memory_bank.py.
"""

from pathlib import Path

# ---- Paths ----
HEALTHY_ROADS_DIR = Path("data/healthy_roads")   # folder of .jpg/.png images, any subfolder depth
OUTPUT_DIR = Path("output/memory_bank")          # where the memory bank artifacts get written

# ---- SAM2 ----
# Download a checkpoint from https://github.com/facebookresearch/sam2 (tiny/small recommended
# for offline speed too, since we just need a road mask, not pixel-perfect segmentation).
SAM2_CHECKPOINT = Path("checkpoints/sam2.1_hiera_small.pt")
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

# ---- DINOv2 ----
DINOV2_MODEL_NAME = "dinov2_vits14"   # small variant -> matches what we'll run on the Pi later
PATCH_SIZE = 14                       # fixed by DINOv2, don't change

# Input image size fed to DINOv2. Must be a multiple of PATCH_SIZE.
# 518 = 37 * 14, a common DINOv2 working resolution.
DINOV2_INPUT_SIZE = 518

# ---- Road ROI heuristic ----
# Used to build the box prompt for SAM2 when you don't have a fixed-camera mask.
# Box covers the bottom portion of the frame (road is usually here), as fractions of (W, H).
ROI_BOX_FRACTIONS = (0.05, 0.35, 0.95, 1.0)  # (x0, y0, x1, y1) as fractions of image width/height

# ---- Coreset subsampling ----
# Keep this fraction of extracted road patch embeddings in the final memory bank.
# Lower = smaller memory bank = faster/lighter on the Pi, but less coverage of "normal" variation.
CORESET_RATIO = 0.10
CORESET_MAX_POINTS = 20000   # hard cap regardless of ratio, keeps Pi-side FAISS index small

# ---- Misc ----
DEVICE = "cuda"   # "cuda" if you have a GPU for this offline step, else "cpu"
BATCH_SIZE = 8
SEED = 42
