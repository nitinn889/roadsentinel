#!/usr/bin/env bash
# RoadSentinel Dataset Download Script
# This script downloads all priority road damage datasets.

set -e

# Target directory for downloads
DOWNLOAD_DIR="RoadSentinel_datasets/downloads"
mkdir -p "$DOWNLOAD_DIR"

echo "=== RoadSentinel Dataset Downloader ==="
echo "Saving archives to: $DOWNLOAD_DIR"

# 1. Pothole Mix v2 (Subfolder containing 4,340 images & masks)
if [ ! -f "$DOWNLOAD_DIR/pothole-mix-v1.0-20220526.zip" ]; then
    echo "[1/6] Downloading Pothole Mix v2 (3.55 GB)..."
    curl -L "https://data.mendeley.com/public-files/datasets/kfth5g2xk3/files/d5b8a18b-b1fa-4637-88ad-56263431ef4e/file_downloaded" -o "$DOWNLOAD_DIR/pothole-mix-v1.0-20220526.zip"
else
    echo "Pothole Mix v2 already exists."
fi

# 2. RDD2022 Full Dataset (13.26 GB)
if [ ! -f "$DOWNLOAD_DIR/RDD2022_released_through_CRDDC2022.zip" ]; then
    echo "[2/6] Downloading RDD2022 Full Dataset (13.26 GB)..."
    curl -L "https://ndownloader.figshare.com/files/38030910" -o "$DOWNLOAD_DIR/RDD2022_released_through_CRDDC2022.zip"
else
    echo "RDD2022 Full Dataset already exists."
fi

# 3. Water-Filled and Dry Potholes (581.2 MB)
echo "[3/6] Downloading Water-Filled and Dry Potholes (581 MB)..."
curl -L "https://data.mendeley.com/public-api/zip/tp95cdvgm8/download/1" -o "$DOWNLOAD_DIR/tp95cdvgm8-1.zip"

# 4. Pothole-600 (300 MB)
if [ ! -f "$DOWNLOAD_DIR/Pothole-600.zip" ]; then
    echo "[4/6] Downloading Pothole-600 (300 MB)..."
    curl -L "https://nas.labsun.org/downloads/2025_tase_mmfseg/Pothole-600.zip" -o "$DOWNLOAD_DIR/Pothole-600.zip"
else
    echo "Pothole-600 already exists."
fi

# 5. MWPD (242 MB)
echo "[5/6] Downloading MWPD (242 MB)..."
curl -L "https://data.mendeley.com/public-api/zip/s5hx9n2jc3/download/2" -o "$DOWNLOAD_DIR/s5hx9n2jc3-2.zip"

# 6. QR4Change (6.08 GB)
echo "[6/6] Downloading QR4Change (6.08 GB)..."
curl -L "https://data.mendeley.com/public-api/zip/zndzygc3p3/download/2" -o "$DOWNLOAD_DIR/zndzygc3p3-2.zip"

echo "All downloads completed successfully!"
