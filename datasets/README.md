# RoadSentinel Dataset System

This directory contains configuration, scripts, metadata, and validation tools for the datasets powering the RoadSentinel UAV-based road condition monitoring pipeline.

## Dataset Suite Layout
When downloaded and prepared using the provided scripts, the directory structure will be organized under the central path:
`RoadSentinel_datasets/` (located outside the Git workspace to avoid committing multi-GB files)

```text
RoadSentinel_datasets/
├── rdd2022/                    # Existing 35-image development subset (untouched)
├── rdd2022_full/
│   ├── China_Drone/            # UAV road damage detection dataset
│   └── India/                  # Indian road domain representation dataset
├── pothole_mix/                # Primary semantic segmentation dataset (4,340 pairs)
├── water_filled_potholes/      # Dry vs. Water-filled/muddy pothole dataset
├── pothole_600/                # Disparity/depth validation dataset
├── chitholian/                 # Full Chitholian pothole dataset
├── pi5_smoketest_subset/       # 75-90 image smoke test subset for Raspberry Pi 5
├── mwpd/                       # Multi-weather pothole dataset
└── qr4change/                  # Indian road environment negatives/references
```

## Available Configurations and Scripts
1. **[DATASET_SOURCES.md](DATASET_SOURCES.md):** Records the official DOIs, verified URLs, download caching paths, and licensing terms for each dataset.
2. **[DATASET_MATRIX.md](DATASET_MATRIX.md):** Master comparison matrix showing task coverage (detection, segmentation, water detection, depth validation, Pi5 smoke testing, etc.).
3. **`download_datasets.sh`:** Automated shell script to download all zip archives.
4. **`prepare_datasets.py`:** Handles zip archive extraction, domain separation, and Raspberry Pi 5 smoke-test subset extraction.
5. **`validate_datasets.py`:** Validates dataset integrity (checking for corruption, filename mismatches, annotation errors, and duplicates).
