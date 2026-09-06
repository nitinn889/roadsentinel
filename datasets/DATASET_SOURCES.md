# RoadSentinel Dataset Sources

This file records the verified sources, download links, licenses, and configurations for all datasets in the RoadSentinel project.

## Priority 1 — Pothole Mix v2
- **Source URL:** [Mendeley Data kfth5g2xk3/2](https://data.mendeley.com/datasets/kfth5g2xk3/2)
- **Direct Cache ZIP URL:** `https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/kfth5g2xk3-2.zip` (7.8 GB, contains both main dataset and RGB-D videos)
- **Direct Subfolder ZIP URL:** `https://data.mendeley.com/public-files/datasets/kfth5g2xk3/files/d5b8a18b-b1fa-4637-88ad-56263431ef4e/file_downloaded` (3.55 GB, `pothole-mix-v1.0-20220526.zip` - primary segmentation dataset)
- **License:** CC BY-NC 3.0 (Attribution-NonCommercial 3.0 Unported)
- **Primary Use:** SAM2 segmentation training/evaluation, DINOv2 feature evaluation.

## Priority 2 & 3 — RDD2022 China_Drone & India
- **Source URL:** [Figshare 21431547](https://doi.org/10.6084/m9.figshare.21431547)
- **Direct Download URL:** `https://ndownloader.figshare.com/files/38030910` (13.26 GB)
- **License:** CC BY-SA 4.0
- **Primary Use:** Drone and ground domain diversity, road defect identification (bounding-box annotations).

## Priority 4 — Water-Filled and Dry Potholes
- **Source URL:** [Mendeley Data tp95cdvgm8/1](https://data.mendeley.com/datasets/tp95cdvgm8/1)
- **Direct Cache ZIP URL:** `https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/tp95cdvgm8-1.zip` (581.2 MB)
- **License:** CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Primary Use:** Water-filled defect handling, muddy/water-obscured road regions.

## Priority 5 — Pothole-600
- **Source URL:** [Pothole-600 Official Page](https://sites.google.com/view/pothole-600/dataset) / [MMFSeg Repo](https://github.com/lab-sun/MMFSeg)
- **Direct Download URL:** `https://nas.labsun.org/downloads/2025_tase_mmfseg/Pothole-600.zip` (300 MB)
- **License:** Academic/Research Use (Fan et al. 2020)
- **Primary Use:** Multi-modal RGB + Disparity depth evaluation.

## Priority 6 — Chitholian Pothole Dataset
- **Source URL:** [Roboflow Pothole](https://public.roboflow.com/object-detection/pothole)
- **License:** CC BY 4.0
- **Primary Use:** Raspberry Pi 5 smoke test subset (75-90 images with COCO annotations).

## Priority 7 — MWPD (Multi-weather Pothole Dataset)
- **Source URL:** [Mendeley Data s5hx9n2jc3/2](https://data.mendeley.com/datasets/s5hx9n2jc3/2)
- **Direct Cache ZIP URL:** `https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/s5hx9n2jc3-2.zip` (242 MB)
- **License:** CC BY 4.0
- **Primary Use:** Weather variation (rain, wet, day, night, twilight) and DINOv2 robustness testing.

## Priority 8 — QR4Change
- **Source URL:** [Mendeley Data zndzygc3p3/2](https://data.mendeley.com/datasets/zndzygc3p3/2)
- **Direct Cache ZIP URL:** `https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zndzygc3p3-2.zip` (6.08 GB)
- **License:** CC BY 4.0
- **Primary Use:** Indian road domain representation, healthy-road reference datasets.

## Additional Road Crack / Distress Datasets
- **Source URL:** [CrackSeg (Roboflow/Ultralytics)](https://github.com/ultralytics/ultralytics)
- **License:** CC BY 4.0
- **Primary Use:** Pavement crack density/severity scoring.

## Additional Temporal / Prediction Datasets
- **Status:** **No verified temporal dataset acquired.**
- **Details:** Public image datasets tracking the *same* physical road segment over long periods showing deterioration progression (cracks becoming potholes) do not currently exist. Accelerometer time-series datasets (AsphaltPavementType) are available but do not contain progression imagery.
