# RoadSentinel — Dataset Expansion & DINOv2 Crater Fine-Tuning Strategy

This guide details the end-to-end strategy and toolkit for expanding road defect datasets, applying domain-adapted data augmentations, and fine-tuning the **DINOv2 + SAM2** vision pipeline to detect severe craters, potholes, and alligator fatigue cracking across both 2D imagery and 3D drone aerial surveys.

---

## 1. External Dataset Ingestion Matrix

RoadSentinel is designed to ingest and unify multiple open-source road inspection datasets:

| Dataset | Modality | Annotations | Target Defect Types | Source / Protocol |
| :--- | :--- | :--- | :--- | :--- |
| **Pothole-600** | Ground 2D RGB (512×512) | Pixel Binary Masks | Deep asphalt craters, cavity edges | IEEE CVPR / CityU Hong Kong |
| **MWPD (Multi-Weather)** | 2D RGB (Multi-weather) | Bounding Boxes | Rain/puddle-filled craters, shadow occlusions | Roboflow / Kaggle |
| **RDD2022 (China Drone)** | Aerial Drone Nadir (100m) | Pascal VOC XML (`D40`) | High-altitude aerial potholes & road fatigue | Crowdsensing / WJD / CRDDC |
| **RDD2022 (India / Japan)** | Vehicle Dashcam 2D RGB | Pascal VOC XML (`D40`, `D20`) | Urban potholes, longitudinal cracks | CRDDC 2022 Benchmark |
| **Water-Filled Potholes** | 2D RGB (Specular/Water) | Category Labels & Masks | Severe water hazard craters | Mendeley Data |

---

## 2. Data Augmentation Strategy for Aerial & Flat Road Surveys

Aerial drone inspection and flat 2D captures exhibit distinct visual artifacts (extreme sun angle variations, asphalt grain variance, motion blur, and tree shadows). The PyTorch augmentation pipeline in `road_health_pipeline/training/finetune_dinov2_crater.py` implements:

1. **Random Shadow Injection (`RandomShadowInjection`)**:
   - Generates randomized polygonal shadow masks with Gaussian blur transitions to simulate tree and building cast shadows.
   - Prevents the DINOv2 feature extractor from confusing shadow boundaries with crater depression rims.

2. **Geometric & Perspective Invariance**:
   - `RandomHorizontalFlip(p=0.5)` & `RandomVerticalFlip(p=0.3)`
   - `RandomRotation(degrees=15)` to handle non-cardinal drone yaw headings.

3. **Photometric Calibration**:
   - `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)` to ensure robust detection under overcast, dawn, noon, and wet asphalt lighting.

---

## 3. DINOv2 Crater Adapter Architecture

Rather than destroying the rich generic semantic representations of the frozen DINOv2 Vision Transformer (`dinov2_vits14`), RoadSentinel uses a **Deep Residual Crater Feature Adapter & Non-Linear Anomaly Classification Head**:

```
Input RGB (518x518)
        │
        ▼
[Frozen DINOv2 Backbone (ViT-S/14)]
        ├──> Patch Tokens (37x37x384)
        └──> Global CLS Token (384)
                    │
                    ▼
[Crater Feature Adapter (LayerNorm + GELU + Linear Residual)]
                    │
                    ├──> Anomaly Classification Logits (P(Crater) vs P(Healthy Road))
                    ├──> Patch-Level Anomaly Scoring Map
                    └──> L2-Normalized Metric Projection (128D) -> FAISS Memory Bank
```

### Loss Formulations:
- **Binary Focal Loss ($\alpha=0.35, \gamma=2.0$)**: Down-weights easy healthy asphalt patches and heavily penalizes missed crater depressions.
- **Cosine Similarity Contrastive Margin**: Maximizes the feature angular separation between healthy road clusters and deep crater cavities.

---

## 4. How to Run 2D Pothole Detection & Validation

To test your pipeline on flat 2D JPEGs without CARLA simulation hardware filters:

```bash
# 1. Run detection on an entire folder of 2D images:
python detect_2d_potholes.py --input RoadSentinel_datasets/pothole_600/Pothole/rgb/ --limit 10

# 2. Run with custom 2D thresholds:
python detect_2d_potholes.py \
  --input path/to/my_images/ \
  --conf 0.15 \
  --min-area 50 \
  --output results/
```

Outputs will be rendered to `results/` containing:
- High-contrast OpenCV bounding boxes with defect IDs, confidence tags, and pixel area.
- SAM2 segmentation masks with semi-transparent alpha overlays and white outer boundary contours.
- `detection_results.json` containing all raw detection metadata.

---

## 5. How to Train and Fine-Tune the Model

```bash
# Run fine-tuning across all discovered datasets:
python road_health_pipeline/training/finetune_dinov2_crater.py \
  --epochs 10 \
  --batch-size 8 \
  --lr 2e-4 \
  --checkpoint road_health_pipeline/checkpoints/dinov2_crater_head.pt
```
