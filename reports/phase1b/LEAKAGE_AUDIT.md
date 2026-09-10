# RoadSentinel Phase 1B: Dataset Leakage and Independence Audit Report

**Audit Document**: `reports/phase1b/LEAKAGE_AUDIT.md`  
**Execution Date**: September 10, 2026  
**Auditor**: Pair-Programming Agent (Antigravity IDE)  
**Status**: **VERIFIED & AUDITED**  

---

## 1. Executive Summary

This audit systematically examined all dataset partitions, foundation model reference banks, calibration sets, and longitudinal progression sequences across the RoadSentinel pipeline to verify that reported performance figures are free from train-test contamination, spatial leakage, or temporal lookahead bias.

### Key Audit Conclusions:
### Key Audit Conclusions:
1. **Zero Exact Filename Overlap**: China Train ($1,921$ images) and China Val ($480$ images) share **0 filenames**. China and India cross-domain share **0 filenames**.
2. **Zero Exact SHA-256 Hash Collisions**: There are **0 identical files** between China Train and China Val, and **0 identical files** between China and India.
3. **Pavement Site Disjointness Verified**: The FHWA LTPP scenario forecasting model exhibits strictly **0 site overlap** between training ($17$ highway sections) and testing ($6$ held-out sections).
4. **Clean Domain Gate Formulation**: The DINOv2 reference bank ($N=1,921$) contains training samples only. Operational thresholds ($p_{99} = 0.4491$, $p_{95} = 0.3804$) were calculated exclusively from the training reference distribution; zero India labels or scores were used for threshold selection.
5. **Identification of Potential Dependency (pHash Near-Duplicates)**: The pHash audit identified **56 cross-split pairs** (Hamming distance $\le 5$) connecting 46 training images to 24 validation images across 16 connected components, reflecting continuous UAV flight video trajectories in RDD2022. These are treated as potential dependency rather than confirmed leakage or verified independence.

---

## 2. Partition Independence Matrix

| Dataset Split Pair | Sample Sizes | Exact Filename Overlap | Exact Hash (SHA-256) Matches | Perceptual Near-Duplicates (pHash $\le 5$) | Independence Status |
|---|---|---|---|---|---|
| **China Train vs. China Val** | $1,921$ vs. $480$ | **0** | **0** | **56** (24 val images) | **UNVERIFIED** (Flight IDs Not Published in RDD2022) |
| **China Splits vs. India Benchmark** | $2,401$ vs. $300$ | **0** | **0** | **0** | **VERIFIED STRICTLY INDEPENDENT** |
| **DINOv2 Ref Bank vs. Evaluated Data** | $1,921$ vs. $780$ | **0** | **0** | **56** (Shares China Train data) | **VERIFIED CLEAN COMPOSITION** (With UAV burst-frame caveat) |
| **XGBoost Train vs. Test Sites** | $17$ sites vs. $6$ sites | **0** | **0** | **0** | **VERIFIED SITE-DISJOINT (0 LEAKAGE)** |
| **Temporal Transitions ($t \rightarrow t+1$)** | $33$ transitions | **N/A** | **N/A** | **N/A** | **VERIFIED CHRONOLOGICAL (NO LOOKAHEAD)** |

---

## 3. Detailed Audit Findings

### 3.1 China UAV In-Domain Train vs. Validation Split & Group Sensitivity
- **Dataset Partitioning**: RDD2022 China_Drone consists of 2,401 total images (1,921 training, 480 validation).
- **Exact Hash Audit**: All 1,921 training images and 480 validation images produce distinct SHA-256 digests. No identical image was copied across splits.
- **Perceptual Duplicate Audit (pHash)**:
  - Using a 64-bit 2D Discrete Cosine Transform (DCT) perceptual hash, exactly **56 cross-split pairs** exhibit a Hamming distance $\le 5$.
  - **Graph of Connected Components**: An undirected graph constructed from the 56 pairs reveals **16 connected components** spanning 46 unique training frames and 24 unique validation frames.
  - **Physical Root Cause**: In RDD2022, drone survey frames are sampled from video flights along road stretches. When a validation flight passes over the same highway section or flies adjacent to a training flight, background fields and road geometry produce high perceptual similarity.
  - **Group Independence Verdict**: Because RDD2022 does not publish drone flight trajectory logs, timestamps, or sequence IDs, sequence/flight independence is marked **`UNVERIFIED`**.
  - **Sensitivity Analysis ([`group_sensitive_metrics.csv`](../../artifacts/phase1b/group_sensitive_metrics.csv))**:
    - Full China Validation ($N=480$): Precision $= 0.6568$, Recall $= 0.7736$, $\text{F1} = 0.7104$, Matched IoU $= 0.8007$.
    - Clean Validation (Excluding 24 connected images, $N=456$): Precision $= 0.6516$, Recall $= 0.7614$, $\text{F1} = 0.7022$, Matched IoU $= 0.8008$.
    - Delta Clean vs. Full: $\Delta\text{F1} = -0.0082$ ($-1.15\%$), and mean matched IoU is identical ($\Delta = +0.0001$). This confirms that the presence of suspected near-duplicate frames does not significantly distort or artificially inflate reported in-domain detector capability.

### 3.2 Cross-Domain India Dashcam Independence & DINOv2 Reference Bank Reconciliation
- **DINOv2 Reference Bank Reconciliation**:
  - The DINOv2 reference bank consists of the 1,921 China training images. The 56 cross-split pHash pairs apply identically to the source images of the reference bank. The previous row reporting "0" for the reference bank compared to 56 for China Train was a metric inconsistency (exact hash collision = 0 vs. pHash = 56).
  - Both rows now correctly reflect the 56 perceptual pairs as potential sequential flight dependencies.
- **Confirmation 1: YOLO Model Training**: Inspection of `yolo/weights/best.pt` and training logs confirms that YOLOv8n was trained exclusively on `yolo/data/rdd2022/images/train` ($1,921$ China UAV images). Zero India images were ever exposed to the detector during weight optimization.
- **Confirmation 2: DINOv2 Domain Gate Independence**:
  - The operational threshold $p_{99} = 0.4491$ was calculated strictly from intra-training leave-one-out $k\text{NN}$ distances.
  - **Zero India images or labels were used to select, tune, or optimize this threshold**.
  - India data was evaluated strictly out-of-sample as a held-out cross-domain probe.

### 3.3 XGBoost Pavement Deterioration Split Independence
- **Database**: FHWA Long-Term Pavement Performance (LTPP InfoPave SDR 40).
- **Partition Rule**: Seeded site-disjoint split (`random_seed=1`).
- **Training Sites (17 sites)**:
  `02-1001`, `02-1004`, `02-1008`, `02-6010`, `02-9035`, `15-1003`, `15-1008`, `15-7080`, `22-3056`, `22-4001`, `36-1011`, `36-1643`, `36-1644`, `36-4018`, `38-2001`, `38-3005`, `44-7401`.
- **Test Sites (6 sites)**:
  `02-1002`, `15-1006`, `36-1008`, `36-4017`, `38-3006`, `38-5002`.
- **Intersection**: $\text{train\_sites} \cap \text{test\_sites} = \emptyset$ (Exactly 0 overlapping test sites).
- **Temporal Directionality**: In each section pair $(t_1, t_2)$, $t_1 < t_2$. Features (`current_severity`, `rainfall_level`, `traffic_level`, `temperature`, `water_exposure`, `days_ahead`) are computed strictly up to $t_1$. No feature incorporates condition measurements from $t_2$.

### 3.4 Temporal Subsystem Validity
- **Camera Identity**: Evaluated across 8 sequences where all frames in a sequence share the exact same camera mounting, orientation, and resolution.
- **Timestamp Monotonicity**: Frame sequences are strictly ordered chronologically ($D_1 \le D_2 \le \dots \le D_N$).
- **Semantic Rule Compliance**: The codebase enforces `NOT_OBSERVED != REPAIRED`. A disappeared defect is logged as unobserved (due to occlusion, lighting, or sensor grazing angle) and is never classified as physically repaired without explicit maintenance metadata.

---

## 4. Summary of Output Artifacts

1. **`artifacts/phase1b/leakage_pairs.csv`**: Full inventory of all 56 suspected near-duplicate pairs, intra-train duplicates, and hash comparisons.
2. **`artifacts/phase1b/split_independence.json`**: Machine-readable JSON summary verifying split disjointness and zero-leakage constraints.
