"""Research Methodology & Evidence Links Component for RoadSentinel Dashboard."""

from __future__ import annotations

from pathlib import Path
import streamlit as st

from data_loader import WORKSPACE_ROOT


def render_methodology():
    st.markdown("## Research Methodology & Scientific Protocols")
    st.markdown(
        "Comprehensive architectural specifications, mathematical formulations, and artifact evidence references "
        "supporting the RoadSentinel research investigation."
    )

    # Methodology Accordion / Tabs
    tabs = st.tabs([
        "Dataset Specifications",
        "Perception Pipelines",
        "Temporal Correspondence",
        "Deterioration Forecasting",
        "Hardware & Reproducibility",
        "Research Artifacts & Evidence",
    ])

    with tabs[0]:
        st.markdown("### Dataset Specifications & Benchmark Protocols")
        st.markdown("""
        #### 1. Common Validation Benchmark (RDD2022 China_Drone)
        - **Source**: Crowdsourced Road Damage Detection Challenge (RDD2022) China_Drone split.
        - **Volume**: 480 validation images (100% held out from YOLO 25-epoch training split of 1,921 images).
        - **Annotations**: 742 ground-truth damage bounding boxes across 5 classes:
          - `D00`: Longitudinal Crack (266 instances)
          - `D10`: Transverse Crack (256 instances)
          - `D20`: Alligator Fatigue Cracking (58 instances)
          - `D40`: Pothole (15 instances — marked low-support)
          - `Repair`: Repaired Surface Patch (147 instances)
        - **Evaluation Protocol**: Bipartite greedy assignment at IoU >= 0.50 (primary) and IoU >= 0.25 (sensitivity).

        #### 2. Experiment A Temporal Dataset
        - **Source**: Controlled synthetic multi-day inspection captures under varying lighting, weather, and camera vantage.
        - **Volume**: 40 physical captures across 4 sequences (`SEG_001` through `SEG_004`, Days 01–10).
        - **Authoritative Ground Truth**: Individual sidecar metadata files recorded at capture time.
        """)

    with tabs[1]:
        st.markdown("### Frozen Perception Architectures")
        st.markdown("""
        #### System A: Supervised YOLOv8n
        - **Architecture**: Ultralytics YOLOv8 nano (3.2M parameters, 8.7 GFLOPs).
        - **Checkpoint**: `yolo/weights/best.pt` (trained for 25 epochs at 512×512 resolution).
        - **Inference Configuration**: `conf = 0.25`, `imgsz = 512`, `iou = 0.70` (NMS).
        - **Role**: High-speed real-time localized damage bounding box detector.

        #### System B: Zero-Shot DINOv2 + SAM2
        - **Vision Transformer**: Meta `dinov2_vits14` (Vision Transformer Small with 14×14 patch tokens).
        - **Segmentation Foundation**: Meta `sam2.1_hiera_small` (Hierarchical Vision Transformer).
        - **Anomaly Detection**: Compares test image patch embeddings against a healthy pavement memory bank extracted from baseline unblemished asphalt.
        - **Suppression**: Heuristic `RoadMarkingSuppressor` removes high-contrast painted road markings and curbs before prompting SAM2.
        - **Role**: Geometry-free anomaly localization and pixel-level boundary segmentation without damage-specific training labels.
        """)

    with tabs[2]:
        st.markdown("### Temporal Tracking Engine & Event Taxonomy")
        st.markdown("""
        #### Multi-Day Bipartite Association
        Tracks are propagated across consecutive observation days ($T-1 \\to T$) using hierarchical matching:
        1. **Mask IoU**: Primary association criterion between SAM2 binary segmentation masks.
        2. **BBox IoU Fallback**: Used when mask overlap is ambiguous or during camera jitter.
        3. **Centroid Distance Fallback**: Normalized Euclidean distance thresholded to 10% of frame diagonal.
        4. **One-to-One Matching**: Solved via Hungarian / greedy matching to prevent multi-defect collapse.

        #### Formal Event Taxonomy
        - `NEW_DEFECT`: Unmatched detection appearing on Day $T$.
        - `MATCHED_EXISTING`: Continuous tracking of a spatial defect across days.
        - `OBSERVED_AREA_INCREASED`: Tracked defect area expands by $\\ge 15\\%$.
        - `OBSERVED_AREA_DECREASED`: Tracked defect area contracts by $\\ge 15\\%$.
        - `NOT_OBSERVED`: Defect observed on Day $T-1$ absent on Day $T$. **Never marked as REPAIRED** without affirmative maintenance metadata.
        """)

    with tabs[3]:
        st.markdown("### Scenario-Conditioned Deterioration Forecasting (XGBoost Model V2)")
        st.markdown("""
        #### Mathematical Formulation
        Given initial surface condition $S_0 \\in [0, 1]$ and operational stressors, predicts future condition:
        $$\\hat{S}_{t+\\Delta t} = f(S_0, R, V, T, W, \\Delta t)$$
        Where:
        - $S_0$: Measured severity index from perception $[0.0 - 0.9]$
        - $R$: Mean precipitation rate (mm/day)
        - $V$: Annual Average Daily Traffic (AADT) volume
        - $T$: Mean ambient pavement temperature (°C)
        - $W$: Subbase moisture / water exposure index $[0.0 - 1.0]$
        - $\\Delta t$: Forecast horizon $\\in \\{30, 60, 90\\}$ days

        #### Monotone Convexity Constraints
        Enforces physical plausibility:
        $$\\frac{\\partial \\hat{S}}{\\partial S_0} \\ge 0, \\quad \\frac{\\partial \\hat{S}}{\\partial R} \\ge 0, \\quad \\frac{\\partial \\hat{S}}{\\partial V} \\ge 0, \\quad \\frac{\\partial \\hat{S}}{\\partial T} \\ge 0, \\quad \\frac{\\partial \\hat{S}}{\\partial W} \\ge 0, \\quad \\frac{\\partial \\hat{S}}{\\partial \\Delta t} \\ge 0$$
        Ensures deterioration never spontaneously reverses without physical maintenance intervention.
        """)

    with tabs[4]:
        st.markdown("### Hardware Environment & Reproducibility")
        st.markdown("""
        - **Host Platform**: Linux x86_64 (Kernel 6.17)
        - **Primary GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (8 GB GDDR6, 140W Max TGP)
        - **CUDA Runtime**: CUDA 13.0 / PyTorch 2.13.0+cu130
        - **Precision**: Full FP32 reference benchmark
        - **Deterministic Seeds**: All batch and benchmark scripts lock `PYTHONHASHSEED=0`, `torch.manual_seed(42)`.
        """)

    with tabs[5]:
        st.markdown("### Direct Repository Artifact Links")
        st.markdown(
            "The following research reports, data manifests, and benchmark tables provide direct scientific verification:"
        )

        artifacts = [
            ("PERCEPTION_METHODOLOGY.md", WORKSPACE_ROOT / "integration/PERCEPTION_METHODOLOGY.md", "Complete mathematical specification of DINOv2 memory bank and SAM2 prompting."),
            ("PERCEPTION_RESEARCH_SUMMARY.md", WORKSPACE_ROOT / "integration/PERCEPTION_RESEARCH_SUMMARY.md", "Final Phase-6 research interpretation, failure modes, and benchmark conclusions."),
            ("TEMPORAL_RESEARCH_SUMMARY.md", WORKSPACE_ROOT / "integration/experiment_a/TEMPORAL_RESEARCH_SUMMARY.md", "Phase-4 temporal analysis, tracking stability CV=0.67%, and environmental confounds."),
            ("FINAL_PERCEPTION_TABLE.csv", WORKSPACE_ROOT / "integration/dashboard_assets/perception/FINAL_PERCEPTION_TABLE.csv", "Frozen headline comparison table (Precision, Recall, F1, Latency, FPS)."),
            ("perception_handoff_manifest.json", WORKSPACE_ROOT / "integration/dashboard_assets/perception/perception_handoff_manifest.json", "Machine-readable perception asset manifest and research freeze declaration."),
            ("scenario_model_v2.json (Config)", WORKSPACE_ROOT / "xgboost/config/scenario_model_v2.json", "XGBoost Model V2 feature definitions, presets, and monotone hyperparameter configuration."),
        ]

        for title, path, desc in artifacts:
            status_tag = "[AVAILABLE]" if path.exists() else "[NOT FOUND]"
            st.markdown(f"- **`{title}`** `{status_tag}`: {desc}<br><small style='color: #64748b;'>Path: <code>{path}</code></small>", unsafe_allow_html=True)
