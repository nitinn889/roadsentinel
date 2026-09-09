"""Failure Mode & Sensitivity Analysis Component for RoadSentinel Dashboard."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from data_loader import (
    BENCHMARK_DIR,
    load_binary_metrics,
)

FIGURES_DIR = BENCHMARK_DIR / "figures"


def render_failure_modes():
    st.markdown("## Failure Mode & Sensitivity Analysis")
    st.markdown(
        "Scientific integrity requires transparently documenting architectural limitations, failure modes, "
        "and sensitivity boundaries rather than showcasing only idealized success cases."
    )

    # 1. DINOv2 + SAM2 Failure Mode Analysis
    st.markdown("### 1. DINOv2 + SAM2 Anomaly Perception Failure Modes")
    st.markdown(
        "DINOv2 foundation embeddings combined with SAM2 prompting operate without supervised road damage annotations. "
        "Through extensive empirical evaluation across 480 benchmark frames and 40 temporal captures, 6 primary failure modes were identified:"
    )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("""
        <div class="metric-card" style="margin-bottom: 12px;">
            <div style="font-weight: 700; color: #f43f5e; font-size: 1.05rem;">1. Coarse Asphalt Aggregate False Positives</div>
            <div style="font-size: 0.88rem; color: #cbd5e1; margin-top: 6px;">
                Natural gravel aggregates, oil stains, and patch stone textures produce high vision transformer patch embedding distances
                relative to the smooth asphalt memory bank, generating false-positive candidate prompts (1,055 non-matched regions).
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="margin-bottom: 12px;">
            <div style="font-weight: 700; color: #f43f5e; font-size: 1.05rem;">2. Thin Crack Patch Dilution</div>
            <div style="font-size: 0.88rem; color: #cbd5e1; margin-top: 6px;">
                DINOv2 utilizes 14×14 pixel non-overlapping patch tokens. For 1–2 pixel hairline cracks, the crack occupies < 10% of the token area,
                diluting the anomalous signal into background asphalt and falling below the 92nd-percentile anomaly threshold.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="margin-bottom: 12px;">
            <div style="font-weight: 700; color: #f43f5e; font-size: 1.05rem;">3. Long Crack SAM2 Fragmentation</div>
            <div style="font-size: 0.88rem; color: #cbd5e1; margin-top: 6px;">
                SAM2 is prompted with centroid point coordinates. When applied to extended continuous longitudinal fissures,
                the prompt generates isolated segmented islands rather than a cohesive full-length defect mask.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        st.markdown("""
        <div class="metric-card" style="margin-bottom: 12px;">
            <div style="font-weight: 700; color: #f43f5e; font-size: 1.05rem;">4. Specular Rain Reflections & Water Sheen</div>
            <div style="font-size: 0.88rem; color: #cbd5e1; margin-top: 6px;">
                Standing water puddles and specular sky reflections alter road surface luminance drastically,
                triggering widespread anomaly spikes (seen in SEG_004 Day 07–08 rain scenarios).
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="margin-bottom: 12px;">
            <div style="font-weight: 700; color: #f43f5e; font-size: 1.05rem;">5. Sunset Grazing Shadow Suppression</div>
            <div style="font-size: 0.88rem; color: #cbd5e1; margin-top: 6px;">
                Under low-angle sunset illumination (< 15° solar elevation), road markings cast heavy shadow penumbras.
                The RoadMarkingSuppressor heuristic misclassifies large asphalt corridors as painted markings, over-suppressing true defects.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card" style="margin-bottom: 12px;">
            <div style="font-weight: 700; color: #f43f5e; font-size: 1.05rem;">6. Perspective Grazing Viewpoint Distortion</div>
            <div style="font-size: 0.88rem; color: #cbd5e1; margin-top: 6px;">
                Unlike nadir UAV imagery, oblique roadside camera angles compress distant road patches,
                reducing pixel resolution per square meter and causing systematic false-positive candidates on road shoulders.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # DINO failure mode breakdown figure
    p_dino_fig = FIGURES_DIR / "fig5_dino_failure_mode_breakdown.png"
    if p_dino_fig.exists():
        st.image(str(p_dino_fig), caption="Figure 5: DINOv2+SAM2 Empirical Failure Mode Category Distribution", use_container_width=True)

    st.markdown("---")

    # 2. YOLOv8n Supervised Failure Modes
    st.markdown("### 2. YOLOv8n Supervised Failure Modes")
    st.markdown(
        "While YOLOv8n achieves high in-domain precision (0.6568) and recall (0.7736), "
        "it exhibits specific vulnerabilities:"
    )

    ycol1, ycol2 = st.columns(2)
    with ycol1:
        st.markdown("""
        - **D20 Alligator Cracking Confusion**: Alligator cracking consists of polygonal fatigue crack lattices. YOLO frequently confuses boundaries with repaired patches or fails to envelope the full interconnected area (F1 = 0.4259).
        - **Roadside Soil & Unpaved Shoulder False Alarms**: In aerial survey images with unpaved verges, dry soil ruts and gravel edges exhibit spatial textures similar to D00 longitudinal cracks.
        - **Thermoplastic Road Marking Wear**: Scuffed, abraded, or peeling lane striping produces high-contrast edges that trigger false D10 transverse crack boxes.
        """)
    with ycol2:
        st.markdown("""
        - **Sub-Resolution Crack Misses**: Downsampling high-resolution images to 512×512 removes sub-millimeter hairline crack fissures from feature maps.
        - **Forward-Facing Dashcam Domain Shift**: When tested on the forward-facing dashcam domain-shift example (India_005086), YOLOv8n trained solely on top-down UAV drone views produced no detections on the examined frame.
        """)

    st.markdown("---")

    # 3. IoU Sensitivity Analysis (Section 21)
    st.markdown("### 3. Localization Sensitivity: IoU 0.50 vs IoU 0.25")
    st.markdown(
        "Evaluating bounding box overlap criteria reveals fundamental differences between rectangular object detection "
        "and prompt-based mask-derived bounding envelopes."
    )

    df50, df25 = load_binary_metrics()

    # Sensitivity comparison table
    sens_data = [
        {
            "Model": "YOLOv8n",
            "IoU 0.50 TP": 574,
            "IoU 0.25 TP": 610,
            "TP Growth": "+36 (+6.3%)",
            "IoU 0.50 Recall": "77.36%",
            "IoU 0.25 Recall": "82.21%",
            "Localization Characteristics": "Tight rectangular anchor-based bounding boxes"
        },
        {
            "Model": "DINOv2 + SAM2",
            "IoU 0.50 TP": 25,
            "IoU 0.25 TP": 74,
            "TP Growth": "+49 (+196.0%)",
            "IoU 0.50 Recall": "3.37%",
            "IoU 0.25 Recall": "9.97%",
            "Localization Characteristics": "Irregular polygon masks converted to loose bounding envelopes"
        }
    ]
    st.dataframe(pd.DataFrame(sens_data), hide_index=True, use_container_width=True)

    # Scientific Caution Callout
    st.markdown("""
    <div class="callout-box">
        <strong>SCIENTIFIC INTERPRETATION (IoU 0.25 SENSITIVITY):</strong><br>
        • Under IoU = 0.50, DINOv2+SAM2 achieves <strong>TP = 25</strong>.<br>
        • Under relaxed IoU = 0.25, DINOv2+SAM2 achieves <strong>TP = 74</strong> (nearly 3× higher).<br>
        • <em>"This indicates that some predicted regions exhibit partial spatial overlap with ground truth but fail the stricter box-localization criterion."</em><br>
        • <strong>Crucial Distinction</strong>: This does <strong>NOT</strong> prove that DINO/SAM is accurate or competitive with YOLOv8n; rather, it highlights that mask-derived bounding boxes for organic crack shapes suffer from loose enclosing geometry compared to ground-truth tight annotation boxes.
    </div>
    """, unsafe_allow_html=True)

    p_sens_fig = FIGURES_DIR / "fig6_iou_sensitivity_comparison.png"
    if p_sens_fig.exists():
        st.image(str(p_sens_fig), caption="Figure 6: Metric Sensitivity Comparison under Strict (0.50) vs Relaxed (0.25) IoU Matching", use_container_width=True)
