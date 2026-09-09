"""Edge Deployment Component for RoadSentinel Dashboard."""

from __future__ import annotations

import streamlit as st


def render_edge_deployment():
    st.markdown("## Edge Deployment & On-Device Acceleration")
    st.markdown(
        "Architecture, target specifications, and evaluation protocols for embedded road survey deployment "
        "on compact edge hardware."
    )

    # Mandatory Pending Notice (Section 22)
    st.markdown("""
    <div style="background: rgba(234, 179, 8, 0.12); border: 2px solid #eab308; border-radius: 14px; padding: 22px 28px; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #facc15; font-family: 'Outfit', sans-serif;">
                RASPBERRY PI 5 EDGE EVALUATION — STATUS: PENDING PHASE 8
            </div>
            <span class="status-pill pending">PENDING PHASE 8</span>
        </div>
        <div style="font-size: 0.95rem; color: #fef08a; margin-top: 10px; line-height: 1.5;">
            <strong>Scientific Integrity Statement:</strong> Physical testing and profiling on the target edge device (Raspberry Pi 5)
            is scheduled exclusively for Phase 8. No synthetic or fabricated on-device latency, FPS, or power metrics are presented here.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Target Hardware Specifications
    st.markdown("### 1. Target Edge Hardware Platform")
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Target Hardware</div>
            <div class="metric-value" style="font-size: 1.4rem; color: #38bdf8;">Raspberry Pi 5</div>
            <div class="metric-delta neutral">8GB LPDDR4X-4267</div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">CPU Architecture</div>
            <div class="metric-value" style="font-size: 1.4rem; color: #a855f7;">Cortex-A76</div>
            <div class="metric-delta neutral">Quad-core @ 2.4 GHz (ARMv8.2)</div>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Accelerator Target</div>
            <div class="metric-value" style="font-size: 1.4rem; color: #34d399;">CPU / NPU Opt</div>
            <div class="metric-delta neutral">ONNX Runtime / NCNN</div>
        </div>
        """, unsafe_allow_html=True)
    with h4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Power Envelope</div>
            <div class="metric-value" style="font-size: 1.4rem; color: #f59e0b;">5V / 5A (25W)</div>
            <div class="metric-delta neutral">Mobile / Drone survey mount</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Planned Evaluation Protocol
    st.markdown("### 2. Planned Phase 8 Edge Evaluation Metrics")
    st.markdown(
        "The following metrics schema is pre-allocated and will be populated directly from on-device profiling benchmarks:"
    )

    proto_df_data = [
        {"Metric Field": "Inference Latency", "Description": "Mean, median, and 95th-percentile inference latency per frame", "Target Threshold": "< 40 ms (>= 25 FPS)", "Status": "Pending Profiling"},
        {"Metric Field": "Throughput (FPS)", "Description": "Sustained frames processed per second during multi-camera stream", "Target Threshold": ">= 25.0 FPS (Real-time)", "Status": "Pending Profiling"},
        {"Metric Field": "RAM Consumption", "Description": "Peak resident memory footprint during model initialization and stream execution", "Target Threshold": "< 1.5 GB", "Status": "Pending Profiling"},
        {"Metric Field": "CPU Utilization", "Description": "Percentage of 4-core load and thermal throttling headroom under 100% duty cycle", "Target Threshold": "< 85% without throttling", "Status": "Pending Profiling"},
        {"Metric Field": "Model Loading Time", "Description": "Cold-start startup latency from disk to runtime memory", "Target Threshold": "< 3.0 seconds", "Status": "Pending Profiling"},
        {"Metric Field": "Prediction Consistency", "Description": "Numerical parity (mAP / IoU deviation) between RTX 5060 FP32 reference and edge INT8/FP16", "Target Threshold": "< 1.5% mAP degradation", "Status": "Pending Profiling"},
    ]
    st.dataframe(proto_df_data, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Optimization Roadmap
    st.markdown("### 3. Model Optimization & Export Pipeline")
    st.markdown("""
    ```
    +---------------------------------------------------------------------------------------+
    |                             FROZEN PYTORCH WEIGHTS                                   |
    |                   yolo/weights/best.pt (YOLOv8n, 6.2 MB FP32)                         |
    +-------------------------------------------+-------------------------------------------+
                                                |
                                                v
    +---------------------------------------------------------------------------------------+
    |                                MODEL CONVERSION & QUANTIZATION                        |
    |  * ONNX Export (opset 17, simplified graph)                                           |
    |  * INT8 Post-Training Quantization (PTQ) via representative calibration frames         |
    |  * NCNN / OpenVINO ARM NEON acceleration for Cortex-A76 vector units                   |
    +-------------------------------------------+-------------------------------------------+
                                                |
                                                v
    +---------------------------------------------------------------------------------------+
    |                             ON-DEVICE BENCHMARK RUNNER                                |
    |  * Automated 100-frame warmup and 1000-frame sustained evaluation                     |
    |  * Hardware sensor logging (CPU temp, core frequency, VPU power, memory RSS)          |
    +---------------------------------------------------------------------------------------+
    ```
    """)

    # Architectural feasibility notes
    st.markdown("""
    <div class="callout-box">
        <strong>Edge Feasibility Assessment:</strong><br>
        • <strong>YOLOv8n</strong>: Highly suited for Raspberry Pi 5 deployment. With 3.2M parameters and 8.7 GFLOPs at 512×512, INT8/ONNX execution is projected to achieve real-time 20–30 FPS on 4-core Cortex-A76.<br>
        • <strong>DINOv2 + SAM2</strong>: Extremely challenging on edge CPUs without dedicated 8+ TOPS NPU acceleration due to Vision Transformer self-attention complexity (~21M parameters in ViT-S/14 + ~22M parameters in SAM2 Hiera-S). Real-time execution would require cloud offloading or an attached edge accelerator (e.g., Hailo-8).
    </div>
    """, unsafe_allow_html=True)
