"""RoadSentinel Dashboard V2 Centralized Data Loader.

Provides offline-first, cached data loading from canonical JSON and CSV assets.
Guarantees defensive fallbacks and fast, responsive dashboard rendering.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

log = logging.getLogger("dashboard_data_loader")

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_METRICS_PATH = WORKSPACE_ROOT / "integration/CANONICAL_RESEARCH_METRICS.json"
DECISIONS_PATH = WORKSPACE_ROOT / "decision_engine/ROAD_HEALTH_DECISIONS.csv"
FORECASTS_PATH = WORKSPACE_ROOT / "integration/primary_goals/goal1_forecasts.csv"
PRIMARY_RESULTS_PATH = WORKSPACE_ROOT / "integration/primary_goals/ROADSENTINEL_PRIMARY_RESULTS.csv"
BENCHMARK_DIR = WORKSPACE_ROOT / "benchmark/final_comparison"
TEMPORAL_DIR = WORKSPACE_ROOT / "integration/experiment_a/temporal"
CROSS_DOMAIN_DIR = WORKSPACE_ROOT / "cross_domain/results"
RELIABILITY_VAL_DIR = WORKSPACE_ROOT / "reliability_validation"
DECISION_ENGINE_DIR = WORKSPACE_ROOT / "decision_engine"


@st.cache_data(show_spinner=False)
def load_canonical_metrics() -> Dict[str, Any]:
    """Load canonical research metrics JSON as single source of truth."""
    if CANONICAL_METRICS_PATH.exists():
        try:
            return json.loads(CANONICAL_METRICS_PATH.read_text(encoding="utf-8"))
        except Exception as err:
            log.warning("Could not read canonical metrics: %s", err)
    return {}


@st.cache_data(show_spinner=False)
def load_decisions_table() -> pd.DataFrame:
    """Load ROAD_HEALTH_DECISIONS.csv for all 40 Experiment A captures."""
    if DECISIONS_PATH.exists():
        return pd.read_csv(DECISIONS_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_goal1_forecasts() -> pd.DataFrame:
    """Load goal1_forecasts.csv (600 scenario forecast records)."""
    if FORECASTS_PATH.exists():
        return pd.read_csv(FORECASTS_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_primary_results() -> pd.DataFrame:
    """Load ROADSENTINEL_PRIMARY_RESULTS.csv (40 captures with Goal 1 & 2 summaries)."""
    if PRIMARY_RESULTS_PATH.exists():
        return pd.read_csv(PRIMARY_RESULTS_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_final_perception_table() -> pd.DataFrame:
    """Load FINAL_PERCEPTION_TABLE.csv (Phase 5/6 Common Benchmark)."""
    p = BENCHMARK_DIR / "FINAL_PERCEPTION_TABLE.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_binary_metrics() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load primary (IoU 0.50) and secondary (IoU 0.25) binary metrics."""
    p50 = BENCHMARK_DIR / "common_binary_metrics.csv"
    p25 = BENCHMARK_DIR / "common_binary_sensitivity_iou25.csv"
    df50 = pd.read_csv(p50) if p50.exists() else pd.DataFrame()
    df25 = pd.read_csv(p25) if p25.exists() else pd.DataFrame()
    return df50, df25


@st.cache_data(show_spinner=False)
def load_temporal_sequences_table() -> pd.DataFrame:
    """Load Phase-4 temporal sequence summary table."""
    p = TEMPORAL_DIR / "tables/table_temporal_sequences.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_cross_domain_dataset() -> pd.DataFrame:
    """Load cross_domain_reliability_dataset.csv (300 Indian frames)."""
    p = CROSS_DOMAIN_DIR / "cross_domain_reliability_dataset.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_reliability_ablation_table() -> pd.DataFrame:
    """Load table_full_ablation.csv (Phase 12 Models A-H across Targets T0-T4)."""
    p = RELIABILITY_VAL_DIR / "tables/table_full_ablation.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_domain_gate_table() -> pd.DataFrame:
    """Load table_domain_gate.csv (Phase 12 Domain Gating metrics)."""
    p = RELIABILITY_VAL_DIR / "tables/table_domain_gate.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_cross_domain_routing_table() -> pd.DataFrame:
    """Load table_cross_domain_routing.csv (Phase 13 Safety Comparison)."""
    p = DECISION_ENGINE_DIR / "tables/table_cross_domain_routing.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_decision_ablation_table() -> pd.DataFrame:
    """Load table_decision_ablation.csv (Phase 13 Systems A to E)."""
    p = DECISION_ENGINE_DIR / "tables/table_decision_ablation.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_segment_decision_summary() -> pd.DataFrame:
    """Load table_segment_decision_summary.csv."""
    p = DECISION_ENGINE_DIR / "tables/table_segment_decision_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_case_studies_table() -> pd.DataFrame:
    """Load table_case_studies.csv."""
    p = DECISION_ENGINE_DIR / "tables/table_case_studies.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_benchmark_summary() -> Dict[str, Any]:
    """Load raw benchmark summary containing all 480 image records."""
    p = BENCHMARK_DIR / "benchmark_summary.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


@st.cache_data(show_spinner=False)
def load_yolo_semantic_metrics() -> pd.DataFrame:
    """Load yolo_semantic_metrics.csv (Class breakdown)."""
    p = BENCHMARK_DIR / "yolo_semantic_metrics.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_panel_examples() -> pd.DataFrame:
    """Load panel_examples.csv (Curated case studies)."""
    p = BENCHMARK_DIR / "panel_examples.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_cross_domain_comparison_table() -> pd.DataFrame:
    """Load table_in_vs_cross_domain.csv."""
    p = WORKSPACE_ROOT / "cross_domain/tables/table_in_vs_cross_domain.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_decision_policy_table() -> pd.DataFrame:
    """Load table_decision_policy.csv."""
    p = DECISION_ENGINE_DIR / "tables/table_decision_policy.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def get_image_path(segment_id: str, day: int) -> Optional[Path]:
    """Resolve physical capture image path for any segment and day."""
    day_str = f"day_{day:02d}"
    # Check temporal_segments directory first
    p_temp = WORKSPACE_ROOT / f"env/output/temporal_segments/{segment_id}/{day_str}.png"
    if p_temp.exists():
        return p_temp
    p = WORKSPACE_ROOT / f"env/output/manual_inspections/{day_str}/{segment_id}/raw.png"
    if p.exists():
        return p
    p_alt = WORKSPACE_ROOT / f"env/output/manual_inspections/{day_str}/{segment_id}/original.jpg"
    if p_alt.exists():
        return p_alt
    return None


@st.cache_data(show_spinner=False)
def discover_temporal_segments() -> List[str]:
    """Dynamically scan env/output/temporal_segments for available segment folders."""
    seg_dir = WORKSPACE_ROOT / "env/output/temporal_segments"
    if seg_dir.exists() and seg_dir.is_dir():
        segments = sorted([
            d.name for d in seg_dir.iterdir()
            if d.is_dir() and d.name.startswith("SEG_")
        ])
        if segments:
            return segments
    # Fallback to primary results if directory scan is empty
    df_prim = load_primary_results()
    if not df_prim.empty and "segment_id" in df_prim.columns:
        return sorted(df_prim["segment_id"].unique().tolist())
    return ["SEG_001", "SEG_002", "SEG_003", "SEG_004"]


@st.cache_data(show_spinner=False)
def discover_segment_observations(segment_id: str) -> List[Dict[str, Any]]:
    """Discover all physical day observations and sidecar metadata for a given segment."""
    seg_dir = WORKSPACE_ROOT / f"env/output/temporal_segments/{segment_id}"
    df_prim = load_primary_results()
    df_dec = load_decisions_table()

    observations: List[Dict[str, Any]] = []

    # If physical directory exists, scan for day_*.png files
    if seg_dir.exists() and seg_dir.is_dir():
        img_files = sorted(
            seg_dir.glob("day_*.png"),
            key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem and p.stem.split("_")[1].isdigit() else 999
        )
        for f in img_files:
            try:
                day_num = int(f.stem.split("_")[1])
            except Exception:
                continue

            meta_file = seg_dir / f"{f.stem}_metadata.json"
            has_meta = meta_file.exists()
            sidecar_data = {}
            if has_meta:
                try:
                    sidecar_data = json.loads(meta_file.read_text(encoding="utf-8"))
                except Exception:
                    has_meta = False

            # Retrieve primary result row
            p_row = df_prim[(df_prim["segment_id"] == segment_id) & (df_prim["day"] == day_num)]
            p_rec = p_row.iloc[0].to_dict() if not p_row.empty else {}

            # Retrieve decision row
            d_row = df_dec[(df_dec["segment"] == segment_id) & (df_dec["day"] == day_num)]
            d_rec = d_row.iloc[0].to_dict() if not d_row.empty else {}

            obs = {
                "segment_id": segment_id,
                "day": day_num,
                "day_label": f"Day {day_num:02d}",
                "image_path": str(f),
                "has_metadata": has_meta,
                "metadata_status": "VALID" if has_meta else "METADATA MISSING",
                "camera_preset": sidecar_data.get("camera_preset") or p_rec.get("camera_preset", "Unknown"),
                "lighting_preset": sidecar_data.get("lighting_preset") or p_rec.get("lighting_preset", "Unknown"),
                "road_health_state": sidecar_data.get("road_health_state") or p_rec.get("road_health_state", "Unknown"),
                "moisture_state": sidecar_data.get("pothole_moisture_state") or ("Waterlogged" if p_rec.get("water_flag") else "Dry"),
                "water_flag": bool(p_rec.get("water_flag", False)),
                "current_severity": float(p_rec.get("current_severity", 0.0)),
                "current_severity_band": d_rec.get("current_severity_band", "LOW"),
                "defect_count": int(p_rec.get("defect_count", 0)),
                "defect_area_ratio": float(p_rec.get("defect_area_ratio", 0.0)),
                "surface_anomaly_score": float(p_rec.get("surface_anomaly_score", 0.0)),
                "temporal_status": p_rec.get("goal2_status", "ELIGIBLE"),
                "temporal_sequence_id": p_rec.get("goal2_sequence_id", "NONE"),
                "exclusion_reason": p_rec.get("goal2_exclusion_reason", "NONE"),
                "decision": d_rec.get("decision", "MONITOR"),
                "forecast_90d_wet": float(p_rec.get("wet_exposure_90d_forecast", 0.0)),
            }
            observations.append(obs)

    # Fallback to primary results if no files on disk
    if not observations and not df_prim.empty:
        sub_df = df_prim[df_prim["segment_id"] == segment_id].sort_values("day")
        for _, row in sub_df.iterrows():
            day_num = int(row["day"])
            obs = {
                "segment_id": segment_id,
                "day": day_num,
                "day_label": f"Day {day_num:02d}",
                "image_path": str(get_image_path(segment_id, day_num) or ""),
                "has_metadata": row.get("metadata_status") == "VALID",
                "metadata_status": row.get("metadata_status", "VALID"),
                "camera_preset": row.get("camera_preset", "Unknown"),
                "lighting_preset": row.get("lighting_preset", "Unknown"),
                "road_health_state": row.get("road_health_state", "Unknown"),
                "moisture_state": "Waterlogged" if row.get("water_flag") else "Dry",
                "water_flag": bool(row.get("water_flag", False)),
                "current_severity": float(row.get("current_severity", 0.0)),
                "current_severity_band": "LOW",
                "defect_count": int(row.get("defect_count", 0)),
                "defect_area_ratio": float(row.get("defect_area_ratio", 0.0)),
                "surface_anomaly_score": float(row.get("surface_anomaly_score", 0.0)),
                "temporal_status": row.get("goal2_status", "ELIGIBLE"),
                "temporal_sequence_id": row.get("goal2_sequence_id", "NONE"),
                "exclusion_reason": row.get("goal2_exclusion_reason", "NONE"),
                "decision": "MONITOR",
                "forecast_90d_wet": float(row.get("wet_exposure_90d_forecast", 0.0)),
            }
            observations.append(obs)

    return observations


@st.cache_data(show_spinner=False)
def load_sequence_daily_summary(sequence_id: str) -> pd.DataFrame:
    """Load daily_summary.csv for a specific temporal sequence."""
    p = TEMPORAL_DIR / f"{sequence_id}/daily_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_sequence_events(sequence_id: str) -> List[Dict[str, Any]]:
    """Load temporal_events.json for a specific temporal sequence."""
    p = TEMPORAL_DIR / f"{sequence_id}/temporal_events.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data.get("events", [])
        except Exception:
            pass
    return []


@st.cache_data(show_spinner=False)
def load_sequence_progression(sequence_id: str) -> Dict[str, Any]:
    """Load progression_summary.json for a specific temporal sequence."""
    p = TEMPORAL_DIR / f"{sequence_id}/progression_summary.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


