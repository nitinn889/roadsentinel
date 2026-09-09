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
    p = WORKSPACE_ROOT / f"env/output/manual_inspections/{day_str}/{segment_id}/raw.png"
    if p.exists():
        return p
    p_alt = WORKSPACE_ROOT / f"env/output/manual_inspections/{day_str}/{segment_id}/original.jpg"
    if p_alt.exists():
        return p_alt
    return None

