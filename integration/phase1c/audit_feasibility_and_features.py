#!/usr/bin/env python3
"""Phase 1C: Data-Feasibility Audit & Temporal-History Feature Construction.

This module performs the rigorous feasibility audit mandated by Prompt 1C:
1. Audits all LTPP SDR 40 records (sites, timestamps, intervals, missingness).
2. Quantifies how many sites and pairs have >= 1 and >= 2 prior inspections.
3. Constructs strictly leak-free temporal-history features using only information
   available at or before the prediction origin (t_current).
4. Generates artifacts/phase1c/temporal_feature_manifest.csv.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("phase1c_audit")

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TABLE_PATH = WORKSPACE_ROOT / "xgboost" / "data" / "processed" / "scenario_training_pairs.csv"
SUMMARY_PATH = WORKSPACE_ROOT / "xgboost" / "data" / "processed" / "scenario_dataset_summary.json"
OUTPUT_DIR = WORKSPACE_ROOT / "artifacts" / "phase1c"
MANIFEST_PATH = OUTPUT_DIR / "temporal_feature_manifest.csv"
FEASIBILITY_SUMMARY_PATH = OUTPUT_DIR / "feasibility_audit_summary.json"


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def build_site_visit_index(df: pd.DataFrame) -> Dict[str, List[Tuple[datetime, float, int]]]:
    """Reconstructs the complete sorted chronological visit history per site.
    
    Each element is (visit_date, mri, construction_no).
    """
    site_visits: Dict[str, Dict[datetime, Tuple[float, int]]] = {}
    
    for _, row in df.iterrows():
        site_id = str(row["site_id"])
        c_no = int(row["construction_no"])
        
        c_date = parse_date(str(row["current_date"]))
        f_date = parse_date(str(row["future_date"]))
        
        c_iri = float(row["current_iri_m_per_km"])
        f_iri = float(row["future_iri_m_per_km"])
        
        if site_id not in site_visits:
            site_visits[site_id] = {}
            
        site_visits[site_id][c_date] = (c_iri, c_no)
        site_visits[site_id][f_date] = (f_iri, c_no)
        
    sorted_site_visits: Dict[str, List[Tuple[datetime, float, int]]] = {}
    for site_id, visits in site_visits.items():
        sorted_list = sorted([(d, v[0], v[1]) for d, v in visits.items()], key=lambda x: x[0])
        sorted_site_visits[site_id] = sorted_list
        
    return sorted_site_visits


def audit_and_extract_features() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    log.info("Loading LTPP scenario table from %s", TABLE_PATH)
    df = pd.read_csv(TABLE_PATH)
    
    # Sort pairs chronologically within each site
    df["current_dt"] = pd.to_datetime(df["current_date"])
    df["future_dt"] = pd.to_datetime(df["future_date"])
    df = df.sort_values(["site_id", "current_dt"]).reset_index(drop=True)
    
    site_visit_index = build_site_visit_index(df)
    
    records: List[Dict[str, Any]] = []
    
    for idx, row in df.iterrows():
        site_id = str(row["site_id"])
        c_date = parse_date(str(row["current_date"]))
        f_date = parse_date(str(row["future_date"]))
        c_iri = float(row["current_iri_m_per_km"])
        f_iri = float(row["future_iri_m_per_km"])
        c_constr = int(row["construction_no"])
        
        # All chronological visits for this site
        all_visits = site_visit_index[site_id]
        first_visit_date = all_visits[0][0]
        
        # Strictly prior visits before c_date
        prior_visits = [(d, iri, constr) for d, iri, constr in all_visits if d < c_date]
        num_prior = len(prior_visits)
        
        # Leakage check: ensure no prior visit has date >= c_date
        future_leak = any(d >= c_date for d, _, _ in prior_visits)
        
        # Base pair identifier
        pair_id = f"{site_id}_{row['current_date']}_{row['future_date']}"
        
        # Temporal features
        time_since_first = (c_date - first_visit_date).days
        
        if num_prior >= 1:
            prev_date, prev_iri, prev_constr = prior_visits[-1]
            time_since_prev = (c_date - prev_date).days
            most_recent_change = c_iri - prev_iri
            annualized_slope = most_recent_change / (time_since_prev / 365.25) if time_since_prev > 0 else 0.0
            same_construction = int(c_constr == prev_constr)
        else:
            prev_iri = np.nan
            time_since_prev = np.nan
            most_recent_change = np.nan
            annualized_slope = np.nan
            same_construction = np.nan
            
        if num_prior >= 2:
            _, second_prev_iri, _ = prior_visits[-2]
        else:
            second_prev_iri = np.nan
            
        # Available severities up to c_date (prior + current)
        history_iris = [v[1] for v in prior_visits] + [c_iri]
        rolling_mean = float(np.mean(history_iris))
        rolling_std = float(np.std(history_iris, ddof=1)) if len(history_iris) > 1 else (0.0 if num_prior == 0 else np.nan)
        
        records.append({
            "pair_id": pair_id,
            "site_id": site_id,
            "state": row["state"],
            "state_code": int(row["state_code"]),
            "shrp_id": str(row["shrp_id"]),
            "construction_no": c_constr,
            "current_date": row["current_date"],
            "future_date": row["future_date"],
            "observed_days_ahead": int(row["observed_days_ahead"]),
            "days_ahead": int(row["days_ahead"]),
            "current_iri_m_per_km": c_iri,
            "future_iri_m_per_km": f_iri,
            "delta_iri_m_per_km": f_iri - c_iri,
            # Scenario covariates
            "rainfall_level": float(row["rainfall_level"]),
            "traffic_level": float(row["traffic_level"]),
            "temperature": float(row["temperature"]),
            "water_exposure": float(row["water_exposure"]),
            "traffic_source": str(row["traffic_source"]),
            # Temporal history features (strictly <= t_current)
            "num_prior_inspections": num_prior,
            "num_available_inspections": num_prior + 1,
            "time_since_first_observation_days": time_since_first,
            "time_since_prev_inspection_days": time_since_prev,
            "prev_iri_m_per_km": prev_iri,
            "second_prev_iri_m_per_km": second_prev_iri,
            "most_recent_iri_change": most_recent_change,
            "annualized_iri_slope": annualized_slope,
            "rolling_iri_mean": rolling_mean,
            "rolling_iri_std": rolling_std,
            "same_construction_no": same_construction,
            # Indicator flags
            "has_at_least_1_prior": num_prior >= 1,
            "has_at_least_2_prior": num_prior >= 2,
            "future_leakage_detected": future_leak,
        })
        
    manifest_df = pd.DataFrame(records)
    
    # Run sanity checks
    assert not manifest_df["future_leakage_detected"].any(), "CRITICAL: Future leakage detected in feature construction!"
    assert len(manifest_df) == 113, f"Expected 113 pairs, got {len(manifest_df)}"
    assert manifest_df["site_id"].nunique() == 23, f"Expected 23 sites, got {manifest_df['site_id'].nunique()}"
    
    # Save manifest
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(MANIFEST_PATH, index=False)
    log.info("Saved temporal feature manifest (%d rows, %d cols) to %s", len(manifest_df), len(manifest_df.columns), MANIFEST_PATH)
    
    # Generate feasibility audit summary dictionary
    site_pair_counts = manifest_df["site_id"].value_counts().to_dict()
    site_visit_counts = {s: len(v) for s, v in site_visit_index.items()}
    
    summary = {
        "dataset_name": "FHWA LTPP SDR 40 (Joined Scenario Deterioration Table)",
        "source_file": "xgboost/data/processed/scenario_training_pairs.csv",
        "total_pairs": len(manifest_df),
        "total_sites": manifest_df["site_id"].nunique(),
        "state_breakdown": manifest_df["state"].value_counts().to_dict(),
        "interval_days": {
            "minimum": int(manifest_df["observed_days_ahead"].min()),
            "maximum": int(manifest_df["observed_days_ahead"].max()),
            "mean": float(manifest_df["observed_days_ahead"].mean()),
            "median": float(manifest_df["observed_days_ahead"].median()),
            "p25": float(manifest_df["observed_days_ahead"].quantile(0.25)),
            "p75": float(manifest_df["observed_days_ahead"].quantile(0.75)),
        },
        "missing_values_per_raw_column": {col: int(df[col].isna().sum()) for col in df.columns if col not in ("current_dt", "future_dt")},
        "temporal_history_sufficiency": {
            "sites_with_at_least_3_chronological_visits": sum(c >= 3 for c in site_visit_counts.values()),
            "sites_with_at_least_4_chronological_visits": sum(c >= 4 for c in site_visit_counts.values()),
            "pairs_with_0_prior_inspections": int((manifest_df["num_prior_inspections"] == 0).sum()),
            "pairs_with_at_least_1_prior_inspection": int((manifest_df["num_prior_inspections"] >= 1).sum()),
            "pairs_with_at_least_2_prior_inspections": int((manifest_df["num_prior_inspections"] >= 2).sum()),
            "percentage_pairs_with_full_lag_history_pct": round(float((manifest_df["num_prior_inspections"] >= 2).mean() * 100), 2),
        },
        "scenario_variable_provenance": {
            "rainfall_level": "FHWA LTPP Virtual Weather Station (VWS) daily precipitation prefix sums (mm/day)",
            "temperature": "FHWA LTPP Virtual Weather Station (VWS) daily mean temperature (deg C)",
            "water_exposure": "Fraction of interval days with daily precipitation > 0 mm",
            "traffic_level": "FHWA LTPP AADTT_LTPPLN heavy truck traffic / day with TRF_REP fallback",
            "days_ahead": "Time elapsed between profile visits (binned to 30-day increments)",
        },
        "causal_claims_permitted": False,
        "counterfactual_capability_validated": False,
        "valid_forecasting_horizon_range_days": [240, 720],
        "rapid_forecasting_30_90_days_supported": False,
        "audit_timestamp": datetime.now().isoformat(),
    }
    
    with open(FEASIBILITY_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved feasibility audit summary to %s", FEASIBILITY_SUMMARY_PATH)
    
    return manifest_df, summary


if __name__ == "__main__":
    audit_and_extract_features()
