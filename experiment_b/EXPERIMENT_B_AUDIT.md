# RoadSentinel — Experiment B Capture Audit & Status Report
## Verification of Controlled Deterioration -> Repair -> Re-Deterioration Capture Set

---

## 1. Audit Summary

An exhaustive filesystem and metadata audit was conducted across all potential capture locations in the repository:
1. `env/output/temporal_segments/` (Authoritative interactive capture root used in Phase 1B/1D)
2. `env/output/manual_inspections/` (Legacy exploratory automated captures)
3. `env/output/temporal_20_day/` (Legacy 20-day headless runs)

### Primary Finding:
- **`SEG_005` (Experiment B Primary)**: **0 / 10 Images** | **0 / 10 Sidecar Metadata JSONs** in `env/output/temporal_segments/`
- **`SEG_006` (Experiment B Replication)**: **0 / 10 Images** | **0 / 10 Sidecar Metadata JSONs** in `env/output/temporal_segments/`
- **Current Status**: **NOT YET CAPTURED BY TEAM LEAD**

---

## 2. Detailed Audit Matrix

| Target Segment | Expected Days | Expected Camera | Expected Lighting | Authoritative Path | Images Present | Sidecars Present | Day 06 Repair Present | Validation Result |
|---|---|---|---|---|---|---|---|---|
| **`SEG_005`** | Day 01–10 | `🔭 Overhead Drone Survey` | `Clear Noon (70° Sun)` | `env/output/temporal_segments/SEG_005/` | **0 / 10** | **0 / 10** | ❌ None | **PENDING TEAM LEAD CAPTURE** |
| **`SEG_006`** | Day 01–10 | `🔭 Overhead Drone Survey` | `Clear Noon (70° Sun)` | `env/output/temporal_segments/SEG_006/` | **0 / 10** | **0 / 10** | ❌ None | **PENDING TEAM LEAD CAPTURE** |

---

## 3. Analysis of Legacy Datasets & Non-Compliance

### 3.1 `env/output/manual_inspections/` (Legacy Sep 6)
- **Severe Environmental Confounding**: Contains randomized weather per day (`Night Highway with Lamps`, `Heavy Rain & Wet Road`, `Dense Atmospheric Fog`, `Overcast Day`).
- **No Scripted Repair Intervention**: On Day 06, condition scores continue monotonic deterioration ($0.025 \rightarrow 0.0467 \rightarrow 0.2156$ on Day 10) rather than a pristine reset.
- **Verdict**: Unsuitable for controlled Experiment B; violates Section 3 (environmental constancy) and Section 4 (Day 06 repair).

### 3.2 `env/output/temporal_20_day/` (Legacy Sep 6)
- **Zero Degradation on Days 01–05**: Manifest condition score is clamped at $0.025$ with 0 defects.
- **No Scripted Repair on Day 06**: Score rises to $0.0467$ with 0 defects, then jumps to single crack on Day 08.
- **Verdict**: Unsuitable for controlled Experiment B.

---

## 4. Required Action for Team Lead

In accordance with the **Absolute Safety Rule** (*Agent must NEVER launch or control Unreal Engine, CARLA, or Interactive Studio*), the Team Lead must launch the RoadSentinel Interactive Studio:

```bash
./launch_studio.sh
```

and perform the 10 manual/procedural captures for **`SEG_005`** and **`SEG_006`** following the protocol below:

### Capture Protocol for Team Lead:
1. **Camera Preset**: Select `🔭 Overhead Drone Survey (SAM 2 Top-Down)` for **ALL 10 DAYS**.
2. **Lighting / Weather**: Select `Clear Noon (70° Sun)` and `Dry Road` for **ALL 10 DAYS**.
3. **SEG_005 Progression**:
   - Day 01: `Pristine (Grade A)` ($\text{Density} \le 2.0$)
   - Day 02: `Minor Wear (Grade B)` ($\text{Density} \approx 3.5$)
   - Day 03: `Moderate Deterioration (Grade C)` ($\text{Density} \approx 5.0$)
   - Day 04: `Severe Breakdown (Grade D)` ($\text{Density} \approx 7.0$)
   - Day 05: `Critical Hazard (Grade F)` ($\text{Density} \approx 9.0$)
   - Day 06: **`Pristine (Grade A)` Reset** ($\text{Density} = 0.0$, Intervention = `REPAIR`)
   - Day 07: `Minor Wear (Grade B)` ($\text{Density} \approx 3.0$)
   - Day 08: `Moderate Deterioration (Grade C)` ($\text{Density} \approx 5.0$)
   - Day 09: `Severe Breakdown (Grade D)` ($\text{Density} \approx 7.0$)
   - Day 10: `Critical Hazard (Grade F)` ($\text{Density} \approx 9.0$)
4. **SEG_006 Replication Progression**: Repeat identical 10-day lifecycle on `SEG_006`.

---

## 5. Pipeline Execution Readiness

All downstream processing modules are tested, typed, and fully ready in the repository:
- `experiment_b/EXPERIMENT_B_METHODOLOGY.md` (Formal methodology)
- `experiment_b/audit_captures.py` (Automated capture integrity auditor)
- `sam2_dino/run_sam2_dino.py` (Frozen DINOv2+SAM2 feature extraction)
- `yolo/` (Frozen YOLOv8n object detection)
- `xgboost/temporal/` (Hierarchical greedy temporal defect tracking)
- `reliability/` (Phase-8 calibrated reliability scoring & risk-coverage evaluation)

Once the Team Lead saves the 20 captures to `env/output/temporal_segments/`, the agent can immediately process the images, generate all tables and figures, produce `lifecycle_handoff.json`, and finalize the research summary.
