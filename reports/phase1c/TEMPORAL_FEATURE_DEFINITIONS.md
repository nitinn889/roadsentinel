# RoadSentinel Phase 1C: Temporal Feature Definitions & Leakage-Free Guarantees

**Document**: `reports/phase1c/TEMPORAL_FEATURE_DEFINITIONS.md`  
**Execution Phase**: Prompt 1C — Temporal-Residual XGBoost Forecasting  
**Supporting Manifest**: [`artifacts/phase1c/temporal_feature_manifest.csv`](../../artifacts/phase1c/temporal_feature_manifest.csv)  
**Date**: September 10, 2026  
**Status**: **MATHEMATICALLY VERIFIED & LEAK-FREE**  

---

## 1. Mathematical Framework & Notation

Let a monitored highway section $s \in S$ possess a sequence of distinct, chronologically ordered inspection visits:
$$\mathcal{V}(s) = \{(t_0, y_0, c_0), (t_1, y_1, c_1), \dots, (t_K, y_K, c_K)\}$$
where $t_i$ is the inspection timestamp ($t_0 < t_1 < \dots < t_K$), $y_i \in [0.0, 1.0]$ is the measured pavement distress severity, and $c_i \in \mathbb{N}$ is the LTPP construction/rehabilitation event number.

When forecasting future deterioration for transition pair $(t_k, t_{k+1})$, the **prediction origin** is $t_{\text{current}} = t_k$, the **current condition** is $y_{\text{current}} = y_k$, and the **forecast target** is $y_{\text{future}} = y_{k+1}$ across elapsed horizon $\Delta t = (t_{k+1} - t_k)$.

The set of historical inspections available at or before the prediction origin is strictly:
$$\mathcal{H}(t_k) = \{(t, y, c) \in \mathcal{V}(s) \mid t \le t_k\}$$
The set of prior inspections strictly preceding the current visit is:
$$\mathcal{H}_{<}(t_k) = \{(t, y, c) \in \mathcal{V}(s) \mid t < t_k\}$$

---

## 2. Formal Temporal Feature Definitions

All temporal features are constructed strictly from $\mathcal{H}(t_k)$ without touching any measurement or exposure occurring in $(t_k, t_{k+1}]$:

### 2.1 Inspection Counts & Chronological Elapsed Times
1. **Prior Inspection Count (`num_prior_inspections`)**:
   $$k = |\mathcal{H}_{<}(t_k)|$$
2. **Total Available Inspections (`num_available_inspections`)**:
   $$N_{\text{avail}} = |\mathcal{H}(t_k)| = k + 1$$
3. **Time Since First Observation (`time_since_first_observation_days`)**:
   $$\tau_{\text{first}} = (t_k - t_0).\text{days}$$
   *(Zero when evaluating the inaugural visit $t_0$.)*
4. **Time Since Previous Inspection (`time_since_prev_inspection_days`)**:
   $$\tau_{\text{prev}} = \begin{cases} (t_k - t_{k-1}).\text{days}, & \text{if } k \ge 1 \\ \text{NaN}, & \text{if } k = 0 \end{cases}$$

### 2.2 Lagged Severities & Recent Trajectory
5. **Previous Severity (`prev_severity`)**:
   $$y_{\text{prev}} = \begin{cases} y_{k-1}, & \text{if } k \ge 1 \\ \text{NaN}, & \text{if } k = 0 \end{cases}$$
6. **Second Previous Severity (`second_prev_severity`)**:
   $$y_{\text{second\_prev}} = \begin{cases} y_{k-2}, & \text{if } k \ge 2 \\ \text{NaN}, & \text{if } k < 2 \end{cases}$$
7. **Most Recent Severity Change (`most_recent_severity_change`)**:
   $$\Delta y_{\text{recent}} = \begin{cases} y_k - y_{k-1}, & \text{if } k \ge 1 \\ \text{NaN}, & \text{if } k = 0 \end{cases}$$
8. **Annualized Severity Slope (`annualized_severity_slope`)**:
   $$m_{\text{recent}} = \begin{cases} \frac{y_k - y_{k-1}}{\tau_{\text{prev}} / 365.25}, & \text{if } k \ge 1 \text{ and } \tau_{\text{prev}} > 0 \\ \text{NaN}, & \text{if } k = 0 \end{cases}$$

### 2.3 Rolling Historical Statistics & Segment Continuity
9. **Rolling Severity Mean (`rolling_severity_mean`)**:
   $$\bar{y}_{\text{hist}} = \frac{1}{k + 1} \sum_{i=0}^k y_i$$
10. **Rolling Severity Variability (`rolling_severity_std`)**:
    $$s_{y} = \begin{cases} \sqrt{\frac{1}{k} \sum_{i=0}^k (y_i - \bar{y}_{\text{hist}})^2}, & \text{if } k \ge 1 \\ 0.0, & \text{if } k = 0 \end{cases}$$
11. **Same Construction Event (`same_construction_no`)**:
    $$I_{\text{constr}} = \begin{cases} 1, & \text{if } k \ge 1 \text{ and } c_k = c_{k-1} \\ 0, & \text{if } k \ge 1 \text{ and } c_k \ne c_{k-1} \\ \text{NaN}, & \text{if } k = 0 \end{cases}$$
    *(Identifies whether an intervening major structural rehabilitation occurred.)*

---

## 3. Strict Non-Fabrication Rule for Unobserved Lags

Prompt 1C explicitly instructs:
> *"Genuine temporal-history features require at least two observations before the prediction origin. If insufficient history exists, do not fabricate lag features."*

In accordance with this rule:
* For pairs with $k = 0$ (the initial pair of a highway section, $N=23$): `prev_severity`, `second_prev_severity`, `most_recent_severity_change`, `annualized_severity_slope`, and `time_since_prev_inspection_days` are set strictly to **`NaN`**.
* For pairs with $k = 1$ (exactly one prior visit, $N=12$): `second_prev_severity` is set strictly to **`NaN`**.
* **Zero Artificial Interpolation**: The pipeline does **not** back-project $y_{k-1} \leftarrow y_k$ or invent constant zero changes.
* **XGBoost Native Missing Value Handling**: Gradient boosted decision trees naturally route `NaN` values to optimal default split directions determined during training, preserving honest representation of missing observational depth without introducing synthetic data bias.

---

## 4. Formal Proof of Leak-Free Feature Construction

To verify absolute mathematical independence from future observations:
1. **Timestamp Monotonicity**: Every element $(t, y, c) \in \mathcal{H}_{<}(t_k)$ satisfies $t < t_k < t_{k+1}$.
2. **Zero Lookahead Audit**: The automated script [`integration/phase1c/audit_feasibility_and_features.py`](../../integration/phase1c/audit_feasibility_and_features.py) asserted for all 113 pairs:
   $$\max_{(t, y, c) \in \mathcal{H}_{<}(t_k)} t < t_k$$
   Result: **`future_leakage_detected = False` for 100% of rows (113/113)**.
3. **Scenario Isolation**: Scenario variables (`rainfall_level`, `temperature`, `water_exposure`, `traffic_level`) describe environmental exposures occurring over the future interval $(t_k, t_{k+1}]$. They are strictly excluded from the temporal-history vector and evaluated only in scenario models M1, M2, and M4.
4. **Quantile Scaler Isolation**: In nested cross-validation, the severity scaling factors $q_{\text{low}}$ and $q_{\text{high}}$ are fitted exclusively on the outer training sites, preventing test fold IRI distributions from leaking into feature scaling.
