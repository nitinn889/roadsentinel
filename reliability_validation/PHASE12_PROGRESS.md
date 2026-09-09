# RoadSentinel Phase 12 Progress & Verification Log

**Phase**: 12 (Reliability Validation + Full Ablation Study)  
**Execution Timestamp**: 2026-09-09  
**Status**: **PASS**  

---

## 1. Completion Checklist

- [x] Multi-target evaluation ($T_0, T_1, T_2, T_3, T_4$) implemented and evaluated.
- [x] Target prevalence documented for China ($N=480$) and India ($N=300$).
- [x] Signal ablation models (Models A through H) trained and cross-validated.
- [x] In-domain 5-fold Stratified CV completed on China_Drone.
- [x] Zero-shot cross-domain evaluation completed on India Dashcam (zero India GT in training).
- [x] 95% Bootstrap Confidence Intervals computed for AUROC and AUPRC.
- [x] Raw DINO $k$-NN distance saturation audited (confirmed non-overlapping distributions).
- [x] DINO Macro Domain Gate evaluated ($\text{AUROC}=1.0000, \text{AUPRC}=1.0000$).
- [x] Selective prediction systems compared (Standalone YOLO vs. Confidence vs. Full Gated Architecture).
- [x] Causally valid temporal feature ablation completed on Experiment A ($N=33$, zero future leakage).
- [x] All 10 publication figures generated in `reliability_validation/figures/`.
- [x] All 6 structured CSV tables generated in `reliability_validation/tables/`.
- [x] Dashboard handoff package established in `integration/dashboard_assets/reliability_validation/`.
- [x] Research questions A–D answered.
- [x] Thesis contribution level and final architecture recommendation established.
- [x] Zero Unreal / CARLA / Retraining violations.

---

## 2. Key Verified Metrics

### Perceptual Targets Prevalence:
- **$T_0$ (F1 > 0)**: China 88.1% Succ / 11.9% Fail; India 2.7% Succ / 97.3% Fail
- **$T_1$ (F1 $\ge 0.50$)**: China 82.1% Succ / 17.9% Fail; India 2.3% Succ / 97.7% Fail
- **$T_2$ (Recall $\ge 0.50$)**: China 85.4% Succ / 14.6% Fail; India 2.7% Succ / 97.3% Fail
- **$T_3$ (Recall $\ge 0.75$)**: China 72.1% Succ / 27.9% Fail; India 1.3% Succ / 98.7% Fail
- **$T_4$ (Recall = 1.0)**: China 71.5% Succ / 28.5% Fail; India 1.3% Succ / 98.7% Fail

### In-Domain Performance by Target:
- **$T_0$**: AUROC **0.8650** [0.798, 0.921], AUPRC **0.7298**, Failure F1 **0.7111** (Model E)
- **$T_1$**: AUROC **0.8166** [0.767, 0.867], AUPRC **0.6490**, Failure F1 **0.5839** (Model B)
- **$T_2$**: AUROC **0.8239** [0.772, 0.880], AUPRC **0.6686**, Failure F1 **0.6341** (Model A)
- **$T_3$**: AUROC **0.6849** [0.631, 0.739], AUPRC **0.5893**, Failure F1 **0.5000** (Model A)
- **$T_4$**: AUROC **0.6777** [0.625, 0.734], AUPRC **0.5850**, Failure F1 **0.4984** (Model A)

### Domain Gating Performance:
- **Raw China Distance**: $0.1841 \pm 0.0847$ (Range $[0.0393, 0.5247]$)
- **Raw India Distance**: $0.8432 \pm 0.0441$ (Range $[0.6405, 0.9261]$)
- **Domain AUROC**: **1.0000** | **Domain AUPRC**: **1.0000**
- **Operating Threshold ($p_{99} = 0.4491$)**: 100.0% India shift detection, 1.46% China false alarms.
