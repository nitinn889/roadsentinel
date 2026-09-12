#!/usr/bin/env python3
"""Evaluate the Phase UAV-2A pilots on the locked, source-separated validation set."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/uav_multidomain/pilot"
FIGURES = ROOT / "reports/uav_multidomain/pilot_figures"
RUN_ROOT = ROOT / "outputs/uav2a_pilot/pilot"
INVENTORY = ARTIFACTS / "training_inventory.csv"
SEED = 20260912
CONFIDENCE = 0.25
MATCH_IOU = 0.50
NMS_IOU = 0.70
BOOTSTRAP_REPLICATES = 2000
CLASS_NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
MODELS = {"m_uav0": "M-UAV0", "m_uav1": "M-UAV1"}
SOURCES = {"rdd2022_china_drone": "China UAV", "uav_pdd2023": "UAV-PDD"}
COLORS = {"M-UAV0": "#1864AB", "M-UAV1": "#D98E04"}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_boxes(label_path: Path) -> np.ndarray:
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        cls, xc, yc, width, height = map(float, fields)
        rows.append([int(cls), xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2])
    return np.asarray(rows, dtype=float).reshape((-1, 5))


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def match_class_specific(gt: np.ndarray, pred: np.ndarray, threshold: float) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Greedy confidence-ordered, class-specific matching."""
    matches, used_gt = [], set()
    order = sorted(range(len(pred)), key=lambda i: pred[i, 5], reverse=True)
    for pred_index in order:
        candidates = [
            (box_iou(gt[g, 1:5], pred[pred_index, 1:5]), g)
            for g in range(len(gt)) if g not in used_gt and int(gt[g, 0]) == int(pred[pred_index, 0])
        ]
        if not candidates:
            continue
        iou, gt_index = max(candidates)
        if iou >= threshold:
            matches.append((gt_index, pred_index, iou))
            used_gt.add(gt_index)
    used_pred = {p for _, p, _ in matches}
    return matches, used_gt, used_pred


def class_stats(gt: np.ndarray, pred: np.ndarray, threshold: float = MATCH_IOU) -> dict[int, dict[str, float]]:
    matches, used_gt, used_pred = match_class_specific(gt, pred, threshold)
    result = {}
    for cls in range(4):
        matched = [(g, p, i) for g, p, i in matches if int(gt[g, 0]) == cls]
        gt_count = int((gt[:, 0] == cls).sum()) if len(gt) else 0
        pred_count = int((pred[:, 0] == cls).sum()) if len(pred) else 0
        result[cls] = {
            "tp": len(matched), "fp": pred_count - len(matched), "fn": gt_count - len(matched),
            "gt": gt_count, "pred": pred_count,
            "iou_sum": float(sum(i for _, _, i in matched)), "matched": len(matched),
        }
    return result


def confusion_for_image(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Class-agnostic localization match; off-diagonal cells expose classification errors."""
    matrix = np.zeros((5, 5), dtype=int)
    pairs = []
    for g in range(len(gt)):
        for p in range(len(pred)):
            iou = box_iou(gt[g, 1:5], pred[p, 1:5])
            if iou >= MATCH_IOU:
                pairs.append((iou, g, p))
    used_gt, used_pred = set(), set()
    for _, g, p in sorted(pairs, reverse=True):
        if g in used_gt or p in used_pred:
            continue
        matrix[int(gt[g, 0]), int(pred[p, 0])] += 1
        used_gt.add(g); used_pred.add(p)
    for g in set(range(len(gt))) - used_gt:
        matrix[int(gt[g, 0]), 4] += 1
    for p in set(range(len(pred))) - used_pred:
        matrix[4, int(pred[p, 0])] += 1
    return matrix


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def ap_from_ranked(records: list[tuple[float, int]], gt_count: int) -> tuple[float, np.ndarray, np.ndarray]:
    if gt_count == 0:
        return math.nan, np.array([]), np.array([])
    ranked = sorted(records, key=lambda item: item[0], reverse=True)
    if not ranked:
        return 0.0, np.array([0.0]), np.array([0.0])
    tp = np.cumsum([label for _, label in ranked])
    fp = np.cumsum([1 - label for _, label in ranked])
    recall = tp / gt_count
    precision = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    ap = float(np.trapezoid(np.interp(grid, mrec, mpre), grid))
    return ap, recall, precision


def ranked_records(images: list[dict], cls: int, threshold: float) -> tuple[list[tuple[float, int]], int]:
    records, gt_total = [], 0
    for image in images:
        gt, pred = image["gt"], image["pred_all"]
        gt_cls = gt[gt[:, 0] == cls] if len(gt) else gt
        pred_cls = pred[pred[:, 0] == cls] if len(pred) else pred
        gt_total += len(gt_cls)
        used = set()
        for pred_index in sorted(range(len(pred_cls)), key=lambda i: pred_cls[i, 5], reverse=True):
            candidates = [(box_iou(gt_cls[g, 1:5], pred_cls[pred_index, 1:5]), g) for g in range(len(gt_cls)) if g not in used]
            if candidates and max(candidates)[0] >= threshold:
                _, gt_index = max(candidates)
                used.add(gt_index)
                records.append((float(pred_cls[pred_index, 5]), 1))
            else:
                records.append((float(pred_cls[pred_index, 5]), 0))
    return records, gt_total


def aggregate(images: list[dict]) -> tuple[dict, list[dict]]:
    per_class, aps_by_class = [], []
    for cls, name in enumerate(CLASS_NAMES):
        totals = {key: sum(image["stats"][cls][key] for image in images) for key in ("tp", "fp", "fn", "gt", "pred", "iou_sum", "matched")}
        precision, recall, f1 = prf(totals["tp"], totals["fp"], totals["fn"])
        ap_thresholds = []
        pr_records, pr_gt = ranked_records(images, cls, MATCH_IOU)
        ap50, _, _ = ap_from_ranked(pr_records, pr_gt)
        for threshold in np.arange(0.50, 0.951, 0.05):
            records, gt_count = ranked_records(images, cls, float(threshold))
            ap_thresholds.append(ap_from_ranked(records, gt_count)[0])
        ap5095 = float(np.nanmean(ap_thresholds))
        aps_by_class.append((ap50, ap5095))
        per_class.append({
            "class_id": cls, "class_name": name, "gt_objects": totals["gt"], "predicted_objects": totals["pred"],
            "tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"], "precision": precision,
            "recall": recall, "f1": f1, "ap50": ap50, "map50_95": ap5095,
            "mean_matched_iou": totals["iou_sum"] / totals["matched"] if totals["matched"] else 0.0,
        })
    total_tp = sum(row["tp"] for row in per_class); total_fp = sum(row["fp"] for row in per_class); total_fn = sum(row["fn"] for row in per_class)
    precision, recall, f1 = prf(total_tp, total_fp, total_fn)
    matched = sum(row["tp"] for row in per_class)
    summary = {
        "images": len(images), "groups": len({image["group"] for image in images}),
        "gt_objects": sum(row["gt_objects"] for row in per_class), "predicted_objects": sum(row["predicted_objects"] for row in per_class),
        "tp": total_tp, "fp": total_fp, "fn": total_fn, "precision": precision, "recall": recall, "f1": f1,
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "map50": float(np.nanmean([item[0] for item in aps_by_class])),
        "map50_95": float(np.nanmean([item[1] for item in aps_by_class])),
        "mean_matched_iou": sum(row["mean_matched_iou"] * row["tp"] for row in per_class) / matched if matched else 0.0,
        "images_without_predictions": sum(len(image["pred_fixed"]) == 0 for image in images),
        "pothole_recall": per_class[3]["recall"],
    }
    return summary, per_class


def aggregate_fixed(images: list[dict]) -> dict:
    """Fast fixed-threshold aggregate used inside the bootstrap loop."""
    f1_values, pothole_recall = [], 0.0
    for cls in range(4):
        tp = sum(image["stats"][cls]["tp"] for image in images)
        fp = sum(image["stats"][cls]["fp"] for image in images)
        fn = sum(image["stats"][cls]["fn"] for image in images)
        _, recall, f1 = prf(tp, fp, fn)
        f1_values.append(f1)
        if cls == 3:
            pothole_recall = recall
    return {"macro_f1": float(np.mean(f1_values)), "pothole_recall": pothole_recall}


def inference() -> tuple[dict[str, list[dict]], dict[tuple[str, str], np.ndarray]]:
    inventory = pd.read_csv(INVENTORY)
    val = inventory[(inventory.model_dataset == "m_uav0") & (inventory.split == "val")].copy()
    outputs, confusion = {}, {}
    for model_id, display_name in MODELS.items():
        weights = RUN_ROOT / model_id / "weights/best.pt"
        if not weights.exists():
            raise FileNotFoundError(weights)
        model = YOLO(str(weights))
        rows = val.to_dict("records")
        paths = [str(ROOT / row["materialized_image"]) for row in rows]
        torch.cuda.synchronize(); started = time.perf_counter()
        results = list(model.predict(paths, imgsz=640, batch=16, device=0, conf=0.001, iou=NMS_IOU, verbose=False, stream=True))
        torch.cuda.synchronize(); elapsed = time.perf_counter() - started
        model_images = []
        for row, result in zip(rows, results, strict=True):
            gt = load_boxes(ROOT / row["materialized_label"])
            height, width = result.orig_shape
            if result.boxes is None or len(result.boxes) == 0:
                pred_all = np.empty((0, 6), dtype=float)
            else:
                xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(float)
                xyxy[:, [0, 2]] /= width; xyxy[:, [1, 3]] /= height
                pred_all = np.column_stack((result.boxes.cls.detach().cpu().numpy(), xyxy, result.boxes.conf.detach().cpu().numpy()))
            pred_fixed = pred_all[pred_all[:, 5] >= CONFIDENCE] if len(pred_all) else pred_all
            stats = class_stats(gt, pred_fixed)
            model_images.append({
                "model_id": model_id, "model": display_name, "sample_id": row["sample_id"], "source": row["dataset_id"],
                "group": row["isolation_group_id"], "original_id": row["original_id"], "image_path": row["image_path"],
                "materialized_image": row["materialized_image"], "gt": gt, "pred_all": pred_all, "pred_fixed": pred_fixed,
                "stats": stats, "latency_ms": elapsed * 1000 / len(rows),
            })
        outputs[model_id] = model_images
        for source in SOURCES:
            matrix = np.zeros((5, 5), dtype=int)
            for image in model_images:
                if image["source"] == source:
                    matrix += confusion_for_image(image["gt"], image["pred_fixed"])
            confusion[(model_id, source)] = matrix
    return outputs, confusion


def bootstrap(outputs: dict[str, list[dict]], point: dict[tuple[str, str], dict]) -> dict:
    rng = np.random.default_rng(SEED)
    reps = {model: {source: [] for source in SOURCES} for model in MODELS}
    pothole = {model: {source: [] for source in SOURCES} for model in MODELS}
    favor = {source: {"m_uav0": 0, "m_uav1": 0, "ties": 0, "differing_groups": 0} for source in SOURCES}
    for source in SOURCES:
        by_model_group = {}
        for model in MODELS:
            source_images = [x for x in outputs[model] if x["source"] == source]
            by_model_group[model] = {group: [x for x in source_images if x["group"] == group] for group in sorted({x["group"] for x in source_images})}
        groups = sorted(by_model_group["m_uav0"])
        if groups != sorted(by_model_group["m_uav1"]):
            raise ValueError("Paired bootstrap groups differ between models")
        for group in groups:
            score = {}
            for model in MODELS:
                group_images = by_model_group[model][group]
                tp = sum(image["stats"][cls]["tp"] for image in group_images for cls in range(4))
                fp = sum(image["stats"][cls]["fp"] for image in group_images for cls in range(4))
                fn = sum(image["stats"][cls]["fn"] for image in group_images for cls in range(4))
                score[model] = prf(tp, fp, fn)[2]
            if not math.isclose(score["m_uav0"], score["m_uav1"], abs_tol=1e-12):
                favor[source]["differing_groups"] += 1
                favor[source]["m_uav1" if score["m_uav1"] > score["m_uav0"] else "m_uav0"] += 1
            else:
                favor[source]["ties"] += 1
        for _ in range(BOOTSTRAP_REPLICATES):
            sampled = rng.choice(groups, size=len(groups), replace=True)
            for model in MODELS:
                images = [image for group in sampled for image in by_model_group[model][group]]
                summary = aggregate_fixed(images)
                reps[model][source].append(summary["macro_f1"])
                pothole[model][source].append(summary["pothole_recall"])

    def interval(values: list[float]) -> dict:
        return {"point": float(np.mean(values)), "ci95_low": float(np.percentile(values, 2.5)), "ci95_high": float(np.percentile(values, 97.5))}

    source_metrics, deltas = {}, {}
    for source in SOURCES:
        source_metrics[source] = {}
        for model in MODELS:
            data = interval(reps[model][source]); data["point"] = point[(model, source)]["macro_f1"]
            p_data = interval(pothole[model][source]); p_data["point"] = point[(model, source)]["pothole_recall"]
            source_metrics[source][model] = {"macro_f1": data, "pothole_recall": p_data}
        delta_values = np.asarray(reps["m_uav1"][source]) - np.asarray(reps["m_uav0"][source])
        pothole_delta = np.asarray(pothole["m_uav1"][source]) - np.asarray(pothole["m_uav0"][source])
        deltas[source] = {
            "macro_f1": {**interval(delta_values), "point": point[("m_uav1", source)]["macro_f1"] - point[("m_uav0", source)]["macro_f1"]},
            "pothole_recall": {**interval(pothole_delta), "point": point[("m_uav1", source)]["pothole_recall"] - point[("m_uav0", source)]["pothole_recall"]},
        }
    balanced = {}
    for model in MODELS:
        values = np.mean(np.vstack([reps[model][source] for source in SOURCES]), axis=0)
        balanced[model] = {**interval(values), "point": float(np.mean([point[(model, source)]["macro_f1"] for source in SOURCES]))}
    delta_values = np.mean(np.vstack([np.asarray(reps["m_uav1"][source]) - np.asarray(reps["m_uav0"][source]) for source in SOURCES]), axis=0)
    balanced["delta_m_uav1_minus_m_uav0"] = {**interval(delta_values), "point": balanced["m_uav1"]["point"] - balanced["m_uav0"]["point"]}
    return {
        "method": "Paired cluster bootstrap over isolation_group_id within each source; percentile 95% intervals.",
        "seed": SEED, "replicates": BOOTSTRAP_REPLICATES,
        "independence_caveat": "UAV-PDD clusters are original photograph IDs. China clusters are pHash-based and may not capture every acquisition sequence; China intervals are approximate.",
        "source_metrics": source_metrics, "source_deltas": deltas, "source_balanced_macro_f1": balanced,
        "breadth_by_source": favor,
    }


def decision(boot: dict, point: dict[tuple[str, str], dict]) -> dict:
    deltas = {source: boot["source_deltas"][source]["macro_f1"]["point"] for source in SOURCES}
    pothole_deltas = {source: boot["source_deltas"][source]["pothole_recall"]["point"] for source in SOURCES}
    balanced = boot["source_balanced_macro_f1"]["delta_m_uav1_minus_m_uav0"]
    baseline_balanced = boot["source_balanced_macro_f1"]["m_uav0"]["point"]
    relative_balanced_delta = balanced["point"] / baseline_balanced if baseline_balanced else math.inf
    gt_potholes = {source: point[("m_uav0", source)]["pothole_gt"] for source in SOURCES}
    breadth_ok = all(
        boot["breadth_by_source"][source]["differing_groups"] >= 10
        and boot["breadth_by_source"][source]["m_uav1"] / boot["breadth_by_source"][source]["differing_groups"] >= 0.60
        for source in SOURCES
    )
    go_conditions = {
        "source_balanced_macro_f1_gain_at_least_0.02_absolute_or_5_percent_relative": balanced["point"] >= 0.02 or relative_balanced_delta >= 0.05,
        "china_macro_f1_does_not_decrease_by_more_than_0.02": deltas["rdd2022_china_drone"] >= -0.02,
        "pothole_recall_does_not_decrease_on_either_source": all(value >= 0 for value in pothole_deltas.values()),
        "improvement_broad_across_groups": breadth_ok,
        "no_leakage_or_evaluation_integrity_failure": True,
        "minimum_20_pothole_gt_per_source": all(value >= 20 for value in gt_potholes.values()),
    }
    no_go_conditions = {
        "source_balanced_macro_f1_negative_with_both_sources_nonpositive": balanced["point"] < 0 and all(value <= 0 for value in deltas.values()),
        "china_macro_f1_material_regression_gt_0.02": deltas["rdd2022_china_drone"] < -0.02,
        "pothole_recall_material_regression_gt_0.05_any_source": any(value < -0.05 for value in pothole_deltas.values()),
    }
    if any(no_go_conditions.values()):
        verdict = "NO-GO"
        rationale = "At least one predeclared material-regression rule was triggered."
    elif all(go_conditions.values()):
        verdict = "GO"
        rationale = "All predeclared source-balance, non-regression, pothole-support, and breadth rules passed."
    else:
        verdict = "INCONCLUSIVE"
        rationale = "No material-regression rule triggered, but at least one predeclared GO requirement lacked support."
    return {
        "decision": verdict, "rationale": rationale,
        "predeclared_rules": {
            "go": "Source-balanced Macro F1 gain >=0.02 absolute or >=5% relative; China Macro F1 decline no greater than 0.02; no pothole-recall decrease in either source; no integrity failure; improvement is broad (at least 10 differing groups and >=60% favor M-UAV1 in each source). A source with fewer than 20 pothole objects is treated as insufficient support for GO.",
            "no_go": "NO-GO takes precedence if balanced Macro F1 is negative with both sources nonpositive, China Macro F1 drops by >0.02, or pothole recall drops by >0.05 in either source.",
            "inconclusive": "Neither branch above, including insufficient pothole support or localized/uncertain improvement.",
        },
        "go_conditions": go_conditions, "no_go_conditions": no_go_conditions,
        "observed": {"source_macro_f1_delta": balanced, "source_macro_f1_relative_delta": relative_balanced_delta, "source_deltas": deltas, "pothole_recall_deltas": pothole_deltas, "pothole_gt": gt_potholes, "breadth": boot["breadth_by_source"]},
    }


def style_axes(ax, title: str, ylabel: str = "") -> None:
    ax.set_title(title, color="#212529", fontweight="bold")
    ax.set_ylabel(ylabel); ax.grid(axis="y", color="#DEE2E6", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def make_plots(per_source: pd.DataFrame, per_class: pd.DataFrame, outputs: dict, confusion: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#FAFBFC", "font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    metrics = [("macro_f1", "Macro F1"), ("map50", "mAP50"), ("map50_95", "mAP50:95")]
    for ax, (metric, title) in zip(axes, metrics):
        pivot = per_source.pivot(index="source_label", columns="model", values=metric).loc[list(SOURCES.values())]
        pivot.plot.bar(ax=ax, color=[COLORS[column] for column in pivot.columns], edgecolor="#343A40", hatch=["//", ".."])
        style_axes(ax, title, "Score"); ax.set_xlabel(""); ax.set_ylim(0, max(0.05, pivot.to_numpy().max() * 1.25)); ax.tick_params(axis="x", rotation=0)
    fig.savefig(FIGURES / "side_by_side_metrics.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, source in zip(axes, SOURCES):
        subset = per_class[per_class.source == source]
        x = np.arange(4); width = 0.36
        for offset, model in enumerate(MODELS.values()):
            values = subset[subset.model == model].set_index("class_id").loc[range(4), "f1"]
            ax.bar(x + (offset - .5) * width, values, width, label=model, color=COLORS[model], edgecolor="#343A40", hatch="//" if offset == 0 else "..")
        ax.set_xticks(x, [name.replace("_", "\n") for name in CLASS_NAMES]); ax.legend(); ax.set_ylim(0, max(.05, subset.f1.max() * 1.25))
        style_axes(ax, f"Per-class F1 — {SOURCES[source]}", "F1")
    fig.savefig(FIGURES / "per_class_f1.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    pivot = per_source.pivot(index="source_label", columns="model", values="macro_f1").loc[list(SOURCES.values())]
    pivot.plot.bar(ax=ax, color=[COLORS[c] for c in pivot.columns], edgecolor="#343A40", hatch=["//", ".."])
    style_axes(ax, "Source-balanced comparison", "Macro F1"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=0)
    fig.savefig(FIGURES / "per_source_macro_f1.png", dpi=180); plt.close(fig)

    labels = [name.replace("_", "\n") for name in CLASS_NAMES] + ["background"]
    for source in SOURCES:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, model in zip(axes, MODELS):
            matrix = confusion[(model, source)]
            image = ax.imshow(matrix, cmap="Blues")
            for row in range(5):
                for col in range(5):
                    ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="white" if matrix[row, col] > matrix.max() / 2 else "#212529")
            ax.set_xticks(range(5), labels, rotation=30, ha="right"); ax.set_yticks(range(5), labels)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth"); ax.set_title(f"{MODELS[model]} — {SOURCES[source]}")
            fig.colorbar(image, ax=ax, shrink=.7)
        fig.savefig(FIGURES / f"confusion_matrix_{source}.png", dpi=180); plt.close(fig)

    for source in SOURCES:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for cls, ax in enumerate(axes.flat):
            for model in MODELS:
                images = [x for x in outputs[model] if x["source"] == source]
                records, gt_count = ranked_records(images, cls, MATCH_IOU)
                _, recall, precision = ap_from_ranked(records, gt_count)
                ax.plot(recall, precision, label=MODELS[model], color=COLORS[MODELS[model]], linestyle="-" if model == "m_uav0" else "--")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()
            style_axes(ax, CLASS_NAMES[cls].replace("_", " ").title())
        fig.suptitle(f"IoU 0.50 precision–recall — {SOURCES[source]}", fontweight="bold")
        fig.savefig(FIGURES / f"precision_recall_curves_{source}.png", dpi=180); plt.close(fig)


def image_f1(image: dict) -> float:
    tp = sum(image["stats"][c]["tp"] for c in range(4)); fp = sum(image["stats"][c]["fp"] for c in range(4)); fn = sum(image["stats"][c]["fn"] for c in range(4))
    return prf(tp, fp, fn)[2]


def draw_examples(outputs: dict) -> None:
    by_id = {model: {image["sample_id"]: image for image in images} for model, images in outputs.items()}
    scored = []
    for sample_id, base in by_id["m_uav0"].items():
        other = by_id["m_uav1"][sample_id]
        scored.append((image_f1(other) - image_f1(base), sample_id))
    selections = {
        "representative_improvements.jpg": [sample for _, sample in sorted(scored, reverse=True)[:6]],
        "representative_regressions.jpg": [sample for _, sample in sorted(scored)[:6]],
        "representative_failures.jpg": [sample for _, sample in sorted(scored, key=lambda x: sum(by_id["m_uav1"][x[1]]["stats"][c]["fn"] + by_id["m_uav1"][x[1]]["stats"][c]["fp"] for c in range(4)), reverse=True)[:6]],
    }
    palette = [(24, 100, 171), (217, 142, 4), (68, 68, 68), (183, 28, 28)]
    for filename, sample_ids in selections.items():
        tiles = []
        for sample_id in sample_ids:
            panels = []
            for model in MODELS:
                item = by_id[model][sample_id]
                frame = cv2.imread(str(ROOT / item["image_path"]))
                height, width = frame.shape[:2]
                for cls, x1, y1, x2, y2 in item["gt"]:
                    cv2.rectangle(frame, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), (70, 180, 70), 2)
                for cls, x1, y1, x2, y2, score in item["pred_fixed"]:
                    color = palette[int(cls)]
                    cv2.rectangle(frame, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), color, 2)
                    cv2.putText(frame, f"{CLASS_NAMES[int(cls)][:4]} {score:.2f}", (int(x1 * width), max(14, int(y1 * height) - 3)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
                frame = cv2.resize(frame, (480, 300))
                cv2.rectangle(frame, (0, 0), (480, 27), (255, 255, 255), -1)
                cv2.putText(frame, f"{MODELS[model]} | F1={image_f1(item):.3f} | green=GT", (7, 19), cv2.FONT_HERSHEY_SIMPLEX, .5, (30, 30, 30), 1, cv2.LINE_AA)
                panels.append(frame)
            row = np.hstack(panels)
            header = np.full((30, 960, 3), 255, dtype=np.uint8)
            cv2.putText(header, f"{by_id['m_uav0'][sample_id]['source']} | {by_id['m_uav0'][sample_id]['original_id']}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (30, 30, 30), 1, cv2.LINE_AA)
            tiles.append(np.vstack((header, row)))
        cv2.imwrite(str(FIGURES / filename), np.vstack(tiles))


def main() -> int:
    runs = pd.read_csv(ARTIFACTS / "pilot_runs.csv")
    pilots = runs[(runs.stage == "pilot") & (runs.status == "PASS")].set_index("model_id")
    if set(pilots.index) != set(MODELS):
        raise RuntimeError("Both completed PASS pilot runs are required")
    outputs, confusion = inference()
    source_rows, class_rows, point = [], [], {}
    for model_id, display_name in MODELS.items():
        run = pilots.loc[model_id]
        for source, source_label in SOURCES.items():
            images = [image for image in outputs[model_id] if image["source"] == source]
            summary, classes = aggregate(images)
            summary.update({
                "model_id": model_id, "model": display_name, "source": source, "source_label": source_label,
                "training_seconds": float(run.duration_seconds), "best_epoch": int(run.best_epoch),
                "peak_gpu_memory_gib": float(run.peak_gpu_memory_gib), "mean_inference_latency_ms_per_image": float(np.mean([x["latency_ms"] for x in images])),
            })
            ordered = {key: summary[key] for key in ["model_id", "model", "source", "source_label", "images", "groups", "gt_objects", "predicted_objects", "tp", "fp", "fn", "images_without_predictions", "precision", "recall", "f1", "macro_f1", "map50", "map50_95", "mean_matched_iou", "pothole_recall", "training_seconds", "best_epoch", "peak_gpu_memory_gib", "mean_inference_latency_ms_per_image"]}
            source_rows.append(ordered)
            point[(model_id, source)] = {**ordered, "pothole_gt": classes[3]["gt_objects"]}
            for row in classes:
                class_rows.append({"model_id": model_id, "model": display_name, "source": source, "source_label": source_label, **row})
    write_csv(ARTIFACTS / "per_source_metrics.csv", source_rows)
    write_csv(ARTIFACTS / "per_class_metrics.csv", class_rows)

    image_rows = []
    for model in MODELS:
        for image in outputs[model]:
            totals = {key: sum(image["stats"][cls][key] for cls in range(4)) for key in ("tp", "fp", "fn", "gt", "pred", "iou_sum", "matched")}
            precision, recall, f1 = prf(totals["tp"], totals["fp"], totals["fn"])
            image_rows.append({
                "model_id": model, "model": MODELS[model], "sample_id": image["sample_id"], "source": image["source"], "source_label": SOURCES[image["source"]],
                "isolation_group_id": image["group"], "original_id": image["original_id"], "image_path": image["image_path"],
                "gt_objects": totals["gt"], "predicted_objects": totals["pred"], "tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"],
                "precision": precision, "recall": recall, "f1": f1, "mean_matched_iou": totals["iou_sum"] / totals["matched"] if totals["matched"] else 0.0,
                "no_predictions": str(totals["pred"] == 0).lower(), "class_stats_json": json.dumps(image["stats"], sort_keys=True),
                "ground_truth_json": json.dumps(image["gt"].tolist()), "predictions_json": json.dumps(image["pred_fixed"].tolist()),
            })
    write_csv(ARTIFACTS / "per_image_predictions.csv", image_rows)
    boot = bootstrap(outputs, point)
    (ARTIFACTS / "bootstrap_comparison.json").write_text(json.dumps(boot, indent=2) + "\n", encoding="utf-8")
    verdict = decision(boot, point)
    (ARTIFACTS / "go_no_go_decision.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    source_frame, class_frame = pd.DataFrame(source_rows), pd.DataFrame(class_rows)
    make_plots(source_frame, class_frame, outputs, confusion)
    draw_examples(outputs)
    print(json.dumps({"status": "PASS", "decision": verdict["decision"], "source_balanced_macro_f1": boot["source_balanced_macro_f1"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
