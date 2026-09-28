"""Shared evaluation code. Every model in this project is scored with the
same function, so results are directly comparable across all five approaches.

Accuracy is reported because it is the actual Zindi leaderboard metric, but
given the severe class imbalance confirmed in the Part 1 EDA (the smallest
class is under 0.5% of the data), accuracy alone cannot show whether a model
has learned anything about the rare classes. Macro-F1 and per-class recall
are used as the primary basis for comparing models for that reason.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.data_prep import ID_TO_LABEL

LABEL_ORDER = [ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]

FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("reports/results")

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
INK = "#0b0b0b"
MUTED = "#898781"


def compute_metrics(y_true, y_pred) -> dict:
    """Compute the full shared metric set for one model's predictions.

    y_true and y_pred are arrays of integer label ids, matching the
    encoding in data/processed/labels.json.
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABEL_ORDER))),
        target_names=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_ORDER))))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class_report": report,
        "confusion_matrix": cm,
    }


def format_metrics_report(metrics: dict, model_name: str) -> str:
    """Render a metrics dict as a plain-text summary for logging or a report."""
    lines = [
        f"Model: {model_name}",
        f"Accuracy (leaderboard metric): {metrics['accuracy']:.4f}",
        f"Macro F1: {metrics['macro_f1']:.4f}",
        f"Weighted F1: {metrics['weighted_f1']:.4f}",
        "",
        "Per-class precision / recall / F1 / support:",
    ]
    for label in LABEL_ORDER:
        row = metrics["per_class_report"][label]
        lines.append(
            f"  {label:30s} "
            f"precision={row['precision']:.3f}  "
            f"recall={row['recall']:.3f}  "
            f"f1={row['f1-score']:.3f}  "
            f"support={int(row['support'])}"
        )
    return "\n".join(lines)


def plot_confusion_matrix(cm: np.ndarray, model_name: str, save: bool = True):
    """Plot a confusion matrix as a single-hue heatmap (sequential magnitude
    encoding, per the project's chart conventions) and optionally save it.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    im = ax.imshow(cm, cmap=cmap)

    ax.set_xticks(range(len(LABEL_ORDER)))
    ax.set_yticks(range(len(LABEL_ORDER)))
    ax.set_xticklabels(LABEL_ORDER, rotation=35, ha="right", fontsize=8, color=MUTED)
    ax.set_yticklabels(LABEL_ORDER, fontsize=8, color=MUTED)
    ax.set_xlabel("Predicted label", color=INK)
    ax.set_ylabel("True label", color=INK)
    ax.set_title(f"Confusion matrix: {model_name}", fontsize=12, loc="left", color=INK)

    max_val = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text_color = "#ffffff" if value > max_val * 0.6 else INK
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8, color=text_color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = model_name.lower().replace(" ", "_")
        plt.savefig(FIGURES_DIR / f"confusion_matrix_{safe_name}.png", dpi=150)

    return fig


def save_metrics(metrics: dict, model_name: str, split: str) -> Path:
    """Save one model's metrics as JSON, in the shared format every part of
    the project should use. Kevin's Results & Visualizations section reads
    every file in reports/results/ to build the five-model comparison table,
    so this format is not something to change independently per part.

    split is which data subset these metrics were computed on, e.g. "val"
    or "test", and is recorded alongside the numbers so it's unambiguous
    later which metrics came from tuning versus the final held-out result.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")
    payload = {
        "model_name": model_name,
        "split": split,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "per_class_report": metrics["per_class_report"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "label_order": LABEL_ORDER,
    }
    out_path = RESULTS_DIR / f"{safe_name}_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path
