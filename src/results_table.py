"""Builds one model's row for the cross-model results table.

The shared table, reports/results/model_comparison.csv, is not edited from
here. If every model appended to it from its own branch, git would report a
merge conflict each time. Instead each model saves its own row file and one
person (Part 4) combines the rows at the end.

The columns start with the ones already used in model_comparison.csv (model,
split, accuracy, macro_f1, the five per-class F1 scores, hardware, notes).
Three are added: clean_accuracy and clean_macro_f1 (scores on the test tweets
with no exact duplicate in train) and train_seconds.
"""

import pandas as pd

from src.metrics import LABEL_ORDER

COLUMNS = (["model", "split", "accuracy", "macro_f1"]
           + [f"f1_{c}" for c in LABEL_ORDER]
           + ["hardware", "notes", "clean_accuracy", "clean_macro_f1", "train_seconds"])


def make_comparison_row(model, split, metrics, clean_metrics=None,
                        train_seconds=None, hardware="", notes="") -> pd.DataFrame:
    """Return a one-row table.

    metrics       : compute_metrics(...) output for the full set
    clean_metrics : compute_metrics(...) output for the clean subset (optional)
    """
    row = {"model": model, "split": split,
           "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
           "hardware": hardware, "notes": notes,
           "clean_accuracy": clean_metrics["accuracy"] if clean_metrics else "",
           "clean_macro_f1": clean_metrics["macro_f1"] if clean_metrics else "",
           "train_seconds": round(train_seconds, 1) if train_seconds is not None else ""}
    for label in LABEL_ORDER:
        row[f"f1_{label}"] = metrics["per_class_report"][label]["f1-score"]
    return pd.DataFrame([row])[COLUMNS]
