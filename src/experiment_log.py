"""Experiment log for the tuning runs.

Every tuning run is recorded with the same columns: what was changed, what we
expected, what happened and what we decided. One row is one configuration,
summarised over its random seeds. The report uses these files to show how
earlier results shaped later experiments.

Each model has its OWN file, reports/results/experiments_<model>.csv. Separate
files mean two people never edit the same file on different branches, so
merging cannot create conflicts. load_all_experiments() joins them for the report.
"""

import csv
from datetime import date
from pathlib import Path

LOG_DIR = Path("reports/results")
COLUMNS = ["exp_id", "date", "owner", "model", "stage", "change", "hypothesis", "n_seeds",
           "val_macro_f1_mean", "val_macro_f1_std", "val_accuracy_mean", "mean_epochs",
           "mean_train_seconds", "decision", "reason"]


def log_path(model: str) -> Path:
    """File for one model's log, e.g. reports/results/experiments_textcnn.csv."""
    return LOG_DIR / f"experiments_{model.lower().replace(' ', '_')}.csv"


def log_experiment(**row) -> None:
    """Append one experiment to the model's log. Every column except `date` is required."""
    missing = [c for c in COLUMNS if c not in row and c != "date"]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    row.setdefault("date", date.today().isoformat())

    path = log_path(row["model"])
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new_file:
            writer.writeheader()  # header only for a brand-new file
        writer.writerow({c: row.get(c, "") for c in COLUMNS})


def reset_log(model: str) -> None:
    """Delete this model's log so that re-running a notebook does not duplicate rows."""
    log_path(model).unlink(missing_ok=True)


def load_all_experiments():
    """Join every model's log into one table (for the report)."""
    import pandas as pd

    files = sorted(LOG_DIR.glob("experiments_*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame(columns=COLUMNS)
