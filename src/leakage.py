"""Cross-split duplicate check, shared by every part of the project.

The shared split was stratified by label only, so a small number of tweets
that are identical after cleaning ended up in different splits. A model can
score those rows by memorizing the training copy. This module defines the
leak-free subset in one place so every model is checked the same way: the
rows of a split whose cleaned text does not also appear in the train split.

The result matches the ID lists in reports/results/val_clean_ids.csv and
test_clean_ids.csv, which were built by hand earlier. Computing the sets here
keeps the check reproducible from the data instead of relying on those files.
"""

import pandas as pd

from src.data_prep import load_split


def load_merged() -> pd.DataFrame:
    """Cleaned tweets joined to their split, one row per tweet."""
    data = pd.read_csv("data/processed/train_cleaned.csv")
    return data.merge(load_split(), on="Tweet_ID")


def leaking_ids(merged: pd.DataFrame, split: str) -> set:
    """IDs of rows in `split` whose cleaned text also appears in train."""
    train_texts = set(merged.loc[merged["split"] == "train", "tweet_clean"])
    rows = merged[merged["split"] == split]
    return set(rows.loc[rows["tweet_clean"].isin(train_texts), "Tweet_ID"])


def leak_free_ids(merged: pd.DataFrame, split: str) -> set:
    """IDs of rows in `split` with no cleaned-text copy in train."""
    rows = merged[merged["split"] == split]
    return set(rows["Tweet_ID"]) - leaking_ids(merged, split)
