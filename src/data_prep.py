"""Shared data preparation: label encoding, text cleaning, and the fixed
train/validation/test split. Every part of the project imports from here
instead of re-implementing its own version, so results stay comparable.
"""

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = Path("splits")
LABELS_PATH = PROCESSED_DIR / "labels.json"
SPLIT_PATH = SPLITS_DIR / "train_val_test_split.csv"

# Canonical label set, normalized to lowercase, sorted alphabetically so the
# integer encoding is fixed and reproducible for everyone.
LABEL_TO_ID = {
    "economic_violence": 0,
    "emotional_violence": 1,
    "harmful_traditional_practice": 2,
    "physical_violence": 3,
    "sexual_violence": 4,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def normalize_label(raw_label: str) -> str:
    """Lowercase a raw label string so casing differences collapse together."""
    return raw_label.strip().lower()


def clean_text(text: str) -> str:
    """Light cleaning appropriate for this dataset.

    The raw tweets already have almost no hashtags, mentions, or URLs, so
    this does not strip those out. It normalizes whitespace and smart quotes
    and lowercases the text, leaving punctuation and emoji intact since they
    may carry signal for a sequential model.
    """
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def load_labeled_data() -> pd.DataFrame:
    """Load Train.csv and attach the normalized label and its integer id."""
    df = pd.read_csv(RAW_DIR / "Train.csv")
    df["label"] = df["type"].map(normalize_label)
    df["label_id"] = df["label"].map(LABEL_TO_ID)
    df["tweet_clean"] = df["tweet"].map(clean_text)
    return df


def build_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Create a fixed stratified 70/15/15 train/val/test split.

    Stratifying on label_id keeps class proportions consistent across all
    three subsets, which matters here since the smallest class has only
    188 examples in the whole dataset.
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label_id"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=seed
    )

    split_map = pd.concat(
        [
            pd.DataFrame({"Tweet_ID": train_df["Tweet_ID"], "split": "train"}),
            pd.DataFrame({"Tweet_ID": val_df["Tweet_ID"], "split": "val"}),
            pd.DataFrame({"Tweet_ID": test_df["Tweet_ID"], "split": "test"}),
        ],
        ignore_index=True,
    )
    return split_map


def save_artifacts(df: pd.DataFrame, split_map: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_DIR / "train_cleaned.csv", index=False)
    split_map.to_csv(SPLIT_PATH, index=False)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(LABEL_TO_ID, f, indent=2)


def load_split() -> pd.DataFrame:
    """Load the shared split file. Every part should call this rather than
    generating its own split.
    """
    return pd.read_csv(SPLIT_PATH)


if __name__ == "__main__":
    data = load_labeled_data()
    splits = build_split(data)
    save_artifacts(data, splits)
    print(f"Saved cleaned data: {len(data)} rows")
    print(f"Saved split map: {splits['split'].value_counts().to_dict()}")
