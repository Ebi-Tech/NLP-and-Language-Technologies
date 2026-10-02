"""Shared input pipeline for the neural models (BiLSTM and TextCNN).

The settings copy notebooks/bidirectional-lstm.ipynb exactly: Keras Tokenizer,
20,000-word vocabulary, <OOV> token for unknown words, maximum length 70,
padding and truncation at the end of the tweet, and "balanced" class weights.
Keeping the input identical means any difference between the BiLSTM and the
TextCNN comes from the architecture and not from how the text was prepared.

Use it from the repository root:
    from src.neural_data import load_neural_data

Per-tweet predictions are saved with the group's helper, src/save_preds.py.
"""

import random

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from src.data_prep import load_split

MAX_WORDS = 20000   # vocabulary size cap
MAX_LEN = 70        # tweets are padded or cut to this many tokens
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and TensorFlow so that a run can be repeated."""
    import tensorflow as tf  # imported here so this module loads without TensorFlow

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_frames() -> dict:
    """Return the train, val and test tables from the shared split file.

    Each table gets an `is_clean` column. It is True when the tweet's exact text
    does NOT also appear in the training set. Duplicates across splits were
    found in Part 2, and scoring on the clean rows shows how much they matter.
    The flag is computed here from the text, so it does not depend on the
    saved *_clean_ids.csv files.
    """
    cleaned = pd.read_csv("data/processed/train_cleaned.csv")
    merged = cleaned.merge(load_split(), on="Tweet_ID", how="inner")
    assert len(merged) == len(cleaned), "split file does not cover every tweet"

    train_texts = set(merged.loc[merged["split"] == "train", "tweet_clean"].str.strip())
    merged["is_clean"] = True  # train rows stay True; only val and test are checked
    for name in ("val", "test"):
        rows = merged["split"] == name
        in_train = merged.loc[rows, "tweet_clean"].str.strip().isin(train_texts)
        merged.loc[rows, "is_clean"] = ~in_train

    return {name: merged[merged["split"] == name].reset_index(drop=True)
            for name in ("train", "val", "test")}


def load_neural_data(max_words: int = MAX_WORDS, max_len: int = MAX_LEN) -> dict:
    """Tokenise and pad every split, and compute the class weights.

    The vocabulary is built from the training tweets only, so nothing from
    validation or test leaks into it.

    The returned dictionary holds, for each split s in {train, val, test}:
        X_s      padded integer sequences, shape (n_tweets, max_len)
        y_s      integer labels 0-4
        clean_s  True where the tweet has no exact duplicate in train
    plus `frames` (the tables), `tokenizer`, `encode` (turns new texts into
    padded sequences with the same tokenizer) and `class_weights`.
    """
    # Imported here so that importing this module does not require TensorFlow.
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    frames = load_frames()
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(frames["train"]["tweet_clean"])

    def encode(texts):
        """Texts -> padded integer sequences (padding and cutting at the end)."""
        return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_len,
                             padding="post", truncating="post")

    out = {"frames": frames, "tokenizer": tokenizer, "encode": encode}
    for split, table in frames.items():
        out[f"X_{split}"] = encode(table["tweet_clean"])
        out[f"y_{split}"] = table["label_id"].values
        out[f"clean_{split}"] = table["is_clean"].values

    # "balanced" weights are inversely proportional to class frequency, so the
    # rare classes count for more in the loss. Computed from train labels only.
    classes = np.unique(out["y_train"])
    weights = compute_class_weight("balanced", classes=classes, y=out["y_train"])
    out["class_weights"] = {int(c): float(w) for c, w in zip(classes, weights)}
    return out
