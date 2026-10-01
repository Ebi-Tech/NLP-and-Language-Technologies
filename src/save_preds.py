"""Saves one model's test predictions in the shared format used by the
cross-model comparison.

Every model writes results/preds_<name>_test.csv with the same columns, so
the comparison stage can read them all the same way. Probabilities are kept,
not just the predicted label, because ROC curves need the model's confidence
and that cannot be recovered later without retraining.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep import ID_TO_LABEL

RESULTS_DIR = Path("results")
LABEL_ORDER = [ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]


def save_preds(model_name, tweet_ids, probs, y_true) -> Path:
    """Write one row per test tweet: ID, probability per class, true and
    predicted label id.

    probs is the model's output over the 5 classes, shape (n_tweets, 5), with
    columns in label_id order. Keras model.predict and sklearn predict_proba
    both return it in that shape already.
    """
    probs = np.asarray(probs)
    if probs.shape[1] != len(LABEL_ORDER):
        raise ValueError(f"expected {len(LABEL_ORDER)} columns, got {probs.shape[1]}")

    df = pd.DataFrame(probs, columns=[f"p_{label}" for label in LABEL_ORDER])
    df.insert(0, "Tweet_ID", np.asarray(tweet_ids))
    df["true_id"] = np.asarray(y_true)
    df["pred_id"] = probs.argmax(axis=1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"preds_{model_name.lower().replace(' ', '_')}_test.csv"
    df.to_csv(out_path, index=False)
    return out_path
