"""Part 1 baselines: majority-class floor and TF-IDF + Logistic Regression.

Both models train on the shared train split and are evaluated on the shared
validation split. The majority-class baseline exists to make the class
imbalance concrete: any real model needs to clear this floor by more than
accuracy alone, since the floor itself can reach a high accuracy purely by
always predicting the dominant class.
"""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_prep import load_split
from src.metrics import compute_metrics, format_metrics_report, plot_confusion_matrix, save_metrics


def load_train_val():
    data = pd.read_csv("data/processed/train_cleaned.csv")
    split_map = load_split()
    merged = data.merge(split_map, on="Tweet_ID")

    train = merged[merged["split"] == "train"]
    val = merged[merged["split"] == "val"]
    return train, val


def run_majority_baseline(train, val):
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(train[["tweet_clean"]], train["label_id"])
    preds = clf.predict(val[["tweet_clean"]])

    metrics = compute_metrics(val["label_id"], preds)
    print(format_metrics_report(metrics, "majority_baseline"))
    plot_confusion_matrix(metrics["confusion_matrix"], "majority_baseline")
    save_metrics(metrics, "majority_baseline", split="val")
    return metrics


def run_tfidf_logreg(train, val):
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipeline.fit(train["tweet_clean"], train["label_id"])
    preds = pipeline.predict(val["tweet_clean"])

    metrics = compute_metrics(val["label_id"], preds)
    print(format_metrics_report(metrics, "tfidf_logreg"))
    plot_confusion_matrix(metrics["confusion_matrix"], "tfidf_logreg")
    save_metrics(metrics, "tfidf_logreg", split="val")
    return metrics


if __name__ == "__main__":
    train, val = load_train_val()
    print(f"train size: {len(train)}, val size: {len(val)}")
    print()

    run_majority_baseline(train, val)
    print()
    run_tfidf_logreg(train, val)
