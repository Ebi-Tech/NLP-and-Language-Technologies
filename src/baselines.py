"""Part 1 baselines: majority-class floor and TF-IDF + Logistic Regression.

Both models train on the shared train split. Configuration choices (here,
class weights on or off) are made on the validation split only, and the test
split is scored once at the end, after the choice is fixed. The final step
also saves per-tweet class probabilities in the shared format, so the
cross-model comparison can draw ROC curves and study errors.
"""

import json

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_prep import load_split
from src.metrics import RESULTS_DIR, compute_metrics, format_metrics_report, plot_confusion_matrix, save_metrics
from src.save_preds import save_preds


def load_train_val():
    data = pd.read_csv("data/processed/train_cleaned.csv")
    split_map = load_split()
    merged = data.merge(split_map, on="Tweet_ID")

    train = merged[merged["split"] == "train"]
    val = merged[merged["split"] == "val"]
    return train, val


def load_train_val_test():
    data = pd.read_csv("data/processed/train_cleaned.csv")
    merged = data.merge(load_split(), on="Tweet_ID")
    return (
        merged[merged["split"] == "train"],
        merged[merged["split"] == "val"],
        merged[merged["split"] == "test"],
    )


def fit_tfidf_logreg(train, class_weight="balanced"):
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, class_weight=class_weight)),
        ]
    )
    return pipeline.fit(train["tweet_clean"], train["label_id"])


def run_majority_baseline(train, val):
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(train[["tweet_clean"]], train["label_id"])
    preds = clf.predict(val[["tweet_clean"]])

    metrics = compute_metrics(val["label_id"], preds)
    print(format_metrics_report(metrics, "majority_baseline"))
    plot_confusion_matrix(metrics["confusion_matrix"], "majority_baseline")
    save_metrics(metrics, "majority_baseline", split="val")
    return metrics


def run_tfidf_logreg(train, val, class_weight="balanced", name="tfidf_logreg"):
    pipeline = fit_tfidf_logreg(train, class_weight)
    preds = pipeline.predict(val["tweet_clean"])

    metrics = compute_metrics(val["label_id"], preds)
    print(format_metrics_report(metrics, name))
    plot_confusion_matrix(metrics["confusion_matrix"], name)
    save_metrics(metrics, name, split="val")
    return metrics


def run_tfidf_final(train, val, test):
    """Pick class weights on or off using validation macro F1, then score the
    test split once with the chosen configuration.

    Ties keep the balanced configuration, which is the one reported in the
    Part 1 validation results. Saves the test metrics, the confusion matrix,
    the per-tweet probabilities, and a small run-info file recording the
    choice.
    """
    configs = {"balanced": "balanced", "none": None}
    fitted, val_metrics = {}, {}
    for label, class_weight in configs.items():
        fitted[label] = fit_tfidf_logreg(train, class_weight)
        val_metrics[label] = compute_metrics(val["label_id"], fitted[label].predict(val["tweet_clean"]))

    save_metrics(val_metrics["none"], "tfidf_logreg_unweighted", split="val")

    chosen = max(val_metrics, key=lambda label: val_metrics[label]["macro_f1"])
    pipeline = fitted[chosen]
    probs = pipeline.predict_proba(test["tweet_clean"])
    test_metrics = compute_metrics(test["label_id"].values, probs.argmax(axis=1))

    save_metrics(test_metrics, "tfidf_logreg", split="test")
    plot_confusion_matrix(test_metrics["confusion_matrix"], "tfidf_logreg_test")
    save_preds("tfidf_logreg", test["Tweet_ID"], probs, test["label_id"])

    info = {
        "model": "TF-IDF + Logistic Regression",
        "config": "class weights" if chosen == "balanced" else "no class weights",
        "val_macro_f1": {label: m["macro_f1"] for label, m in val_metrics.items()},
        "chosen_on": "validation macro F1",
    }
    with open(RESULTS_DIR / "tfidf_logreg_run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    return {"val_metrics": val_metrics, "chosen": chosen, "test_metrics": test_metrics, "pipeline": pipeline}


if __name__ == "__main__":
    train, val, test = load_train_val_test()
    print(f"train size: {len(train)}, val size: {len(val)}, test size: {len(test)}")
    print()

    run_majority_baseline(train, val)
    print()
    run_tfidf_logreg(train, val)
    print()
    result = run_tfidf_final(train, val, test)
    print(f"chosen on validation: {result['chosen']}")
    print(format_metrics_report(result["test_metrics"], "tfidf_logreg (test)"))
