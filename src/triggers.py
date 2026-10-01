"""Finds the words each class can be guessed from, and blanks them out of text.

Part 1 showed the TF-IDF baseline is close to a keyword lookup, with features
like raped, fgm and beats carrying each class. This pulls that word list out of
the same pipeline so it can be used as evidence rather than assumed.

Terms appearing in more than max_df of the training tweets are dropped first.
Without that the top features are words like he and my, which are common
everywhere and say nothing about a class.
"""

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_prep import ID_TO_LABEL

LABEL_ORDER = [ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]


def top_trigger_terms(texts, labels, top=10, max_df=0.10, verbose=True) -> dict:
    """Return {class label: [its top terms]}, using the Part 1 baseline config."""
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(texts, labels)

    X = pipe["tfidf"].transform(texts)
    doc_freq = np.asarray((X > 0).sum(axis=0)).ravel() / X.shape[0]
    too_common = doc_freq > max_df

    vocab = np.array(pipe["tfidf"].get_feature_names_out())
    per_class = {}
    for i, label in enumerate(LABEL_ORDER):
        coef = pipe["clf"].coef_[i].copy()
        coef[too_common] = -np.inf
        per_class[label] = list(vocab[coef.argsort()[::-1][:top]])
        if verbose:
            print(f"{label}: {', '.join(per_class[label])}")

    return per_class


def trigger_pattern(terms) -> re.Pattern:
    """One case-insensitive regex matching any of the terms as whole words."""
    return re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in sorted(terms)) + r")\b",
        re.IGNORECASE,
    )


def mask_terms(text, pattern, token="") -> str:
    """Blank out every trigger term in one piece of text."""
    return re.sub(r"\s+", " ", pattern.sub(token, str(text))).strip()
