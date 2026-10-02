"""Stress tests for the "classification without keywords" question.

Both tests change the cleaned tweet text, so they work for any model: tokenise
the changed text with that model's own pipeline, predict, and compare the
macro-F1 with the unchanged text.

1. mask_keywords : replace the strongest class-indicating words with a neutral
   placeholder. A model that mostly looks up keywords should drop sharply.
2. shuffle_words : randomly reorder the words inside each tweet. A model that
   really uses word order should drop; one that does not, will not.
"""

import random
import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data_prep import ID_TO_LABEL

# A made-up word that no tweet contains, so every tokenizer treats it as unknown.
PLACEHOLDER = "xxmask"
_WORD = re.compile(r"[a-z0-9']+")


def top_keywords(train_texts, train_labels, n_per_class: int = 10, min_df: int = 5) -> dict:
    """Return the words that most strongly indicate each class.

    A class-balanced TF-IDF + logistic regression model is fitted on the TRAIN
    split only (so nothing from validation or test leaks in), and the words
    with the largest coefficient for each class are returned. English stop
    words ("he", "my", "was", ...) and words seen in fewer than `min_df`
    training tweets are skipped, so masking removes content words, not grammar.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), token_pattern=r"[a-z0-9']+", min_df=min_df)
    X = vectorizer.fit_transform(train_texts)
    model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, train_labels)

    vocab = np.array(vectorizer.get_feature_names_out())
    # "he's" is a stop word in disguise, so strip a trailing 's before checking.
    is_content_word = np.array(
        [(w[:-2] if w.endswith("'s") else w) not in ENGLISH_STOP_WORDS for w in vocab])

    keywords = {}
    for row, class_id in enumerate(model.classes_):
        best_first = np.argsort(model.coef_[row])[::-1]   # largest coefficient first
        picked = [j for j in best_first if is_content_word[j]][:n_per_class]
        keywords[ID_TO_LABEL[int(class_id)]] = vocab[picked].tolist()
    return keywords


def mask_keywords(texts, keywords) -> list:
    """Replace every word in `keywords` with the placeholder."""
    wanted = set(keywords)
    return [_WORD.sub(lambda m: PLACEHOLDER if m.group(0) in wanted else m.group(0), t)
            for t in texts]


def shuffle_words(texts, seed: int = 0) -> list:
    """Randomly reorder the words of each tweet (the same seed gives the same shuffle)."""
    rng = random.Random(seed)
    shuffled = []
    for text in texts:
        words = text.split()
        rng.shuffle(words)
        shuffled.append(" ".join(words))
    return shuffled


def count_masked(texts, keywords) -> list:
    """How many words keyword masking replaces in each text."""
    wanted = set(keywords)
    return [sum(1 for w in _WORD.findall(t) if w in wanted) for t in texts]


def mask_random(texts, counts, seed: int = 0) -> list:
    """Mask the same number of randomly chosen content words per text.

    This is the control for mask_keywords. Masking 50 words might break a model
    simply because the text is damaged, not because those particular words
    carried the class. Masking the same count of random content words separates
    the two: if the random version hurts just as much, the keyword result says
    nothing about keywords.
    """
    rng = random.Random(seed)
    out = []
    for text, k in zip(texts, counts):
        words = _WORD.findall(text)
        content = [i for i, w in enumerate(words) if w not in ENGLISH_STOP_WORDS]
        chosen = set(rng.sample(content, min(k, len(content))))
        seen = [-1]

        def replace(match):
            seen[0] += 1
            return PLACEHOLDER if seen[0] in chosen else match.group(0)

        out.append(_WORD.sub(replace, text))
    return out
