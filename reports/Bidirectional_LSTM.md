# Part 2: BiLSTM Model

## Why a BiLSTM

A BiLSTM reads a sentence in both directions, from the start forward and from the end backward, and combines what it learns from both passes. This matters for this dataset specifically because a tweet's meaning can depend on words that come later in the sentence, not just earlier ones, something a model that only reads forward would miss half of.

The case for trying a recurrent model at all comes from the baseline result in Part 1. TF-IDF and Logistic Regression, a model with no sense of word order whatsoever, reached 99.70% accuracy here almost entirely by matching individual trigger words like "raped" or "fgm". That result on its own does not prove a sequence-aware model is unnecessary, it proves the opposite is worth testing directly: does paying attention to word order and sentence structure actually help on this dataset, or is it separable by vocabulary alone regardless of architecture. Zhang, Robinson, and Tepper found that adding a recurrent component that tracks word order improved hate speech detection results over a convolution-only model on most of the datasets they tested [1], which is direct support for including a recurrent model in this comparison rather than assuming the classical baseline's near-perfect score already settles the question.

## Preparing the data

Every tweet was turned into a fixed sequence of 70 word-index numbers. The 70-word cutoff comes from the length analysis in Part 1, which found tweets rarely exceed 68 words, so 70 keeps almost every tweet whole without wasting space on padding that goes much further than needed.

The vocabulary was built only from the training split, never from validation or test, so the model never gets an unfair hint about words it should not have seen yet. This produced a vocabulary of 34,625 words, smaller than Part 1's 37,767, which makes sense since the training split is about 70% of the full dataset, fewer tweets naturally means fewer unique words.

Class weights were calculated and passed into training to address the severe imbalance already documented in Part 1. The two rarest classes received weights around 36 and 42, meaning a mistake on either counts roughly 36 to 42 times more than a mistake on the majority class during training, which gets a weight under 0.25. Without this, the model has little reason to learn the rare classes at all, the same failure the majority-class baseline already demonstrated.

## A duplicate leakage problem in the shared split

Before training, I checked whether any tweets were duplicated across the train, validation, and test sets, since the TF-IDF baseline's unusually high score made this worth verifying rather than assuming. The check found real leakage: 113 duplicate tweets shared between train and validation, 104 between train and test, and 29 between validation and test.

I could not fix the shared split without affecting every other model built on it, so I built an additional check instead: alongside my normal validation and test scores, I also computed scores using only the rows that do not have a duplicate sitting in the training set. For my BiLSTM, this leak-free score (macro F1 0.9747) came out very close to the normal score (macro F1 0.9780), so the leakage does not appear to meaningfully inflate this particular model's results. Clean ID lists for both validation and test have been shared with the rest of the group so the same check can be run consistently across all four models.

## Architecture

The final model is an embedding layer (128 dimensions), followed by a bidirectional LSTM, a dropout layer, and a dense output layer with five outputs, one per class. Kim's work on convolutional sentence classification and the broader literature on recurrent models for short-text classification both informed the overall design choice of a single, moderately sized recurrent layer rather than a deeper stack, given the dataset's size [2].

## Hyperparameter comparison

Two configurations were trained and compared directly on validation data rather than accepting a single run as final:

| Configuration | Validation accuracy | Validation macro F1 |
|---|---|---|
| 64-unit BiLSTM | 0.9971 | 0.9746 |
| 32-unit BiLSTM | 0.9975 | 0.9780 |

The 64-unit version reached near-perfect training accuracy (0.9998) while validation stayed lower, a larger gap than the 32-unit version showed (0.9991 training versus 0.9975 validation), pointing to more overfitting in the larger model. The smaller, 32-unit configuration won on both measures and was selected as the final model.

## Results

Final test set performance, evaluated once with the selected 32-unit configuration:

| Metric | Score |
|---|---|
| Accuracy | 0.9968 |
| Macro F1 | 0.9723 |

Per-class F1 ranged from 0.93 (`emotional_violence`, the lowest) to 1.00 (`physical_violence` and `sexual_violence`, the two largest classes). The main source of error was 11 `sexual_violence` tweets misclassified as `emotional_violence`, the single largest confusion in the test set's confusion matrix. `harmful_traditional_practice` reached perfect recall (1.00) but precision of only 0.90, from a small number of other classes being misclassified into it.

Accuracy alone is reported because it is the leaderboard metric for this challenge, but macro F1 is the metric this model's quality is actually judged on, following the same reasoning laid out in Part 1: a model can score high on accuracy while ignoring minority classes entirely, and macro F1 does not allow that to hide [3].

## What this means going into the final comparison

The BiLSTM's test macro F1 (0.9723) sits close to but slightly below the TF-IDF baseline's validation macro F1 (0.9841), though these two numbers are not from the same evaluation set and are not directly comparable without the baseline also being checked on held-out test data. What is clear is that a model aware of word order does not obviously outperform a model that only matches vocabulary on this particular dataset, which fits with Part 1's finding that the five categories are largely separable by a narrow set of trigger words. Whether the remaining two models in this study (TextCNN and the transformer) close that gap or confirm it is the open question the final comparison needs to answer.