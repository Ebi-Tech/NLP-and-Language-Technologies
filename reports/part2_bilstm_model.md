# Part 2: BiLSTM Model

## Why a BiLSTM

A BiLSTM reads a sentence in both directions, from the start forward and from the end backward, and combines what it learns from both passes. This matters for this dataset specifically because a tweet's meaning can depend on words that come later in the sentence, not just earlier ones, something a model that only reads forward would miss half of.

The case for trying a recurrent model at all comes from the baseline result in Part 1. TF-IDF and Logistic Regression, a model with no sense of word order whatsoever, reached 99.70% accuracy here almost entirely by matching individual trigger words like "raped" or "fgm". That result on its own does not prove a sequence-aware model is unnecessary, it proves the opposite is worth testing directly: does paying attention to word order and sentence structure actually help on this dataset, or is it separable by vocabulary alone regardless of architecture. Zhang, Robinson, and Tepper combined convolutional and gated recurrent networks for hate speech detection on Twitter and reported that their method captures both word sequence and order information in short texts and outperforms previously reported results on 6 of 7 datasets by between 1 and 13% in F1 [1], which is consistent with including a recurrent model in this comparison rather than assuming the classical baseline's near-perfect score already settles the question.

## Preparing the data

Every tweet was turned into a fixed sequence of 70 word-index numbers. The 70-word cutoff comes from the length analysis in Part 1, which found tweets rarely exceed 68 words, so 70 keeps almost every tweet whole without wasting space on padding that goes much further than needed.

The vocabulary was built only from the training split, never from validation or test, so the model never gets an unfair hint about words it should not have seen yet. This produced a vocabulary of 34,625 words, smaller than Part 1's 37,767, which makes sense since the training split is about 70% of the full dataset, fewer tweets naturally means fewer unique words.

Class weights were calculated and passed into training to address the severe imbalance already documented in Part 1. The two rarest classes received weights around 36 and 42, meaning a mistake on either counts roughly 36 to 42 times more than a mistake on the majority class during training, which gets a weight under 0.25. Without this, the model has little reason to learn the rare classes at all, the same failure the majority-class baseline already demonstrated.

## A duplicate leakage problem in the shared split

Before training, I checked whether any tweets were duplicated across the train, validation, and test sets, since the TF-IDF baseline's unusually high score made this worth verifying rather than assuming. The check found real leakage: 113 duplicate tweets shared between train and validation, 104 between train and test, and 29 between validation and test.

I could not fix the shared split without affecting every other model built on it, so I built an additional check instead: alongside my normal validation and test scores, I also computed scores using only the rows that do not have a duplicate sitting in the training set. For my final BiLSTM, the leak-free test score (macro F1 0.9685) came out very close to the normal test score (macro F1 0.9693), so the leakage does not appear to meaningfully inflate this particular model's results. Clean ID lists for both validation and test have been shared with the rest of the group so the same check can be run consistently across all four models.

## Architecture

The final model is an embedding layer (128 dimensions), followed by a bidirectional LSTM, a dropout layer, and a dense output layer with five outputs, one per class. Kim's work on convolutional sentence classification and the broader literature on recurrent models for short-text classification both informed the overall design choice of a single, moderately sized recurrent layer rather than a deeper stack, given the dataset's size [2].

## Hyperparameter comparison

Two configurations were compared on validation data, first in a single run each and then across three seeds each, because a single run turned out not to be reliable:

| Configuration | Runs | Validation macro F1 (mean and standard deviation) | Validation accuracy (mean) |
|---|---|---|---|
| 64-unit BiLSTM | 3 seeds | 0.9760, 0.0085 | 0.9973 |
| 32-unit BiLSTM | 3 seeds | 0.9726, 0.0038 | 0.9966 |

These numbers come from `experiments_bilstm.csv`. The single-run comparison made first gave a validation macro F1 of 0.9620 for 64 units and 0.9674 for 32 units, a gap that the seeds show to be within noise: the 64-unit mean is higher by 0.0034, inside the 0.005 tolerance the group uses for ties, and the seed alone moves a configuration's validation macro F1 by about 0.009 (32 units) to 0.021 (64 units). Under that rule a tie goes to the configuration with fewer parameters, so the 32-unit BiLSTM (2,601,541 parameters against 2,659,461) was selected, not because it scored higher.

## Results

Final test set performance of the selected 32-unit configuration, retrained with a fixed seed (seed 42, stopped after 9 epochs with the epoch 6 weights restored), whose saved predictions are in `preds_bilstm_test.csv`:

| Metric | Score |
|---|---|
| Accuracy | 0.9960 |
| Macro F1 | 0.9693 |
| Macro F1 on the leak-free test rows | 0.9685 |

Per-class F1 ranged from 0.928 (`emotional_violence`, the lowest) to 0.998 (`sexual_violence`, the largest class), with 0.941 for `economic_violence`, 0.983 for `harmful_traditional_practice` and 0.997 for `physical_violence`. The main source of error was 14 `sexual_violence` tweets misclassified as `emotional_violence`, the single largest confusion in the test set's confusion matrix. `economic_violence` and `harmful_traditional_practice` reached perfect recall (1.00) but precision of only 0.89 and 0.97, from 4 and 1 tweets of other classes being misclassified into them.

The test set was scored more than once for this model. Two earlier unseeded runs of the same configuration scored macro F1 values of 0.9723 and 0.9657, and the model was then retrained with a fixed seed so that its weights, tokenizer and predictions could be saved and reproduced. The seeded model's score is the one reported, and no configuration was chosen from test scores.

Accuracy alone is reported because it is the leaderboard metric for this challenge, but macro F1 is the metric this model's quality is actually judged on, following the same reasoning laid out in Part 1: a model can score high on accuracy while ignoring minority classes entirely, and macro F1 does not allow that to hide [3].

## What this means going into the final comparison

The BiLSTM's test macro F1 (0.9693) sits below the TF-IDF baseline's test macro F1 (0.9813), measured on the same test set, and below the TextCNN (0.9978) and DistilBERT (0.9954). A model aware of word order therefore does not outperform a model that only matches vocabulary on this particular dataset, which fits with Part 1's finding that the five categories are largely separable by a narrow set of trigger words. The stress tests add two details. When the word order is shuffled, the BiLSTM's macro F1 falls from 0.9693 to 0.9295, more than DistilBERT's fall of 0.0097, so it does use some word order. When the strongest keywords are masked, it falls to 0.1993, so it relies on the same trigger words as the other models. These results are discussed in the Results and Discussion sections.

## References

[1] Z. Zhang, D. Robinson, and J. Tepper, "Detecting hate speech on Twitter using a convolution-GRU based deep neural network," in *Proc. 15th Extended Semantic Web Conf. (ESWC)*, Heraklion, Greece, 2018, pp. 745-760.

[2] Y. Kim, "Convolutional neural networks for sentence classification," in *Proc. 2014 Conf. Empirical Methods in Natural Language Processing (EMNLP)*, Doha, Qatar, 2014, pp. 1746-1751, doi: 10.3115/v1/D14-1181.

[3] M. Sokolova and G. Lapalme, "A systematic analysis of performance measures for classification tasks," *Inf. Process. Manag.*, vol. 45, no. 4, pp. 427-437, 2009, doi: 10.1016/j.ipm.2009.03.002.
