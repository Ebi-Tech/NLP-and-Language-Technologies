# Methodology input: TF-IDF + Logistic Regression (Part 1)

Filled in using the team's methodology input template. Every number comes from the saved files in `reports/results/` (`tfidf_logreg_val.json`, `tfidf_logreg_unweighted_val.json`, `tfidf_logreg_test.json`, `tfidf_logreg_run_info.json`, `preds_tfidf_logreg_test.csv`) and from `notebooks/02_baseline_models.ipynb`.

**Your name and model:** Jean, TF-IDF + Logistic Regression (the Part 1 baseline). The majority-class classifier is the floor and needs no training.

**1. Input representation**
- scikit-learn `TfidfVectorizer` on `tweet_clean`, unigrams and bigrams, vocabulary capped at the 20,000 most frequent n-grams, fitted on the train split only. Unknown words are ignored (there is no OOV token). The default token pattern keeps words of 2 or more characters, so single-character words, punctuation and emoji are not features.
- No sequence length or padding: each tweet is one sparse L2-normalised vector of length 20,000. No embeddings.
- No cleaning beyond the shared `clean_text` function.

**2. Architecture and why it differs from the other models**
- TF-IDF (20,000 features) feeding a multinomial logistic regression, with 100,005 trainable parameters (5 x 20,000 weights plus 5 intercepts).
- It captures the presence and weight of single words and two-word phrases anywhere in the tweet, which is the keyword evidence found in Part 1. It cannot capture word order beyond two-word windows, longer context, similarity between words, or interactions between features.

**3. Training setup**
- L-BFGS solver, L2 penalty with C = 1.0, tolerance 1e-4, maximum 1,000 iterations; it converged after 39. There is no learning rate, batch size or epochs.
- Loss is multinomial cross-entropy with `class_weight="balanced"`. From the train counts (152, 456, 132, 4,162 and 22,853) the weights are 36.52, 12.17, 42.05, 1.33 and 0.24 for economic, emotional, harmful traditional practice, physical and sexual violence.
- Fully deterministic: no random seed is involved, and rerunning reproduces identical files.

**4. What you tuned** (validation only, 1 run each; seeds do not apply because training is deterministic)

| ID | Change | Val macro F1 | Val accuracy | Val errors | Decision |
|---|---|---|---|---|---|
| TF1 | Balanced class weights | 0.9841 | 0.9970 | 18 | selected |
| TF2 | No class weights | 0.9000 | 0.9906 | 56 | not selected |

- The expectation that weights would raise macro F1 was stated in the Part 1 report before the comparison was run. Without weights, rare-class recall falls to 0.607 (`harmful_traditional_practice`), 0.753 (`emotional_violence`) and 0.818 (`economic_violence`). The gap of 0.084 is far above a 0.005 tie tolerance.
- Not done: C, the n-gram range, the vocabulary size and the minimum document frequency were not tuned.

**5. Cost and environment**
- About 11 seconds per fit (19 seconds unweighted) on a Windows laptop CPU, no GPU. Python 3.13.3, scikit-learn 1.8.0, pandas 3.0.0, numpy 2.2.6.
- Nothing was dropped. A refit of the same settings on Colab (scikit-learn 1.6.1) gave test macro F1 0.9815 versus 0.9813 locally, probably because of library versions (not tested).

**6. Test results** (scored once, after the class-weight decision)
- Clean-test macro F1 0.9808 (5,831 tweets with no copy in train), full-test macro F1 0.9813, accuracy 0.9970 (18 errors in 5,948 tweets).
- Weakest class: `emotional_violence`, F1 0.961 (precision 0.925, recall 1.000, 98 test tweets; 8 tweets of other classes were wrongly assigned to it). All 18 errors are tweets of the two largest classes predicted as something else: 14 from `sexual_violence` and 4 from `physical_violence`.
