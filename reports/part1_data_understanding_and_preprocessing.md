# Part 1: Data Understanding and Preprocessing

## Problem Introduction

Gender-based violence is a widespread and under-reported issue, and social media has become a place where survivors and observers describe these experiences in their own words, often outside formal reporting channels. The Gender-Based Violence Tweet Classification Challenge asks for a model that reads a tweet and classifies it into one of five categories: sexual violence, physical violence, emotional violence, economic violence, or harmful traditional practice. Critically, the challenge is framed as classification "without using keywords," meaning a model that succeeds only by matching a handful of trigger words has not actually solved the problem.

The practical relevance of this task is direct. A system that can reliably categorize this kind of content could support content moderation, help route disclosures to the right kind of support service, or assist researchers and NGOs monitoring the prevalence and framing of different forms of violence at scale, something that is not feasible to do manually across the volume of text social media produces. Because the categories carry real consequences for anyone acting on a model's output, understanding exactly where and why a model fails matters as much as its headline score.

This is fundamentally a sequential modelling problem: the meaning of a tweet depends on the order and combination of its words, not just their presence. A bag-of-words view can pick up on surface vocabulary, but distinguishing categories that share vocabulary (for example, violence described in a domestic setting versus a stranger setting) plausibly requires modelling word order and longer-range context, which motivates evaluating models that are explicitly designed to capture sequential dependencies.

## Dataset Overview

The dataset is provided by Zindi and consists of `Train.csv` (39,650 labeled tweets), `Test.csv` (15,581 unlabeled tweets used only for the leaderboard submission), and `SampleSubmission.csv` (the required submission format). The full exploratory analysis behind this section is in `notebooks/01_exploratory_data_analysis.ipynb`, which is annotated so that every result is explained in place; this document summarizes the findings and states their consequences for preprocessing and modelling.

The organizers' own starter notebook (kept for reference in `reference/StarterNotebook.ipynb`) states that the Zindi leaderboard metric is accuracy, and simultaneously demonstrates why that is a weak signal on its own: a simple Naive Bayes baseline reached 88% accuracy while predicting only 3 of the 5 categories, never once predicting the two rarest classes. This is confirmed and quantified in our own analysis below, and it directly shapes the evaluation strategy for the whole project (see the separate Evaluation Metrics section of the final report), where accuracy is reported because it is the actual leaderboard metric, but macro-F1 and per-class recall are used as the primary basis for comparing models.

## Key Findings and Their Implications

### Severe long-tail class imbalance

The five classes are not evenly represented: `sexual_violence` accounts for 82.34% of the data, `physical_violence` for 15.00%, and the remaining three classes together for only 2.66%, with the two smallest, `economic_violence` and `harmful_traditional_practice`, at 217 and 188 examples respectively (see `reports/figures/class_distribution.png`). This is the single most consequential property of the dataset for this project. It means:

Accuracy alone cannot be trusted to reflect model quality, since a model can score well above 80% while ignoring three of the five classes entirely, exactly as the organizers' own baseline demonstrates.

The train/validation/test split cannot be a naive random split. We used a fixed stratified 70/15/15 split (built in `src/data_prep.py` and shared as `splits/train_val_test_split.csv`), which keeps class proportions consistent across all three subsets down to the smallest class, verified directly rather than assumed. Without stratification, a random split could plausibly leave the validation or test set with only a handful of `harmful_traditional_practice` examples, making per-class metrics for that class meaningless by chance alone.

Any imbalance-handling strategy (class weighting, oversampling, or similar) applied during model training is a first-class design decision in this project, not an afterthought, and needs to be documented and compared across the five approaches rather than picked once and left unexamined.

### Long-form text, not short tweets

Median length is 43 words (224 characters), with a minimum of 2 words and a maximum of 68 words (306 characters); even the 25th percentile is 26 words (see `reports/figures/length_histogram.png`). The length distribution is not smooth: it clusters heavily between 45 and 58 words before a hard ceiling at 68, suggesting the organizers likely truncated the text to a fixed length before release. This has two direct consequences: first, this dataset is better understood as short-document classification than as short-text or tweet-style classification, which changes which prior literature is actually relevant to cite in the related-work review; second, a maximum sequence length around 60 to 70 tokens for the neural models should retain nearly all content in nearly every example, so there is little benefit to a longer setting and little content lost at that cutoff.

### Length itself varies by class

Breaking length down by class shows that `physical_violence` is systematically much shorter (mean 23.30 words, median 19.0) than every other class, which cluster between roughly 34 and 42 words on average (see `reports/figures/length_by_class_boxplot.png`). This is a genuine finding from the data, not an assumption carried in from elsewhere, and it means length itself carries some class signal. This is flagged here so it can be checked explicitly during error analysis: a model could be partly responding to how long an example is rather than to its content, particularly for `physical_violence`.

### Large, sparse vocabulary with a long rare-word tail

The cleaned corpus contains 37,767 unique tokens, and 50.7% of them appear exactly once. The most frequent words are almost entirely common function words, with one notable exception: `raped` is the 7th most frequent word overall, consistent with `sexual_violence` dominating the dataset. This shapes two decisions: it argues for subword tokenization in the neural models rather than pure whole-word tokenization, since the rarest classes have very few labeled examples from which to learn whole-word representations, and it is a direct reminder of the risk the challenge brief itself names, that a model could look artificially strong on the majority class by keying on a narrow set of high-frequency words rather than genuinely learning to distinguish categories.

### Minimal raw-Twitter noise, with a small amount of genuine noise

Unlike much social-media text, this dataset contains essentially no hashtags, mentions, or URLs, indicating it was already cleaned by the organizers before release. The noise that does exist is different in kind: 3 rows contain a garbled encoding character, about 10.7% of rows contain emoji, and 345 groups of tweets are identical once case and whitespace differences are normalized away, of which only 2 groups carry conflicting labels across their duplicates. These findings directly shaped the cleaning function in `src/data_prep.py`: effort went into whitespace and smart-quote normalization and lowercasing rather than into stripping social-media artifacts that are mostly not present, and emoji were deliberately retained rather than stripped, since they may carry emotional signal relevant to classification.

## Preprocessing Decisions

The shared preprocessing pipeline in `src/data_prep.py` makes four decisions, each justified directly by the findings above rather than by default convention.

Label normalization comes first: the raw `type` column mixes casing (`sexual_violence`, `Physical_violence`, `Harmful_Traditional_practice`), which was collapsed to a single lowercase canonical form and mapped to a fixed integer encoding, saved as `data/processed/labels.json`. This was verified to cover all five raw label variants with zero unmapped values, so every part of the project encodes labels identically.

Text cleaning is deliberately light: whitespace collapsing, smart-quote normalization, and lowercasing, without hashtag, mention, or URL stripping, since the EDA confirmed those are not meaningfully present, and without emoji removal, since the EDA gave no reason to treat emoji as noise rather than signal.

The train/validation/test split is a fixed, stratified 70/15/15 split, generated once and shared as `splits/train_val_test_split.csv`, so every one of the five modelling approaches trains, tunes, and is finally evaluated on identical data. This was chosen over stratified k-fold cross-validation because a shared, single split is far simpler for four people working across different frameworks (classical scikit-learn, Keras or PyTorch for the recurrent and convolutional models, and Hugging Face for the transformer) to use identically, and because k-fold would multiply the cost of fine-tuning the transformer model by the number of folds, which works against keeping training time reasonable.

Tokenization strategy is left open at this stage, to be chosen per model architecture, but the EDA directly narrows that choice: a maximum sequence length near 68 tokens is sufficient to avoid truncating content, and subword tokenization is preferred over whole-word tokenization for the neural models, given the long rare-word tail and the scarcity of examples for the smallest classes.

## Baseline Results

Two baselines were trained on the shared split to anchor the rest of the comparison: a majority-class classifier and a TF-IDF plus Logistic Regression model with balanced class weights, chosen specifically because an unweighted model would likely default toward the majority class given the imbalance documented above. Both are implemented in `src/baselines.py`, fully reproduced with explanations in `notebooks/02_baseline_models.ipynb`, and scored with the shared metrics code in `src/metrics.py`.

The majority-class baseline reaches 82.34% accuracy and a macro F1 of only 0.1806, since it always predicts `sexual_violence` and scores exactly zero precision, recall, and F1 on every other class. This is the numerical confirmation of the imbalance argument above: an 82% accurate model here has learned nothing at all about four of the five categories.

The TF-IDF and Logistic Regression baseline reaches 99.70% accuracy and a macro F1 of 0.9841, with every class, including the two rarest, scoring above 0.94 on precision, recall, and F1. This result is high enough to demand investigation rather than acceptance at face value. Reading off the highest-weighted features the trained model relies on for each class shows why: the top features are close to literal trigger words, `raped` and `rape` for `sexual_violence`, `fgm` and `child marriage` for `harmful_traditional_practice`, `beats` and `husband` for `physical_violence`, `fired`, `job`, and `boss` for `economic_violence`, and `humiliated`/`insulted` for `emotional_violence`. This model is closer to a keyword lookup table than to genuine language understanding, and it explains the near-perfect score directly: these categories are largely separable by a narrow, learnable vocabulary.

This finding matters beyond Part 1. The challenge brief explicitly frames the task as classification "without using keywords," and this baseline's success is itself evidence of the exact shortcut the brief warns against. It also means this classical baseline is not a weak strawman; it may be difficult for any model to substantially outscore it on accuracy or macro F1 using the same validation set. The genuinely useful comparison for the neural models built in Parts 2 through 4 is therefore not only whether they beat this headline score, but whether they generalize better than this baseline on the harder subset of examples that do not contain an obvious trigger word, which is worth building directly into the shared error analysis in a later stage.

## What This Means Going Into Model Selection

Four properties of this dataset should carry the most weight in choosing and justifying the five modelling approaches. The severity and shape of the class imbalance makes imbalance-handling an explicit, comparable design decision across models rather than an implementation detail. The long-form nature of the text favors architectures and literature suited to longer documents over short-text-specific approaches. The confirmed keyword-driven shortcut, demonstrated directly by the TF-IDF baseline's own feature weights rather than assumed from the challenge brief alone, sets a genuinely strong classical baseline and reframes the comparison: the value of the neural models is not simply outscoring this baseline, but showing whether they generalize beyond the same narrow vocabulary it relies on. Finally, the class-specific difference in tweet length, particularly `physical_violence` being systematically shorter, is a confound to check for explicitly during error analysis, so that apparent model strength or weakness on that class is not mistaken for something it is not.
