# Part 3: Methodology (report section draft)

> **Status: draft for Ajak to edit.** Sentences marked **[CONFIRM]** describe another person's work and must be checked with that person before submission. Numbers in `{braces}` are filled from `reports/results/experiments_textcnn.csv` and `reports/results/model_comparison.csv` once the final runs are done. Write this in your own words before submitting: you may be asked to defend every line of it.

## 4. Methodology

### 4.1 Overview

All five approaches are trained and scored on the same fixed split and compared with the same metric code, so that differences in results can be attributed to the model rather than to the data handling. The study compares one classical baseline that ignores word order (TF-IDF with logistic regression) against three neural architectures that model word order in different ways: a bidirectional LSTM (whole-sequence recurrence), a TextCNN (local windows of words), and a fine-tuned DistilBERT (self-attention over the whole sequence), with a majority-class classifier as a floor.

### 4.2 Data preparation shared by every model

**Split.** The 39,650 labelled tweets were divided once into stratified train, validation and test sets (70/15/15, seed 42; 27,755 / 5,947 / 5,948 tweets). Stratification keeps all five classes present in every subset even for the smallest class (188 tweets in total). A single split was used instead of k-fold cross-validation because four people work in different frameworks and because fine-tuning a transformer k times was too costly.

**Cleaning.** Text is lowercased, smart quotes are normalised and whitespace collapsed. Emoji and punctuation are kept. The exploratory analysis found almost no hashtags, mentions or URLs, so nothing was stripped for those.

**Duplicates and the clean test subset.** Exact-text duplicates cross the split: 113 distinct texts appear in both train and validation, 104 in both train and test, and 29 in both validation and test. Regenerating the split would have invalidated results already produced, so the split was kept. Instead every model is scored twice on the test set: on all 5,948 tweets, and on the 5,831 tweets whose exact text does not appear in the training set (the *clean* subset). The clean subset is the headline result.

**Class imbalance.** The classes are severely imbalanced (`sexual_violence` 82.3%, `physical_violence` 15.0%, the other three together 2.7%). The neural models use scikit-learn's `balanced` class weights (for example 42.1 for `harmful_traditional_practice` and 0.24 for `sexual_violence`), passed to the loss during training. The effect of removing the weights is tested as an experiment (Stage C, below). {DistilBERT: **[CONFIRM]** how Kevin handled imbalance.}

### 4.3 Sequence representation

**Neural models (BiLSTM, TextCNN).** Words are mapped to integers with the Keras `Tokenizer` (vocabulary capped at 20,000 words, out-of-vocabulary words mapped to `<OOV>`), fitted on the training split only. Sequences are padded or truncated at the end to 70 tokens, which covers every tweet in the dataset (maximum length 68 words). Each token is embedded by a 128-dimensional embedding layer learned from scratch; no pretrained word vectors were used, so both models start from the same representation. The pipeline is implemented once (`src/neural_data.py`) and shared by both models.

**TF-IDF model.** Unigram and bigram TF-IDF features (20,000 features) feed a class-balanced logistic regression. It has no access to the order of words beyond two-word windows.

**DistilBERT.** **[CONFIRM with Kevin]** The Hugging Face tokenizer for {checkpoint name} with {max length} subword tokens; only light cleaning.

### 4.4 The three neural approaches and why they differ

**BiLSTM.** **[CONFIRM with Divine; matches notebooks/03 at the time of writing]** Embedding (128) → bidirectional LSTM ({32} units per direction, selected over 64) → dropout (0.3) → softmax. It reads the tweet left-to-right and right-to-left and compresses it into one vector, so it can in principle use word order across the whole tweet. Trained with Adam (default learning rate), batch size 64, up to 15 epochs, early stopping on validation loss (patience 3). The notebook does not set random seeds and reports one run per configuration, so BiLSTM numbers carry no seed-to-seed spread. **[CONFIRM]**

**TextCNN.** Embedding (128) → several parallel 1-D convolutions of different widths, each followed by ReLU and global max-pooling → concatenation → dropout (0.5) → softmax [2]. Each filter is a detector for a pattern of *k* consecutive words, and max-pooling keeps only the strongest match anywhere in the tweet. The model therefore sees local word patterns but has no memory of the rest of the sequence, which makes it a deliberate contrast with the BiLSTM. Training matches the BiLSTM (Adam 1e-3, batch 64, up to 15 epochs, early stopping on validation loss with patience 3, best weights restored) so that the architecture is the main difference. Dropout follows Kim (2014).

**DistilBERT.** **[CONFIRM with Kevin]** A pretrained transformer fine-tuned for the five classes. Self-attention lets every token attend to every other token, and pretraining supplies language knowledge the other two models must learn from 27,755 tweets.

### 4.5 Experimental design

**Protocol.** Every model is tuned on the validation set only. The test set is used once, after all decisions are fixed.

**Seeds and selection rule.** Each configuration is trained with {2 or 3} random seeds and compared by mean validation macro-F1. Because the three rare classes have only 28 to 33 validation tweets each, a single misclassified tweet moves a rare-class F1 by about 0.03, so configurations within 0.005 macro-F1 of the best are treated as tied and the one with fewer parameters is chosen. This rule was fixed before any training.

**TextCNN tuning, in three linked stages** (every run is logged in `reports/results/experiments_textcnn.csv` with its hypothesis and decision):

| Stage | Question | Configurations | Hypothesis written in advance |
|---|---|---|---|
| A | How much word order does the classifier need? | Kernel widths `{1}`, `{2}`, `{3,4,5}`, `{3,4,5,6,7}`; 100 filters each | Width-1 kernels (single-word detectors) match wider kernels, because a bag-of-words model already reaches about 99.7% accuracy |
| B | Does capacity matter? | 50, 100, 200 filters, using Stage A's winning kernels | Little, above a small number |
| C | Do class weights matter? | Stage B's winner with and without `balanced` weights | Weights raise macro-F1 through rare-class recall |

Results and the reasoning that connected the stages are in Section 5. {Fill in: what Stage A found, and how it shaped the choice of kernels in Stage B.}

**Stress tests.** Two tests probe whether a model does more than keyword lookup. *Keyword masking* replaces about 50 content words (the ten strongest class-indicative words per class according to a TF-IDF + logistic regression model fitted on the training split, excluding English stop words) by a placeholder in the test tweets. *Word shuffling* randomly permutes the word order inside each test tweet. Both are applied to the clean test subset and to every model; the TF-IDF model is the reference. {Fill in after all models are finished: Divine and Kevin run `src/robustness.py` on their models.}

### 4.6 Evaluation

Macro-averaged F1 is the primary metric, because it gives each class equal weight and so cannot hide a failure on a rare class behind the 82% majority. Accuracy is reported because it is the competition's leaderboard metric. Per-class precision, recall and F1, the confusion matrix and one-vs-rest ROC curves are reported alongside. Learning curves (train and validation loss per epoch) are shown for each neural model. Metrics are computed by one shared function (`src/metrics.py`) so no model is scored differently. **Limitation to state in the report:** with 28 to 32 test tweets in each of the smallest classes, differences of less than about 0.01 macro-F1 between models are within what one or two tweets can change, so rankings are supported by the seed means and standard deviations, not by single runs.

### 4.7 Implementation and reproducibility

Python with TensorFlow/Keras (neural models), scikit-learn (TF-IDF, metrics, class weights), and Hugging Face Transformers (DistilBERT). Library versions are printed at the top of each notebook: TensorFlow {version}, scikit-learn {version}, pandas {version}, NumPy {version}. Hardware: {GPU name on Colab}. The code is at {GitHub link}; every notebook starts from the same setup cell and runs on Google Colab. Seeds are fixed, but GPU operations can be non-deterministic, so results are reported as mean and standard deviation across seeds.

### 4.8 AI-use disclosure (draft: edit so that it is true)

AI assistance (Claude, Anthropic) was used for: {list exactly what you used it for, for example: planning the repository structure, drafting the shared helper modules in `src/`, writing the first version of the TextCNN notebook, drafting this section}. The group reviewed {how}, re-ran all training on Google Colab, and is responsible for the work. {Each member: state what you wrote and ran yourself.}

## References for this section (IEEE)

The consolidated list is Divine's. Numbers 1 to 5 below are the ones already used in `notebooks/03_bidirectional-lstm.ipynb`; the entries from 6 onward are new and should be appended to the consolidated list in this order. Verify each entry against the source before submitting.

[2] Y. Kim, "Convolutional neural networks for sentence classification," in *Proc. 2014 Conf. Empirical Methods in Natural Language Processing (EMNLP)*, Doha, Qatar, 2014, pp. 1746-1751. (already in the list)

[6] Y. Zhang and B. Wallace, "A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification," in *Proc. 8th Int. Joint Conf. Natural Language Processing (IJCNLP)*, 2017, pp. 253-263. (arXiv:1510.03820)

[7] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf. Learning Representations (ICLR)*, 2015.

[8] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, no. 56, pp. 1929-1958, 2014.

[9] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825-2830, 2011.

[10] M. Abadi *et al.*, "TensorFlow: A system for large-scale machine learning," in *Proc. 12th USENIX Symp. Operating Systems Design and Implementation (OSDI)*, 2016, pp. 265-283.

[11] Zindi, "Gender-Based Violence Tweet Classification Challenge," 2021. [Online]. Available: https://zindi.africa/competitions/gender-based-violence-tweet-classification-challenge
