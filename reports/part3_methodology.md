# Part 3: Methodology

## 4. Methodology

### 4.1 Overview

All five approaches are trained and scored on the same fixed split and compared with the same metric code, so that differences in results can be attributed to the model rather than to the data handling. The study compares one classical baseline that ignores word order (TF-IDF with logistic regression) against three neural architectures that model word order in different ways: a bidirectional LSTM (whole-sequence recurrence), a TextCNN (local windows of words), and a fine-tuned DistilBERT (self-attention over the whole sequence), with a majority-class classifier as a floor. The data come from the Zindi Gender-Based Violence Tweet Classification Challenge, released under a CC-BY 1.0 license [11].

### 4.2 Data preparation shared by every model

**Split.** The 39,650 labelled tweets were divided once into stratified train, validation and test sets (70/15/15, seed 42; 27,755 / 5,947 / 5,948 tweets). Stratification keeps all five classes present in every subset even for the smallest class (188 tweets in total). A single split was used instead of k-fold cross-validation because four people work in different frameworks and because fine-tuning a transformer k times was too costly.

**Cleaning.** Text is lowercased, smart quotes are normalised and whitespace collapsed. Emoji and punctuation are kept. The exploratory analysis found almost no hashtags, mentions or URLs, so nothing was stripped for those.

**Duplicates and the clean test subset.** Exact-text duplicates cross the split: 113 distinct texts appear in both train and validation, 104 in both train and test, and 29 in both validation and test. Regenerating the split would have invalidated results already produced, so the split was kept. Instead every model is scored twice on the test set: on all 5,948 tweets, and on the 5,831 tweets whose exact text does not appear in the training set (the *clean* subset). The clean subset is the headline result.

**Class imbalance.** The classes are severely imbalanced (`sexual_violence` 82.3%, `physical_violence` 15.0%, the other three together 2.7%). The BiLSTM is trained with scikit-learn's `balanced` class weights [9] (for example 42.1 for `harmful_traditional_practice` and 0.24 for `sexual_violence`), passed to the loss during training. The TextCNN and DistilBERT were each trained with and without these weights, and in both cases the unweighted model was kept (for the TextCNN this is Stage C below). The BiLSTM and the other two neural models therefore differ in loss weighting as well as in architecture, and the Results section states this when comparing them. **[CONFIRM with Kevin: the DistilBERT choice was made on validation data, not test data.]**

### 4.3 Sequence representation

**Neural models (BiLSTM, TextCNN).** Words are mapped to integers with the Keras `Tokenizer` (vocabulary capped at 20,000 words, out-of-vocabulary words mapped to `<OOV>`), fitted on the training split only. Sequences are padded or truncated at the end to 70 tokens, which covers every tweet in the dataset (maximum length 68 words). Each token is embedded by a 128-dimensional embedding layer learned from scratch; no pretrained word vectors were used, so both models start from the same representation. The pipeline is implemented once (`src/neural_data.py`) and shared by both models.

**TF-IDF model.** Unigram and bigram TF-IDF features (20,000 features) feed a class-balanced logistic regression. It has no access to the order of words beyond two-word windows.

**DistilBERT.** The Hugging Face tokenizer for `distilbert-base-uncased`, with sequences padded or truncated to 128 subword tokens. **[CONFIRM with Kevin: whether any cleaning beyond the shared `clean_text` function was applied.]**

### 4.4 The three neural approaches and why they differ

**BiLSTM.** Embedding (128) → bidirectional LSTM (32 units per direction, selected over 64) → dropout (0.3) → softmax. It reads the tweet left-to-right and right-to-left and compresses it into one vector, so it can in principle use word order across the whole tweet. Trained with Adam (default learning rate), batch size 64, up to 15 epochs, early stopping on validation loss (patience 3). The notebook sets no random seed and reports one run per configuration, so BiLSTM numbers carry no seed-to-seed spread.

**TextCNN.** The architecture follows Kim [2]: embedding (128) → parallel 1-D convolutions, each followed by ReLU and global max-pooling → concatenation → dropout (0.5) [8] → softmax. Each filter is a detector for a pattern of *k* consecutive words, and max-pooling keeps only the strongest match anywhere in the tweet. The model therefore sees local word patterns but has no memory of the rest of the sequence, which makes it a deliberate contrast with the BiLSTM. Kernel widths and filter counts were tuned (Section 4.5) following the practitioner guidance in [6]. The configuration selected by the pre-set rule is a single convolution of width 1 with 50 filters, trained without class weights. A width-1 convolution with global max-pooling looks at one word at a time and discards position, so the selected model is in effect a learned keyword detector. Training matches the BiLSTM (Adam [7] with learning rate 1e-3, batch 64, up to 15 epochs, early stopping on validation loss with patience 3, best weights restored) so that the architecture is the main difference.

**DistilBERT.** A pretrained transformer (`distilbert-base-uncased`) fine-tuned for the five classes: 3 epochs, AdamW with learning rate 2e-5, batch size 32, one run with PyTorch seed 42, 16.7 minutes on a Tesla T4. Self-attention lets every token attend to every other token, and pretraining supplies language knowledge the other two models must learn from 27,755 tweets.

### 4.5 Experimental design

**Protocol.** Every model is tuned on the validation set only. The test set is used once, after all decisions are fixed.

**Seeds and selection rule.** The TextCNN configurations are each trained with 3 random seeds (0, 1, 2) and compared by mean validation macro-F1; standard deviations are sample standard deviations over the three seeds. Because the three rare classes have only 28 to 33 validation tweets each, a single misclassified tweet moves a rare-class F1 by about 0.03, so configurations within 0.005 macro-F1 of the best are treated as tied and the one with fewer parameters is chosen. This rule was fixed before any training. The reported TextCNN is the seed with the best validation macro-F1 (seed 0), chosen without looking at the test set.

**TextCNN tuning, in three linked stages** (every run is logged in `reports/results/experiments_textcnn.csv` with its hypothesis and decision):

| Stage | Question | Configurations | Hypothesis written in advance | Outcome (mean validation macro-F1 ± std) |
|---|---|---|---|---|
| A | How much word order does the classifier need? | Kernel widths `{1}`, `{2}`, `{3,4,5}`, `{3,4,5,6,7}`; 100 filters each | Width-1 kernels (single-word detectors) match wider kernels, because a bag-of-words model already reaches about 99.7% accuracy | Supported. 0.9821 ± 0.0044, 0.9801 ± 0.0054, 0.9823 ± 0.0046, 0.9849 ± 0.0033: all within 0.005, so the tie rule chose `{1}` |
| B | Does capacity matter? | 50, 100, 200 filters, using Stage A's winning kernels | Little, above a small number | Supported. 0.9831 ± 0.0057, 0.9821 ± 0.0044, 0.9831 ± 0.0102: tied, so 50 filters was chosen |
| C | Do class weights matter? | Stage B's winner with and without `balanced` weights | Weights raise macro-F1 through rare-class recall | Rejected. Unweighted 0.9951 ± 0.0042 against weighted 0.9831 ± 0.0057, a gap of 0.012, above both the tie tolerance and the seed spread |

Stage A shaped Stage B only through the tie rule: the widest kernel set had the highest mean, but its lead over width 1 (0.0028) was smaller than the seed spread, so the simplest set carried forward. Results and the reasoning that connected the stages are in Section 5.

**Stress tests.** Two tests probe whether a model does more than keyword lookup. *Keyword masking* replaces the strongest class-indicative content words (up to ten per class, taken from a TF-IDF + logistic regression model fitted on the training split, excluding English stop words and words seen in fewer than five training tweets) by a placeholder in the test tweets. *Word shuffling* randomly permutes the word order inside each test tweet (seed 42). Both tests are run on the full test set with the same settings as the DistilBERT run, so the numbers are directly comparable, and the clean-subset macro-F1 is reported next to each score. The TF-IDF model is the reference. The TextCNN and DistilBERT runs are complete; the BiLSTM run is pending. Because a width-1 convolution with global max-pooling ignores word order by construction, shuffling cannot change the selected TextCNN's predictions, and the identical scores before and after shuffling are reported but not interpreted as evidence about word order.

### 4.6 Evaluation

Macro-averaged F1 is the primary metric, because it gives each class equal weight and so cannot hide a failure on a rare class behind the 82% majority. Accuracy is reported because it is the competition's leaderboard metric. Per-class precision, recall and F1, the confusion matrix and one-vs-rest ROC curves are reported alongside. Learning curves (train and validation loss per epoch) are shown for each neural model. Metrics are computed by one shared function (`src/metrics.py`) so no model is scored differently. **Limitation to state in the report:** with 28 to 32 test tweets in each of the smallest classes, differences of less than about 0.01 macro-F1 between models are within what one or two tweets can change, so rankings are supported by the seed means and standard deviations, not by single runs. The BiLSTM and DistilBERT have no seed spread, so comparisons with them cannot use one.

### 4.7 Implementation and reproducibility

Python with TensorFlow/Keras [10] (BiLSTM, TextCNN), scikit-learn [9] (TF-IDF, metrics, class weights), and Hugging Face Transformers with PyTorch (DistilBERT). Library versions are printed at the top of each notebook: TensorFlow 2.20.0, scikit-learn 1.6.1, pandas 2.2.3, NumPy 2.1.3. Hardware: Tesla T4 GPU on Google Colab. The code is at https://github.com/Ebi-Tech/NLP-and-Language-Technologies and every notebook starts from the same setup cell. The TextCNN uses fixed seeds (0, 1, 2) and is reported as mean and standard deviation across them, because GPU operations can be non-deterministic. The BiLSTM sets no seed and DistilBERT is a single run (seed 42).

### 4.8 AI-use disclosure (draft: edit so that it is true)

AI assistance (Claude, Anthropic) was used for: {confirm the earlier uses, for example planning the repository structure, drafting the shared helper modules in `src/`, writing the first version of the TextCNN notebook, drafting this section}; and, in the final stage, adding the download and push cells to the TextCNN notebook, checking the result files against the repository, and drafting corrections to `DECISIONS.md`, `README.md` and this section after the run. The group reviewed {how}, re-ran all training on Google Colab, and is responsible for the work. {Each member: state what you wrote and ran yourself.}

## References for this section (IEEE)

The consolidated list is Divine's. Numbers 1 to 5 below are the ones already used in `notebooks/03_bidirectional-lstm.ipynb`; the entries from 6 onward are new and should be appended to the consolidated list in this order. Verify each entry against the source before submitting.

[2] Y. Kim, "Convolutional neural networks for sentence classification," in *Proc. 2014 Conf. Empirical Methods in Natural Language Processing (EMNLP)*, Doha, Qatar, 2014, pp. 1746-1751. (already in the list)

[6] Y. Zhang and B. Wallace, "A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification," in *Proc. 8th Int. Joint Conf. Natural Language Processing (IJCNLP)*, 2017, pp. 253-263. (arXiv:1510.03820)

[7] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf. Learning Representations (ICLR)*, 2015.

[8] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, no. 56, pp. 1929-1958, 2014.

[9] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825-2830, 2011.

[10] M. Abadi *et al.*, "TensorFlow: A system for large-scale machine learning," in *Proc. 12th USENIX Symp. Operating Systems Design and Implementation (OSDI)*, 2016, pp. 265-283.

[11] Zindi, "Gender-Based Violence Tweet Classification Challenge," 2021. [Online]. Available: https://zindi.africa/competitions/gender-based-violence-tweet-classification-challenge