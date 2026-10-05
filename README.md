# NLP-and-Language-Technologies

## Formative Assignment 2: GBV Tweet Classification (Sequential Modelling)

This is a group project comparing five sequential and classical modelling approaches on the Gender-Based Violence Tweet Classification Challenge dataset from Zindi.

**Content warning.** The dataset contains real, first-person descriptions of gender-based violence, including graphic descriptions of sexual abuse. Read with that in mind before opening the raw data files.

**Data attribution.** The dataset is provided by Zindi under a CC-BY 1.0 license, which permits this kind of research and educational use and redistribution. Source: [Gender-Based Violence Tweet Classification Challenge, Zindi](https://zindi.world/competitions/gender-based-violence-tweet-classification-challenge).

## Repo Structure

```text
DECISIONS.md                    running decision log, one row per decision that affects more than one person
requirements.txt                Python packages for the notebooks (see Setup)
data/
  raw/                          original Train.csv / Test.csv (do not edit)
  processed/                    cleaned data produced by src/data_prep.py
splits/
  train_val_test_split.csv      shared fixed split, use this exact file
src/
  data_prep.py                  cleaning + label encoding (shared, import this)
  metrics.py                    shared evaluation functions (shared, import this)
  baselines.py                  majority-class + TF-IDF/LogReg baselines
  leakage.py                    cross-split duplicate check and leak-free subsets (shared)
  save_preds.py                 saves per-tweet class probabilities in the shared format
  neural_data.py                shared tokenizer/padding/class-weight pipeline for the neural models
  textcnn.py                    TextCNN model and training function
  experiment_log.py             writes the per-model tuning log
  results_table.py              builds one model's row for the comparison table
  robustness.py                 keyword-masking and word-shuffling stress tests
notebooks/
  01_exploratory_data_analysis.ipynb   exploratory data analysis, fully annotated
  02_baseline_models.ipynb             baseline model training and evaluation
  03_bidirectional-lstm.ipynb          Part 2: BiLSTM
  04_textcnn.ipynb                     Part 3: TextCNN
  05_distilbert.ipynb                  Part 4: DistilBERT
  06_comparison.ipynb                  Part 4: cross-model comparison, reads reports/results/preds_*_test.csv
reports/
  part1_data_understanding_and_preprocessing.md   written report section for Part 1
  part2_bilstm_model.md          written report section for Part 2 (BiLSTM)
  part3_methodology.md           written Methodology section (Part 3)
  part4_results_and_discussion.md   written Results, Discussion and Error Analysis sections (Part 4)
  figures/                       all chart and confusion matrix images
  results/                       every result file, one set per model (see Conventions below)
reference/
  StarterNotebook.ipynb          organizer-provided starter notebook (background only)
```

The `reference/` folder holds background material from the competition organizers, like the original starter notebook. It sits apart from the dataset because it is not part of the pipeline.

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` covers every notebook, including `torch` and `transformers` for `05_distilbert.ipynb` on a local machine. On Colab, PyTorch and TensorFlow are already installed, and `05_distilbert.ipynb` installs `transformers` in its own first cell.

### Google Colab

Open a notebook from `notebooks/` in Colab. For the neural models, choose Runtime, then Change runtime type, then a GPU. The first cell clones this repository, moves into it and adds it to the import path, so the notebook runs from a fresh runtime. TensorFlow is already installed on Colab and is not reinstalled. Run notebooks from the repository root, because the imports from `src/` use paths relative to it.

## Reproducing Part 3 (TextCNN)

Open `notebooks/04_textcnn.ipynb` in Colab and switch the runtime to a GPU. The saved run used a Tesla T4. Run the cells in order. Training covers 7 configurations with 3 seeds each, at 20 to 39 seconds on average per configuration on a T4, followed by one test evaluation and the stress tests.

Every output file starts with `textcnn`, or is `experiments_textcnn.csv` or `confusion_matrix_textcnn.png`, and is written to `reports/results/` or `reports/figures/`. Step 11 of the notebook zips those files for download. Step 12 pushes them to the branch, using a GitHub token stored in Colab Secrets as `GITHUB_TOKEN`. Never paste the token into a cell.

## Reproducing Parts 2 and 4

`notebooks/03_bidirectional-lstm.ipynb` (BiLSTM) compares 32 and 64 units over 3 seeds each, then retrains the selected 32-unit model with seed 42 and saves it. It was run on a Kaggle GPU whose model was not recorded. `notebooks/05_distilbert.ipynb` fine-tunes DistilBERT on a Colab Tesla T4 with and without class weights, and the selected run took 15.4 minutes. `notebooks/06_comparison.ipynb` reads the saved prediction files and rebuilds the comparison tables and figures without retraining anything.

## Results at a Glance

Macro F1 on the 5,831 test tweets with no copy in the training set, the headline metric:

| Model | Clean macro F1 | Errors in 5,948 test tweets |
|---|---|---|
| TextCNN | 0.9977 | 3 |
| DistilBERT | 0.9952 | 4 |
| TF-IDF + logistic regression | 0.9808 | 18 |
| BiLSTM | 0.9685 | 24 |
| Majority class | 0.1822 | 1,050 |

Masking the fifty strongest class keywords drops every trained model to a macro F1 of about 0.20 to 0.40, so the scores depend heavily on a small set of trigger words. The full tables, tuning experiments and error analysis are in the four files under `reports/`.

## Shared Artifacts

Several files in this repo are shared infrastructure, so do not regenerate them on your own branch.

`splits/train_val_test_split.csv` holds the fixed stratified train, validation and test split that all five models train and evaluate against, which keeps results comparable across parts. `src/data_prep.py` handles text cleaning and label encoding the same way for everyone.

`src/metrics.py` computes the metrics used to score every model and saves them to `reports/results/` in one JSON format. That way nobody scores their own model differently from the rest, and the five-model comparison can be built by reading those files directly instead of collecting numbers from each person.

`src/leakage.py` defines the leak-free subset of each split, meaning the rows with no cleaned-text copy in train, so the duplicate check runs the same way for every model. `src/save_preds.py` saves each model's per-tweet class probabilities on the test split in one shared format, and the five-model comparison reads those files.

## Conventions Every Model Follows

### Data and splits

Use `splits/train_val_test_split.csv`. The neural models load it through `src/neural_data.load_neural_data()`.

### Tuning and testing

Tune on validation only, and use the test set once, after every decision is made. Run each tuning configuration with at least 2 seeds (3 preferred) and report the mean and standard deviation. One run cannot separate differences of 0.005 macro-F1 when the rare classes have 28 to 33 validation tweets.

Report two test scores: the full test set and the *clean* subset, which holds the tweets whose exact text is not in the training set. The headline score is the macro F1 on the clean subset, and accuracy, which is the leaderboard metric, is reported alongside.

### Result files

Result files are per model and live in `reports/results/`. Do not edit another person's file or append to a file that several people write to, because that causes merge conflicts. Each model saves these files:

* `<model>_test.json` and `<model>_clean_test.json` with `src/metrics.save_metrics`
* `preds_<model>_test.csv` with `src/save_preds.save_preds`
* `experiments_<model>.csv` with `src/experiment_log.log_experiment`
* `<model>_comparison_row.csv` with `src/results_table.make_comparison_row`
* `<model>_masked_test.json`, `<model>_random_masked_test.json` and `<model>_shuffled_test.json` from the stress tests in `src/robustness.py` (full test set, seed 42 for the random and shuffle tests)
* `bilstm_model.keras` and `bilstm_tokenizer.pkl`, the saved final BiLSTM, from which its predictions and stress tests can be reproduced

Part 4 builds `model_comparison_all.csv`, `roc_auc_all.csv` and `stress_test_comparison.csv` from these files, in `06_comparison.ipynb`, together with `comparison_confusion_matrices.png`, `comparison_roc.png` and `comparison_stress_tests.png`.

### Decisions and stress tests

Record any decision that affects more than one person in `DECISIONS.md`.

Read each stress test against the architecture. Shuffling cannot change the output of a model that ignores word order by construction, such as the final TextCNN with width-1 kernels (`DECISIONS.md` #25), so that result says nothing about the dataset.

## Writing Style

This applies to the README, report sections, notebook text and code comments.

Write in plain paragraphs and keep lists short and flat. Use a list only for items that really are a list, such as prerequisites or file names, and let headers do the structural work. Never change a code block or terminal command when you edit the prose around it.

Keep the tone direct and practical, the way you would explain something to a teammate. Skip filler and transition phrases, and say what a step does and why. Do not use em dashes or en dashes, and make sure no broken characters end up in the text.
