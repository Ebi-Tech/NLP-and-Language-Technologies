# NLP-and-Language-Technologies

## Formative Assignment 2: GBV Tweet Classification (Sequential Modelling)

This is a group project comparing five sequential and classical modelling approaches on the Gender-Based Violence Tweet Classification Challenge dataset from Zindi.

**Content warning.** The dataset contains real, first-person descriptions of gender-based violence, including graphic descriptions of sexual abuse. Read with that in mind before opening the raw data files.

**Data attribution.** The dataset is provided by Zindi. Check the licence terms on the competition page before redistributing the data. Source: [Gender-Based Violence Tweet Classification Challenge, Zindi](https://zindi.africa/competitions/gender-based-violence-tweet-classification-challenge).

## Repo Structure

```text
data/
  raw/                          original Train.csv / Test.csv (do not edit)
  processed/                    cleaned data produced by src/data_prep.py
splits/
  train_val_test_split.csv      shared fixed split, use this exact file
src/
  data_prep.py                  cleaning + label encoding (shared, import this)
  metrics.py                    shared evaluation functions (shared, import this)
  baselines.py                  majority-class + TF-IDF/LogReg baselines
  save_preds.py                 saves per-tweet test predictions in the shared format
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
reports/
  part1_data_understanding_and_preprocessing.md   written report section for Part 1
  part3_methodology.md           draft of the Methodology section (Part 3)
  figures/                       all chart and confusion matrix images
  results/                       every result file, one set per model (see Conventions below)
reference/
  StarterNotebook.ipynb          organizer-provided starter notebook (background only)
```

The `reference/` folder holds background material from the competition organizers, like the original starter notebook, kept separate from the actual dataset since it's not part of the pipeline.

## Setup

```bash
pip install -r requirements.txt
```

**Google Colab.** Open a notebook from `notebooks/` in Colab and, for the neural models, choose Runtime, then Change runtime type, then a GPU. The first cell clones this repository, moves into it and adds it to the import path, so the notebook runs from a fresh runtime. TensorFlow is already installed on Colab and is not reinstalled. Run notebooks from the repository root, because `src/` imports use paths relative to it.

## Shared Artifacts

Several files in this repo are shared infrastructure, not something to regenerate on your own branch. `splits/train_val_test_split.csv` holds the fixed stratified train, validation, and test split that all five models train and evaluate against, so results stay comparable across parts. `src/data_prep.py` handles text cleaning and label encoding the same way for everyone. `src/metrics.py` computes the evaluation metrics used to score every model and saves them to `reports/results/` in one consistent JSON format, so nobody ends up scoring their own model differently from the rest, and the five-model comparison in the Results stage can be built by reading those files directly rather than re-collecting numbers from each person.

## Conventions Every Model Follows

1. **Same data.** Use `splits/train_val_test_split.csv`. The neural models load it through `src/neural_data.load_neural_data()`.
2. **Tune on validation only. Use the test set once,** after every decision is made.
3. **Several seeds.** Run each tuning configuration with at least 2 seeds (3 preferred) and report mean and standard deviation. One run cannot separate differences of 0.005 macro-F1 when the rare classes have 28 to 33 validation tweets.
4. **Report two test scores:** the full test set and the *clean* subset (tweets whose exact text is not in the training set). The clean subset is the headline.
5. **Result files are per model and live in `reports/results/`.** Do not edit another person's file or append to a file several people write to, because that causes merge conflicts. Each model saves:
   - `<model>_test.json` and `<model>_clean_test.json` with `src/metrics.save_metrics`
   - `preds_<model>_test.csv` with `src/save_preds.save_preds`
   - `experiments_<model>.csv` with `src/experiment_log.log_experiment`
   - `<model>_comparison_row.csv` with `src/results_table.make_comparison_row`

   The combined `model_comparison.csv` is built from these by Part 4.
6. **Record decisions** in `DECISIONS.md`.

## Branch Workflow

Part 1, covering baselines and EDA, gets pushed straight to `main` since it's the foundation everything else depends on. Parts 2 through 4 each work on their own branch off `main` and get reviewed before merging in.
