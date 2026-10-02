# NLP-and-Language-Technologies

## Formative Assignment 2: GBV Tweet Classification (Sequential Modelling)

This is a group project comparing five sequential and classical modelling approaches on the Gender-Based Violence Tweet Classification Challenge dataset from Zindi.

**Content warning.** The dataset contains real, first-person descriptions of gender-based violence, including graphic descriptions of sexual abuse. Read with that in mind before opening the raw data files.

**Data attribution.** The dataset is provided by Zindi under a CC-BY 1.0 license, which permits this kind of research and educational use and redistribution. Source: [Gender-Based Violence Tweet Classification Challenge, Zindi](https://zindi.africa/competitions/gender-based-violence-tweet-classification-challenge).

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
  leakage.py                    cross-split duplicate check and leak-free subsets (shared)
  save_preds.py                 saves per-tweet class probabilities in the shared format
  robustness.py                 keyword-masking and word-shuffling stress tests
notebooks/
  01_exploratory_data_analysis.ipynb   exploratory data analysis, fully annotated
  02_baseline_models.ipynb             baseline model training and evaluation
reports/
  part1_data_understanding_and_preprocessing.md   written report section for Part 1
  figures/                       all chart and confusion matrix images
  results/                       per-model metrics as JSON, one file per model and split
reference/
  StarterNotebook.ipynb          organizer-provided starter notebook (background only)
```

The `reference/` folder holds background material from the competition organizers, like the original starter notebook, kept separate from the actual dataset since it's not part of the pipeline.

## Setup

```bash
pip install -r requirements.txt
```

## Shared Artifacts

Several files in this repo are shared infrastructure, not something to regenerate on your own branch. `splits/train_val_test_split.csv` holds the fixed stratified train, validation, and test split that all five models train and evaluate against, so results stay comparable across parts. `src/data_prep.py` handles text cleaning and label encoding the same way for everyone. `src/metrics.py` computes the evaluation metrics used to score every model and saves them to `reports/results/` in one consistent JSON format, so nobody ends up scoring their own model differently from the rest, and the five-model comparison in the Results stage can be built by reading those files directly rather than re-collecting numbers from each person. `src/leakage.py` defines the leak-free subset of each split (rows with no cleaned-text copy in train), so the duplicate check is run the same way for every model, and `src/save_preds.py` saves each model's per-tweet class probabilities on the test split in one shared format, which the five-model comparison reads.

## Branch Workflow

Part 1, covering baselines and EDA, gets pushed straight to `main` since it's the foundation everything else depends on. Parts 2 through 4 each work on their own branch off `main` and get reviewed before merging in.
