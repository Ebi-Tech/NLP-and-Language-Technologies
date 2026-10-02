# Methodology input template (one per model owner)

Send this to Divine, Kevin and Jean. I need it filled in to write section 4 (Methodology) of the report. Keep each field to 2 to 5 lines. Use numbers, not adjectives. Anything you cannot fill in, write "not done".

**Your name and model:**

**1. Input representation**
- Tokenizer and vocabulary (type, size cap, what happens to unknown words)
- Maximum sequence length and padding or truncation
- Embeddings (random or pretrained, dimension)
- Any cleaning beyond the shared `clean_text` function

**2. Architecture and why it differs from the other models**
- Layers and sizes, in order
- One sentence on what this architecture can capture that the others cannot, and one on what it cannot

**3. Training setup**
- Optimizer, learning rate, batch size, maximum epochs
- Loss and how class imbalance is handled
- Stopping rule and what is restored at the end
- Regularisation (dropout, weight decay)

**4. What you tuned**
- Each configuration you tried, what you expected, the mean validation macro-F1 (and std over seeds), and what you decided. A copy of your rows from `reports/results/experiments_<model>.csv` is enough.
- Number of seeds per configuration

**5. Cost and environment**
- Training time per run, hardware, and library versions (your notebook prints them)
- Anything that did not work or was dropped, and why

**6. Test results (fill in only after tuning is final)**
- Clean-test macro-F1, full-test macro-F1, accuracy, weakest class

---

## Example, filled in for the TextCNN (use this as the level of detail; do not copy its content)

**1. Input representation.** Keras Tokenizer, 20,000-word cap, unknown words mapped to `<OOV>`, vocabulary fitted on train only. Max length 70, post-padding and post-truncation. Embeddings random-initialised, 128-d, learned from scratch. No extra cleaning.

**2. Architecture.** Embedding, then parallel Conv1D layers with widths {W} and {N} filters each (ReLU), global max-pool per branch, concatenate, dropout 0.5, softmax over 5 classes. Captures local word patterns anywhere in the tweet; cannot use context beyond its widest window.

**3. Training.** Adam, lr 1e-3, batch 64, up to 15 epochs, class weights as chosen in Stage C (final TextCNN: none), early stopping on validation loss with patience 3 and best weights restored, dropout 0.5.

**4. Tuning.** Stage A kernel widths ({1}, {2}, {3,4,5}, {3,4,5,6,7}), Stage B filters (50/100/200), Stage C class weights on/off. 3 seeds each. Rows A1 to C1 in `reports/results/experiments_<model>.csv`.

**5. Cost.** About {X} seconds per run on {GPU}. TensorFlow {version}.

**6. Test.** Clean-test macro-F1 {mean ± std}, weakest class {class}.
