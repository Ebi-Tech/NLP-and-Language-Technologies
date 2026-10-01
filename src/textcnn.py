"""TextCNN for the GBV tweet classification task (Part 3).

The architecture follows Kim (2014): several 1-D convolutions of different
widths run in parallel over the word embeddings, each followed by global
max-pooling. The pooled features are joined, passed through dropout and fed to
a softmax layer.

Two functions live here:
    build_textcnn    - builds and compiles the Keras model
    train_and_score  - trains one configuration with one seed and scores it on
                       the VALIDATION set (the test set is never used here)
"""

import time

import numpy as np
from sklearn.metrics import f1_score

from src.neural_data import MAX_LEN, MAX_WORDS, set_seed


def build_textcnn(kernel_sizes=(3, 4, 5), n_filters=100, embed_dim=128, dropout=0.5,
                  max_words=MAX_WORDS, max_len=MAX_LEN, n_classes=5, lr=1e-3):
    """Build the TextCNN.

    kernel_sizes : widths of the convolution filters, e.g. (3, 4, 5). A filter of
                   width k looks at k consecutive words at a time, so width 1 is
                   a single-word detector and wider filters see short phrases.
    n_filters    : number of filters per width.
    """
    # Imported inside the function so that importing this module does not load
    # TensorFlow (which is slow) until a model is actually needed.
    from tensorflow import keras
    from tensorflow.keras import layers

    tokens = keras.Input(shape=(max_len,), name="tokens")

    # Each word index becomes a learned vector. The embeddings start random,
    # exactly as in the BiLSTM, so only the architecture differs between them.
    embedded = layers.Embedding(max_words, embed_dim, name="embedding")(tokens)

    # One branch per kernel width: convolution, then keep only the strongest
    # response of each filter anywhere in the tweet (global max-pooling).
    branches = []
    for k in kernel_sizes:
        x = layers.Conv1D(n_filters, k, activation="relu", name=f"conv_k{k}")(embedded)
        x = layers.GlobalMaxPooling1D(name=f"pool_k{k}")(x)
        branches.append(x)

    # Concatenate needs at least two inputs, so a single branch is used as is.
    features = branches[0] if len(branches) == 1 else layers.Concatenate(name="concat")(branches)
    features = layers.Dropout(dropout, name="dropout")(features)
    probs = layers.Dense(n_classes, activation="softmax", name="classifier")(features)

    model = keras.Model(tokens, probs, name="textcnn_k" + "-".join(map(str, kernel_sizes)))
    # Labels are integers 0-4 (not one-hot), hence the "sparse" loss.
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_score(data, *, kernel_sizes, n_filters, seed, class_weight="balanced",
                    dropout=0.5, epochs=15, batch_size=64, patience=3, verbose=0):
    """Train one configuration with one seed and score it on validation.

    data         : the dictionary returned by src.neural_data.load_neural_data()
    class_weight : "balanced" uses the weights computed from the training labels;
                   anything else (for example None) trains without weights.
    epochs       : upper limit only. Early stopping usually ends training sooner.
    patience     : epochs without a better validation loss before stopping.

    Returns a dictionary with the trained model, the Keras history (for the
    learning curves), validation macro-F1 and accuracy, the number of epochs
    actually run, training time in seconds and the parameter count.
    """
    from tensorflow import keras

    set_seed(seed)  # same seed gives the same starting weights and shuffling
    model = build_textcnn(kernel_sizes, n_filters, dropout=dropout)

    # Stop when validation loss has not improved for `patience` epochs and go
    # back to the weights from the best epoch.
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                               restore_best_weights=True)
    weights = data["class_weights"] if class_weight == "balanced" else None

    start = time.time()
    history = model.fit(data["X_train"], data["y_train"],
                        validation_data=(data["X_val"], data["y_val"]),
                        epochs=epochs, batch_size=batch_size, class_weight=weights,
                        callbacks=[early_stop], verbose=verbose)
    seconds = time.time() - start

    # Score on validation with the restored best weights. Macro-F1 is the
    # headline metric because it weights all five classes equally.
    val_pred = model.predict(data["X_val"], verbose=0).argmax(axis=1)
    return {
        "model": model,
        "history": history.history,
        "val_macro_f1": f1_score(data["y_val"], val_pred, average="macro", zero_division=0),
        "val_accuracy": float((val_pred == data["y_val"]).mean()),
        "epochs_run": len(history.history["loss"]),
        "train_seconds": seconds,
        "n_params": int(model.count_params()),
    }


def mean_std(values):
    """Mean and sample standard deviation (0.0 if there is only one value)."""
    v = np.asarray(values, dtype=float)
    std = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return float(v.mean()), std
