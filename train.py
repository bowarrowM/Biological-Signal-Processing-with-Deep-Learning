"""
Training entry point.

Usage:
    python train.py --ravdess-path /data/ravdess --crema-path /data/crema \
                    --tess-path /data/tess --savee-path /data/savee
    # or via environment variables:
    RAVDESS_PATH=... CREMA_PATH=... TESS_PATH=... SAVEE_PATH=... python train.py
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from config import load_config, require_paths
from data import FEATURE_DIM, MAX_LEN
from model import build_model
import ravdess
import crema
import tess
import savee

cfg = load_config()
require_paths(cfg, "ravdess_path", "crema_path", "tess_path", "savee_path")

SEED       = cfg["training"]["random_seed"]
TEST_SIZE  = cfg["training"]["test_size"]
VAL_SIZE   = cfg["training"]["val_size"]
LR         = cfg["training"]["learning_rate"]
BATCH_SIZE = cfg["training"]["batch_size"]
EPOCHS          = cfg["training"]["epochs"]
LABEL_SMOOTHING = cfg["training"]["label_smoothing"]
ES_PAT     = cfg["training"]["early_stop_patience"]
LR_PAT     = cfg["training"]["reduce_lr_patience"]
LR_FACTOR  = cfg["training"]["reduce_lr_factor"]
MIN_LR     = cfg["training"]["min_lr"]

CHECKPOINT_PATH    = cfg["model"]["checkpoint_path"]
LABEL_ENCODER_PATH = cfg["model"]["label_encoder_path"]
SCALER_PATH        = cfg["model"]["scaler_path"]

os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_all_data() -> tuple[list, list]:
    """
    Aggregate samples from all datasets.
    To add a new dataset: import its module and append its load() results here.
    """
    data, labels = [], []

    for name, module, path_key in [
        ("RAVDESS", ravdess, "ravdess_path"),
        ("CREMA-D", crema,   "crema_path"),
        ("TESS",    tess,    "tess_path"),
        ("SAVEE",   savee,   "savee_path"),
    ]:
        print(f"Loading {name}...")
        d, l = module.load(cfg["data"][path_key])
        data   += d
        labels += l
        print(f"  {len(d)} segments loaded.")

    return data, labels


def evaluate(model, X_test, y_test, label_encoder):
    y_pred      = np.argmax(model.predict(X_test), axis=1)
    y_true      = np.argmax(y_test, axis=1)
    class_names = label_encoder.classes_

    print(f"\nTest accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    per_class_f1 = dict(zip(class_names.tolist(), f1_score(y_true, y_pred, average=None).round(3).tolist()))
    print("Per-class F1:", per_class_f1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(CHECKPOINT_PATH), "confusion_matrix.png"), dpi=150)
    plt.show()


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"],     label="Train")
    ax1.plot(history.history["val_accuracy"], label="Val")
    ax1.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"],     label="Train")
    ax2.plot(history.history["val_loss"], label="Val")
    ax2.set(title="Loss", xlabel="Epoch", ylabel="Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(CHECKPOINT_PATH), "training_curves.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    data, labels = load_all_data()
    print(f"\nTotal segments: {len(data)}")

    X = pad_sequences(data, maxlen=MAX_LEN, padding="post", dtype="float32", truncating="post")

    label_encoder = LabelEncoder()
    y = to_categorical(label_encoder.fit_transform(labels))

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved → {LABEL_ENCODER_PATH}")
    print(f"Classes: {list(label_encoder.classes_)}")

    # 3-way stratified split: train / val / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=y_train,
        random_state=SEED,
    )
    print(f"Split → train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # Normalize per-feature using training set statistics
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, FEATURE_DIM)).reshape(X_train.shape)
    X_val   = scaler.transform(X_val.reshape(-1, FEATURE_DIM)).reshape(X_val.shape)
    X_test  = scaler.transform(X_test.reshape(-1, FEATURE_DIM)).reshape(X_test.shape)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved → {SCALER_PATH}")

    class_counts  = np.sum(y_train, axis=0)
    total         = np.sum(class_counts)
    class_weights = {i: total / (len(class_counts) * c) for i, c in enumerate(class_counts)}

    model = build_model(input_shape=(MAX_LEN, FEATURE_DIM), n_classes=y.shape[1])
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=ES_PAT, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=LR_FACTOR, patience=LR_PAT, min_lr=MIN_LR),
            ModelCheckpoint(CHECKPOINT_PATH, monitor="val_loss", save_best_only=True, mode="min"),
        ],
    )

    evaluate(model, X_test, y_test, label_encoder)
    plot_history(history)
