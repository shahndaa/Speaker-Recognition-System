"""
Train the speaker recognition model end-to-end.

Usage:
    python -m src.train
    python -m src.train --dataset_dir /path/to/16000_pcm_speeches --epochs 20
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from . import config
from .dataset import make_datasets, save_class_names
from .model import build_model
from .utils import set_global_seed


def plot_training_curves(history: dict, out_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (test set)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the speaker recognition model.")
    parser.add_argument("--dataset_dir", type=str, default=str(config.DATASET_DIR))
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    set_global_seed(config.RANDOM_SEED)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading dataset from: {args.dataset_dir}")
    from pathlib import Path
    train_ds, val_ds, test_ds, class_names = make_datasets(Path(args.dataset_dir))
    print(f"      Found {len(class_names)} speakers: {class_names}")

    save_class_names(class_names)

    print("[2/5] Building model...")
    model = build_model(input_shape=(config.SAMPLES_PER_TRACK, 1), num_classes=len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(config.MODEL_PATH), monitor="val_accuracy", save_best_only=True,
        ),
    ]

    print("[3/5] Training...")
    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    train_time = time.time() - start
    print(f"      Training finished in {train_time:.1f}s")

    with open(config.TRAINING_HISTORY_PATH, "w") as f:
        json.dump(history.history, f, indent=2)
    plot_training_curves(history.history, config.TRAINING_CURVES_PATH)

    print("[4/5] Evaluating on held-out test set...")
    y_true, y_pred = [], []
    for batch_audio, batch_labels in test_ds:
        preds = model.predict(batch_audio, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(batch_labels.numpy())

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, config.CONFUSION_MATRIX_PATH)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "training_time_seconds": train_time,
        "classification_report": report,
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[5/5] Done.")
    print(f"      Test accuracy: {test_acc*100:.2f}%")
    print(f"      Model saved to: {config.MODEL_PATH}")
    print(f"      Metrics saved to: {config.METRICS_PATH}")
    print(f"      Confusion matrix saved to: {config.CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
