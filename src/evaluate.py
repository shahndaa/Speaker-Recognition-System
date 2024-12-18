"""Evaluate the saved checkpoint on the held-out test set only (no training).
Lightweight and fast — safe to run without risking long-running memory issues."""
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src import config
from src.dataset import make_datasets, save_class_names
from src.model import AudioToFFT
from src.train import plot_confusion_matrix


def main():
    print("Building datasets (same seed -> same split as training)...")
    _, _, test_ds, class_names = make_datasets(config.DATASET_DIR)
    save_class_names(class_names)

    print("Loading best checkpoint...")
    model = tf.keras.models.load_model(str(config.MODEL_PATH), custom_objects={"AudioToFFT": AudioToFFT})

    y_true, y_pred = [], []
    for batch_audio, batch_labels in test_ds:
        preds = model.predict(batch_audio, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(batch_labels.numpy())

    test_acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_txt = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, config.CONFUSION_MATRIX_PATH)

    metrics = {
        "test_accuracy": test_acc,
        "num_test_samples": len(y_true),
        "classification_report": report_dict,
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(report_txt)
    print(f"\nFINAL TEST ACCURACY: {test_acc*100:.2f}%  (on {len(y_true)} held-out samples)")


if __name__ == "__main__":
    main()
