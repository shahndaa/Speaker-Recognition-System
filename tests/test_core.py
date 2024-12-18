"""
Lightweight unit tests that don't require the (large, not-committed) dataset —
they just check the model builds correctly and the label save/load round-trips.

Run with: pytest tests/
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.model import build_model
from src.dataset import save_class_names, load_class_names


def test_model_builds_and_runs_forward_pass():
    model = build_model(input_shape=(16000, 1), num_classes=5)
    dummy_batch = tf.random.uniform((2, 16000, 1))
    output = model(dummy_batch)
    assert output.shape == (2, 5)
    # softmax outputs should sum to ~1 per sample
    np.testing.assert_allclose(np.sum(output.numpy(), axis=1), [1.0, 1.0], atol=1e-4)


def test_class_names_round_trip(tmp_path: Path = None):
    tmp_path = tmp_path or Path(tempfile.mkdtemp())
    path = tmp_path / "label_encoder.json"
    names = ["Alice", "Bob", "Carol"]

    save_class_names(names, path)
    loaded = load_class_names(path)

    assert loaded == names
    assert json.loads(path.read_text())["class_names"] == names
