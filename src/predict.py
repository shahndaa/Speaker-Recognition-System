"""
Run the trained model on one or more .wav files (1 second, 16kHz, mono).

Usage:
    python -m src.predict path/to/clip.wav
    python -m src.predict clip1.wav clip2.wav
"""
from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf

from . import config
from .dataset import load_class_names
from .model import AudioToFFT


def load_audio(path: str) -> tf.Tensor:
    audio = tf.audio.decode_wav(
        tf.io.read_file(path), desired_channels=1, desired_samples=config.SAMPLES_PER_TRACK
    ).audio
    return tf.squeeze(audio, axis=-1)  # (SAMPLES_PER_TRACK,)


def predict(paths: list[str]) -> list[dict]:
    class_names = load_class_names()
    model = tf.keras.models.load_model(str(config.MODEL_PATH), custom_objects={"AudioToFFT": AudioToFFT})

    batch = tf.stack([load_audio(p) for p in paths])
    batch = tf.expand_dims(batch, axis=-1)  # (batch, samples, 1)
    probs = model.predict(batch, verbose=0)

    results = []
    for path, p in zip(paths, probs):
        idx = int(np.argmax(p))
        results.append({
            "file": path,
            "predicted_speaker": class_names[idx],
            "confidence": float(p[idx]),
            "all_probabilities": {name: round(float(prob), 4) for name, prob in zip(class_names, p)},
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict the speaker of one or more wav files.")
    parser.add_argument("audio_files", nargs="+", help="Path(s) to 1-second, 16kHz, mono .wav files")
    args = parser.parse_args()

    for r in predict(args.audio_files):
        print(f"\n{r['file']}")
        print(f"  -> Predicted speaker: {r['predicted_speaker']}  (confidence: {r['confidence']:.2%})")


if __name__ == "__main__":
    main()
