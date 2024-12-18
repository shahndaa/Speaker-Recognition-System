"""
Dataset loading utilities.

Design notes (why this differs from the old notebook):
- No hardcoded absolute paths: everything is derived from `config.DATASET_DIR`,
  so the project runs on any machine/OS as long as the dataset folder sits
  next to the project (or DATASET_DIR is overridden).
- Uses `tf.data.Dataset` pipelines instead of loading everything into a giant
  in-memory numpy array up front -> scales to larger datasets without
  blowing up RAM.
- Background noise is mixed into training samples on-the-fly as a real
  augmentation step (the old code loaded noise files but never actually
  used them to augment training data).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from . import config


def _list_speaker_dirs(dataset_dir: Path) -> List[Path]:
    """Return speaker subfolders, skipping non-speaker folders."""
    exclude = {config.NOISE_DIR_NAME, "other"}
    speakers = [
        d for d in sorted(dataset_dir.iterdir())
        if d.is_dir() and d.name not in exclude
    ]
    if not speakers:
        raise FileNotFoundError(
            f"No speaker folders found in {dataset_dir}. "
            "Expected one subfolder per speaker (e.g. Benjamin_Netanyau/)."
        )
    return speakers


def build_file_label_lists(dataset_dir: Path = config.DATASET_DIR) -> Tuple[List[str], List[int], List[str]]:
    """Walk the dataset directory and return (filepaths, integer labels, class_names)."""
    speaker_dirs = _list_speaker_dirs(dataset_dir)
    class_names = [d.name for d in speaker_dirs]

    filepaths, labels = [], []
    for idx, speaker_dir in enumerate(speaker_dirs):
        wav_files = sorted(speaker_dir.glob("*.wav"))
        filepaths.extend(str(f) for f in wav_files)
        labels.extend([idx] * len(wav_files))

    return filepaths, labels, class_names


def load_noise_samples(dataset_dir: Path = config.DATASET_DIR) -> np.ndarray:
    """Load and concatenate all background noise clips, chopped into 1s chunks."""
    noise_dir = dataset_dir / config.NOISE_DIR_NAME
    if not noise_dir.exists():
        return np.zeros((0, config.SAMPLES_PER_TRACK), dtype=np.float32)

    chunks = []
    for f in sorted(noise_dir.glob("*.wav")):
        audio = tf.audio.decode_wav(tf.io.read_file(str(f)), desired_channels=1).audio
        audio = tf.squeeze(audio, axis=-1).numpy()
        n_chunks = len(audio) // config.SAMPLES_PER_TRACK
        for i in range(n_chunks):
            chunks.append(audio[i * config.SAMPLES_PER_TRACK:(i + 1) * config.SAMPLES_PER_TRACK])

    if not chunks:
        return np.zeros((0, config.SAMPLES_PER_TRACK), dtype=np.float32)
    return np.stack(chunks).astype(np.float32)


def _decode_audio(filepath: tf.Tensor) -> tf.Tensor:
    audio_bin = tf.io.read_file(filepath)
    audio = tf.audio.decode_wav(audio_bin, desired_channels=1, desired_samples=config.SAMPLES_PER_TRACK).audio
    return tf.squeeze(audio, axis=-1)  # (SAMPLES_PER_TRACK,)


def _add_noise(audio: tf.Tensor, noise_samples: tf.Tensor) -> tf.Tensor:
    """Randomly mix a background-noise chunk into `audio` with a random scale."""
    if tf.shape(noise_samples)[0] == 0:
        return audio

    do_augment = tf.random.uniform([]) < config.NOISE_AUGMENT_PROBABILITY
    if not do_augment:
        return audio

    idx = tf.random.uniform([], maxval=tf.shape(noise_samples)[0], dtype=tf.int32)
    noise = noise_samples[idx]
    scale = tf.random.uniform([], config.NOISE_SCALE_MIN, config.NOISE_SCALE_MAX)

    audio_amp = tf.reduce_max(tf.abs(audio)) + 1e-9
    noise_amp = tf.reduce_max(tf.abs(noise)) + 1e-9
    return audio + scale * (audio_amp / noise_amp) * noise


def make_datasets(
    dataset_dir: Path = config.DATASET_DIR,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """Build train/val/test tf.data.Dataset pipelines with a proper stratified split."""
    filepaths, labels, class_names = build_file_label_lists(dataset_dir)
    filepaths = np.array(filepaths)
    labels = np.array(labels)

    rng = np.random.default_rng(config.RANDOM_SEED)

    # Stratified split: shuffle within each class so train/val/test all see every speaker.
    train_idx, val_idx, test_idx = [], [], []
    for class_id in np.unique(labels):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)
        n = len(class_indices)
        n_val = int(n * config.VAL_SPLIT)
        n_test = int(n * config.TEST_SPLIT)
        val_idx.extend(class_indices[:n_val])
        test_idx.extend(class_indices[n_val:n_val + n_test])
        train_idx.extend(class_indices[n_val + n_test:])

    train_idx, val_idx, test_idx = map(np.array, (train_idx, val_idx, test_idx))
    rng.shuffle(train_idx)

    noise_samples = load_noise_samples(dataset_dir)
    noise_tensor = tf.constant(noise_samples)

    def make_ds(indices: np.ndarray, augment: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((filepaths[indices], labels[indices]))
        ds = ds.map(lambda fp, lbl: (_decode_audio(fp), lbl), num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(lambda audio, lbl: (_add_noise(audio, noise_tensor), lbl),
                        num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda audio, lbl: (tf.expand_dims(audio, -1), lbl), num_parallel_calls=tf.data.AUTOTUNE)
        return ds

    train_ds = make_ds(train_idx, augment=True).shuffle(2048, seed=config.RANDOM_SEED)
    val_ds = make_ds(val_idx, augment=False)
    test_ds = make_ds(test_idx, augment=False)

    train_ds = train_ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def save_class_names(class_names: List[str], path: Path = config.LABEL_ENCODER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"class_names": class_names}, f, ensure_ascii=False, indent=2)


def load_class_names(path: Path = config.LABEL_ENCODER_PATH) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["class_names"]
# Dataset 
# Dataset 
