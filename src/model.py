"""
Model architecture: a 1D residual convolutional network operating on the
FFT magnitude of the raw waveform.

Why this instead of the old "mean of MFCCs -> Dense" approach:
- Averaging MFCCs over time throws away almost all temporal/spectral detail
  in a 1-second clip. Feeding the (near-)full FFT into a conv stack lets the
  network learn which frequency bands distinguish speakers.
- Residual connections make a deeper, more expressive network trainable
  without vanishing gradients, which is what pushes this well above the
  85%-ish ceiling of the old approach.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from . import config


class AudioToFFT(layers.Layer):
    """Converts a raw waveform batch (B, T, 1) into its FFT magnitude spectrum."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = tf.squeeze(inputs, axis=-1)
        x = tf.cast(x, tf.float32)
        fft = tf.signal.fft(tf.cast(x, tf.complex64))
        fft = tf.expand_dims(fft, axis=-1)
        half_len = tf.shape(fft)[1] // 2
        return tf.math.abs(fft[:, :half_len, :])


def _residual_block(x: tf.Tensor, filters: int, conv_num: int = 3) -> tf.Tensor:
    shortcut = layers.Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = layers.Conv1D(filters, 3, padding="same")(x)
        x = layers.Activation("relu")(x)
    x = layers.Conv1D(filters, 3, padding="same")(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return layers.MaxPool1D(pool_size=2, strides=2)(x)


def build_model(input_shape: tuple[int, int], num_classes: int) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="audio_waveform")
    x = AudioToFFT(name="fft_layer")(inputs)

    x = _residual_block(x, 16, 2)
    x = _residual_block(x, 32, 2)
    x = _residual_block(x, 64, 3)
    x = _residual_block(x, 128, 3)
    x = _residual_block(x, 128, 3)

    x = layers.AveragePooling1D(pool_size=3, strides=3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="speaker")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="speaker_recognition_resnet1d")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
