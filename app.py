"""
Streamlit demo for the Speaker Recognition System.

Run locally:
    streamlit run app.py

Deploy for free on Streamlit Community Cloud by connecting this GitHub repo
and pointing it at this file (see README for the full steps).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import streamlit as st
import tensorflow as tf

from src import config
from src.dataset import load_class_names
from src.model import AudioToFFT

st.set_page_config(page_title="Speaker Recognition", page_icon="🎙️", layout="centered")


@st.cache_resource
def get_model_and_labels():
    model = tf.keras.models.load_model(str(config.MODEL_PATH), custom_objects={"AudioToFFT": AudioToFFT})
    class_names = load_class_names()
    return model, class_names


def decode_wav_bytes(file_bytes: bytes) -> tf.Tensor:
    """Decode wav bytes into a fixed-length (SAMPLES_PER_TRACK, 1) tensor,
    trimming or zero-padding so any 16kHz mono clip works, not just exact 1s ones."""
    audio, _ = tf.audio.decode_wav(file_bytes, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    target_len = config.SAMPLES_PER_TRACK
    length = tf.shape(audio)[0]
    audio = tf.cond(
        length >= target_len,
        lambda: audio[:target_len],
        lambda: tf.pad(audio, [[0, target_len - length]]),
    )
    return tf.expand_dims(audio, axis=-1)


def predict_from_bytes(file_bytes: bytes):
    model, class_names = get_model_and_labels()
    audio = decode_wav_bytes(file_bytes)
    batch = tf.expand_dims(audio, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    return class_names, probs


st.title("🎙️ Speaker Recognition System")
st.write(
    "A 1D residual CNN trained on the FFT spectrum of raw audio, identifying "
    "**who is speaking** from a short clip. Trained on the "
    "[Speaker Recognition Dataset](https://www.kaggle.com/datasets/kongaevans/speaker-recognition-dataset) "
    "— test accuracy **95.29%**."
)

st.subheader("Try it")
tab_upload, tab_samples = st.tabs(["Upload your own .wav", "Try a sample clip"])

audio_bytes = None

with tab_upload:
    uploaded = st.file_uploader("Upload a 16kHz mono .wav clip (~1 second)", type=["wav"])
    if uploaded is not None:
        audio_bytes = uploaded.read()
        st.audio(audio_bytes, format="audio/wav")

with tab_samples:
    sample_files = sorted(glob.glob(os.path.join("assets", "samples", "*.wav")))
    if sample_files:
        labels = [os.path.basename(f).replace("_sample.wav", "").replace("_", " ") for f in sample_files]
        choice = st.selectbox("Pick a bundled sample", labels)
        chosen_path = sample_files[labels.index(choice)]
        with open(chosen_path, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/wav")
    else:
        st.info("No bundled sample clips found in assets/samples/.")

if audio_bytes is not None:
    with st.spinner("Running inference..."):
        class_names, probs = predict_from_bytes(audio_bytes)

    top_idx = int(np.argmax(probs))
    st.success(f"**Predicted speaker: {class_names[top_idx]}**  ({probs[top_idx]:.2%} confidence)")

    st.bar_chart({name: float(p) for name, p in zip(class_names, probs)})

st.divider()
st.caption(
    "Model: 5 residual Conv1D blocks over the FFT magnitude of a 16kHz waveform. "
    "See the [GitHub repo](.) for training code, evaluation report, and the confusion matrix."
)
# Stream 
# Stream2 
# Stream3 
# Cache 
# Stream 
# Stream2 
