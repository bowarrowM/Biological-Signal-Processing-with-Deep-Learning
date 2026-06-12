"""
Gradio demo for Speech Emotion Recognition.

Usage:
    python app.py
    # Opens at http://127.0.0.1:7860
"""

import pickle

import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import load_config
from data import FEATURE_DIM, MAX_LEN, process_audio_file

cfg = load_config()

# ── load artifacts once at startup ───────────────────────────────────────────

model = tf.keras.models.load_model(cfg["model"]["checkpoint_path"])

with open(cfg["model"]["label_encoder_path"], "rb") as f:
    label_encoder = pickle.load(f)

with open(cfg["model"]["scaler_path"], "rb") as f:
    scaler = pickle.load(f)

EMOTION_EMOJI = {
    "angry":   "😠",
    "disgust": "🤢",
    "fear":    "😨",
    "happy":   "😊",
    "neutral": "😐",
    "sad":     "😢",
    "surprise":"😲",
}


# ── prediction function ───────────────────────────────────────────────────────

def predict(audio_path: str) -> dict:
    if audio_path is None:
        return {}

    segments = process_audio_file(audio_path, label="")
    if not segments:
        return {}

    X = pad_sequences(segments, maxlen=MAX_LEN, padding="post", dtype="float32", truncating="post")
    X = scaler.transform(X.reshape(-1, FEATURE_DIM)).reshape(X.shape)

    # Average across all 2-second segments of the file
    probs   = model.predict(X, verbose=0).mean(axis=0)
    classes = label_encoder.classes_

    return {f"{EMOTION_EMOJI.get(cls, '')} {cls}": float(prob)
            for cls, prob in zip(classes, probs)}


# ── interface ─────────────────────────────────────────────────────────────────

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        label="Record or upload audio",
    ),
    outputs=gr.Label(
        num_top_classes=7,
        label="Detected emotion",
    ),
    title="🎙️ Speech Emotion Recognition",
    description=(
        "Speak into your microphone or upload a **.wav** file. "
        "The model detects one of 7 emotions: "
        "angry, disgust, fear, happy, neutral, sad, surprise.\n\n"
        "Trained on RAVDESS · CREMA-D · TESS · SAVEE — 93% test accuracy."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
