"""
Inference script — predict the emotion of a single audio file.

Usage:
    python predict.py path/to/audio.wav
    python predict.py path/to/audio.wav --top 3   # show top-3 predictions
"""

import argparse
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import load_config
from data import MAX_LEN, FEATURE_DIM, process_audio_file

cfg = load_config()


def load_artifacts():
    model = tf.keras.models.load_model(cfg["model"]["checkpoint_path"])

    with open(cfg["model"]["label_encoder_path"], "rb") as f:
        label_encoder = pickle.load(f)

    with open(cfg["model"]["scaler_path"], "rb") as f:
        scaler = pickle.load(f)

    return model, label_encoder, scaler


def predict(audio_path: str, top_n: int = 1) -> list[tuple[str, float]]:
    """
    Returns a list of (emotion, confidence) tuples, sorted by confidence descending.
    """
    model, label_encoder, scaler = load_artifacts()

    # Use a placeholder label — augmentation won't fire for unknown labels
    segments = process_audio_file(audio_path, label="")

    if not segments:
        raise ValueError(f"No audio segments extracted from {audio_path!r}")

    X = pad_sequences(segments, maxlen=MAX_LEN, padding="post", dtype="float32", truncating="post")
    X = scaler.transform(X.reshape(-1, FEATURE_DIM)).reshape(X.shape)

    # Average predictions across all segments of the file
    probs = model.predict(X, verbose=0).mean(axis=0)

    classes = label_encoder.classes_
    ranked  = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to a .wav file")
    parser.add_argument("--top", type=int, default=1, help="Number of top predictions to show")
    args = parser.parse_args()

    results = predict(args.audio, top_n=args.top)

    print(f"\nFile: {args.audio}")
    print(f"Predicted: {results[0][0].upper()}  ({results[0][1]*100:.1f}%)\n")

    if args.top > 1:
        print("All scores:")
        for emotion, confidence in results:
            bar = "█" * int(confidence * 30)
            print(f"  {emotion:<10} {confidence*100:5.1f}%  {bar}")
