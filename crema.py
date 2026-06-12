"""
CREMA-D loader.

Layout: flat directory, filenames like 1001_DFA_ANG_XX.wav
Emotion is field index 2 of the '_'-delimited filename.
"""

import os

from data import process_audio_file

EMOTION_MAP = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}


def load(path: str) -> tuple[list, list]:
    data, labels = [], []
    for file in os.listdir(path):
        if not file.endswith(".wav"):
            continue
        parts = file.split("_")
        if len(parts) < 3:
            continue
        emotion = EMOTION_MAP.get(parts[2])
        if emotion is None:
            continue
        for feat in process_audio_file(os.path.join(path, file), emotion):
            data.append(feat)
            labels.append(emotion)
    return data, labels
