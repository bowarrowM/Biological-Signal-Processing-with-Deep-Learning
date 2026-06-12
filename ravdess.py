"""
RAVDESS loader.

Layout: one sub-folder per actor, filenames like 03-01-03-01-01-01-01.wav
Emotion is field index 2 of the '-'-delimited filename (1-indexed).
"""

import os

from data import process_audio_file

EMOTION_MAP = {
    1: "neutral", 2: "neutral", 3: "happy",   4: "sad",
    5: "angry",   6: "fear",    7: "disgust",  8: "surprise",
}


def load(path: str) -> tuple[list, list]:
    data, labels = [], []
    for folder in sorted(os.listdir(path)):
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if not file.endswith(".wav"):
                continue
            try:
                emotion = EMOTION_MAP[int(file.split("-")[2])]
            except (IndexError, KeyError, ValueError):
                continue
            for feat in process_audio_file(os.path.join(folder_path, file), emotion):
                data.append(feat)
                labels.append(emotion)
    return data, labels
