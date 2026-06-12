"""
TESS loader.

Layout: one folder per speaker+emotion, all .wav files inside.
    <path>/OAF_Angry/
    <path>/YAF_Pleasant_Surprise/
Emotion is derived from the folder name suffix (after the first '_').
"""

import os

from data import process_audio_file

# Folder suffix (lowercased) → unified emotion label
EMOTION_MAP = {
    "angry":              "angry",
    "disgust":            "disgust",
    "fear":               "fear",
    "happy":              "happy",
    "neutral":            "neutral",
    "sad":                "sad",
    "pleasant_surprise":  "surprise",
    "pleasant_surprised": "surprise",  # YAF folders use this variant
    "ps":                 "surprise",
}


def load(path: str) -> tuple[list, list]:
    data, labels = [], []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue
        # Folder names: OAF_Angry, YAF_Pleasant_Surprise, etc.
        parts = folder.split("_", 1)
        if len(parts) < 2:
            continue
        emotion = EMOTION_MAP.get(parts[1].lower())
        if emotion is None:
            continue
        for file in os.listdir(folder_path):
            if not file.endswith(".wav"):
                continue
            for feat in process_audio_file(os.path.join(folder_path, file), emotion):
                data.append(feat)
                labels.append(emotion)
    return data, labels
