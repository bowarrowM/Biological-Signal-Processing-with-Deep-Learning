"""
SAVEE loader.

Layout: one sub-folder per speaker (DC, JE, JK, KL), files named by
emotion code + sentence number, e.g. a01.wav, sa01.wav, su01.wav.

Some Kaggle uploads use a flat layout with a speaker prefix:
e.g. DC_a01.wav — both are handled automatically.
"""

import os
import re

from data import process_audio_file

EMOTION_MAP = {
    "a":  "angry",
    "d":  "disgust",
    "f":  "fear",
    "h":  "happy",
    "n":  "neutral",
    "sa": "sad",
    "su": "surprise",
}


def _parse_emotion(filename: str) -> str | None:
    """Extract emotion from filenames like a01.wav or DC_a01.wav."""
    stem = os.path.splitext(filename)[0].lower()
    if "_" in stem:
        stem = stem.split("_", 1)[1]   # strip speaker prefix
    match = re.match(r"([a-z]+)", stem)
    return EMOTION_MAP.get(match.group(1)) if match else None


def load(path: str) -> tuple[list, list]:
    data, labels = [], []

    has_wavs = any(f.endswith(".wav") for f in os.listdir(path))

    if has_wavs:
        # Flat layout: all files in one directory
        entries = [(path, f) for f in os.listdir(path) if f.endswith(".wav")]
    else:
        # Nested layout: speaker sub-folders
        entries = [
            (os.path.join(path, folder), file)
            for folder in os.listdir(path)
            if os.path.isdir(os.path.join(path, folder))
            for file in os.listdir(os.path.join(path, folder))
            if file.endswith(".wav")
        ]

    for folder_path, file in entries:
        emotion = _parse_emotion(file)
        if emotion is None:
            continue
        for feat in process_audio_file(os.path.join(folder_path, file), emotion):
            data.append(feat)
            labels.append(emotion)

    return data, labels
