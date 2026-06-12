"""
Shared audio processing: augmentation and feature extraction.

Each dataset has its own loader file (ravdess.py, crema.py, tess.py, savee.py)
that imports process_audio_file from here.
"""

import librosa
import numpy as np

from config import load_config

cfg = load_config()

SR          = cfg["audio"]["sample_rate"]
SEG_DUR     = cfg["audio"]["segment_duration"]
SEG_SAMPLES = SEG_DUR * SR
MAX_LEN     = cfg["audio"]["max_len"]
N_MFCC      = cfg["audio"]["n_mfcc"]
N_MELS      = cfg["audio"]["n_mels"]
FEATURE_DIM = N_MFCC + N_MELS + 2  # +2 for ZCR and RMS

WEAK_CLASSES = cfg["augmentation"]["weak_classes"]  # {emotion: strategy}


# ── augmentation ──────────────────────────────────────────────────────────────

def _augment(audio: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "pitch_up":
        return librosa.effects.pitch_shift(y=audio, sr=SR, n_steps=2)
    if strategy == "pitch_down":
        return librosa.effects.pitch_shift(y=audio, sr=SR, n_steps=-2)
    if strategy == "time_stretch_slow":
        return librosa.effects.time_stretch(y=audio, rate=0.9)
    if strategy == "time_stretch_fast":
        return librosa.effects.time_stretch(y=audio, rate=1.1)
    if strategy == "background_noise":
        return audio + 0.003 * np.random.randn(len(audio))
    raise ValueError(f"Unknown augmentation strategy: {strategy!r}")


def targeted_augmentation(audio: np.ndarray, label: str) -> np.ndarray:
    strategy = WEAK_CLASSES.get(label)
    return _augment(audio, strategy) if strategy else audio


# ── feature extraction ────────────────────────────────────────────────────────

def extract_features(audio: np.ndarray) -> np.ndarray:
    """Returns (timesteps, FEATURE_DIM) array of MFCC + log-Mel + ZCR + RMS."""
    mfccs   = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC).T
    melspec = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS)
    melspec = librosa.power_to_db(melspec, ref=np.max).T
    zcr     = librosa.feature.zero_crossing_rate(y=audio).T          # (timesteps, 1)
    rms     = librosa.feature.rms(y=audio).T                         # (timesteps, 1)
    min_len = min(mfccs.shape[0], melspec.shape[0], zcr.shape[0], rms.shape[0])
    return np.concatenate(
        (mfccs[:min_len], melspec[:min_len], zcr[:min_len], rms[:min_len]), axis=1
    )


def process_audio_file(file_path: str, label: str) -> list[np.ndarray]:
    """Segment one audio file into 2-second windows and extract features."""
    audio, _ = librosa.load(file_path, sr=SR)
    features = []
    for start in range(0, len(audio), SEG_SAMPLES):
        segment = audio[start : start + SEG_SAMPLES]
        if len(segment) < SEG_SAMPLES:
            segment = np.pad(segment, (0, SEG_SAMPLES - len(segment)), mode="constant")
        segment = targeted_augmentation(segment, label)
        features.append(extract_features(segment))
    return features
