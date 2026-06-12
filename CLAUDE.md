# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Speech Emotion Recognition (SER) using a hybrid **CNN-LSTM** model trained on **RAVDESS** and **CREMA-D** audio datasets. Achieves ~74% accuracy across 7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprise).

## Setup

```bash
pip install -r requirements.txt
```

## Running the training pipeline

Set dataset paths, then run:

```bash
# Via environment variables
export RAVDESS_PATH=/path/to/ravdess
export CREMA_PATH=/path/to/crema
python BitirmeSON.py

# Or via CLI flags
python BitirmeSON.py --ravdess-path /path/to/ravdess --crema-path /path/to/crema
```

Outputs to `checkpoints/`: `best_model.keras`, `label_encoder.pkl`, `confusion_matrix.png`, `training_curves.png`.

## Configuration

All hyperparameters live in `config.yaml`. `config.py` loads it and applies overrides.

**Path override priority:** CLI arg > env var > `config.yaml` value.

When adding a new dataset, add its path key under `data:` in `config.yaml` and register it in `_PATH_OVERRIDES` in `config.py`.

Augmentation strategies per emotion are defined in `config.yaml` under `augmentation.weak_classes`. Supported values: `pitch_up`, `pitch_down`, `time_stretch_slow`, `time_stretch_fast`, `background_noise`. Emotions not listed receive no augmentation.

## Dataset format

- **RAVDESS**: actor sub-folders, filenames like `03-01-03-01-01-01-01.wav` — emotion is field index 2 (1-indexed → 8 classes mapped to 7, with `1` and `2` both → `neutral`)
- **CREMA-D**: flat directory of `.wav` files, emotion is field index 2 of `_`-delimited name (e.g. `1001_DFA_ANG_XX.wav`)

## Architecture (BitirmeSON.py)

**Pipeline:**
1. Load `.wav` → resample to 22,050 Hz → 2-second sliding window segments
2. Targeted augmentation for `weak_classes` (configurable per-emotion strategy)
3. Feature extraction per segment: 40 MFCC + 40 Mel-spectrogram (dB) → `(timesteps, 80)`
4. Pad/truncate to `max_len=100` → `X` shape `(N, 100, 80)`
5. 3-way train/val/test split (stratified); inverse-frequency class weights on train set

**Model:**
```
Input(100, 80)
→ Conv1D(64, k=3) → BN → MaxPool(2) → Dropout
→ Conv1D(128, k=3) → BN → MaxPool(2) → Dropout
→ BiLSTM(128, return_sequences=True) → Dropout
→ BiLSTM(64) → Dropout
→ Dense(n_classes, softmax)
```
L2 regularization on all learnable layers. Adam, categorical crossentropy.

**Callbacks:** `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` — all monitor `val_loss` on the **validation split** (separate from test set).

## Script roles

- `train.py` — entry point; orchestrates data loading, training, evaluation
- `data.py` — augmentation, feature extraction, dataset loaders (`load_ravdess`, `load_crema`)
- `model.py` — `build_model()` only; swap architecture here without touching anything else
- `config.py` — config loader (yaml + CLI + env overrides)
- `config.yaml` — all hyperparameters and dataset path stubs
- `BitirmeSON.py` — retired; redirects to `train.py`
- `RAV_CREMA_RawData_Organisation.py` — legacy exploratory script, not part of the pipeline

## Adding a new dataset

1. Add its path key under `data:` in `config.yaml`
2. Register the CLI flag + env var in `config.py` → `_PATH_OVERRIDES`
3. Add its emotion map constant and a `load_<name>(path)` function in `data.py`
4. Call the loader inside `load_all_data()` in `train.py` and concatenate into `data`/`labels`
