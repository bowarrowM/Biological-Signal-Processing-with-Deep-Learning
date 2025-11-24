# Biological Signal Processing with Deep Learning

A deep learning project for **Speech Emotion Recognition (SER)** using a hybrid **CNN-LSTM** architecture trained on audio data from **RAVDESS** and **CREMA-D** datasets.

This model uses **MFCC + Mel-Spectrogram features**, 2-second segmented audio windows, targeted audio augmentation, and class balancing to improve robustness across emotion classes.

Language: Python

Libraries: librosa, numpy, pandas, matplotlib, seaborn, scikit-learn, tensorflow


## Features

- **Audio preprocessing** with `librosa`
- **MFCC + Mel-Spectrogram** feature extraction
- **2-second sliding window segmentation**
- **Emotion-dependent targeted augmentations**:
  - Happy → Pitch shift +2  
  - Sad → Time stretch 0.9  
  - Fear → Pitch shift −2  
  - Disgust → Time stretch 1.1  
  - Neutral → Background noise
- **CNN layers** for spatial feature extraction
- **Bidirectional LSTM layers** for temporal modeling
- **Class weighting** for imbalanced dataset
- **Model checkpointing** + early stopping
- **Evaluation** with accuracy, classification report, F1, and confusion matrix

---

## Dataset Structure

The project uses:
- **RAVDESS**  
- **CREMA-D**

## Audio Processing Pipeline
- All audio is resampled to 22,050 Hz.

- Each window is padded if too short.

- Applied targeted emotional augmentation for weaker classes: "surprise", "fear", "happy", "disgust"


## Feature extraction

For each audio segment:

- 40 MFCC coefficients

- 40 Mel filterbank features (in dB)

- Combined feature vector: (timesteps × 80 features)

## Padding

- Sequences are padded/truncated to: max_len = 100 timesteps

## Model Architecture
"""

Input → 
Conv1D(64) → BN → MaxPool → Dropout
Conv1D(128) → BN → MaxPool → Dropout
BiLSTM(128) → Dropout
BiLSTM(64) → Dropout
Dense(softmax)

Regularization: l2(0.0001)

Optimizer: Adam (lr=0.0003)

Loss: Categorical crossentropy

"""

## Training

Training uses:

Early stopping

Learning rate reduction

Best model checkpoint:

## Model Results

Accuracy : 0.74

"""
F1 scores:

ANGRY: 0.73
DİSGUST: 1.00
FEAR: 0.54
HAPPY: 0.97
NEUTRAL: 0.61
SAD: 0.55
SURPRISE: 0.82

"""
