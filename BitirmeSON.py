import librosa
import librosa.display
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D, 
                                     BatchNormalization, Bidirectional, LSTM, Input)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score

SAMPLE_RATE = 22050
SEGMENT_DURATION = 2
SAMPLES_PER_SEGMENT = SEGMENT_DURATION * SAMPLE_RATE
max_len = 100
n_mfcc = 40
n_mels = 40

weak_classes = ["surprise", "fear", "happy","disgust"]

ravdess_path = r"C:\Users\melik\Desktop\rvcrma\ravdessraw\\"
crema_path = r"C:\Users\melik\Desktop\rvcrma\cremaraw\\"

emotion_map_ravdess = {
    1: 'neutral', 2: 'neutral', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'
}

emotion_map_crema = {
    'SAD': 'sad', 'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear', 'HAP': 'happy', 'NEU': 'neutral'
}

def add_background_noise(audio, noise_factor=0.003):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def pitch_shift_audio(audio, sr=SAMPLE_RATE, n_steps=2):
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def time_stretch_audio(audio, rate=1.1):
    return librosa.effects.time_stretch(y=audio, rate=rate)

def targeted_augmentation(audio, label):
    if label == "happy":
        audio = pitch_shift_audio(audio, sr=SAMPLE_RATE, n_steps=2)
    elif label == "sad":
        audio = time_stretch_audio(audio, rate=0.9)
    elif label == "fearful":
        audio = pitch_shift_audio(audio, sr=SAMPLE_RATE, n_steps=-2)
    elif label == "disgust":
        audio = time_stretch_audio(audio, rate=1.1)
    elif label == "neutral":
        audio = add_background_noise(audio)
    return audio

def extract_features(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=n_mfcc).T
    melspec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=n_mels)
    melspec_db = librosa.power_to_db(melspec, ref=np.max).T
    min_len = min(mfccs.shape[0], melspec_db.shape[0])
    return np.concatenate((mfccs[:min_len], melspec_db[:min_len]), axis=1)

def process_audio_file(file_path, label):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    total_samples = len(audio)
    features = []

    for start in range(0, total_samples, SAMPLES_PER_SEGMENT):
        end = start + SAMPLES_PER_SEGMENT
        segment = audio[start:end]
        if len(segment) < SAMPLES_PER_SEGMENT:
            segment = np.pad(segment, (0, SAMPLES_PER_SEGMENT - len(segment)), mode='constant')
        if label in weak_classes:
            segment = targeted_augmentation(segment, label)
        features.append(extract_features(segment))
    return features

def load_dataset(path, emotion_map, is_crema=False):
    data, labels = [], []
    if is_crema:
        for file in os.listdir(path):
            if file.endswith(".wav"):
                emotion = emotion_map[file.split('_')[2]]
                file_path = os.path.join(path, file)
                features = process_audio_file(file_path, emotion)
                for feat in features:
                    data.append(feat)
                    labels.append(emotion)
    else:
        for folder in os.listdir(path):
            for file in os.listdir(os.path.join(path, folder)):
                if file.endswith(".wav"):
                    emotion = emotion_map[int(file.split('-')[2])]
                    file_path = os.path.join(path, folder, file)
                    features = process_audio_file(file_path, emotion)
                    for feat in features:
                        data.append(feat)
                        labels.append(emotion)
    return data, labels

data_ravdess, labels_ravdess = load_dataset(ravdess_path, emotion_map_ravdess)
data_crema, labels_crema = load_dataset(crema_path, emotion_map_crema, is_crema=True)

data = data_ravdess + data_crema
labels = labels_ravdess + labels_crema

X = pad_sequences(data, maxlen=max_len, padding='post', dtype='float32', truncating='post')
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

class_counts = np.sum(y_train, axis=0)
total = np.sum(class_counts)
class_weights = {i: total/(len(class_counts)*class_counts[i]) for i in range(len(class_counts))}
print(f"class weights: {class_weights}")

regularizer = l2(0.0001)
inp = Input(shape=(max_len, n_mfcc + n_mels))

x = Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=regularizer)(inp)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)

x = Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=regularizer)(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)

x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Dropout(0.3)(x)

x = Bidirectional(LSTM(64))(x)
x = Dropout(0.3)(x)

out = Dense(y.shape[1], activation='softmax', kernel_regularizer=regularizer)(x)

model = Model(inputs=inp, outputs=out)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
checkpoint = ModelCheckpoint(
    'best_model_mfcc_melspec.keras',
    monitor='val_loss',           
    save_best_only=True,        
    mode='min'                     
)

history = model.fit(
    X_train, y_train,
    epochs=100,  
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

model.load_weights('best_model_mfcc_melspec.keras')

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"Model Accuracy: {accuracy}\n")
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test_classes, y_pred_classes)
class_names = label_encoder.classes_

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.title('Eğitim + Doğrulama Doğruluğu')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Test Loss')
plt.legend()

plt.tight_layout()
plt.show()

f1 = f1_score(y_test_classes, y_pred_classes, average=None)
print(f"F1-Score: {f1}")