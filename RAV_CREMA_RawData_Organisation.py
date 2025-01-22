#
import pandas as pd
import numpy as np
import tensorflow as tf 
import os
import sys
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM,BatchNormalization , GRU
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


ravdess = r"C:\Users\melik\Desktop\rvcrma\ravdessraw\\"
ravdess_emotions = os.listdir(ravdess)
print(ravdess_emotions)

rv_emotion = []
rv_path = []
for i in ravdess_emotions:
    actor = os.listdir(ravdess + i)
    for f in actor:
        part = f.split('.')[0].split('-')
        rv_emotion.append(int(part[2]))
        rv_path.append(ravdess + i + '/' + f)
        print(actor[0])
print(part[0])
print(rv_path[0])
print(int(part[2]))
print(f)

emotion_df = pd.DataFrame(rv_emotion, columns=['Emotions'])
path_df = pd.DataFrame(rv_path, columns=['Path'])
rv_df = pd.concat([emotion_df, path_df], axis=1)
rv_df.Emotions.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust',
                             8:'surprise'},
                            inplace=True)

print("|_____________________________________________________________|")
print(rv_df.Emotions.value_counts())
print("|_____________________________________________________________|")



crema = r"C:\Users\melik\Desktop\rvcrma\cremaraw\\"
crema_directory_list = os.listdir(crema)
file_emotion = []
file_path = []
for file in crema_directory_list:
    file_path.append(crema + file)
    part=file.split('_')
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')
        
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
path_df = pd.DataFrame(file_path, columns=['Path'])
crema_df = pd.concat([emotion_df, path_df], axis=1)
crema_df.head()
print(crema_df.Emotions.value_counts())
print("|_____________________________________________________________|")

combined_path = pd.concat([rv_df, crema_df], axis = 0)
combined_path.to_csv("combined_path.csv",index=False)
combined_path.head()

plt.title('Count of Emotions', size=16)
sns.countplot(combined_path.Emotions)
plt.ylabel('Count', size=12)
plt.xlabel('Emotions', size=12)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.show()
data,sr = librosa.load(file_path[0])
sr
