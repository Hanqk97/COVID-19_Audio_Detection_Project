import librosa
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

AUDIO_PATH = 'Data/Augmented'
FEATURES_PATH = 'Data/Features'
os.makedirs(FEATURES_PATH, exist_ok=True)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    return np.concatenate([mfccs.mean(axis=1), zcr.mean(), rms.mean()])

metadata = pd.read_csv('Data/metadata_augmented.csv')
features = []

for _, row in metadata.iterrows():
    file_path = os.path.join(AUDIO_PATH, row['TYPE'], row['COVID_STATUS'], f"{row['SUB_ID']}.flac")
    if os.path.exists(file_path):
        feature_vector = extract_features(file_path)
        features.append([row['SUB_ID'], row['COVID_STATUS'], *feature_vector])

columns = ['SUB_ID', 'COVID_STATUS'] + [f'feature_{i}' for i in range(15)]
pd.DataFrame(features, columns=columns).to_csv(os.path.join(FEATURES_PATH, 'features.csv'), index=False)

