import os
import pandas as pd
import librosa
import numpy as np

# Base directory and output paths
BASE_DIR = '/Users/jonahgloss/Downloads/Data/Processed'
folders = ['Speech', 'Breathing', 'Cough']
CSV_OUTPUT = '/Users/jonahgloss/Desktop/features.csv'

# Helper function to extract features from audio files
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    features = {
        'mfcc_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1).tolist(),
        'zcr': np.mean(librosa.feature.zero_crossing_rate(y)),
        'rms': np.mean(librosa.feature.rms(y=y)),
        'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
    }
    flat_features = {k: v if isinstance(v, float) else np.mean(v) for k, v in features.items()}
    return flat_features

# Main script to process files
data = []
for folder in folders:
    folder_path = os.path.join(BASE_DIR, folder)
    for subfolder in ['positive', 'negative']:
        subfolder_path = os.path.join(folder_path, subfolder)
        label = subfolder  # positive or negative
        class_type = folder.lower()  # speech, breathing, cough

        for file in os.listdir(subfolder_path):
            if file.endswith('.flac'):
                file_path = os.path.join(subfolder_path, file)
                try:
                    features = extract_features(file_path)
                    features['filename'] = file
                    features['class_type'] = class_type
                    features['label'] = label
                    data.append(features)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Save features to CSV
df = pd.DataFrame(data)
df.to_csv(CSV_OUTPUT, index=False)
print(f"Features saved to {CSV_OUTPUT}")

