import librosa
import numpy as np

def load_audio(file_path, sr=44100):
    return librosa.load(file_path, sr=sr)

def normalize_features(features):
    return (features - np.mean(features, axis=0)) / np.std(features, axis=0)
