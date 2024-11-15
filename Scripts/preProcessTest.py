import os
import pandas as pd
import librosa
import numpy as np
import soundfile as sf
from torch_audiomentations import Compose, AddColoredNoise, PitchShift
import torch

# Paths
AUDIO_PATH = 'Data/AUDIO'
AUGMENTED_OUTPUT_PATH = 'Data/Augmented'
os.makedirs(AUGMENTED_OUTPUT_PATH, exist_ok=True)

# Load metadata
metadata = pd.read_csv('Data/metadata_update.csv')
sample_rate = 44100

# Augmentation setup
wav_augment = Compose([
    AddColoredNoise(p=0.5, output_type='tensor'),
    PitchShift(min_transpose_semitones=-3, max_transpose_semitones=3, p=0.5, sample_rate=sample_rate, output_type='tensor')
])

def apply_librosa_time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)

def augment_audio(y, sr):
    y_tensor = torch.tensor(y).unsqueeze(0).unsqueeze(0)
    y_augmented = wav_augment(y_tensor, sr).squeeze().numpy()
    stretch_rate = 1.0 + np.random.uniform(-0.2, 0.2)
    return apply_librosa_time_stretch(y_augmented, rate=stretch_rate)

# Augment data
metadata_augmented = []
for _, row in metadata.iterrows():
    subject_id = row['SUB_ID']
    covid_status = row['COVID_STATUS']
    for file_type in ['breathing', 'cough', 'speech']:
        file_path = os.path.join(AUDIO_PATH, file_type, f"{subject_id}.flac")
        if os.path.exists(file_path):
            y, sr = librosa.load(file_path, sr=sample_rate)
            output_dir = os.path.join(AUGMENTED_OUTPUT_PATH, file_type, covid_status)
            os.makedirs(output_dir, exist_ok=True)
            sf.write(os.path.join(output_dir, f"{subject_id}.flac"), y, sr)
            if covid_status == 'positive':
                for i in range(4):
                    y_aug = augment_audio(y, sr)
                    sf.write(os.path.join(output_dir, f"{subject_id}_aug_{i+1}.flac"), y_aug, sr)
