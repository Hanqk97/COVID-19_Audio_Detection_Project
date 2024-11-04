import os
import pandas as pd
import librosa
import numpy as np
import soundfile as sf
from torch_audiomentations import Compose, AddColoredNoise, PitchShift, TimeStretch
import torch

# Paths
AUDIO_PATH = 'Data/AUDIO'  # Path to the original audio directory
AUGMENTED_OUTPUT_PATH = 'Data/Augmented'  # Path for augmented data
os.makedirs(AUGMENTED_OUTPUT_PATH, exist_ok=True)

# Load metadata and filter for COVID-positive samples
metadata = pd.read_csv('metadata_update.csv')

# Define WavAugment transformations
wav_augment = Compose([
    AddColoredNoise(p=0.5),
    PitchShift(min_transpose_semitones=-2, max_transpose_semitones=2, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5)
])

# Function to augment a single audio file
def apply_wav_augment(y, sr):
    waveform = torch.tensor(y).unsqueeze(0)  # Convert to tensor format for WavAugment
    augmented_waveform = wav_augment(waveform, sr)
    return augmented_waveform.squeeze().numpy()

def augment_and_save(file_path, output_dir, sr=44100):
    y, _ = librosa.load(file_path, sr=sr)
    y_augmented = apply_wav_augment(y, sr)  # Apply WavAugment

    # Save original and augmented files in respective positive or negative folders
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    original_path = os.path.join(output_dir, f"{file_name}.flac")
    augmented_path = os.path.join(output_dir, f"{file_name}_augmented.flac")
    sf.write(original_path, y, sr)
    sf.write(augmented_path, y_augmented, sr)
    return original_path, augmented_path

# Set up metadata storage
metadata_augmented = []
total_files = len(metadata)
processed_count = 0

# Process each audio file based on metadata and save in structured folders
for _, row in metadata.iterrows():
    file_type = 'breathing' if 'breathing' in row['SUB_ID'] else 'cough' if 'cough' in row['SUB_ID'] else 'speech'
    covid_status = 'positive' if row['COVID_STATUS'] == 'p' else 'negative'
    output_dir = os.path.join(AUGMENTED_OUTPUT_PATH, file_type, covid_status)
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(AUDIO_PATH, f"{row['SUB_ID']}.flac")
    if os.path.exists(file_path):
        if covid_status == 'positive':
            # Augment positive samples and save both original and augmented
            original_path, augmented_path = augment_and_save(file_path, output_dir)
            # Add metadata for both original and augmented files
            metadata_augmented.append([row['SUB_ID'], row['COVID_STATUS'], row['GENDER'], 0])  # Original
            metadata_augmented.append([f"{row['SUB_ID']}_augmented", row['COVID_STATUS'], row['GENDER'], 1])  # Augmented
            processed_count += 1
            print(f"Processed {processed_count}/{total_files} (augmented positive sample).")
        else:
            # For negative samples, simply copy the original file
            y, sr = librosa.load(file_path, sr=44100)
            sf.write(os.path.join(output_dir, f"{row['SUB_ID']}.flac"), y, sr)
            metadata_augmented.append([row['SUB_ID'], row['COVID_STATUS'], row['GENDER'], 0])
            processed_count += 1
            print(f"Processed {processed_count}/{total_files} (negative sample).")

# Save metadata_augmented.csv
metadata_df = pd.DataFrame(metadata_augmented, columns=['SUB_ID', 'COVID_STATUS', 'GENDER', 'augmented'])
metadata_df.to_csv('metadata_augmented.csv', index=False)

print("Augmentation and metadata generation complete.")
