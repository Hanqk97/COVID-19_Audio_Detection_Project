import os
import pandas as pd
import librosa
import numpy as np
import soundfile as sf
from torch_audiomentations import Compose, AddColoredNoise, PitchShift
import torch

# Paths
AUDIO_PATH = 'Data/AUDIO'  # Path to the original audio directory
AUGMENTED_OUTPUT_PATH = 'Data/Augmented'  # Path for augmented data
os.makedirs(AUGMENTED_OUTPUT_PATH, exist_ok=True)

# Load metadata and filter for COVID-positive samples
metadata = pd.read_csv('metadata_update.csv')
sample_rate = 44100  # Set sample rate for consistency

# Define WavAugment transformations with output_type and sample_rate
wav_augment = Compose(
    transforms=[
        AddColoredNoise(p=0.5, output_type='tensor'),
        PitchShift(min_transpose_semitones=-2, max_transpose_semitones=2, p=0.5, sample_rate=sample_rate, output_type='tensor')
    ]
)

# Alternative time stretching using librosa
def apply_librosa_time_stretch(y, rate=1.2):
    return librosa.effects.time_stretch(y, rate=rate)

# Function to apply WavAugment and optional time stretching
def augment_and_save(file_path, output_dir, sr=sample_rate):
    y, _ = librosa.load(file_path, sr=sr)
    print(f"Loaded file: {file_path}")
    
    # Apply WavAugment transformations
    waveform = torch.tensor(y).unsqueeze(0)  # Convert to tensor format for WavAugment
    y_augmented = wav_augment(waveform, sr).squeeze().numpy()

    # Apply additional time stretching using librosa
    y_time_stretched = apply_librosa_time_stretch(y, rate=1.2)  # Stretch by 20%

    # Save original, augmented, and time-stretched files
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    original_path = os.path.join(output_dir, f"{file_name}.flac")
    augmented_path = os.path.join(output_dir, f"{file_name}_augmented.flac")
    time_stretched_path = os.path.join(output_dir, f"{file_name}_time_stretched.flac")
    
    sf.write(original_path, y, sr)
    print(f"Saved original: {original_path}")
    sf.write(augmented_path, y_augmented, sr)
    print(f"Saved augmented: {augmented_path}")
    sf.write(time_stretched_path, y_time_stretched, sr)
    print(f"Saved time-stretched: {time_stretched_path}")

    return original_path, augmented_path, time_stretched_path

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
            # Augment positive samples and save original, augmented, and time-stretched files
            original_path, augmented_path, time_stretched_path = augment_and_save(file_path, output_dir)
            
            # Add metadata for original, augmented, and time-stretched files
            metadata_augmented.append([row['SUB_ID'], row['COVID_STATUS'], row['GENDER'], 0])  # Original
            metadata_augmented.append([f"{row['SUB_ID']}_augmented", row['COVID_STATUS'], row['GENDER'], 1])  # Augmented
            metadata_augmented.append([f"{row['SUB_ID']}_time_stretched", row['COVID_STATUS'], row['GENDER'], 1])  # Time-stretched
            
            processed_count += 1
            print(f"Processed {processed_count}/{total_files} (augmented positive sample).")
        else:
            # For negative samples, simply copy the original file
            y, sr = librosa.load(file_path, sr=44100)
            sf.write(os.path.join(output_dir, f"{row['SUB_ID']}.flac"), y, sr)
            metadata_augmented.append([row['SUB_ID'], row['COVID_STATUS'], row['GENDER'], 0])
            processed_count += 1
            print(f"Processed {processed_count}/{total_files} (negative sample).")
    else:
        print(f"File not found: {file_path}")

# Save metadata_augmented.csv
metadata_df = pd.DataFrame(metadata_augmented, columns=['SUB_ID', 'COVID_STATUS', 'GENDER', 'augmented'])
metadata_df.to_csv('metadata_augmented.csv', index=False)

print("Augmentation and metadata generation complete.")
