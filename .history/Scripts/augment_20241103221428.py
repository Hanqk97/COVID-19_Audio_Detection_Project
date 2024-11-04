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

# Load metadata
metadata = pd.read_csv('Data/metadata_update.csv')
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
def augment_audio(y, sr):
    waveform = torch.tensor(y).unsqueeze(0)  # Convert to tensor format for WavAugment
    y_augmented = wav_augment(waveform, sr).squeeze().numpy()
    y_time_stretched = apply_librosa_time_stretch(y, rate=1.2)  # Stretch by 20%
    return y_augmented, y_time_stretched

# Set up metadata storage
metadata_augmented = []
total_files = len(metadata)
processed_count = 0

# Process each subject in metadata
for _, row in metadata.iterrows():
    subject_id = row['SUB_ID']
    covid_status = 'positive' if row['COVID_STATUS'] == 'p' else 'negative'
    gender = row['GENDER']
    
    # Define output directory for this subject based on COVID status
    for file_type in ['breathing', 'cough', 'speech']:
        file_path = os.path.join(AUDIO_PATH, file_type, f"{subject_id}.flac")
        output_dir = os.path.join(AUGMENTED_OUTPUT_PATH, file_type, covid_status)
        os.makedirs(output_dir, exist_ok=True)
        
        if os.path.exists(file_path):
            if covid_status == 'positive':
                # Load the original file
                y, sr = librosa.load(file_path, sr=sample_rate)
                
                # Save original positive file
                original_path = os.path.join(output_dir, f"{subject_id}.flac")
                sf.write(original_path, y, sr)
                metadata_augmented.append([subject_id, row['COVID_STATUS'], gender, file_type, 0])  # Original
                
                # Generate augmentations for positive samples
                for i in range(1, 4):  # Generate three augmented versions for balancing
                    y_augmented, y_time_stretched = augment_audio(y, sr)
                    
                    # Save augmented files
                    augmented_path = os.path.join(output_dir, f"{subject_id}_aug_{i}.flac")
                    time_stretched_path = os.path.join(output_dir, f"{subject_id}_time_stretched_{i}.flac")
                    
                    sf.write(augmented_path, y_augmented, sr)
                    sf.write(time_stretched_path, y_time_stretched, sr)
                    
                    # Add metadata for augmented files
                    metadata_augmented.append([f"{subject_id}_aug_{i}", row['COVID_STATUS'], gender, file_type, 1])
                    metadata_augmented.append([f"{subject_id}_time_stretched_{i}", row['COVID_STATUS'], gender, file_type, 1])
                    
                processed_count += 1
                print(f"Processed {processed_count}/{total_files} (positive sample with augmentations).")
            
            else:
                # For negative samples, simply copy the original file
                y, sr = librosa.load(file_path, sr=sample_rate)
                negative_output_path = os.path.join(output_dir, f"{subject_id}.flac")
                sf.write(negative_output_path, y, sr)
                
                # Add metadata for original negative file
                metadata_augmented.append([subject_id, row['COVID_STATUS'], gender, file_type, 0])
                processed_count += 1
                print(f"Processed {processed_count}/{total_files} (negative sample).")
        else:
            print(f"File not found: {file_path}")

# Save metadata_augmented.csv in the Data directory
metadata_df = pd.DataFrame(metadata_augmented, columns=['SUB_ID', 'COVID_STATUS', 'GENDER', 'TYPE', 'augmented'])
metadata_df.to_csv('Data/metadata_augmented.csv', index=False)

print("Augmentation and metadata generation complete.")
