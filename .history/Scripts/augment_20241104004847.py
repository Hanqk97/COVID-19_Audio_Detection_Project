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

# Count positive and negative samples from the metadata
total_positive = metadata[metadata['COVID_STATUS'] == 'p'].shape[0]
total_negative = metadata[metadata['COVID_STATUS'] == 'n'].shape[0]
target_positive_count = total_negative  # We need exactly as many positives as negatives, i.e., 793
augment_count_needed = target_positive_count - total_positive  # Required number of additional subjects

# Define WavAugment transformations with output_type and sample_rate
wav_augment = Compose(
    transforms=[
        AddColoredNoise(p=0.5, output_type='tensor'),
        PitchShift(min_transpose_semitones=-3, max_transpose_semitones=3, p=0.5, sample_rate=sample_rate, output_type='tensor')
    ]
)

# Alternative time stretching using librosa
def apply_librosa_time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)

# Function to apply a single augmentation to create a distinct variation
def augment_audio(y, sr):
    # Convert the audio to the required shape [batch_size, num_channels, num_samples]
    y_tensor = torch.tensor(y).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, num_samples]
    
    # Apply WavAugment transformations
    y_augmented = wav_augment(y_tensor, sr).squeeze().numpy()  # Remove batch and channel dimensions
    
    # Vary the time stretch rate to add further distinction
    stretch_rate = 1.0 + (np.random.uniform(-0.2, 0.2))
    y_time_stretched = apply_librosa_time_stretch(y_augmented, rate=stretch_rate)
    
    return y_time_stretched

# Set up metadata storage
metadata_augmented = []
total_positives_created = total_positive  # Start with the current count of positive samples

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
            y, sr = librosa.load(file_path, sr=sample_rate)
            
            if covid_status == 'positive':
                # Save original positive file
                original_path = os.path.join(output_dir, f"{subject_id}.flac")
                sf.write(original_path, y, sr)
                metadata_augmented.append([subject_id, row['COVID_STATUS'], gender, file_type, 0])  # Original
                
                # Determine required number of augmentations for this subject
                remaining_to_create = target_positive_count - total_positives_created
                num_augmentations = min(4, remaining_to_create)  # Start with up to 4 augmentations
                
                # Generate augmentations for this subject
                for i in range(num_augmentations):
                    if total_positives_created >= target_positive_count:
                        break  # Stop if we reach the target positive count
                        
                    augmented_version = augment_audio(y, sr)
                    augmented_subject_id = f"{subject_id}_aug_{i + 1}"
                    augmented_path = os.path.join(output_dir, f"{augmented_subject_id}_{file_type}.flac")
                    sf.write(augmented_path, augmented_version, sr)
                    metadata_augmented.append([augmented_subject_id, row['COVID_STATUS'], gender, file_type, 1])
                    
                    if file_type == 'speech':  # Increment count only after all 3 types are done for a subject
                        total_positives_created += 1
                
                print(f"Processed positive sample: {subject_id} with {num_augmentations} augmentations.")
            else:
                # For negative samples, simply copy the original file
                negative_output_path = os.path.join(output_dir, f"{subject_id}.flac")
                sf.write(negative_output_path, y, sr)
                metadata_augmented.append([subject_id, row['COVID_STATUS'], gender, file_type, 0])
                print(f"Processed negative sample: {subject_id} - {file_type}")
        else:
            print(f"File not found: {file_path}")

# Save metadata_augmented.csv in the Data directory
metadata_df = pd.DataFrame(metadata_augmented, columns=['SUB_ID', 'COVID_STATUS', 'GENDER', 'TYPE', 'augmented'])
metadata_df.to_csv('Data/metadata_augmented.csv', index=False)

print("Augmentation and metadata generation complete.")
print(f"Total positive subjects now: {total_positives_created}, total target: {target_positive_count}")
