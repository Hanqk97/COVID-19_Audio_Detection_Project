import os
import librosa
import shutil
import numpy as np
import torch
import soundfile as sf
from torch_audiomentations import Compose, AddColoredNoise, PitchShift

# Define paths
BASE_PATH = 'data/segmented'
OUTPUT_PATH = 'data/augmented'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Augmentation pipeline
augmentation_pipeline = Compose(
    transforms=[
        AddColoredNoise(p=0.5),
        PitchShift(min_transpose_semitones=-3, max_transpose_semitones=3, p=0.5, sample_rate=44100)
    ]
)

def apply_time_stretch(audio, rate, sr=44100):
    """
    Applies time stretching to a waveform read from a .flac file.

    Args:
        audio (numpy.ndarray): Audio waveform.
        rate (float): Stretching factor (e.g., 1.2 for 20% faster, 0.8 for 20% slower).
        sr (int): Sampling rate of the audio.

    Returns:
        numpy.ndarray: Time-stretched audio waveform.
    """
    if rate <= 0:
        raise ValueError("Stretching rate must be positive.")

    # Convert waveform to spectrogram (STFT)
    spectrogram = librosa.stft(audio)

    # Apply time stretching to the spectrogram
    stretched_spectrogram = librosa.phase_vocoder(spectrogram, rate)

    # Convert back to audio waveform
    stretched_audio = librosa.istft(stretched_spectrogram)
    return stretched_audio

# Initialize counters
counts = {
    'positive': {'male': 0, 'female': 0},
    'negative': {'male': 0, 'female': 0},
}

# Function to count subfolders
def count_subjects_and_copy(base_path, output_path, covid_status, gender):
    """Count subjects and copy files."""
    gender_path = os.path.join(base_path, covid_status, gender)
    output_gender_path = os.path.join(output_path, covid_status, gender)
    os.makedirs(output_gender_path, exist_ok=True)

    if os.path.exists(gender_path):
        subject_folders = [f for f in os.listdir(gender_path) if os.path.isdir(os.path.join(gender_path, f))]
        counts[covid_status][gender] += len(subject_folders)

        # Copy original files
        for i, subject in enumerate(subject_folders, 1):
            subject_input_path = os.path.join(gender_path, subject)
            subject_output_path = os.path.join(output_gender_path, subject)
            os.makedirs(subject_output_path, exist_ok=True)

            for file in os.listdir(subject_input_path):
                if file.endswith('.flac'):
                    shutil.copy(os.path.join(subject_input_path, file), subject_output_path)

            # Print progress
            print(f"Copied {i}/{len(subject_folders)} {gender} {covid_status} subjects.")

        return subject_folders
    return []

# Count positive and negative subjects, and copy data
positive_male_subjects = count_subjects_and_copy(BASE_PATH, OUTPUT_PATH, 'positive', 'male')
positive_female_subjects = count_subjects_and_copy(BASE_PATH, OUTPUT_PATH, 'positive', 'female')
negative_male_subjects = count_subjects_and_copy(BASE_PATH, OUTPUT_PATH, 'negative', 'male')
negative_female_subjects = count_subjects_and_copy(BASE_PATH, OUTPUT_PATH, 'negative', 'female')

# Calculate augmentation targets
total_negatives = len(negative_male_subjects) + len(negative_female_subjects)
target_positives = total_negatives
total_positives = len(positive_male_subjects) + len(positive_female_subjects)
augment_needed = target_positives - total_positives

# Maintain the male:female ratio in augmentation
male_ratio = len(positive_male_subjects) / total_positives
female_ratio = len(positive_female_subjects) / total_positives

target_positive_male = int(round(male_ratio * target_positives))
target_positive_female = target_positives - target_positive_male

# Augmentation function
def augment_audio(file_path, output_dir, subject_id, file_type, aug_index):
    """Applies augmentation to an audio file and saves augmented versions."""
    y, sr = librosa.load(file_path, sr=44100)
    augmented_audio = augmentation_pipeline(torch.tensor(y).unsqueeze(0).unsqueeze(0), sr).squeeze().numpy()
    
    # Add time stretching
    stretch_rate = 1.0 + (np.random.uniform(-0.2, 0.2))
    augmented_audio = apply_time_stretch(augmented_audio, rate=stretch_rate)

    # Save augmented file
    augmented_file_name = f"{subject_id}_AUG{aug_index}_{file_type}.flac"
    augmented_path = os.path.join(output_dir, augmented_file_name)
    sf.write(augmented_path, augmented_audio, sr)
    print(f"Saved augmented file: {augmented_path}")

# Balance the positive samples while maintaining the ratio
def balance_and_augment(subjects, target_count, gender, covid_status):
    """Balance the dataset by augmenting subjects."""
    output_dir = os.path.join(OUTPUT_PATH, covid_status, gender)
    current_count = len(subjects)

    for i, subject in enumerate(subjects, 1):
        subject_path = os.path.join(BASE_PATH, covid_status, gender, subject)

        # Augment each file type for this subject
        for aug_index in range(1, 5):  # Generate up to 4 augmentations
            augmented_subject_id = f"{subject}_AUG{aug_index}"
            augmented_subject_path = os.path.join(output_dir, augmented_subject_id)
            os.makedirs(augmented_subject_path, exist_ok=True)

            for file_type in ['breathing', 'cough', 'speech']:
                file_path = os.path.join(subject_path, f"{subject}_{file_type}.flac")
                if os.path.exists(file_path):
                    augment_audio(file_path, augmented_subject_path, subject, file_type, aug_index)

            current_count += 1
            if current_count >= target_count:
                break

        # Print progress
        print(f"Processed {i}/{len(subjects)} {gender} {covid_status} subjects.")

# Augment positive samples
balance_and_augment(positive_male_subjects, target_positive_male, 'male', 'positive')
balance_and_augment(positive_female_subjects, target_positive_female, 'female', 'positive')

# Print final results
print(f"Positive male: Original={len(positive_male_subjects)}, Augmented target={target_positive_male}")
print(f"Positive female: Original={len(positive_female_subjects)}, Augmented target={target_positive_female}")
print(f"Negative male: {len(negative_male_subjects)}")
print(f"Negative female: {len(negative_female_subjects)}")
print("Dataset augmentation and balancing complete!")
