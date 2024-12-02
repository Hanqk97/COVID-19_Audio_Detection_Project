import os
import librosa
import numpy as np
import torch
import glob
import soundfile as sf
from torch_audiomentations import Compose, AddColoredNoise, PitchShift
import shutil

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

# Function: Add Echo
def add_echo(audio, sr, delay=0.2, decay=0.4):
    """Simulate echo effect."""
    n_delay = int(sr * delay)
    echo_signal = np.zeros_like(audio)
    echo_signal[n_delay:] = audio[:-n_delay] * decay
    return audio + echo_signal

# Function: Apply Resampling
def resample_audio(audio, orig_sr, target_sr):
    """Resample audio to a new sample rate."""
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


# Function: Shift Audio
def shift_audio(audio, shift_max=0.2):
    """Shift audio in time."""
    shift = np.random.uniform(-shift_max, shift_max) * len(audio)
    shift = int(shift)
    if shift > 0:
        audio = np.r_[audio[shift:], np.zeros(shift)]
    else:
        audio = np.r_[np.zeros(-shift), audio[:shift]]
    return audio

# Function to count subjects and copy files
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

            print(f"Copied {i}/{len(subject_folders)} {gender} {covid_status} subjects.")

        return subject_folders
    return []

# Augmentation function
def augment_audio(file_path, output_dir, subject_id, file_type, aug_index):
    """Applies augmentation to an audio file and saves augmented versions."""
    try:
        print(f"Processing file: {file_path}")
        # Load audio
        y, sr = librosa.load(file_path, sr=44100)

        # Apply augmentations
        augmented_audio = augmentation_pipeline(
            torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0), sr
        ).squeeze().numpy()

        # Add echo effect
        augmented_audio = add_echo(augmented_audio, sr)

        # Apply random resampling
        target_sr = np.random.choice([32000, 44100, 48000])
        augmented_audio = resample_audio(augmented_audio, sr, target_sr)

        # Apply time shifting
        augmented_audio = shift_audio(augmented_audio)

        # Save augmented file
        augmented_file_name = f"{subject_id}_AUG{aug_index}_{file_type}.flac"
        augmented_path = os.path.join(output_dir, augmented_file_name)
        sf.write(augmented_path, augmented_audio, target_sr)
        print(f"Saved augmented file: {augmented_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Initialize counters
counts = {
    'positive': {'male': 0, 'female': 0},
    'negative': {'male': 0, 'female': 0},
}

# Count subjects
positive_male_subjects = count_subjects_and_copy(BASE_PATH, OUTPUT_PATH, 'positive', 'male')
positive_female_subjects = count_subjects_and_copy(BASE_PATH, OUTPUT_PATH, 'positive', 'female')
negative_male_subjects = count_subjects_and_copy(BASE_PATH, OUTPUT_PATH, 'negative', 'male')
negative_female_subjects = count_subjects_and_copy(BASE_PATH, OUTPUT_PATH, 'negative', 'female')

# Calculate augmentation targets
total_negatives = len(negative_male_subjects) + len(negative_female_subjects)
male_ratio = len(negative_male_subjects) / total_negatives
female_ratio = len(negative_female_subjects) / total_negatives

# Calculate target counts for positive samples
target_positive_male = int(round(total_negatives * male_ratio))
target_positive_female = total_negatives - target_positive_male

# Calculate the number of augmentations needed
current_positive_male = len(positive_male_subjects)
current_positive_female = len(positive_female_subjects)

augment_male_needed = max(0, target_positive_male - current_positive_male)
augment_female_needed = max(0, target_positive_female - current_positive_female)

# Updated augmentation logic
def balance_and_augment(subjects, augment_needed, gender, covid_status):
    """Augment audio files to balance the dataset."""
    output_dir = os.path.join(OUTPUT_PATH, covid_status, gender)
    current_count = len(subjects)

    # Augment subjects only if needed
    for subject in subjects:
        if augment_needed <= 0:
            break

        subject_path = os.path.join(BASE_PATH, covid_status, gender, subject)
        for aug_index in range(1, 5):  # Generate up to 4 augmentations
            if augment_needed <= 0:
                break

            augmented_subject_id = f"{subject}_AUG{aug_index}"
            augmented_subject_path = os.path.join(output_dir, augmented_subject_id)
            os.makedirs(augmented_subject_path, exist_ok=True)

            for file_type in ['breathing', 'cough', 'speech']:
                search_pattern = os.path.join(subject_path, f"{subject}*{file_type}*.flac")
                matching_files = glob.glob(search_pattern)

                if matching_files:
                    file_path = matching_files[0]
                    print(f"Processing {file_type} file for subject {subject}: {file_path}")
                    augment_audio(file_path, augmented_subject_path, subject, file_type, aug_index)
                else:
                    print(f"No matching file found for {file_type} in {subject_path}. Skipping.")

            augment_needed -= 1

# Augment positive samples
balance_and_augment(positive_male_subjects, augment_male_needed, 'male', 'positive')
balance_and_augment(positive_female_subjects, augment_female_needed, 'female', 'positive')

# Final results
print(f"Positive male: Original={len(positive_male_subjects)}, Augmented target={target_positive_male}")
print(f"Positive female: Original={len(positive_female_subjects)}, Augmented target={target_positive_female}")
print(f"Negative male: {len(negative_male_subjects)}")
print(f"Negative female: {len(negative_female_subjects)}")
print("Dataset augmentation and balancing complete!")
