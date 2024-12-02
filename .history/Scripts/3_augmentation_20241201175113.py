import os
import json
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

# Utility functions for augmentation
def add_echo(audio, sr, delay=0.2, decay=0.4):
    n_delay = int(sr * delay)
    echo_signal = np.zeros_like(audio)
    echo_signal[n_delay:] = audio[:-n_delay] * decay
    return audio + echo_signal

def resample_audio(audio, orig_sr, target_sr):
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

def shift_audio(audio, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))
    if shift > 0:
        return np.r_[audio[shift:], np.zeros(shift)]
    else:
        return np.r_[np.zeros(-shift), audio[:shift]]

# Augment audio function
def augment_audio(file_path, output_dir, subject_id, file_type, aug_index):
    try:
        y, sr = librosa.load(file_path, sr=44100)

        augmented_audio = augmentation_pipeline(
            torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0), sr
        ).squeeze().numpy()

        augmented_audio = add_echo(augmented_audio, sr)
        target_sr = np.random.choice([32000, 44100, 48000])
        augmented_audio = resample_audio(augmented_audio, sr, target_sr)
        augmented_audio = shift_audio(augmented_audio)

        augmented_file_name = f"{subject_id}_AUG{aug_index}_{file_type}.flac"
        sf.write(os.path.join(output_dir, augmented_file_name), augmented_audio, target_sr)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Calculate target augmentation
def calculate_targets(positive_male, positive_female, negative_male, negative_female):
    total_negatives = negative_male + negative_female
    male_ratio = positive_male / (positive_male + positive_female)
    female_ratio = positive_female / (positive_male + positive_female)

    target_positive_male = int(round(total_negatives * male_ratio))
    target_positive_female = total_negatives - target_positive_male

    return target_positive_male, target_positive_female

# Main augmentation function
def balance_and_augment(subjects, target_count, gender, covid_status):
    output_dir = os.path.join(OUTPUT_PATH, covid_status, gender)
    os.makedirs(output_dir, exist_ok=True)
    current_count = len(subjects)

    for subject in subjects:
        subject_path = os.path.join(BASE_PATH, covid_status, gender, subject)
        if current_count >= target_count:
            break

        for file_type in ["breathing", "cough", "speech"]:
            search_pattern = os.path.join(subject_path, f"{subject}*{file_type}*.flac")
            matching_files = glob.glob(search_pattern)

            if matching_files:
                file_path = matching_files[0]
                for aug_index in range(1, 5):  # Generate up to 4 augmentations per file
                    if current_count >= target_count:
                        break
                    augment_audio(file_path, output_dir, subject, file_type, aug_index)
                    current_count += 1

# Count and copy subjects
def count_subjects(base_path, covid_status, gender):
    path = os.path.join(base_path, covid_status, gender)
    if os.path.exists(path):
        return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    return []

# Count original data
positive_male_subjects = count_subjects(BASE_PATH, 'positive', 'male')
positive_female_subjects = count_subjects(BASE_PATH, 'positive', 'female')
negative_male_subjects = count_subjects(BASE_PATH, 'negative', 'male')
negative_female_subjects = count_subjects(BASE_PATH, 'negative', 'female')

# Calculate target counts
target_positive_male, target_positive_female = calculate_targets(
    len(positive_male_subjects), len(positive_female_subjects),
    len(negative_male_subjects), len(negative_female_subjects)
)

# Augment data to balance
balance_and_augment(positive_male_subjects, target_positive_male, "male", "positive")
balance_and_augment(positive_female_subjects, target_positive_female, "female", "positive")

# Final report
print(f"Final counts:")
print(f"Positive male: Original={len(positive_male_subjects)}, Augmented target={target_positive_male}")
print(f"Positive female: Original={len(positive_female_subjects)}, Augmented target={target_positive_female}")
print(f"Negative male: {len(negative_male_subjects)}")
print(f"Negative female: {len(negative_female_subjects)}")
