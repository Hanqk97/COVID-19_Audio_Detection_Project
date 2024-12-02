import os
import librosa
import shutil
import torch
import soundfile as sf
from torch_audiomentations import Compose, AddColoredNoise, PitchShift

# Define paths
BASE_PATH = 'data/reclassified'
OUTPUT_PATH = 'data/augmented'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Augmentation pipeline (without TimeStretch)
augmentation_pipeline = Compose(
    transforms=[
        AddColoredNoise(p=0.5),
        PitchShift(min_transpose_semitones=-3, max_transpose_semitones=3, p=0.5, sample_rate=44100)
    ]
)

# Function to apply time stretching using librosa
def apply_time_stretch(audio, rate):
    return librosa.effects.time_stretch(audio, rate)

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
def augment_audio(file_path, output_dir, subject_id, num_augmentations=4):
    """Applies augmentation to an audio file and saves augmented versions."""
    y, sr = librosa.load(file_path, sr=44100)
    for i in range(num_augmentations):
        # Apply augmentations
        augmented_audio = augmentation_pipeline(torch.tensor(y).unsqueeze(0).unsqueeze(0), sr).squeeze().numpy()
        
        # Add time stretching
        stretch_rate = 1.0 + (np.random.uniform(-0.2, 0.2))
        augmented_audio = apply_time_stretch(augmented_audio, rate=stretch_rate)

        # Save augmented file
        augmented_file_name = f"{subject_id}_AUG{i + 1}.flac"
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
        subject_output_path = os.path.join(output_dir, subject)
        os.makedirs(subject_output_path, exist_ok=True)

        # Perform augmentation if needed
        if current_count < target_count:
            for file in os.listdir(subject_path):
                if file.endswith('.flac'):
                    augment_audio(os.path.join(subject_path, file), subject_output_path, subject)
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