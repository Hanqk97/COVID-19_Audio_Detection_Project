import os
import librosa
import numpy as np
import soundfile as sf
from collections import defaultdict
from torch_audiomentations import Compose, AddColoredNoise, PitchShift

# Augmentation pipeline
augmentation_pipeline = Compose([
    AddColoredNoise(p=0.5),
    PitchShift(min_transpose_semitones=-2, max_transpose_semitones=2, p=0.5, sample_rate=44100)
])

# Augmentation function
def augment_audio(y, sr, num_augmentations=3):
    augmented_clips = []
    for _ in range(num_augmentations):
        augmented = augmentation_pipeline(torch.tensor(y).unsqueeze(0).unsqueeze(0), sr).squeeze().numpy()
        augmented_clips.append(augmented)
    return augmented_clips

# Fixed-length Cough Segmentation
def segment_cough(file_path, output_dir, segment_length=3, sample_rate=44100):
    """
    Segments cough audio into a single fixed-length chunk for each ID.
    """
    y, sr = librosa.load(file_path, sr=sample_rate)
    segment_samples = int(segment_length * sr)

    # Truncate or pad to fixed length
    if len(y) > segment_samples:
        # Truncate to the first `segment_length` seconds
        y = y[:segment_samples]
    elif len(y) < segment_samples:
        # Pad with silence to reach `segment_length` seconds
        padding = segment_samples - len(y)
        y = np.pad(y, (0, padding), mode="constant")

    # Save the fixed-length segment
    os.makedirs(output_dir, exist_ok=True)
    segment_path = os.path.join(output_dir, f"{os.path.basename(file_path).split('.')[0]}.flac")
    sf.write(segment_path, y, sr)
    print(f"Saved fixed-length cough segment: {segment_path}")

# Fixed-length Breathing Segmentation
def segment_breathing(file_path, output_dir, segment_length=6, sample_rate=44100):
    """
    Segments breathing audio into a single fixed-length chunk for each ID.
    """
    y, sr = librosa.load(file_path, sr=sample_rate)
    segment_samples = int(segment_length * sr)

    # Truncate or pad to fixed length
    if len(y) > segment_samples:
        # Truncate to the first `segment_length` seconds
        y = y[:segment_samples]
    elif len(y) < segment_samples:
        # Pad with silence to reach `segment_length` seconds
        padding = segment_samples - len(y)
        y = np.pad(y, (0, padding), mode="constant")

    # Save the fixed-length segment
    os.makedirs(output_dir, exist_ok=True)
    segment_path = os.path.join(output_dir, f"{os.path.basename(file_path).split('.')[0]}_fixed.flac")
    sf.write(segment_path, y, sr)
    print(f"Saved fixed-length breathing segment: {segment_path}")

# Speech processing (unchanged)
def process_speech(file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    segment_path = os.path.join(output_dir, os.path.basename(file_path))
    y, sr = librosa.load(file_path, sr=44100)
    sf.write(segment_path, y, sr)

# Count results
def count_results(output_dir):
    """
    Counts the number of positive and negative files for each audio type.
    """
    counts = defaultdict(lambda: {"positive": 0, "negative": 0})

    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.flac'):
                if "positive" in root.lower():
                    if "cough" in file.lower():
                        counts["cough"]["positive"] += 1
                    elif "breathing" in file.lower():
                        counts["breathing"]["positive"] += 1
                    elif "speech" in file.lower():
                        counts["speech"]["positive"] += 1
                elif "negative" in root.lower():
                    if "cough" in file.lower():
                        counts["cough"]["negative"] += 1
                    elif "breathing" in file.lower():
                        counts["breathing"]["negative"] += 1
                    elif "speech" in file.lower():
                        counts["speech"]["negative"] += 1

    return counts

# Main processing function
def process_dataset(input_dir, output_dir):
    """
    Processes and segments all audio files in the dataset, ensuring fixed-length segments.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.flac'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                if "cough" in file.lower():
                    segment_cough(file_path, output_subdir, segment_length=3)  # Fixed-length cough
                elif "breathing" in file.lower():
                    segment_breathing(file_path, output_subdir, segment_length=6)  # Fixed-length breathing
                elif "speech" in file.lower():
                    process_speech(file_path, output_subdir)  # Entire speech file

    # Count results
    counts = count_results(output_dir)
    print("\nFinal Counts:")
    for audio_type, categories in counts.items():
        print(f"\n{audio_type.capitalize()}:")
        for category, count in categories.items():
            print(f"  {category.capitalize()}: {count} files")

# Define directories
INPUT_DIR = "data/reclassified"
OUTPUT_DIR = "data/segmented"

# Process dataset
process_dataset(INPUT_DIR, OUTPUT_DIR)
