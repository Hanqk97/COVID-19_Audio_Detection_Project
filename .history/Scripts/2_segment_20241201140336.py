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

# Updated Cough Segmentation
def segment_cough(file_path, output_dir, sample_rate=44100, energy_threshold=0.05, min_duration=0.3):
    """
    Segments cough audio into bursts based on energy thresholds and silence removal.

    Args:
        file_path (str): Path to the audio file.
        output_dir (str): Directory to save segmented audio.
        sample_rate (int): Sampling rate.
        energy_threshold (float): Energy threshold for detecting coughs.
        min_duration (float): Minimum duration (seconds) for valid cough bursts.
    """
    # Load audio file
    y, sr = librosa.load(file_path, sr=sample_rate)

    # Apply band-pass filter for cough-specific frequencies (300–800 Hz)
    y = librosa.effects.preemphasis(y, coef=0.97)

    # Detect segments with non-silent audio
    intervals = librosa.effects.split(y, top_db=30)  # Remove silence
    bursts = []

    for start, end in intervals:
        # Calculate energy of the segment
        segment = y[start:end]
        duration = (end - start) / sr
        segment_energy = np.sum(segment**2) / len(segment)
        if segment_energy > energy_threshold and duration >= min_duration:
            bursts.append((start, end))

    # Save each detected burst
    os.makedirs(output_dir, exist_ok=True)
    for idx, (start, end) in enumerate(bursts):
        segment = y[start:end]
        segment_path = os.path.join(output_dir, f"{os.path.basename(file_path).split('.')[0]}_seg{idx + 1}.flac")
        sf.write(segment_path, segment, sr)
        print(f"Saved cough segment: {segment_path}")

# Updated Breathing Segmentation
def segment_breathing(file_path, output_dir, segment_length=6, overlap=0.5, sample_rate=44100):
    """
    Segments breathing audio into fixed-length intervals.

    Args:
        file_path (str): Path to the audio file.
        output_dir (str): Directory to save segmented audio.
        segment_length (float): Length of each segment in seconds.
        overlap (float): Overlap ratio between segments.
        sample_rate (int): Sampling rate.

    Returns:
        None
    """
    # Load audio file
    y, sr = librosa.load(file_path, sr=sample_rate)

    # Segment parameters
    segment_samples = int(segment_length * sr)
    hop_samples = int(segment_samples * (1 - overlap))
    total_samples = len(y)

    os.makedirs(output_dir, exist_ok=True)

    # Segment breathing audio
    for i in range(0, total_samples - segment_samples + 1, hop_samples):
        start = i
        end = i + segment_samples
        segment = y[start:end]
        segment_path = os.path.join(output_dir, f"{os.path.basename(file_path).split('.')[0]}_seg{i // hop_samples + 1}.flac")
        sf.write(segment_path, segment, sr)
        print(f"Saved breathing segment: {segment_path}")

# Speech processing (logical segmentation)
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
    Processes and segments all audio files in the dataset, then counts the results.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.flac'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                if "cough" in file.lower():
                    segment_cough(file_path, output_subdir)  # Updated segmentation
                elif "breathing" in file.lower():
                    segment_breathing(file_path, output_subdir)  # Updated segmentation
                elif "speech" in file.lower():
                    process_speech(file_path, output_subdir)

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
