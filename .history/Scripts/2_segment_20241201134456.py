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

# Updated Cough Segmentation Function
def segment_cough_advanced(file_path, output_dir, energy_threshold=0.02, min_duration=0.3, sample_rate=44100):
    """
    Segments cough audio into individual bursts using advanced methods.

    Args:
        file_path (str): Path to the audio file.
        output_dir (str): Directory to save segmented audio.
        energy_threshold (float): Energy threshold to detect active regions.
        min_duration (float): Minimum duration for active regions (in seconds).
        sample_rate (int): Sampling rate.

    Returns:
        None
    """
    # Load audio file
    y, sr = librosa.load(file_path, sr=sample_rate)

    # Apply high-pass filter to remove low-frequency noise
    y_filtered = librosa.effects.preemphasis(y, coef=0.97)

    # Frame parameters
    frame_length = int(sr * 0.02)  # 20ms frames
    hop_length = frame_length // 2

    # Compute energy per frame
    energy = np.array([
        np.sum(np.abs(y_filtered[i:i + frame_length])**2) for i in range(0, len(y_filtered), hop_length)
    ])
    energy = energy / np.max(energy)  # Normalize energy

    # Detect active frames
    active_frames = np.where(energy > energy_threshold)[0]

    # Group contiguous active frames into segments
    segments = []
    segment_start = None
    for i in range(len(active_frames)):
        if segment_start is None:
            segment_start = active_frames[i] * hop_length
        if i == len(active_frames) - 1 or active_frames[i + 1] != active_frames[i] + 1:
            segment_end = (active_frames[i] + 1) * hop_length
            duration = (segment_end - segment_start) / sr
            if duration >= min_duration:
                segments.append((segment_start, segment_end))
            segment_start = None

    # Save segments
    os.makedirs(output_dir, exist_ok=True)
    for idx, (start, end) in enumerate(segments):
        segment = y[int(start):int(end)]
        segment_filename = os.path.basename(file_path).replace('.flac', f'_seg{idx + 1}.flac')
        segment_path = os.path.join(output_dir, segment_filename)
        sf.write(segment_path, segment, sr)
        print(f"Saved cough segment: {segment_path}")

# Breathing segmentation
def segment_breathing(file_path, output_dir, segment_length=6, sample_rate=44100, overlap=0.5):
    y, sr = librosa.load(file_path, sr=sample_rate)
    segment_samples = int(segment_length * sr)
    hop_samples = int(segment_samples * (1 - overlap))
    total_samples = len(y)

    os.makedirs(output_dir, exist_ok=True)
    for i in range(0, total_samples - segment_samples + 1, hop_samples):
        segment = y[i:i+segment_samples]
        segment_path = os.path.join(output_dir, f"breathing_seg{i + 1}.flac")
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
                    segment_cough_advanced(file_path, output_subdir)  # Use updated segmentation
                elif "breathing" in file.lower():
                    segment_breathing(file_path, output_subdir)
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
