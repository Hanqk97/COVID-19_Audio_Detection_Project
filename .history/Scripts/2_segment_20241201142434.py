import os
import librosa
import numpy as np
import soundfile as sf
from collections import defaultdict

# Utility: Pad or truncate to a fixed length
def pad_or_truncate(y, target_length, sr):
    target_samples = int(target_length * sr)
    if len(y) > target_samples:
        return y[:target_samples]
    elif len(y) < target_samples:
        padding = target_samples - len(y)
        return np.pad(y, (0, padding), mode="constant")
    return y

# Cough segmentation with representative segment selection
def segment_cough(file_path, output_dir, segment_length=3, sample_rate=44100):
    """
    Processes cough audio to ensure at least one representative 3-second segment per ID.
    """
    y, sr = librosa.load(file_path, sr=sample_rate)
    segment_samples = int(segment_length * sr)

    if len(y) <= segment_samples:
        # Pad if the audio is shorter than the desired segment length
        y = pad_or_truncate(y, segment_length, sr)
        os.makedirs(output_dir, exist_ok=True)
        segment_path = os.path.join(output_dir, f"{os.path.basename(file_path).split('.')[0]}_padded.flac")
        sf.write(segment_path, y, sr)
        print(f"Saved padded cough segment: {segment_path}")
    else:
        # Split into multiple segments
        segments = [y[i:i + segment_samples] for i in range(0, len(y), segment_samples) if len(y[i:i + segment_samples]) == segment_samples]

        if not segments:
            print(f"No valid segments detected for {file_path}. Falling back to default.")
            selected_segment = y[:segment_samples]
        else:
            # Select the most representative segment (highest energy)
            selected_segment = max(segments, key=lambda s: np.sum(s**2))

        os.makedirs(output_dir, exist_ok=True)
        segment_path = os.path.join(output_dir, f"{os.path.basename(file_path).split('.')[0]}_selected.flac")
        sf.write(segment_path, selected_segment, sr)
        print(f"Saved representative or fallback cough segment: {segment_path}")

# Breathing segmentation with representative segment selection
def segment_breathing(file_path, output_dir, segment_length=6, sample_rate=44100):
    """
    Processes breathing audio to ensure at least one representative 6-second segment per ID.
    """
    y, sr = librosa.load(file_path, sr=sample_rate)

    if len(y) <= segment_length * sr:
        # Pad if the audio is shorter than the desired segment length
        y = pad_or_truncate(y, segment_length, sr)
        os.makedirs(output_dir, exist_ok=True)
        segment_path = os.path.join(output_dir, f"{os.path.basename(file_path).split('.')[0]}_padded.flac")
        sf.write(segment_path, y, sr)
        print(f"Saved padded breathing segment: {segment_path}")
    else:
        # Split into multiple segments
        segments = []
        for i in range(0, len(y), segment_length * sr):
            segment = y[i:i + segment_length * sr]
            if len(segment) == segment_length * sr:
                segments.append(segment)

        # Select the most representative segment (highest energy)
        selected_segment = max(segments, key=lambda s: np.sum(s**2))
        os.makedirs(output_dir, exist_ok=True)
        segment_path = os.path.join(output_dir, f"{os.path.basename(file_path).split('.')[0]}_selected.flac")
        sf.write(segment_path, selected_segment, sr)
        print(f"Saved representative breathing segment: {segment_path}")

# Speech processing remains unchanged
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
    Processes and segments all audio files in the dataset.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.flac'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                if "cough" in file.lower():
                    segment_cough(file_path, output_subdir, segment_length=3)
                elif "breathing" in file.lower():
                    segment_breathing(file_path, output_subdir, segment_length=6)
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
