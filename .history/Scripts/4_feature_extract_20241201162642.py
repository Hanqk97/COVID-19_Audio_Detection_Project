import os
import json
import librosa
import numpy as np

# Utility: Aggregate features for traditional ML
def aggregate_features(features):
    """Compute statistical aggregations for a feature set."""
    return {
        "mean": np.mean(features, axis=1).tolist(),
        "variance": np.var(features, axis=1).tolist(),
        "skewness": np.mean(features**3, axis=1).tolist(),
        "min": np.min(features, axis=1).tolist(),
        "max": np.max(features, axis=1).tolist(),
        "std": np.std(features, axis=1).tolist(),
        "rms": np.sqrt(np.mean(features**2, axis=1)).tolist(),
        "iqr": (np.percentile(features, 75, axis=1) - np.percentile(features, 25, axis=1)).tolist(),
        "median": np.median(features, axis=1).tolist()
    }

# Utility: Compute Log-Mel spectrogram
def compute_log_mel(y, sr, n_mels=128):
    """Compute log-mel spectrogram for a given audio signal."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

# Feature extraction for an audio file
def extract_features(file_path, sample_rate=16000, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=sample_rate)
    features = {}

    # Log-Mel spectrogram
    log_mel = compute_log_mel(y, sr)
    features["log_mel"] = log_mel.tolist()

    # Aggregate features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features["mfcc"] = aggregate_features(mfcc)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid"] = aggregate_features(spectral_centroid)
    rms = librosa.feature.rms(y=y)
    features["rms"] = aggregate_features(rms)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features["chroma"] = aggregate_features(chroma)

    return features

# Process a single user ID
def process_id(input_dir, output_dir, id_name):
    features = {"cough": {}, "breathing": {}, "speech": {}}

    # Iterate over files for this ID
    for file_name in os.listdir(input_dir):
        if "cough" in file_name.lower():
            features["cough"] = extract_features(os.path.join(input_dir, file_name))
        elif "breathing" in file_name.lower():
            features["breathing"] = extract_features(os.path.join(input_dir, file_name))
        elif "speech" in file_name.lower():
            features["speech"] = extract_features(os.path.join(input_dir, file_name))

    # Save features to JSON
    output_file = os.path.join(output_dir, f"{id_name}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(features, f, indent=4)
    print(f"Saved features for ID {id_name} to {output_file}")

# Process all user IDs
def process_dataset(input_dir, output_dir):
    for label in ["negative", "positive"]:
        for gender in ["male", "female"]:
            base_path = os.path.join(input_dir, label, gender)
            if not os.path.exists(base_path):
                continue

            # Iterate over IDs
            for id_name in os.listdir(base_path):
                id_path = os.path.join(base_path, id_name)
                if os.path.isdir(id_path):  # Ensure it's a directory
                    print(f"Processing ID: {id_name} in {label}/{gender}")
                    process_id(id_path, output_dir, id_name)

# Define input and output directories
INPUT_DIR = "data/augmented"
OUTPUT_DIR = "data/extracted_features"

# Run the pipeline
process_dataset(INPUT_DIR, OUTPUT_DIR)
