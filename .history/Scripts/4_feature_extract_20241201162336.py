import os
import json
import librosa
import numpy as np

# Utility: Aggregate features for traditional ML
def aggregate_features(features):
    return {
        "mean": np.mean(features, axis=1).tolist(),
        "variance": np.var(features, axis=1).tolist(),
        "skewness": np.mean(features**3, axis=1).tolist(),
        "min": np.min(features, axis=1).tolist(),
        "max": np.max(features, axis=1).tolist(),
        "std": np.std(features, axis=1).tolist(),
        "iqr": (np.percentile(features, 75, axis=1) - np.percentile(features, 25, axis=1)).tolist(),
    }

# Utility: Compute Log-Mel spectrogram
def compute_log_mel(y, sr, n_mels=128):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

# Feature extraction for all types
def extract_features(file_path, n_mfcc=13, sr=16000):
    y, sr = librosa.load(file_path, sr=sr)
    features = {}

    # Log-Mel spectrogram
    log_mel = compute_log_mel(y, sr)
    features["log_mel"] = log_mel.tolist()

    # Aggregate features
    features["mfcc"] = aggregate_features(librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mfcc))
    features["delta"] = aggregate_features(librosa.feature.delta(librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mfcc)))
    features["chroma"] = aggregate_features(librosa.feature.chroma_stft(y, sr=sr))
    features["rms"] = aggregate_features(librosa.feature.rms(y=y))

    return features

# Process a single ID
def process_id(input_dir, output_dir, id_name):
    features = {"cough": {}, "breathing": {}, "speech": {}}

    # Infer gender and label from the directory structure
    parts = os.path.normpath(input_dir).split(os.sep)
    label = parts[-3]  # "positive" or "negative"
    gender = parts[-2]  # "male" or "female"

    # Add gender and label to features
    features["gender"] = gender
    features["label"] = label

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".flac"):
            try:
                if "cough" in file_name.lower():
                    features["cough"] = extract_features(os.path.join(input_dir, file_name))
                elif "breathing" in file_name.lower():
                    features["breathing"] = extract_features(os.path.join(input_dir, file_name))
                elif "speech" in file_name.lower():
                    features["speech"] = extract_features(os.path.join(input_dir, file_name))
                print(f"Processed {file_name} for ID {id_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Save features as JSON
    output_file = os.path.join(output_dir, f"{id_name}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(features, f, indent=4)
    print(f"Saved features for ID {id_name} to {output_file}")

# Process the dataset
def process_dataset(input_dir, output_dir):
    for root, dirs, _ in os.walk(input_dir):
        for id_name in dirs:
            id_path = os.path.join(root, id_name)
            print(f"Processing ID: {id_name} in {root}")
            process_id(id_path, output_dir, id_name)

# Run the pipeline
INPUT_DIR = "data/augmented"
OUTPUT_DIR = "data/extracted_features"
process_dataset(INPUT_DIR, OUTPUT_DIR)
