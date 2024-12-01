import os
import json
import librosa
import numpy as np
from vggish import VGGish  # Assuming a VGGish wrapper is available

# Initialize VGGish
vggish = VGGish()

# Utility: Aggregate features for traditional ML
def aggregate_features(features):
    return {
        "mean": np.mean(features, axis=1).tolist(),
        "variance": np.var(features, axis=1).tolist(),
        "skewness": np.mean(features**3, axis=1).tolist()
    }

# Utility: Compute Log-Mel spectrogram
def compute_log_mel(y, sr, n_mels=128):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

# Feature extraction for cough
def extract_cough_features(file_path, sample_rate=16000):
    y, sr = librosa.load(file_path, sr=sample_rate)
    features = {}

    # VGGish embeddings (for both ML and DL models)
    features["vggish"] = vggish.forward(y, sr).mean(axis=0).tolist()

    # Log-Mel spectrogram (for CNN/ResNet)
    log_mel = compute_log_mel(y, sr)
    features["log_mel"] = log_mel.tolist()  # Store as 2D matrix

    # Aggregate features (for SVM/Random Forest)
    features["mfcc"] = aggregate_features(librosa.feature.mfcc(y, sr=sr, n_mfcc=40))
    features["spectral_centroid"] = aggregate_features(librosa.feature.spectral_centroid(y, sr=sr))
    features["energy"] = aggregate_features(librosa.feature.rms(y=y))

    return features

# Feature extraction for breathing
def extract_breathing_features(file_path, sample_rate=16000, segment_length=6):
    y, sr = librosa.load(file_path, sr=sample_rate)
    features = {}

    # VGGish embeddings
    features["vggish"] = vggish.forward(y, sr).mean(axis=0).tolist()

    # Log-Mel spectrogram
    log_mel = compute_log_mel(y, sr)
    features["log_mel"] = log_mel.tolist()

    # Aggregate features
    features["mfcc"] = aggregate_features(librosa.feature.mfcc(y, sr=sr, n_mfcc=20))
    features["rms"] = aggregate_features(librosa.feature.rms(y=y))

    return features

# Feature extraction for speech
def extract_speech_features(file_path, sample_rate=16000):
    y, sr = librosa.load(file_path, sr=sample_rate)
    features = {}

    # VGGish embeddings
    features["vggish"] = vggish.forward(y, sr).mean(axis=0).tolist()

    # Log-Mel spectrogram
    log_mel = compute_log_mel(y, sr)
    features["log_mel"] = log_mel.tolist()

    # Aggregate features
    features["mfcc"] = aggregate_features(librosa.feature.mfcc(y, sr=sr, n_mfcc=13))
    features["delta"] = aggregate_features(librosa.feature.delta(librosa.feature.mfcc(y, sr=sr, n_mfcc=13)))
    features["chroma"] = aggregate_features(librosa.feature.chroma_stft(y, sr=sr))

    return features

# Process a single ID
def process_id(input_dir, output_dir, id_name):
    features = {"cough": {}, "breathing": {}, "speech": {}}

    for audio_type in ["cough", "breathing", "speech"]:
        file_path = os.path.join(input_dir, audio_type, f"{id_name}_{audio_type}.flac")
        if os.path.exists(file_path):
            try:
                if audio_type == "cough":
                    features[audio_type] = extract_cough_features(file_path)
                elif audio_type == "breathing":
                    features[audio_type] = extract_breathing_features(file_path)
                elif audio_type == "speech":
                    features[audio_type] = extract_speech_features(file_path)
                print(f"Processed {audio_type} for ID {id_name}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

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
            process_id(id_path, output_dir, id_name)

# Run the pipeline
INPUT_DIR = "data/segmented"
OUTPUT_DIR = "data/extracted_features"
process_dataset(INPUT_DIR, OUTPUT_DIR)
