import os
import librosa
import numpy as np
import json
from scipy.stats import skew, kurtosis

# Define paths
AUDIO_PATH = 'data/Augmented'
OUTPUT_PATH = 'data/extracted_feature'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Feature extraction function
def extract_features(file_path):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=44100)
        
        # Ensure the audio has sufficient duration
        if len(y) < sr:
            print(f"Audio too short: {file_path}")
            return None

        # MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = mfccs.mean(axis=1).astype(float)
        mfccs_var = mfccs.var(axis=1).astype(float)
        mfccs_skew = skew(mfccs, axis=1).astype(float)
        mfccs_kurt = kurtosis(mfccs, axis=1).astype(float)
        
        # Root Mean Square Energy (RMS)
        rms = librosa.feature.rms(y=y)
        rms_mean = float(rms.mean())
        rms_var = float(rms.var())

        # Zero Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = float(zcr.mean())
        zcr_var = float(zcr.var())

        # Spectral Features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        
        # Harmonic Features
        harmonic, percussive = librosa.effects.hpss(y)
        hnr = librosa.feature.rms(y=harmonic).mean() / (librosa.feature.rms(y=percussive).mean() + 1e-6)

        # Aggregate Features
        features = {
            # MFCCs
            'mfccs_mean': mfccs_mean.tolist(),
            'mfccs_var': mfccs_var.tolist(),
            'mfccs_skew': mfccs_skew.tolist(),
            'mfccs_kurt': mfccs_kurt.tolist(),

            # RMS
            'rms_mean': rms_mean,
            'rms_var': rms_var,

            # ZCR
            'zcr_mean': zcr_mean,
            'zcr_var': zcr_var,

            # Spectral
            'spectral_centroid_mean': float(spectral_centroid.mean()),
            'spectral_bandwidth_mean': float(spectral_bandwidth.mean()),
            'spectral_rolloff_mean': float(spectral_rolloff.mean()),
            'spectral_contrast_mean': spectral_contrast.mean(axis=1).tolist(),
            'spectral_flatness_mean': float(spectral_flatness.mean()),

            # Harmonic-to-Noise Ratio
            'hnr': float(hnr)
        }
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Count total files for progress tracking
total_files = sum([len(files) for _, _, files in os.walk(AUDIO_PATH) if any(file.endswith('.flac') for file in files)])
processed_files = 0

# Process all audio files in the directory
for root, dirs, files in os.walk(AUDIO_PATH):
    for file in files:
        if file.endswith(".flac"):
            # Increment processed files count
            processed_files += 1

            # Extract type (breathing/cough/speech) and label (negative/positive) from the path
            path_parts = root.split(os.sep)
            if len(path_parts) >= 3:  # Ensure path structure is as expected
                audio_type = path_parts[-2]  # Extract the type
                label = path_parts[-1]       # Extract the label
                file_id = file.split('.')[0] # Extract the file ID (without extension)
                
                # Full file path
                file_path = os.path.join(root, file)
                
                # Print progress
                print(f"Processing {processed_files}/{total_files}: {file_path}")
                
                # Extract features
                features = extract_features(file_path)
                if features:
                    # Add metadata to features
                    features['type'] = audio_type
                    features['label'] = label
                    features['file_id'] = file_id
                    
                    # Save as JSON
                    type_folder = os.path.join(OUTPUT_PATH, audio_type)
                    os.makedirs(type_folder, exist_ok=True)  # Create folder for type
                    output_file = os.path.join(type_folder, f"{file_id}.json")
                    
                    with open(output_file, 'w') as f:
                        json.dump(features, f, indent=4)
            else:
                print(f"Unexpected path structure: {root}")

print(f"Processing complete! {processed_files}/{total_files} files processed.")
