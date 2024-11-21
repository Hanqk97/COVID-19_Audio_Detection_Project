import os
import librosa
import numpy as np
import json

# Define paths
AUDIO_PATH = 'data/Augmented'
OUTPUT_PATH = 'data/extracted_feature'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Feature extraction function
def extract_features(file_path):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=44100)
        
        # MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = mfccs.mean(axis=1).astype(float)  # Convert to native Python float
        mfccs_var = mfccs.var(axis=1).astype(float)    # Convert to native Python float
        
        # Root Mean Square Energy (RMS)
        rms = float(librosa.feature.rms(y=y).mean())
        
        # Zero Crossing Rate (ZCR)
        zcr = float(librosa.feature.zero_crossing_rate(y).mean())
        
        # Spectral Features
        spectral_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        spectral_bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
        spectral_rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
        
        # Combine features into a dictionary
        features = {
            'mfccs_mean': mfccs_mean.tolist(),
            'mfccs_var': mfccs_var.tolist(),
            'rms': rms,
            'zcr': zcr,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff
        }
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Process all audio files in the directory
for root, dirs, files in os.walk(AUDIO_PATH):
    for file in files:
        if file.endswith(".flac"):
            # Extract type (breathing/cough/speech) and label (negative/positive) from the path
            path_parts = root.split(os.sep)
            if len(path_parts) >= 3:  # Ensure path structure is as expected
                audio_type = path_parts[-2]  # Extract the type
                label = path_parts[-1]       # Extract the label
                file_id = file.split('.')[0] # Extract the file ID (without extension)
                
                # Full file path
                file_path = os.path.join(root, file)
                
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
