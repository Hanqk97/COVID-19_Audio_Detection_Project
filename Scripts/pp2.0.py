import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define paths
DATA_DIR = '/Users/jonahgloss/Downloads/extracted_feature'


# Load all JSON files and aggregate features
def load_data(data_dir):
    features, labels = [], []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    # Combine all features into one vector
                    feature_vector = np.concatenate([
                        np.array(data['mfccs_mean']),
                        np.array(data['mfccs_var']),
                        np.array(data['spectral_contrast_mean']),
                        np.array([
                            data['rms_mean'], data['rms_var'],
                            data['zcr_mean'], data['zcr_var'],
                            data['spectral_centroid_mean'],
                            data['spectral_bandwidth_mean'],
                            data['spectral_rolloff_mean'],
                            data['spectral_flatness_mean'],
                            data['hnr']
                        ])
                    ])
                    features.append(feature_vector)
                    labels.append(data['label'])

    return np.array(features), np.array(labels)


# Load the data
X, y = load_data(DATA_DIR)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
