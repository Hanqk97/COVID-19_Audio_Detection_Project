import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report


# Data loading function
def load_data(data_dir):
    features, labels = [], []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    feature_vector = np.concatenate([
                        data['mfccs_mean'],
                        data['mfccs_var'],
                        data['spectral_contrast_mean'],
                        [
                            data['rms_mean'], data['rms_var'],
                            data['zcr_mean'], data['zcr_var'],
                            data['spectral_centroid_mean'],
                            data['spectral_bandwidth_mean'],
                            data['spectral_rolloff_mean'],
                            data['spectral_flatness_mean'],
                            data['hnr']
                        ]
                    ])
                    features.append(feature_vector)
                    labels.append(data['label'])

    return np.array(features), np.array(labels)


# Load data
DATA_DIR = '/Users/jonahgloss/Downloads/extracted_feature'
X, y = load_data(DATA_DIR)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Predict using the trained SVM model
y_pred = svm_model.predict(X_test)

# Generate classification report
print("SVM Classification Report:")
print(classification_report(y_test, y_pred))

# Print success message
print("SVM model trained and evaluated successfully!")


