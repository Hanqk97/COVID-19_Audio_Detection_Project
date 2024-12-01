import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical


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

# One-hot encode labels
y_encoded = to_categorical(y_encoded)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Reshape for CNN (1D Convolution)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build CNN model
cnn_model = models.Sequential([
    layers.Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(128, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')  # For binary classification (COVID or not)
])

# Compile the model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test)
print(f"CNN Test Accuracy: {test_accuracy * 100:.2f}%")

# Print success message
print("CNN model trained successfully!")
