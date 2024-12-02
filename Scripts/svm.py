import os
import json
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score

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

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

grid = GridSearchCV(SVC(class_weight='balanced', random_state=42), param_grid, refit=True, verbose=3, cv=5)
grid.fit(X_train, y_train)

# Best model from grid search
best_svm_model = grid.best_estimator_

# Predict using the best SVM model
y_pred = best_svm_model.predict(X_test)

# Generate classification report
print("SVM Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred):.4f}")

# Print success message
print("SVM model trained and evaluated successfully!")

