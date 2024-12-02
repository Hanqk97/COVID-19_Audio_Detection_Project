import os
import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt
from datetime import datetime

# Load and preprocess data
def read_data(base_folder):
    data = []
    for label in ['negative', 'positive']:
        label_folder = os.path.join(base_folder, label)
        for gender in ['male', 'female']:
            gender_folder = os.path.join(label_folder, gender)
            if not os.path.exists(gender_folder):
                continue
            file_list = [f for f in os.listdir(gender_folder) if f.endswith('.json')]
            for file in file_list:
                file_path = os.path.join(gender_folder, file)
                with open(file_path, 'r') as f:
                    sample = json.load(f)
                features = []
                features.extend(sample['cough']['mfcc']['mean'])
                features.extend(sample['breathing']['mfcc']['mean'])
                features.extend(sample['speech']['mfcc']['mean'])
                label_value = 1 if label == 'positive' else 0
                data.append((features, label_value))
    return data

# Plot and save confusion matrix
def plot_confusion_matrix(cm, classes, output_path):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = "d"
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Train and evaluate XGBoost
def train_and_evaluate(data, model_name="XGBoost"):
    results_dir = f"result/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(results_dir, exist_ok=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(data), 1):
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]

        X_train = np.array([item[0] for item in train_data])
        y_train = np.array([item[1] for item in train_data])
        X_val = np.array([item[0] for item in val_data])
        y_val = np.array([item[1] for item in val_data])

        # Train XGBoost model
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, learning_rate=0.1)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)

        metrics.append((f1, precision, recall, accuracy))
        plot_confusion_matrix(cm, ["Negative", "Positive"], os.path.join(results_dir, f"cm_fold{fold}.png"))

    avg_metrics = np.mean(metrics, axis=0)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump({
            "f1_score": avg_metrics[0],
            "precision": avg_metrics[1],
            "recall": avg_metrics[2],
            "accuracy": avg_metrics[3]
        }, f, indent=4)

# Main execution
if __name__ == "__main__":
    base_folder = "data/extracted_features"
    data = read_data(base_folder)
    train_and_evaluate(data)
