import os
import json
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score, balanced_accuracy_score
)
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import pandas as pd

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

# Load IDs from CSV files
def read_ids_from_csv_list(folder_path, pattern):
    file_paths = sorted(glob.glob(os.path.join(folder_path, pattern)))
    id_lists = []
    for file_path in file_paths:
        ids = pd.read_csv(file_path).iloc[:, 0].tolist()
        print(f"IDs loaded from {file_path}: {ids[:10]}... (total {len(ids)})")  # Print first 10 IDs for debugging
        id_lists.append(ids)
    return id_lists

def read_data_for_ids(base_folder, ids):
    """
    Reads data for a given list of IDs by iterating through all JSON files in the folder structure.
    """
    data = []
    ids_set = set(ids)  # Convert to set for faster lookup
    for label in ['negative', 'positive']:
        for gender in ['male', 'female']:
            label_gender_folder = os.path.join(base_folder, label, gender)
            if not os.path.exists(label_gender_folder):
                continue
            
            # Iterate through all JSON files in the folder
            for file_name in os.listdir(label_gender_folder):
                if not file_name.endswith('.json'):
                    continue
                file_id = file_name.replace('.json', '')  # Extract ID from filename
                
                if file_id in ids_set:
                    json_path = os.path.join(label_gender_folder, file_name)
                    try:
                        with open(json_path, 'r') as f:
                            sample = json.load(f)
                    except Exception as e:
                        print(f"Error reading JSON file {json_path}: {e}. Skipping.")
                        continue

                    # Extract features
                    features = []
                    features.extend(sample['cough']['mfcc']['mean'])
                    features.extend(sample['breathing']['mfcc']['mean'])
                    features.extend(sample['speech']['mfcc']['mean'])

                    # Determine label (positive: 1, negative: 0)
                    label_value = 1 if label == 'positive' else 0
                    data.append((features, label_value))

                    # Remove the ID from the list to avoid re-processing
                    ids_set.remove(file_id)

    # Check if any IDs were not found
    if ids_set:
        print(f"Warning: The following IDs were not found in the directory structure: {ids_set}")

    return data


# Train and evaluate XGBoost using train/validation splits
def train_and_evaluate_with_csv_pairs(base_folder, train_val_folder, model_name="XGBoost"):
    train_id_lists = read_ids_from_csv_list(train_val_folder, "train_*.csv")
    val_id_lists = read_ids_from_csv_list(train_val_folder, "val_*.csv")

    for split_idx, (train_ids, val_ids) in enumerate(zip(train_id_lists, val_id_lists)):
        print(f"Processing split {split_idx}...")

        # Load data for the current split
        train_data = read_data_for_ids(base_folder, train_ids)
        val_data = read_data_for_ids(base_folder, val_ids)

        if not train_data or not val_data:
            print(f"Split {split_idx}: Train or validation data missing.")
            continue

        # Prepare datasets
        X_train = np.array([item[0] for item in train_data])
        y_train = np.array([item[1] for item in train_data])
        X_val = np.array([item[0] for item in val_data])
        y_val = np.array([item[1] for item in val_data])

        # Train model
        print(f"Training model for split {split_idx}...")
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, learning_rate=0.1)
        model.fit(X_train, y_train)

        # Predict and evaluate
        print(f"Evaluating model for split {split_idx}...")
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        cm = confusion_matrix(y_val, y_pred)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = f"result/{model_name}/{model_name}_{timestamp}_split_{split_idx}/"
        os.makedirs(results_dir, exist_ok=True)
        plot_confusion_matrix(cm, ["Negative", "Positive"], os.path.join(results_dir, "confusion_matrix.png"))

        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            json.dump({
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "auc": auc
            }, f, indent=4)


# Load the saved XGBoost model
def load_saved_model(model_path):
    print(f"Loading model from {model_path}")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

# Test the saved model on validation sets
def test_saved_model_with_val_sets(saved_model_path, base_folder, train_val_folder, model_name="XGBoost"):
    val_id_lists = read_ids_from_csv_list(train_val_folder, "val_*.csv")
    saved_model = load_saved_model(saved_model_path)

    for split_idx, val_ids in enumerate(val_id_lists):
        print(f"Testing saved model on split {split_idx}...")

        # Load validation data
        val_data = read_data_for_ids(base_folder, val_ids)
        if not val_data:
            print(f"Split {split_idx}: Validation data missing.")
            continue

        # Prepare datasets
        X_val = np.array([item[0] for item in val_data])
        y_val = np.array([item[1] for item in val_data])

        # Predict and evaluate
        y_pred = saved_model.predict(X_val)
        y_pred_proba = saved_model.predict_proba(X_val)[:, 1]
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        cm = confusion_matrix(y_val, y_pred)

        # Save augmented results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = f"result/{model_name}/aug_{model_name}_{timestamp}_split_{split_idx}/"
        os.makedirs(results_dir, exist_ok=True)
        plot_confusion_matrix(cm, ["Negative", "Positive"], os.path.join(results_dir, "confusion_matrix.png"))

        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            json.dump({
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "auc": auc
            }, f, indent=4)

        print(f"Results for split {split_idx} saved to {results_dir}")

# Main execution for testing the saved model
if __name__ == "__main__":
    base_folder = "data/extracted_features"
    train_val_folder = "data/raw_data/LISTS"
    saved_model_path = "result/best_model/XGBoost.json"
    train_and_evaluate_with_csv_pairs(base_folder, train_val_folder)
    test_saved_model_with_val_sets(saved_model_path, base_folder, train_val_folder)
    
