import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score
)
from joblib import dump, load

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

# Read IDs from CSV files
def read_ids_from_csv_list(folder_path, pattern):
    file_paths = sorted(glob.glob(os.path.join(folder_path, pattern)))
    id_lists = []
    for file_path in file_paths:
        ids = pd.read_csv(file_path).iloc[:, 0].tolist()
        print(f"IDs loaded from {file_path}: {ids[:10]}... (total {len(ids)})")
        id_lists.append(ids)
    return id_lists

# Read data for given IDs
def read_data_for_ids(base_folder, ids):
    data = []
    ids_set = set(ids)
    for label in ['negative', 'positive']:
        for gender in ['male', 'female']:
            label_gender_folder = os.path.join(base_folder, label, gender)
            if not os.path.exists(label_gender_folder):
                continue
            for file_name in os.listdir(label_gender_folder):
                if not file_name.endswith('.json'):
                    continue
                file_id = file_name.replace('.json', '')
                if file_id in ids_set:
                    json_path = os.path.join(label_gender_folder, file_name)
                    try:
                        with open(json_path, 'r') as f:
                            sample = json.load(f)
                        features = []
                        features.extend(sample['cough']['mfcc']['mean'])
                        features.extend(sample['breathing']['mfcc']['mean'])
                        features.extend(sample['speech']['mfcc']['mean'])
                        label_value = 1 if label == 'positive' else 0
                        data.append((features, label_value))
                        ids_set.remove(file_id)
                    except Exception as e:
                        print(f"Error reading JSON file {json_path}: {e}. Skipping.")
    if ids_set:
        print(f"Warning: The following IDs were not found in the directory structure: {ids_set}")
    return data

# Train and evaluate SVM
def train_and_evaluate_with_csv_pairs(base_folder, train_val_folder, model_name="SVM"):
    train_id_lists = read_ids_from_csv_list(train_val_folder, "train_*.csv")
    val_id_lists = read_ids_from_csv_list(train_val_folder, "val_*.csv")

    results_dir = f"result/{model_name}/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(results_dir, exist_ok=True)

    for split_idx, (train_ids, val_ids) in enumerate(zip(train_id_lists, val_id_lists)):
        print(f"Processing split {split_idx}...")

        train_data = read_data_for_ids(base_folder, train_ids)
        val_data = read_data_for_ids(base_folder, val_ids)
        if not train_data or not val_data:
            print(f"Split {split_idx}: Train or validation data missing.")
            continue

        # Prepare data
        X_train = np.array([item[0] for item in train_data])
        y_train = np.array([item[1] for item in train_data])
        X_val = np.array([item[0] for item in val_data])
        y_val = np.array([item[1] for item in val_data])

        # Define the SVM pipeline
        model = Pipeline([
            ('scaler', StandardScaler()),  # Standardize features
            ('svm', SVC(probability=True, kernel='rbf', C=1.0, gamma='scale'))  # SVM with RBF kernel
        ])

        # Train the SVM model
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]  # Probabilities for ROC AUC

        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        cm = confusion_matrix(y_val, y_pred)

        # Save the best model for each split
        best_model_path = os.path.join(results_dir, f"SVM_split_{split_idx}.joblib")
        dump(model, best_model_path)
        print(f"Best model for split {split_idx} saved to {best_model_path} with F1 score: {f1:.4f}")

        # Save metrics and confusion matrix
        plot_confusion_matrix(cm, ["Negative", "Positive"], os.path.join(results_dir, f"cm_split_{split_idx}.png"))
        with open(os.path.join(results_dir, f"metrics_split_{split_idx}.json"), "w") as f:
            json.dump({
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "auc": auc
            }, f, indent=4)

# Test the saved model
def test_saved_model_with_val_sets(saved_model_path, base_folder, train_val_folder, model_name="SVM"):
    """
    Test the saved SVM model on validation sets.
    """
    val_id_lists = read_ids_from_csv_list(train_val_folder, "val_*.csv")

    results_dir = f"result/{model_name}/aug_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(results_dir, exist_ok=True)

    # Load the saved SVM model
    model = load(saved_model_path)

    for split_idx, val_ids in enumerate(val_id_lists):
        print(f"Testing saved model on split {split_idx}...")

        val_data = read_data_for_ids(base_folder, val_ids)
        if not val_data:
            print(f"Split {split_idx}: Validation data missing.")
            continue

        # Prepare validation data
        X_val = np.array([item[0] for item in val_data])
        y_val = np.array([item[1] for item in val_data])

        # Predict and evaluate
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        cm = confusion_matrix(y_val, y_pred)

        # Save results for this split
        split_results_dir = os.path.join(results_dir, f"split_{split_idx}")
        os.makedirs(split_results_dir, exist_ok=True)

        plot_confusion_matrix(cm, ["Negative", "Positive"], os.path.join(split_results_dir, "confusion_matrix.png"))
        with open(os.path.join(split_results_dir, "metrics.json"), "w") as f:
            json.dump({
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "auc": auc
            }, f, indent=4)

        print(f"Results for split {split_idx} saved to {split_results_dir}")

# Main execution
if __name__ == "__main__":
    base_folder = "data/extracted_features"
    train_val_folder = "data/raw_data/LISTS"
    saved_model_path = "result/SVM/SVM_split_0.joblib"

    # Train and evaluate the model
    train_and_evaluate_with_csv_pairs(base_folder, train_val_folder, model_name="SVM")

    # Test the saved model
    test_saved_model_with_val_sets(saved_model_path, base_folder, train_val_folder, model_name="SVM")
