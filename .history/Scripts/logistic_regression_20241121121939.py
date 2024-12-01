import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.table import Table
import random

# Configuration
CLASS_WEIGHT = {0: 1, 1: 1}  # Adjust for class imbalance if needed
K_FOLDS = 5  # Number of folds for cross-validation

def load_data_with_id(base_folder):
    """
    Load and preprocess data from JSON files, organized by IDs.
    Returns features, labels, and IDs as numpy arrays.
    """
    features = []
    labels = []
    ids = []
    
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    record = json.load(f)
                ids.append(record["file_id"])  # Extract ID
                labels.append(1 if record["label"] == "positive" else 0)  # Positive = 1, Negative = 0
                feature_vector = []
                for key in ["mfccs_mean", "mfccs_var", "rms", "zcr", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff"]:
                    value = record[key]
                    feature_vector.extend(value if isinstance(value, list) else [value])
                features.append(feature_vector)

    return np.array(features, dtype=float), np.array(labels, dtype=int), np.array(ids)

def stratified_split_by_id(ids, labels, k_folds):
    """
    Perform stratified K-Fold split ensuring consistent ID-based distribution.
    """
    unique_ids = np.array(sorted(set(ids), key=lambda x: x[0]))  # Sort by first letter of ID
    id_to_label = {uid: labels[np.where(ids == uid)[0][0]] for uid in unique_ids}

    unique_labels = np.array([id_to_label[uid] for uid in unique_ids])
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    id_folds = []
    for train_ids, val_ids in skf.split(unique_ids, unique_labels):
        id_folds.append((unique_ids[train_ids], unique_ids[val_ids]))
    return id_folds

def filter_data_by_ids(features, labels, ids, selected_ids):
    """
    Filter features, labels, and IDs by selected IDs.
    """
    mask = np.isin(ids, selected_ids)
    return features[mask], labels[mask], ids[mask]

def normalize_data(features):
    """
    Normalize feature data using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)

def train_and_evaluate_model(features, labels, ids, id_folds, output_dir):
    """
    Train and evaluate model using stratified K-Fold validation.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = []

    for fold, (train_ids, val_ids) in enumerate(id_folds, start=1):
        print(f"Processing Fold {fold}/{K_FOLDS}...")

        # Split data by IDs
        X_train, y_train, _ = filter_data_by_ids(features, labels, ids, train_ids)
        X_val, y_val, _ = filter_data_by_ids(features, labels, ids, val_ids)

        # Normalize features
        X_train = normalize_data(X_train)
        X_val = normalize_data(X_val)

        # Train Logistic Regression model
        model = LogisticRegression(max_iter=1000, class_weight=CLASS_WEIGHT)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_val)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_val, y_pred)
        conf_matrix = confusion_matrix(y_val, y_pred, labels=[0, 1])

        # Save metrics and plot results
        metrics = {"Precision": precision, "Recall": recall, "F1 Score": f1, "Balanced Accuracy": balanced_acc}
        save_metrics_and_plot(metrics, conf_matrix, os.path.join(output_dir, f"fold_{fold}"), f"Fold {fold} Metrics")
        all_metrics.append(metrics)

    # Compute average metrics
    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
    avg_metrics_file = os.path.join(output_dir, "average_metrics.json")
    with open(avg_metrics_file, "w") as f:
        json.dump(avg_metrics, f, indent=4)
    print("Cross-validation complete. Results saved.")

def save_metrics_and_plot(metrics, conf_matrix, output_path, title):
    """
    Save metrics as a JSON file and plot the confusion matrix.
    """
    metrics_file = output_path + "_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    table_data = [
        ["", "Predicted Positive", "Predicted Negative"],
        ["Actual Positive", f"TP: {conf_matrix[1, 1]}", f"FN: {conf_matrix[1, 0]}"],
        ["Actual Negative", f"FP: {conf_matrix[0, 1]}", f"TN: {conf_matrix[0, 0]}"],
    ]
    table_colors = [
        ["w", "#ADD8E6", "#ADD8E6"],
        ["#FFFFE0", "#98FB98", "#FFCCCB"],
        ["#FFFFE0", "#FFCCCB", "#98FB98"],
    ]

    table = Table(ax, bbox=[0, 0, 1, 1])
    for i, row in enumerate(table_data):
        for j, cell_text in enumerate(row):
            table.add_cell(i, j, width=1 / len(row), height=1 / len(table_data), text=cell_text, loc="center",
                           facecolor=table_colors[i][j])
    ax.add_table(table)

    metrics_text = "\n".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
    plt.figtext(0.95, 0.5, metrics_text, fontsize=10, ha="left", va="center", bbox=dict(facecolor="white", alpha=0.8))
    plt.title(title, fontsize=12, fontweight="bold", pad=20)

    img_file = output_path + "_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(img_file, bbox_inches="tight")
    plt.close(fig)

def main():
    """
    Main function to load data, split into folds, and train/evaluate model.
    """
    input_dir = "Data/extracted_feature"  # Folder containing extracted features
    output_dir = "results/logistic_regression/audio_classification"  # Folder to save results
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    features, labels, ids = load_data_with_id(input_dir)
    print(f"Loaded {len(labels)} examples.")

    # Stratified split by ID
    print("Splitting data into stratified folds...")
    id_folds = stratified_split_by_id(ids, labels, K_FOLDS)

    # Train and evaluate model
    train_and_evaluate_model(features, labels, ids, id_folds, output_dir)

if __name__ == "__main__":
    main()
