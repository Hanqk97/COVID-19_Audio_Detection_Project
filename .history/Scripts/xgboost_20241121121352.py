import os
import json
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from matplotlib.table import Table
from datetime import datetime

# Configuration Variables
K_FOLDS = 5  # Number of folds for cross-validation
INPUT_DIR = "data/extracted_feature"
OUTPUT_DIR = f"results/audio_classification_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_audio_features(input_dir):
    """
    Load audio features and labels from JSON files.
    """
    features = []
    labels = []
    ids = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    record = json.load(f)
                ids.append(record["file_id"])  # Use file ID for stratified splitting
                labels.append(1 if record["label"] == "positive" else 0)  # Positive = 1, Negative = 0
                feature_vector = []
                for key in ["mfccs_mean", "mfccs_var", "rms", "zcr", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff"]:
                    value = record[key]
                    feature_vector.extend(value if isinstance(value, list) else [value])
                features.append(feature_vector)

    return np.array(features, dtype=float), np.array(labels, dtype=int), np.array(ids)

def normalize_features(features):
    """
    Normalize feature data using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)

def stratified_split(ids, labels, k_folds):
    """
    Perform stratified K-Fold split ensuring consistent ID-based distribution.
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    return [(train_idx, val_idx) for train_idx, val_idx in skf.split(ids, labels)]

def train_and_evaluate_xgboost(features, labels, ids, splits, output_dir):
    """
    Train and evaluate XGBoost model using stratified K-Fold validation.
    """
    metrics_per_fold = []
    confusion_matrices = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"Processing Fold {fold}/{K_FOLDS}...")

        # Split data
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Normalize data
        X_train = normalize_features(X_train)
        X_val = normalize_features(X_val)

        # Train XGBoost model
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100, learning_rate=0.1)
        model.fit(X_train, y_train)

        # Predict on validation set
        y_pred = model.predict(X_val)
        conf_matrix = confusion_matrix(y_val, y_pred, labels=[0, 1])
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_val, y_pred)

        metrics = {"Precision": precision, "Recall": recall, "F1 Score": f1, "Balanced Accuracy": balanced_acc}
        metrics_per_fold.append(metrics)
        confusion_matrices.append(conf_matrix)

        # Save results for the fold
        fold_output_path = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_output_path, exist_ok=True)
        save_metrics_and_conf_matrix(metrics, conf_matrix, fold_output_path, f"Fold {fold} Metrics")

    # Compute and save average metrics
    avg_metrics = {key: np.mean([m[key] for m in metrics_per_fold]) for key in metrics_per_fold[0]}
    save_average_metrics(avg_metrics, confusion_matrices, output_dir)

def save_metrics_and_conf_matrix(metrics, conf_matrix, output_path, title):
    """
    Save metrics and confusion matrix plot.
    """
    # Save metrics as JSON
    with open(os.path.join(output_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    table_data = [
        ["", "Predicted Positive", "Predicted Negative"],
        ["Actual Positive", f"TP: {conf_matrix[1, 1]}", f"FN: {conf_matrix[1, 0]}"],
        ["Actual Negative", f"FP: {conf_matrix[0, 1]}", f"TN: {conf_matrix[0, 0]}"],
    ]
    table_colors = [["w", "#ADD8E6", "#ADD8E6"], ["#FFFFE0", "#98FB98", "#FFCCCB"], ["#FFFFE0", "#FFCCCB", "#98FB98"]]
    table = Table(ax, bbox=[0, 0, 1, 1])
    for i, row in enumerate(table_data):
        for j, cell_text in enumerate(row):
            table.add_cell(i, j, width=1 / len(row), height=1 / len(table_data), text=cell_text, loc="center", facecolor=table_colors[i][j])
    ax.add_table(table)

    metrics_text = "\n".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
    plt.figtext(0.95, 0.5, metrics_text, fontsize=10, ha="left", va="center", bbox=dict(facecolor="white", alpha=0.8))
    plt.title(title, fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "confusion_matrix.png"))
    plt.close(fig)

def save_average_metrics(avg_metrics, confusion_matrices, output_dir):
    """
    Save average metrics across all folds.
    """
    # Compute average confusion matrix
    avg_conf_matrix = np.sum(confusion_matrices, axis=0)
    avg_metrics_path = os.path.join(output_dir, "average_metrics.json")
    with open(avg_metrics_path, "w") as f:
        json.dump(avg_metrics, f, indent=4)
    print("Average metrics saved:", avg_metrics)

def main():
    """
    Main function to train and evaluate XGBoost for audio classification.
    """
    # Load data
    print("Loading audio features...")
    features, labels, ids = load_audio_features(INPUT_DIR)
    print(f"Loaded {len(labels)} samples.")

    # Perform stratified splits
    splits = stratified_split(ids, labels, K_FOLDS)

    # Train and evaluate
    train_and_evaluate_xgboost(features, labels, np.arange(len(labels)), splits, OUTPUT_DIR)

if __name__ == "__main__":
    main()
