import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import pandas as pd
from torchvision.models import resnet18

# Dataset class for PyTorch
class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, label = self.data[idx]
        feature_array = np.array(features)
        height = int(np.sqrt(len(feature_array)))  # Reshape features for ResNet input
        width = len(feature_array) // height
        reshaped_features = feature_array.reshape((1, height, width))  # Single channel
        return torch.tensor(reshaped_features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ResNet Model
class ResNetModel(nn.Module):
    def __init__(self, input_channels):
        super(ResNetModel, self).__init__()
        self.resnet = resnet18(pretrained=False)
        # Adjust first conv layer for single-channel input
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        # Adjust output layer for binary classification
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, x):
        return self.resnet(x)

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

# Read data
def read_ids_from_csv_list(folder_path, pattern):
    file_paths = sorted(glob.glob(os.path.join(folder_path, pattern)))
    id_lists = []
    for file_path in file_paths:
        ids = pd.read_csv(file_path).iloc[:, 0].tolist()
        print(f"IDs loaded from {file_path}: {ids[:10]}... (total {len(ids)})")
        id_lists.append(ids)
    return id_lists

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

# Train and evaluate the model
def train_and_evaluate_with_csv_pairs(base_folder, train_val_folder, model_name="ResNet"):
    train_id_lists = read_ids_from_csv_list(train_val_folder, "train_*.csv")
    val_id_lists = read_ids_from_csv_list(train_val_folder, "val_*.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = f"result/{model_name}/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(results_dir, exist_ok=True)

    for split_idx, (train_ids, val_ids) in enumerate(zip(train_id_lists, val_id_lists)):
        print(f"Processing split {split_idx}...")

        train_data = read_data_for_ids(base_folder, train_ids)
        val_data = read_data_for_ids(base_folder, val_ids)
        if not train_data or not val_data:
            print(f"Split {split_idx}: Train or validation data missing.")
            continue

        train_loader = DataLoader(AudioDataset(train_data), batch_size=16, shuffle=True)
        val_loader = DataLoader(AudioDataset(val_data), batch_size=16, shuffle=False)

        sample_features = train_data[0][0]
        height = int(np.sqrt(len(sample_features)))
        width = len(sample_features) // height

        model = ResNetModel(1).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_f1 = 0
        best_model_path = os.path.join(results_dir, f"best_model_split_{split_idx}.pth")
        for epoch in range(200):
            model.train()
            total_loss = 0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds)

        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved for split {split_idx} with F1 score: {best_f1:.4f}")

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
def test_saved_model_with_val_sets(saved_model_path, base_folder, train_val_folder, model_name="ResNet"):
    val_id_lists = read_ids_from_csv_list(train_val_folder, "val_*.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = f"result/{model_name}/aug_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(results_dir, exist_ok=True)

    sample_features = read_data_for_ids(base_folder, val_id_lists[0])[0][0]
    height = int(np.sqrt(len(sample_features)))
    width = len(sample_features) // height
    model = ResNetModel(1).to(device)
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()

    for split_idx, val_ids in enumerate(val_id_lists):
        print(f"Testing saved model on split {split_idx}...")

        val_data = read_data_for_ids(base_folder, val_ids)
        if not val_data:
            print(f"Split {split_idx}: Validation data missing.")
            continue

        val_loader = DataLoader(AudioDataset(val_data), batch_size=16, shuffle=False)

        y_true, y_pred = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                y_true.extend(labels.cpu().numpy)
                y_pred.extend(preds)

        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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
    saved_model_path = "result/best_model/ResNet.pth"

    # Test the saved model
    test_saved_model_with_val_sets(saved_model_path, base_folder, train_val_folder)

    # Train and evaluate the model
    train_and_evaluate_with_csv_pairs(base_folder, train_val_folder)

    
