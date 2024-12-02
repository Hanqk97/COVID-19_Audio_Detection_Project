import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import (
    f1_score, recall_score, precision_score, accuracy_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from datetime import datetime
import matplotlib.pyplot as plt

# Define CNN Model
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Load Data
def load_data(data_dir):
    features, labels = [], []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                    log_mel = np.array(data["cough"]["log_mel"], dtype=np.float32)  # Example: using log_mel
                    label = 1 if data["label"] == "positive" else 0
                    features.append(log_mel)
                    labels.append(label)
    return np.array(features), np.array(labels)

# Evaluate Model
def evaluate_model(model, dataloader, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
            preds = (outputs > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return (
        f1_score(all_labels, all_preds),
        recall_score(all_labels, all_preds),
        precision_score(all_labels, all_preds),
        accuracy_score(all_labels, all_preds),
        roc_auc_score(all_labels, all_preds),
        confusion_matrix(all_labels, all_preds),
        total_loss / len(dataloader)
    )

# Train and Validate Model
def train_model(features, labels, output_dir):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = []
    fold = 1

    for train_idx, val_idx in kfold.split(features):
        print(f"Fold {fold}...")
        # Prepare Data
        train_features, train_labels = features[train_idx], labels[train_idx]
        val_features, val_labels = features[val_idx], labels[val_idx]
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_features).unsqueeze(1), torch.tensor(train_labels)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(val_features).unsqueeze(1), torch.tensor(val_labels)
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

        # Initialize Model
        model = AudioCNN().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train
        model.train()
        for epoch in range(10):
            total_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

        # Validate
        metrics.append(evaluate_model(model, val_loader, criterion))
        fold += 1

    # Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, f"CNN_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    metrics_file = os.path.join(result_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    # Generate Confusion Matrix Image
    best_fold = np.argmax([m[0] for m in metrics])  # Best F1-Score
    _, _, _, _, _, best_cm, _ = metrics[best_fold]
    disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=["Negative", "Positive"])
    disp.plot()
    plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
    plt.close()

    print("Training complete. Results saved.")

# Main
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features, labels = load_data("data/extracted_features")
train_model(features, labels, "results")
