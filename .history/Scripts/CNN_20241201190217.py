import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime

# Dataset Class
class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, label = self.data[idx]
        feature_array = np.array(features)
        height = int(np.sqrt(len(feature_array)))  # Use square dimensions for simplicity
        width = len(feature_array) // height
        reshaped_features = feature_array.reshape((1, height, width))  # Add channel dimension
        print(f"Reshaped feature shape: {reshaped_features.shape}")  # Debugging print
        return torch.tensor(reshaped_features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# Read and preprocess data
def read_data(base_folder):
    print("Starting to read data...")
    data = []
    total_files = 0

    # Traverse the directory structure
    for label in ['negative', 'positive']:
        label_folder = os.path.join(base_folder, label)
        for gender in ['male', 'female']:
            gender_folder = os.path.join(label_folder, gender)
            if not os.path.exists(gender_folder):
                continue

            # Collect all JSON files
            file_list = [f for f in os.listdir(gender_folder) if f.endswith('.json')]
            total_files += len(file_list)

            for file in file_list:
                file_path = os.path.join(gender_folder, file)
                try:
                    with open(file_path, 'r') as f:
                        sample = json.load(f)
                    
                    # Extract features
                    features = []
                    features.extend(sample['cough']['mfcc']['mean'])  # Cough features
                    features.extend(sample['breathing']['mfcc']['mean'])  # Breathing features
                    features.extend(sample['speech']['mfcc']['mean'])  # Speech features
                    
                    # Determine label
                    label_value = 1 if label == 'positive' else 0
                    data.append((features, label_value))
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    
    print(f"Data reading completed. Total files read: {total_files}")
    print(f"Total samples loaded: {len(data)}")
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

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CNN Model
class CNNModel(nn.Module):
    def __init__(self, input_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        # Dynamically compute the flattened size
        self.fc1_input_dim = self._compute_flattened_dim(input_dim)
        self.fc1 = nn.Linear(self.fc1_input_dim, 256)
        self.fc2 = nn.Linear(256, 2)

    def _compute_flattened_dim(self, input_dim, height, width):
        dummy_input = torch.zeros(1, 1, height, width)  # Dummy batch of size 1
        x = self.pool(self.relu(self.conv1(dummy_input)))
        print("After Conv1 + Pool:", x.shape)
        x = self.pool(self.relu(self.conv2(x)))
        print("After Conv2 + Pool:", x.shape)
        x = self.pool(self.relu(self.conv3(x)))
        print("After Conv3 + Pool:", x.shape)
        return x.numel()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        print("Input shape:", x.shape)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        print("After Conv1 + Pool:", x.shape)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        print("After Conv2 + Pool:", x.shape)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        print("After Conv3 + Pool:", x.shape)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training and evaluation updates
def train_and_evaluate(data, model_name="CNN"):
    print("Starting model training and evaluation...")
    results_dir = f"result/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(results_dir, exist_ok=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = []
    fold = 1

    for train_idx, val_idx in kf.split(data):
        print(f"Starting fold {fold}...")
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]

        train_loader = DataLoader(AudioDataset(train_data), batch_size=16, shuffle=True)
        val_loader = DataLoader(AudioDataset(val_data), batch_size=16, shuffle=False)

        input_dim = int(np.sqrt(len(train_data[0][0])))  # Assuming square log-mel spectrogram
        model = CNNModel(input_dim).to(device)  # Use device
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("Starting training...")
        for epoch in range(20):  # Increased epochs for better training
            model.train()
            total_loss = 0.0
            for i, (features, labels) in enumerate(train_loader):
                features, labels = features.to(device), labels.to(device)  # Use device
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/20, Loss: {total_loss:.4f}")

        print("Starting validation...")
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)  # Use device
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

        metrics.append((f1, precision, recall, accuracy, auc))
        print(f"Fold {fold} metrics: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, "
              f"Accuracy={accuracy:.4f}, AUC={auc:.4f}")

        plot_confusion_matrix(cm, ["Negative", "Positive"], os.path.join(results_dir, f"cm_fold{fold}.png"))
        fold += 1

    # Save average metrics
    avg_metrics = np.mean(metrics, axis=0)
    print("\nAverage metrics:")
    print(f"F1 Score: {avg_metrics[0]:.4f}")
    print(f"Precision: {avg_metrics[1]:.4f}")
    print(f"Recall (Sensitivity): {avg_metrics[2]:.4f}")
    print(f"Accuracy: {avg_metrics[3]:.4f}")
    print(f"AUC: {avg_metrics[4]:.4f}")

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump({
            "f1_score": avg_metrics[0],
            "precision": avg_metrics[1],
            "recall": avg_metrics[2],
            "accuracy": avg_metrics[3],
            "auc": avg_metrics[4]
        }, f, indent=4)

    print(f"Metrics saved to {results_dir}")

# Main execution
if __name__ == "__main__":
    data = read_data("data/extracted_features")
    train_and_evaluate(data)
