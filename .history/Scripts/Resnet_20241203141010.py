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
from torchvision.models import resnet18

# Dataset Class
class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, label = self.data[idx]
        feature_array = np.array(features)
        height = int(np.sqrt(len(feature_array)))  # Calculate height
        width = len(feature_array) // height      # Calculate width
        reshaped_features = feature_array.reshape((1, height, width))  # Add channel dimension
        return torch.tensor(reshaped_features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Read and preprocess data
def read_data(base_folder):
    print("Starting to read data...")
    data = []
    total_files = 0

    for label in ['negative', 'positive']:
        label_folder = os.path.join(base_folder, label)
        for gender in ['male', 'female']:
            gender_folder = os.path.join(label_folder, gender)
            if not os.path.exists(gender_folder):
                continue

            file_list = [f for f in os.listdir(gender_folder) if f.endswith('.json')]
            total_files += len(file_list)

            for file in file_list:
                file_path = os.path.join(gender_folder, file)
                try:
                    with open(file_path, 'r') as f:
                        sample = json.load(f)
                    
                    features = []
                    features.extend(sample['cough']['mfcc']['mean'])
                    features.extend(sample['breathing']['mfcc']['mean'])
                    features.extend(sample['speech']['mfcc']['mean'])
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

# ResNet Model
class ResNetModel(nn.Module):
    def __init__(self, input_channels):
        super(ResNetModel, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, x):
        return self.resnet(x)

# Training and evaluation
def train_and_evaluate(data, model_name="ResNet"):
    print("Starting model training and evaluation...")
    results_dir = f"result/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(results_dir, exist_ok=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = []

    sample_features = data[0][0]
    input_length = len(sample_features)
    height = int(np.sqrt(input_length))
    width = input_length // height

    best_f1 = 0  # Track the best F1 score
    best_model_path = os.path.join(results_dir, "best_model.pth")  # Path to save the best model

    for fold, (train_idx, val_idx) in enumerate(kf.split(data), 1):
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        train_loader = DataLoader(AudioDataset(train_data), batch_size=16, shuffle=True)
        val_loader = DataLoader(AudioDataset(val_data), batch_size=16, shuffle=False)

        model = ResNetModel(1).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print(f"Starting fold {fold}...")
        for epoch in range(200):
            model.train()
            total_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        # Evaluate the model
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds)

        # Calculate metrics
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        # Save the best model based on F1 score
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved for fold {fold} with F1 score: {best_f1:.4f}")

        metrics.append((f1, precision, recall, accuracy, auc))
        plot_confusion_matrix(cm, ["Negative", "Positive"], os.path.join(results_dir, f"cm_fold{fold}.png"))

    avg_metrics = np.mean(metrics, axis=0)
    print("\nAverage metrics:")
    print(f"F1 Score: {avg_metrics[0]:.4f}")
    print(f"Precision: {avg_metrics[1]:.4f}")
    print(f"Recall (Sensitivity): {avg_metrics[2]:.4f}")
    print(f"Accuracy: {avg_metrics[3]:.4f}")
    print(f"AUC: {avg_metrics[4]:.4f}")

    # Save final metrics
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump({
            "f1_score": avg_metrics[0],
            "precision": avg_metrics[1],
            "recall": avg_metrics[2],
            "accuracy": avg_metrics[3],
            "auc": avg_metrics[4]
        }, f, indent=4)

    print(f"Best model saved at {best_model_path} with F1 score: {best_f1:.4f}")


# Main execution
if __name__ == "__main__":
    data = read_data("data/extracted_features")
    train_and_evaluate(data)
