#pip install numpy pandas scikit-learn librosa torch matplotlib

import pandas as pd
from sklearn.model_selection import train_test_split

# Load augmented metadata
metadata = pd.read_csv('Data/metadata_augmented.csv')

# Map COVID_STATUS to numerical labels
metadata['label'] = metadata['COVID_STATUS'].map({'p': 1, 'n': 0})

# Split into train, validation, and test sets
train_data, test_data = train_test_split(metadata, test_size=0.2, random_state=42, stratify=metadata['label'])
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['label'])

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")
