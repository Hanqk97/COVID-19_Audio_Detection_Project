import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Define paths
FEATURES_PATH = '/Users/jonahgloss/Downloads/Data/metadata_augmented.csv'  # Update this to your CSV path if needed
AUDIO_BASE_PATH = '/Users/jonahgloss/Downloads/Data/Processed/'
MODELS_PATH = '/Users/jonahgloss/Downloads/Data/Processed/'
os.makedirs(MODELS_PATH, exist_ok=True)

# Load and preprocess data
data = pd.read_csv(FEATURES_PATH)

# Function to construct the audio file path
def get_audio_file_path(row):
    """
    Constructs the correct path to the audio file based on the CSV row information.
    """
    folder = "positive" if row['COVID_STATUS'] == 'p' else "negative"
    subfolder = row['TYPE']  # Either breathing, cough, or speech
    augmentation_suffix = f"_aug_{row['augmented']}" if row['augmented'] == 1 else ""
    file_name = f"{row['SUB_ID']}"
    #print(file_name)# Adjust suffix here if needed
    return os.path.join(AUDIO_BASE_PATH, subfolder, folder, file_name)

# Generate audio file paths
data['audio_path'] = data.apply(get_audio_file_path, axis=1)

# Check for missing audio files
missing_files = data[~data['audio_path'].apply(os.path.exists)]
if not missing_files.empty:
    print(f"Missing audio files:\n{missing_files[['SUB_ID', 'audio_path']]}")

# Filter out rows with missing files
data = data[data['audio_path'].apply(os.path.exists)]

# Raise an error if no files are left
if data.empty:
    raise ValueError("No audio files found! Check the CSV and file paths.")

# Extract features and labels
# (Modify as needed to load actual audio features from your files)
X = data.iloc[:, 2:].values  # Assumes features start at the 3rd column (adjust as needed)
y = (data['COVID_STATUS'] == 'p').astype(int).values  # Binary encoding: 1 for positive, 0 for negative

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize features to improve model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, f"{MODELS_PATH}/scaler.pkl")

# Function to train and evaluate a model
def train_and_evaluate_model(model, model_name):
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, f"{MODELS_PATH}/{model_name}.pkl")
    print(f"{model_name} saved to {MODELS_PATH}/{model_name}.pkl")

    # Evaluate model
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Train Random Forest with hyperparameter tuning
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
rf_model = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='accuracy', n_jobs=-1)
train_and_evaluate_model(rf_model, "rf_model")

# Train Support Vector Machine with hyperparameter tuning
svm_pipeline = Pipeline([
    ("svc", SVC(probability=True, random_state=42))
])
svm_params = {
    "svc__C": [0.1, 1, 10],
    "svc__kernel": ["linear", "rbf"],
    "svc__gamma": ["scale", "auto"]
}
svm_model = GridSearchCV(svm_pipeline, svm_params, cv=3, scoring='accuracy', n_jobs=-1)
train_and_evaluate_model(svm_model, "svm_model")

