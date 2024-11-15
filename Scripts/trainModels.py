import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

FEATURES_PATH = 'Data/Features/features.csv'
MODELS_PATH = 'models'
os.makedirs(MODELS_PATH, exist_ok=True)

# Load data
data = pd.read_csv(FEATURES_PATH)
X = data.iloc[:, 2:].values
y = (data['COVID_STATUS'] == 'positive').astype(int).values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, f"{MODELS_PATH}/rf_model.pkl")

# Train SVM
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, f"{MODELS_PATH}/svm_model.pkl")

# Evaluate
y_pred = rf_model.predict(X_test)
print("Random Forest Performance:")
print(classification_report(y_test, y_pred))
