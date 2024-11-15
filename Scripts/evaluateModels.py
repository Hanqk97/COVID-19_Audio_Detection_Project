import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
import joblib

FEATURES_PATH = 'Data/Features/features.csv'
MODELS_PATH = 'models'

# Load data
data = pd.read_csv(FEATURES_PATH)
X = data.iloc[:, 2:].values
y = (data['COVID_STATUS'] == 'positive').astype(int).values

# Load models
rf_model = joblib.load(f"{MODELS_PATH}/rf_model.pkl")
svm_model = joblib.load(f"{MODELS_PATH}/svm_model.pkl")

# Evaluate
for model, name in [(rf_model, "Random Forest"), (svm_model, "SVM")]:
    y_prob = model.predict_proba(X)[:, 1]
    auc_score = roc_auc_score(y, y_prob)
    print(f"{name} AUC-ROC: {auc_score}")
