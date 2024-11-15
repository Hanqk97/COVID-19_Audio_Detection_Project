from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Train SVM
svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
svm_model.fit(train_features, train_labels)

# Validate
val_predictions = svm_model.predict(val_features)
print("Validation Accuracy:", accuracy_score(val_labels, val_predictions))
print(classification_report(val_labels, val_predictions))
