# ==========================================
# NAIVE BAYES - IRIS DATASET
# ==========================================

# 1. Import Libraries
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# 2. Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# 3. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Train Model
model = GaussianNB()
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)

# 7. Metrics Calculation

# For multi-class, we use macro average
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
error_rate = 1 - accuracy

print("\nAccuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)

# 8. TP, FP, TN, FN (for each class)

print("\nClass-wise TP, FP, TN, FN:\n")

for i in range(len(cm)):
    TP = cm[i][i]
    FP = sum(cm[:, i]) - TP
    FN = sum(cm[i, :]) - TP
    TN = sum(sum(cm)) - (TP + FP + FN)

    print(f"Class {i}:")
    print("TP:", TP, "FP:", FP, "FN:", FN, "TN:", TN)
    print()