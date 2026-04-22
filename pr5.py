# ==========================================
# LOGISTIC REGRESSION - SOCIAL NETWORK ADS
# ==========================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# 2. Load Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
print("Dataset Preview:\n", dataset.head())

# 3. Select Features & Target
X = dataset.iloc[:, [2, 3]].values   # Age, Salary
y = dataset.iloc[:, 4].values        # Purchased

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# 5. Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 6. Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. Prediction
y_pred = model.predict(X_test)

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# 9. Extract TP, FP, TN, FN
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

print("\nTP:", TP)
print("FP:", FP)
print("TN:", TN)
print("FN:", FN)

# 10. Performance Metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
error_rate = 1 - accuracy
precision = TP / (TP + FP)
recall = TP / (TP + FN)

print("\nAccuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)

# ==========================================
# 11. CONFUSION MATRIX HEATMAP (SAVE AS PNG)
# ==========================================

plt.figure(figsize=(6, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['No', 'Yes'],
            yticklabels=['No', 'Yes'])

plt.title(f"Confusion Matrix (Accuracy = {accuracy:.2f})")
plt.xlabel("Predicted")
plt.ylabel("Actual")


plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')

plt.close()  

# ==========================================
# CONCLUSION
# ==========================================
print("\nConclusion: Logistic Regression model achieved high accuracy and effectively classified user purchase behavior.")
