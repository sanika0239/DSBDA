import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("housingData.csv")
df.columns = df.columns.str.upper()
print(df.head())
print(df.info())

# ------------------------------------------
# Handle Missing Values
# ------------------------------------------
print("\nMissing Values Before:\n", df.isnull().sum())

df.fillna(df.mean(), inplace=True)

print("\nMissing Values After:\n", df.isnull().sum())


print(df.describe())
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig("heatmap.png")
plt.close()
print("\n✅ Heatmap saved as heatmap.png")



sns.pairplot(df[['RM', 'LSTAT', 'PTRATIO', 'MEDV']])
plt.savefig("pairplot.png")
plt.close()

print("✅ Pairplot saved as pairplot.png")


X = df.drop('MEDV', axis=1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)



y_pred = model.predict(X_test)


comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
print("\nActual vs Predicted:\n", comparison.head())


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nRMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nFeature Coefficients:\n", coefficients)