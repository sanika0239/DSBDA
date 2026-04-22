# ==========================================
# IRIS DATASET - EXPLORATORY DATA ANALYSIS
# ==========================================

# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset
df = sns.load_dataset('iris')

# Display first few rows
print("First 5 rows of dataset:")
print(df.head())

# ==========================================
# 1. Features and Their Types
# ==========================================
print("\nFeature Types:")
print(df.dtypes)

print("\nFeature Description:")
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"{col} → Categorical (Nominal)")
    else:
        print(f"{col} → Numeric (Continuous)")

# ==========================================
# 2. Histograms for Each Feature
# ==========================================
print("\nGenerating Histograms...")

df.hist(figsize=(10,8))
plt.suptitle("Histograms of Iris Features")
plt.show()

# ==========================================
# 3. Boxplots for Each Feature
# ==========================================
print("\nGenerating Boxplots...")

for col in df.columns[:-1]:  # exclude species
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# ==========================================
# 4. Compare Distributions & Identify Outliers
# ==========================================
print("\nStatistical Summary:")
print(df.describe())

print("\nOutlier Detection using IQR Method:")

for col in df.columns[:-1]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    print(f"\n{col}:")
    print(f"Lower Bound = {lower_bound}, Upper Bound = {upper_bound}")
    print(f"Number of Outliers = {len(outliers)}")

# ==========================================
# FINAL INFERENCE (PRINTED)
# ==========================================
print("\nFinal Inference:")
print("1. All features except 'species' are numeric.")
print("2. Petal length and width show clear separation between species.")
print("3. Sepal width contains some outliers.")
print("4. Dataset is clean and suitable for machine learning.")