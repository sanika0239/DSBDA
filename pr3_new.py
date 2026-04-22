
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("insurance.csv")
print("First 5 rows:\n")
print(df.head())



bins = [0, 25, 40, 60, 100]
labels = ["Young (0-25)", "Adult (26-40)", "Middle Age (41-60)", "Senior (60+)"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
print("\nAge groups created successfully.\n")


grouped_stats = df.groupby("age_group")["charges"].agg(
    ["mean", "median", "min", "max", "std"]
)
print("Summary Statistics of Income (charges) grouped by Age Groups:\n")
print(grouped_stats)

mean_income_list = df.groupby("age_group")["charges"].mean().tolist()

print("\nMean Income List (per Age Group):")
print(mean_income_list)




# Skewed distribution
plt.figure()
plt.hist(df["charges"], bins=30)
plt.title("Histogram of Insurance Charges (Skewed Distribution)")
plt.xlabel("Charges")
plt.ylabel("Frequency")
plt.savefig("charges_skewed_hist.png")
plt.close()



url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df = pd.read_csv(url, header=None, names=column_names)

setosa = df[df["species"] == "Iris-setosa"]
versicolor = df[df["species"] == "Iris-versicolor"]
virginica = df[df["species"] == "Iris-virginica"]

print("=== Iris-setosa Statistics ===")
print(setosa.describe())

print("\n=== Iris-versicolor Statistics ===")
print(versicolor.describe())

print("\n=== Iris-virginica Statistics ===")
print(virginica.describe())


# Keep only numeric columns
numeric_df = df.drop(columns=["species"])

# Correlation matrix
corr = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(6, 5))
plt.imshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar()
plt.title("Correlation Heatmap - Iris Dataset")
plt.tight_layout()
plt.savefig("iris_correlation_heatmap.png")  
plt.close()