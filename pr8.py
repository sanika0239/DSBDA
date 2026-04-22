# Import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load inbuilt Titanic dataset
df = sns.load_dataset('titanic')

# Display first few rows
print(df.head())

# Box plot: Age distribution w.r.t gender and survival
plt.figure(figsize=(8,6))

sns.boxplot(x='sex', y='age', hue='survived', data=df)

# Labels and title
plt.title("Age Distribution by Gender and Survival (Titanic Dataset)")
plt.xlabel("Gender")
plt.ylabel("Age")

plt.legend(title="Survived", labels=["No (0)", "Yes (1)"])

plt.show()