import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("Titanic-Dataset.csv")
print("Dataset Loaded Successfully!\n")

print("First 5 Rows of Dataset:\n")
print(df.head())


print("\n--- Checking Missing Values ---")
print(df.isnull().sum())

print("\n--- Dataset Dimensions ---")
print("Shape of Dataset:", df.shape)

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Dataset Info ---")
print(df.info())


# print("\n--- Variable Descriptions ---")
# print("""
# PassengerId : Unique ID of passenger (Integer)
# Survived    : Survival (0 = No, 1 = Yes) (Categorical/Numeric)
# Pclass      : Ticket class (1, 2, 3) (Categorical)
# Name        : Passenger name (String)
# Sex         : Gender (Categorical)
# Age         : Age in years (Numeric)
# SibSp       : Number of siblings/spouses aboard (Integer)
# Parch       : Number of parents/children aboard (Integer)
# Ticket      : Ticket number (String)
# Fare        : Ticket fare (Float)
# Cabin       : Cabin number (String - contains missing values)
# Embarked    : Port of Embarkation (Categorical - contains missing values)
# """)


print("\n--- Data Types Before Conversion ---")
print(df.dtypes)

df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')

print("\n--- Data Types After Conversion ---")
print(df.dtypes)


# df['Age'].fillna(df['Age'].mean(), inplace=True)

# df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


label_encoder = LabelEncoder()

df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
df['Pclass'] = df['Pclass'].cat.codes

print("\n--- After Encoding Categorical Variables ---")
print(df.head())


scaler = StandardScaler()
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\n--- After Normalization ---")
print(df.head())

print("\nProgram Executed Successfully!")