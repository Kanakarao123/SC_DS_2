# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("train.csv")

# ----------------------------
# Initial Exploration
# ----------------------------
print("Initial Data Info:\n")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary Statistics:\n", df.describe())

# ----------------------------
# Data Cleaning
# ----------------------------

# Fill missing Age with median
df.loc[df['Age'].isna(), 'Age'] = df['Age'].median()


# Fill missing Embarked with mode
df.loc[df['Embarked'].isna(), 'Embarked'] = df['Embarked'].mode()[0]


# Drop Cabin column due to too many missing values
df.drop('Cabin', axis=1,inplace=True)

# ----------------------------
# Feature Engineering
# ----------------------------

# Create FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Extract Title from Name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt','Col','Don','Dr',
                                   'Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Drop unused columns
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1,inplace=True)

# Convert categorical variables using one-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

# ----------------------------
# Exploratory Data Analysis (EDA)
# ----------------------------

# Set plot style
sns.set(style='whitegrid')

# 1. Survival rate by gender
plt.figure(figsize=(6,4))
sns.barplot(x='Sex_male', y='Survived', data=df)
plt.title('Survival Rate by Gender (1=Male)')
plt.show()

# 2. Survival rate by Pclass
plt.figure(figsize=(6,4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.show()

# 3. Age distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# 4. Survival by Age
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True, palette='Set2')
plt.title('Survival by Age')
plt.show()

# 5. Survival by Family Size
plt.figure(figsize=(6,4))
sns.barplot(x='FamilySize', y='Survived', data=df)
plt.title('Survival Rate by Family Size')
plt.show()

# ----------------------------
# Save cleaned dataset (optional)
# ----------------------------
filename = "titanic_cleaned.csv"
df.to_csv(filename, index=False)
print(f"Cleaned dataset saved to '{filename}'")
