# t# Telco Customer Churn Analysis – Capstone Project

# 📌 1. Introduction

"Churn" refers to customers leaving a service. In the telecom industry, understanding the factors that lead to churn helps businesses retain customers, improve service quality, and increase profitability.

In this notebook, we will:
- Load and clean the Telco Customer Churn dataset
- Explore the data with visualizations
- Identify key churn factors

---

# 📥 2. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Configure default styling
sns.set(style='whitegrid')
%matplotlib inline

# 📂 3. Load Dataset
# Note: You can upload 'WA_Fn-UseC_-Telco-Customer-Churn.csv' manually to run this locally.
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()

# 🧹 4. Data Cleaning

# Drop customerID (not useful for analysis)
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric (some non-numeric values exist)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check for missing values
missing_values = df.isnull().sum()

# Drop rows with missing TotalCharges
df.dropna(inplace=True)

# Convert 'Churn' to binary
# Yes -> 1, No -> 0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Convert 'SeniorCitizen' to categorical
df['SeniorCitizen'] = df['SeniorCitizen'].map({1: 'Yes', 0: 'No'})

# Print cleaned data info
df.info()

# ✅ Cleaned data is ready for EDA and modeling

# 📊 5. Exploratory Data Analysis (EDA)

# Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='Set2')
plt.title('Churn Count')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# Churn by Gender
plt.figure(figsize=(6, 4))
sns.countplot(x='gender', hue='Churn', data=df, palette='Set1')
plt.title('Churn by Gender')
plt.ylabel('Count')
plt.show()

# Churn by Contract Type
plt.figure(figsize=(8, 5))
sns.countplot(x='Contract', hue='Churn', data=df, palette='Set3')
plt.title('Churn by Contract Type')
plt.ylabel('Count')
plt.xticks(rotation=15)
plt.show()

# Monthly Charges Distribution by Churn
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df[df['Churn'] == 0]['MonthlyCharges'], label='No Churn', fill=True)
sns.kdeplot(data=df[df['Churn'] == 1]['MonthlyCharges'], label='Churn', fill=True)
plt.title('Monthly Charges Distribution by Churn')
plt.xlabel('Monthly Charges')
plt.legend()
plt.show()

# Total Charges Distribution by Churn
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df[df['Churn'] == 0]['TotalCharges'], label='No Churn', fill=True)
sns.kdeplot(data=df[df['Churn'] == 1]['TotalCharges'], label='Churn', fill=True)
plt.title('Total Charges Distribution by Churn')
plt.xlabel('Total Charges')
plt.legend()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 🤖 6. Predictive Modeling

# Convert categorical columns to numeric using Label Encoding
categorical_cols = df.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Split data into features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n📈 Model Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
elco-churn-analysis
