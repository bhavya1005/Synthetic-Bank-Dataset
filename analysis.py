import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

# Set visual style for plots
sns.set(style="whitegrid")

# ---------------------------
# 1. Load the Dataset
# ---------------------------

# Define the path to the CSV file
csv_file_path = 'customers.csv'

# Check if the file exists
if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"The file {csv_file_path} does not exist in the current directory.")

# Load the dataset with proper NA handling
df = pd.read_csv(csv_file_path, na_values=[''])

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Display DataFrame shape
print("\nDataset Shape:", df.shape)

# ---------------------------
# 2. Perform Basic Data Cleaning
# ---------------------------

# a. Convert 'Created_At' to datetime
df['Created_At'] = pd.to_datetime(df['Created_At'])

# b. Convert categorical columns to 'category' dtype
categorical_cols = ['Last_Name', 'State', 'Role']
for col in categorical_cols:
    df[col] = df[col].astype('category')

print("\nUpdated Data Types:")
print(df.dtypes)

# c. Handle Missing Values

# Identify numerical columns with missing values
numerical_cols = ['Income', 'Credit_Score', 'Age']

# Impute missing numerical values with median (avoiding inplace=True to prevent FutureWarning)
for col in numerical_cols:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)
    print(f"Imputed missing values in '{col}' with median value {median_value}")

# ---------------------------
# 3. Add a Gender Column
# ---------------------------

# Assuming 'Role' represents Gender:
# - 'Primary' -> 'Male'
# - 'Secondary' -> 'Female'

# Create a mapping dictionary
role_to_gender = {
    'Primary': 'Male',
    'Secondary': 'Female'
}

# Map 'Role' to 'Gender'
df['Gender'] = df['Role'].map(role_to_gender)

# Check if any unmapped roles exist
unmapped_roles = df['Gender'].isnull().sum()
if unmapped_roles > 0:
    print(f"Warning: {unmapped_roles} records have unmapped 'Role' values.")

# Display the first few rows with the new 'Gender' column
print("\nDataset with 'Gender' column:")
print(df.head())

# ---------------------------
# 4. Create Visualizations
# ---------------------------

# a. Number of People vs. State

plt.figure(figsize=(15, 8))
sns.countplot(data=df, y='State', order=df['State'].value_counts().index, palette='viridis')
plt.title('Number of People per State')
plt.xlabel('Count')
plt.ylabel('State')
plt.tight_layout()
plt.show()

# b. Income vs. State

plt.figure(figsize=(15, 8))
sns.boxplot(data=df, x='State', y='Income', palette='coolwarm')
plt.title('Income Distribution per State')
plt.xlabel('State')
plt.ylabel('Income')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# c. Credit Scores vs. State

plt.figure(figsize=(15, 8))
sns.boxplot(data=df, x='State', y='Credit_Score', palette='coolwarm')
plt.title('Credit Score Distribution per State')
plt.xlabel('State')
plt.ylabel('Credit Score')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# d. Percentage of Males and Females in the Dataset

# Calculate gender counts
gender_counts = df['Gender'].value_counts()

# Plot pie chart
plt.figure(figsize=(6,6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'], startangle=140)
plt.title('Percentage of Males and Females')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# ---------------------------
# 5. Provide Basic Statistical Summaries
# ---------------------------

print("\n--- Basic Statistical Summaries ---\n")

# a. Descriptive Statistics for Numerical Columns
print("Descriptive Statistics for Numerical Columns:")
print(df[numerical_cols + ['Income_per_Age']].describe())

# b. Frequency Counts for Categorical Columns
print("\nFrequency Counts for Categorical Columns:")
for col in categorical_cols + ['Gender']:
    print(f"\n{col} Value Counts:")
    print(df[col].value_counts())

# c. Correlation Analysis (Optional)
# Note: Excluding non-numeric columns to prevent errors
numeric_df = df.select_dtypes(include=[np.number])

# Compute correlation matrix
corr_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(corr_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# ---------------------------
# 6. Use Appropriate Encoding in Visualizations
# ---------------------------

# For visualizations, encoding may not be strictly necessary as plots can handle categorical data.
# However, for some advanced visualizations or machine learning tasks, encoding is essential.
# Below is an example of encoding for potential future use.

# a. One-Hot Encoding for 'Last_Name' and 'State'
# Note: 'Last_Name' has 1,000 unique categories which can lead to high dimensionality.
# Consider frequency encoding or target encoding for better efficiency.

# For demonstration, we'll proceed with One-Hot Encoding but drop first to avoid multicollinearity
df_encoded = pd.get_dummies(df, columns=['Last_Name', 'State'], drop_first=True)
print("\nApplied One-Hot Encoding to 'Last_Name' and 'State'.")

# b. Label Encoding for 'Gender'
# Since 'Gender' is binary, label encoding is straightforward
le_gender = LabelEncoder()
df_encoded['Gender'] = le_gender.fit_transform(df_encoded['Gender'])

print("Applied Label Encoding to 'Gender'.")

# Display the first few rows of the encoded DataFrame
print("\nFirst 5 rows of the Encoded DataFrame:")
print(df_encoded.head())

# ---------------------------
# End of Analysis
# ---------------------------
