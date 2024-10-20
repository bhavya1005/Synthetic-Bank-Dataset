import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import os

# ---------------------------
# 1. Load the Dataset
# ---------------------------

# Define the path to the CSV file
csv_file_path = 'customers.csv'

# Check if the file exists
if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"The file '{csv_file_path}' does not exist in the current directory.")

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

# c. Handle Missing Values Using Mean Imputation with SimpleImputer

# Identify numerical columns with missing values
numerical_cols = ['Income', 'Credit_Score', 'Age']

# Initialize SimpleImputer with mean strategy
imputer = SimpleImputer(strategy='mean')

# Fit and transform the numerical columns
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

print("\nImputed missing values in numerical columns with mean values.")

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
# 4. Add a Household Identifier (Optional)
# ---------------------------

# Create a unique household identifier by combining Last_Name and Address
df['Household_ID'] = df['Last_Name'].astype(str) + '_' + df['Address'].astype(str)

# ---------------------------
# 5. Create Visualizations with Plotly
# ---------------------------

# a. Number of People vs. State
household_size_state = df.groupby('State').size().reset_index(name='Number_of_People')

fig1 = px.bar(
    household_size_state.sort_values('Number_of_People', ascending=False),
    x='Number_of_People',
    y='State',
    orientation='h',
    title='Number of People per State',
    labels={'Number_of_People': 'Count', 'State': 'State'},
    color='Number_of_People',
    color_continuous_scale='Viridis'
)
fig1.update_layout(yaxis={'categoryorder':'total ascending'})
fig1.show()

# b. Income vs. State
fig2 = px.box(
    df,
    x='State',
    y='Income',
    title='Income Distribution per State',
    labels={'Income': 'Income', 'State': 'State'},
    color='State',
    color_discrete_sequence=px.colors.sequential.RdBu
)
fig2.update_xaxes(categoryorder='total descending')
fig2.show()

# c. Credit Scores vs. State
fig3 = px.box(
    df,
    x='State',
    y='Credit_Score',
    title='Credit Score Distribution per State',
    labels={'Credit_Score': 'Credit Score', 'State': 'State'},
    color='State',
    color_discrete_sequence=px.colors.sequential.PuBu
)
fig3.update_xaxes(categoryorder='total descending')
fig3.show()

# d. Percentage of Males and Females in the Dataset
gender_counts = df['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']

fig4 = px.pie(
    gender_counts,
    names='Gender',
    values='Count',
    title='Percentage of Males and Females',
    color='Gender',
    color_discrete_map={'Male': 'lightblue', 'Female': 'lightcoral'}
)
fig4.show()

# ---------------------------
# 6. Provide Basic Statistical Summaries
# ---------------------------

print("\n--- Basic Statistical Summaries ---\n")

# a. Descriptive Statistics for Numerical Columns
print("Descriptive Statistics for Numerical Columns:")
# Ensure 'Income_per_Age' exists before describing
if 'Income_per_Age' in df.columns:
    print(df[numerical_cols + ['Income_per_Age']].describe())
else:
    # Create 'Income_per_Age' if not present
    df['Income_per_Age'] = df['Income'] / df['Age']
    print(df[numerical_cols + ['Income_per_Age']].describe())

# b. Frequency Counts for Categorical Columns
print("\nFrequency Counts for Categorical Columns:")
for col in categorical_cols + ['Gender']:
    print(f"\n{col} Value Counts:")
    print(df[col].value_counts())

# c. Correlation Analysis

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])

# Compute the correlation matrix
corr_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(corr_matrix)

# Visualize the correlation matrix with Plotly
fig5 = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale='RdBu',
    title='Correlation Matrix of Numerical Features'
)
fig5.show()

# ---------------------------
# 7. Use Appropriate Encoding in Visualizations
# ---------------------------

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
# 8. Group Households and Display Household Information
# ---------------------------

# a. Group Households by Same Address and Same Last Name
# This ensures that households are uniquely identified by both last name and address.

# Group by 'Last_Name' and 'Address'
household_group = df.groupby(['Last_Name', 'Address']).agg(
    Number_of_People=('First_Name', 'count'),
    Total_Income=('Income', 'sum')
).reset_index()

print("\n--- Household Information ---\n")
print(household_group.head())

# b. Print the Total Number of Households
number_of_households = household_group.shape[0]
print(f"\nTotal number of households: {number_of_households}")

# c. Sort and Display Household Information
household_group_sorted = household_group.sort_values(by='Total_Income', ascending=False)

print("\n--- Sorted Household Information (Top 5) ---\n")
print(household_group_sorted.head())

# d. Save Household Information to CSV
household_group_sorted.to_csv('household_information.csv', index=False)
print("\nHousehold information saved to 'household_information.csv'.")

# ---------------------------
# 9. Additional Visualizations with Plotly (Optional)
# ---------------------------

# a. Distribution of Number of People per Household
fig6 = px.histogram(
    household_group,
    x='Number_of_People',
    nbins=range(1, household_group['Number_of_People'].max()+2),
    title='Distribution of Number of People per Household',
    labels={'Number_of_People': 'Number of People', 'count': 'Frequency'},
    color='Number_of_People',
    color_continuous_scale='Blues'
)
fig6.update_xaxes(dtick=1)
fig6.show()

# b. Distribution of Total Income per Household
fig7 = px.histogram(
    household_group,
    x='Total_Income',
    nbins=50,
    title='Distribution of Total Income per Household',
    labels={'Total_Income': 'Total Income', 'count': 'Frequency'},
    opacity=0.75,
    color_discrete_sequence=['lightgreen']
)
fig7.update_layout(bargap=0.1)
fig7.show()
