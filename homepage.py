# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import os

# ---------------------------
# 1. App Configuration
# ---------------------------

st.set_page_config(
    page_title="Bank Household Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# 2. Title and Introduction
# ---------------------------

st.title("🏦 Bank Household Analysis Dashboard")
st.markdown("""
Welcome to the **Bank Household Analysis Dashboard**. This dashboard provides insights into how banks determine households based on customer data. 

**Key Features of This Dashboard:**
- **Dataset Overview:** View the complete dataset.
- **Summary Statistics:** Get a statistical summary of key variables.
- **Customer Distribution:** Visualize the number of customers in each state.
- **Correlation Heatmap:** Understand relationships between numerical features.

*Navigate through the sidebar to explore different aspects of the data.*
""")

# ---------------------------
# 3. Load and Clean the Dataset
# ---------------------------

@st.cache_data
def load_and_clean_data(filepath):
    # Check if the file exists
    if not os.path.exists(filepath):
        st.error(f"The file '{filepath}' does not exist in the current directory.")
        return None
    
    # Load the dataset
    df = pd.read_csv(filepath, na_values=[''])
    
    # Data Cleaning
    # a. Convert 'Created_At' to datetime
    if 'Created_At' in df.columns:
        df['Created_At'] = pd.to_datetime(df['Created_At'], errors='coerce')
    
    # b. Convert categorical columns to 'category' dtype
    categorical_cols = ['Last_Name', 'State', 'Role']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # c. Handle Missing Values Using Mean Imputation with SimpleImputer
    numerical_cols = ['Income', 'Credit_Score', 'Age']
    imputer = SimpleImputer(strategy='mean')
    for col in numerical_cols:
        if col in df.columns:
            df[col] = imputer.fit_transform(df[[col]])
    
    # d. Add Gender Column Assuming 'Role' Represents Gender
    role_to_gender = {
        'Primary': 'Male',
        'Secondary': 'Female'
    }
    if 'Role' in df.columns:
        df['Gender'] = df['Role'].map(role_to_gender)
        # Add 'Unknown' to categories before filling NaNs
        if df['Gender'].dtype.name == 'category':
            df['Gender'] = df['Gender'].cat.add_categories(['Unknown'])
        else:
            df['Gender'] = df['Gender'].astype('category')
            df['Gender'] = df['Gender'].cat.add_categories(['Unknown'])
        # Fill NaN values with 'Unknown'
        df['Gender'] = df['Gender'].fillna('Unknown')
    else:
        df['Gender'] = 'Unknown'
        df['Gender'] = df['Gender'].astype('category')
    
    # e. Add Income_per_Age Feature
    if 'Income' in df.columns and 'Age' in df.columns:
        df['Income_per_Age'] = df['Income'] / df['Age']
    
    return df

# Load the data
data_filepath = 'customers.csv'
df = load_and_clean_data(data_filepath)

if df is not None:
    # Display Dataset Information
    st.header("📊 Dataset Overview")
    st.write(f"**Total Records:** {df.shape[0]}")
    st.write(f"**Total Features:** {df.shape[1]}")
    
    # Display the dataset
    st.subheader("🔍 View the Dataset")
    st.dataframe(df.head(100))  # Display first 100 rows for better performance
    
    # Display Summary Statistics
    st.subheader("📈 Summary Statistics")
    st.write(df.describe())
    
    # ---------------------------
    # 4. Plot: Number of Customers in Each State
    # ---------------------------
    
    st.subheader("📍 Number of Customers per State")
    
    # Aggregate data
    state_counts = df['State'].value_counts().reset_index()
    state_counts.columns = ['State', 'Number_of_Customers']
    
    # Create Plotly bar chart
    fig_customers_state = px.bar(
        state_counts.sort_values('Number_of_Customers', ascending=False),
        x='Number_of_Customers',
        y='State',
        orientation='h',
        title='Number of Customers in Each State',
        labels={'Number_of_Customers': 'Number of Customers', 'State': 'State'},
        color='Number_of_Customers',
        color_continuous_scale='Viridis'
    )
    fig_customers_state.update_layout(yaxis={'categoryorder':'total ascending'})
    
    st.plotly_chart(fig_customers_state, use_container_width=True)
    
    # ---------------------------
    # 5. Plot: Correlation Heatmap
    # ---------------------------
    
    st.subheader("🧮 Correlation Heatmap")
    
    # Compute correlation matrix
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    # Create Plotly heatmap
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        title='Correlation Matrix of Numerical Features'
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # ---------------------------
    # 6. Footer or Additional Information (Optional)
    # ---------------------------
    
    st.markdown("---")
    st.markdown("""
    **Note:** This dashboard provides a preliminary analysis of how households are determined based on customer data. Further insights and detailed analyses can be explored in subsequent pages.
    """)
