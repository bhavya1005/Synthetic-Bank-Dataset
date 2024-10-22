import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import SimpleImputer
import os

st.set_page_config(
    page_title="Customer Household Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Data Loading and Cleaning
@st.cache_data
def load_and_clean_data(filepath):
    """
    Loads and cleans the customer dataset.

    Parameters:
    - filepath (str): Path to the 'customers.csv' file.

    Returns:
    - pd.DataFrame: Cleaned main DataFrame.
    - pd.DataFrame: Cleaned Household DataFrame.
    """
    df = pd.read_csv(filepath, na_values=['', ' ', 'NA', 'NaN'])
    if 'Created_At' in df.columns:
        df['Created_At'] = pd.to_datetime(df['Created_At'], errors='coerce')

    categorical_cols = ['Last_Name', 'State']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
        else:
            st.warning(f"Warning: Column '{col}' not found in the dataset.")

    numerical_cols = ['Income', 'Credit_Score', 'Age']
    imputer = SimpleImputer(strategy='mean')
    for col in numerical_cols:
        if col in df.columns:
            df[col] = imputer.fit_transform(df[[col]])
        else:
            st.warning(f"Warning: Numerical column '{col}' not found in the dataset.")
    
    for col in categorical_cols:
        if col in df.columns:
            if 'Unknown' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(['Unknown'])
            df[col] = df[col].fillna('Unknown')
        else:
            st.warning(f"Warning: Categorical column '{col}' not found in the dataset.")
            
    household_df = df.groupby(['Last_Name', 'Address', 'State'], observed=True).agg({
        'Income': 'sum',
        'Credit_Score': 'mean',
        'Age': 'mean',
        'First_Name': 'count'
    }).reset_index()
    household_df.rename(columns={
        'Income': 'Total_Income',
        'Credit_Score': 'Average_Credit_Score',
        'Age': 'Average_Age',
        'First_Name': 'Number_of_People'
    }, inplace=True)

    string_cols = ['First_Name', 'Last_Name', 'Address']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            st.warning(f"Warning: String column '{col}' not found in the dataset.")

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.title()
        else:
            st.warning(f"Warning: Categorical column '{col}' not found in the dataset.")

    missing_values = df.isnull().sum()
    if missing_values.any():
        st.warning("There are still missing values in the dataset:")
        st.write(missing_values[missing_values > 0])
    else:
        st.success("No missing values remain in the dataset.")

    return df, household_df

data_filepath = 'customers.csv'
df, household_df = load_and_clean_data(data_filepath)

st.sidebar.title("📑 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Customer Insights", "Household Detection", "Future Studies"])

def home_page():
    st.title("🏦 Bank Household Analysis Dashboard")
    st.markdown("""
    ## **Introduction**
    Welcome to the **Bank Household Analysis Dashboard**.
    """)

    if df is not None:
        st.write(df.describe())
        st.markdown("---")
        st.header("🔥 Correlation Heatmap")
        corr = df[['Income', 'Credit_Score', 'Age']].corr()
        fig_heatmap = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Viridis',
            title='Correlation Matrix'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.error("Dataset not loaded. Please ensure 'customers.csv' is in the project directory.")

def customer_insights_page():
    if df is None:
        st.error("Dataset not loaded. Please ensure 'customers.csv' is in the project directory.")
        return
    st.title("📊 Customer Insights")
    st.markdown("---")
    st.sidebar.header("🔧 Filter Options")

    selected_states = st.sidebar.multiselect(
        "Select States:",
        options=sorted(df['State'].unique()),
        default=sorted(df['State'].unique())
    )

    min_age = int(df['Age'].min())
    max_age = int(df['Age'].max())
    selected_age = st.sidebar.slider(
        "Select Age Range:",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age)
    )

    min_income = int(df['Income'].min())
    max_income = int(df['Income'].max())
    selected_income = st.sidebar.slider(
        "Select Income Range:",
        min_value=min_income,
        max_value=max_income,
        value=(min_income, max_income)
    )

    min_credit = int(df['Credit_Score'].min())
    max_credit = int(df['Credit_Score'].max())
    selected_credit = st.sidebar.slider(
        "Select Credit Score Range:",
        min_value=min_credit,
        max_value=max_credit,
        value=(min_credit, max_credit)
    )

    filtered_df = df[
        (df['State'].isin(selected_states)) &
        (df['Age'] >= selected_age[0]) &
        (df['Age'] <= selected_age[1]) &
        (df['Income'] >= selected_income[0]) &
        (df['Income'] <= selected_income[1]) &
        (df['Credit_Score'] >= selected_credit[0]) &
        (df['Credit_Score'] <= selected_credit[1])
    ]
    st.sidebar.markdown(f"### Total Customers: {filtered_df.shape[0]}")
    st.markdown(f"### 🔎 Displaying {filtered_df.shape[0]} out of {df.shape[0]} customers based on selected filters.")
    st.markdown("---")

def household_detection_page():
    if df is None or household_df is None:
        st.error("Dataset not loaded. Please ensure 'customers.csv' is in the project directory.")
        return
    st.title("🏠 Household Detection")
    st.markdown("---")
    st.sidebar.header("🔧 Filter Options")

    selected_states_for_plot = st.sidebar.multiselect(
        "Select States for Household Size Analysis:",
        options=sorted(household_df['State'].unique()),
        default=sorted(household_df['State'].unique())
    )

    filtered_household_state_df = household_df[household_df['State'].isin(selected_states_for_plot)]

    household_size_counts = filtered_household_state_df['Number_of_People'].value_counts().reset_index()
    household_size_counts.columns = ['Number_of_People', 'Number_of_Households']
    household_size_counts = household_size_counts.sort_values(by='Number_of_People')

    fig_household_size = px.bar(
        household_size_counts,
        x='Number_of_People',
        y='Number_of_Households',
        title='Number of Households by Household Size',
        labels={'Number_of_People': 'Number of People in Household', 'Number_of_Households': 'Number of Households'},
        color='Number_of_Households',
        color_continuous_scale='Viridis'
    )

    fig_household_size.update_layout(
        xaxis_title='Number of People in Household',
        yaxis_title='Number of Households',
        hovermode='closest'
    )

    fig_household_size.update_traces(
        hovertemplate='<b>Household Size: %{x}</b><br>Number of Households: %{y}<extra></extra>'
    )

    st.plotly_chart(fig_household_size, use_container_width=True)

def future_studies_page():
    st.title("🔮 Future Studies")
    st.markdown("## **Conclusion**")

if page == "Home":
    home_page()
elif page == "Customer Insights":
    customer_insights_page()
elif page == "Household Detection":
    household_detection_page()
elif page == "Future Studies":
    future_studies_page()
else:
    st.error("Page not found.")
