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
# Load the dataset 
    df = pd.read_csv(filepath, na_values=['', ' ', 'NA', 'NaN'])
#  Convert 'Created_At' to datetime if it exists
    if 'Created_At' in df.columns:
        df['Created_At'] = pd.to_datetime(df['Created_At'], errors='coerce')
 # Convert specified columns to 'category' dtype
    categorical_cols = ['Last_Name', 'State']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
        else:
            st.warning(f"Warning: Column '{col}' not found in the dataset.")

#  Handling Missing Values
# Identifying numerical and categorical columns
    numerical_cols = ['Income', 'Credit_Score', 'Age']
# Imputing missing numerical values with the mean
    imputer = SimpleImputer(strategy='mean')
    for col in numerical_cols:
        if col in df.columns:
            df[col] = imputer.fit_transform(df[[col]])
        else:
            st.warning(f"Warning: Numerical column '{col}' not found in the dataset.")
 # Handling missing categorical values by filling with 'Unknown'
    for col in categorical_cols:
        if col in df.columns:
            # Add 'Unknown' category if it's not already present
            if 'Unknown' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(['Unknown'])
            # Fill NaN with 'Unknown'
            df[col] = df[col].fillna('Unknown')
        else:
            st.warning(f"Warning: Categorical column '{col}' not found in the dataset.")
            
# Household detection 
# Creating a household dataframe
# Households are determined by shared Last_Name and Address
    household_df = df.groupby(['Last_Name', 'Address', 'State'], observed=True).agg({
        'Income': 'sum',
        'Credit_Score': 'mean',
        'Age': 'mean',
        'First_Name': 'count'  # Using 'First_Name' count as number of people in household
    }).reset_index()
    household_df.rename(columns={
        'Income': 'Total_Income',
        'Credit_Score': 'Average_Credit_Score',
        'Age': 'Average_Age',
        'First_Name': 'Number_of_People'
    }, inplace=True)

# Data Cleaning
# Convert 'Created_At' column to datetime
    if 'Created_At' in df.columns:
        df['Created_At'] = pd.to_datetime(df['Created_At'], errors='coerce')

# Trimming whitespace from string columns to ensure consistency
    string_cols = ['First_Name', 'Last_Name', 'Address']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            st.warning(f"Warning: String column '{col}' not found in the dataset.")
# Capitalizing categorical variables for consistency
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.title()
        else:
            st.warning(f"Warning: Categorical column '{col}' not found in the dataset.")

# Handling remaining missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        st.warning("There are still missing values in the dataset:")
        st.write(missing_values[missing_values > 0])
    else:
        st.success("No missing values remain in the dataset.")

# Return the cleaned DataFrames
    return df, household_df
data_filepath = 'customers.csv'
df, household_df = load_and_clean_data(data_filepath)

## Streamlit Dashboard
st.sidebar.title("📑 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Customer Insights", "Household Detection", "Future Studies"])
# Home Page
def home_page():
    st.title("🏦 Bank Household Analysis Dashboard")
    st.markdown("""
    ## **Introduction**
    The goal of this project is to determine how large banks wiyth 
    Welcome to the **Bank Household Analysis Dashboard**. This project aims to provide comprehensive insights into how banks determine households based on customer data. By analyzing various demographic and financial metrics, we seek to uncover patterns and relationships that can inform banking strategies and decision-making.
    
    ## **Dataset Overview**
    
    The dataset used in this analysis contains information about bank customers, including their personal details, financial metrics, and household information. Key columns include:
    
    - **First_Name:** Customer's first name.
    - **Last_Name:** Customer's last name.
    - **Address:** Customer's residential address.
    - **Age:** Customer's age.
    - **Income:** Customer's annual income.
    - **Credit_Score:** Customer's credit score.
    - **State:** State where the customer resides.
    - **Created_At:** Date when the customer record was created.
    
    ## **Summary Statistics**
    """)

    if df is not None:
        st.write(df.describe())
        st.markdown("---")    
# Correlation Heatmap
        st.header("🔥 Correlation Heatmap")
        st.markdown("""
        The heatmap below illustrates the correlation between different numerical variables in the dataset. Correlation coefficients range from -1 to 1, indicating the strength and direction of the relationship between variables.
        """)
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
    st.markdown("""
    This dasboard is helpful in exploring customer demographics and financial metrics to understand the distribution and relationships within the dataset. Use the interactive filters on the sidebar to customize the data view.
    """)
    st.markdown("---")  # Horizontal line for separation
    st.sidebar.header("🔧 Filter Options")
# State Filter
    selected_states = st.sidebar.multiselect(
        "Select States:",
        options=sorted(df['State'].unique()),
        default=sorted(df['State'].unique())
    )
# Age Range Filter
    min_age = int(df['Age'].min())
    max_age = int(df['Age'].max())
    selected_age = st.sidebar.slider(
        "Select Age Range:",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age)
    )
 # Income Range Filter
    min_income = int(df['Income'].min())
    max_income = int(df['Income'].max())
    selected_income = st.sidebar.slider(
        "Select Income Range:",
        min_value=min_income,
        max_value=max_income,
        value=(min_income, max_income)
    )
# Credit Score Range Filter
    min_credit = int(df['Credit_Score'].min())
    max_credit = int(df['Credit_Score'].max())
    selected_credit = st.sidebar.slider(
        "Select Credit Score Range:",
        min_value=min_credit,
        max_value=max_credit,
        value=(min_credit, max_credit)
    )
 # Apply Filters
    filtered_df = df[
        (df['State'].isin(selected_states)) &
        (df['Age'] >= selected_age[0]) &
        (df['Age'] <= selected_age[1]) &
        (df['Income'] >= selected_income[0]) &
        (df['Income'] <= selected_income[1]) &
        (df['Credit_Score'] >= selected_credit[0]) &
        (df['Credit_Score'] <= selected_credit[1])
    ]
 # Display number of records after filtering
    st.sidebar.markdown(f"### Total Customers: {filtered_df.shape[0]}")
    st.markdown(f"### 🔎 Displaying {filtered_df.shape[0]} out of {df.shape[0]} customers based on selected filters.")
    st.markdown("---")

 # ---------------------------
 # 1. Number of Customers in Each State
# ---------------------------

    st.header("📍 Number of Customers in Each State")
    st.markdown("""
    This bar chart shows the distribution of customers across different states.
    """)

    state_counts = filtered_df['State'].value_counts().reset_index()
    state_counts.columns = ['State', 'Number_of_Customers']

    fig_state = px.bar(
        state_counts.sort_values('Number_of_Customers', ascending=False),
        x='Number_of_Customers',
        y='State',
        orientation='h',
        title='Number of Customers in Each State',
        labels={'Number_of_Customers': 'Number of Customers', 'State': 'State'},
        color='State',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    fig_state.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Number of Customers",
        yaxis_title="State",
        showlegend=False
    )

    st.plotly_chart(fig_state, use_container_width=True)

    st.markdown("**Insights:** This plot helps identify states with high and low customer concentrations, informing targeted marketing and growth strategies.")
    st.markdown("---")

# ---------------------------
# 2. Income vs. State Interactive Plot
# ---------------------------

    st.header("💰 Income Distribution among Different States")
    st.markdown("""
    This plot shows theb relationship between customer income and their residing state.
    """)
    fig_income_state = px.histogram(
        filtered_df,
        x='State',
        y='Income',
        title='Income Distribution by State',
        labels={'Income': 'Annual Income', 'State': 'State'},
        color='State',
        barmode='group',
        color_discrete_sequence=px.colors.sequential.Viridis
    )

    st.plotly_chart(fig_income_state, use_container_width=True)

    st.markdown("""
    **Insights:**
    - **Income Variability:** Assess how income levels vary across different states.
    - **High-Earning States:** Identify states with significantly higher incomes, aiding in region-specific financial strategies.
    """)
    st.header("💰 Income vs. Credit Score Scatterplot")
    st.markdown("""
    This scatterplot shows the relationship between customer income and credit score.
    """)

    fig_income_credit = px.scatter(
        filtered_df,
        x='Income',
        y='Credit_Score',
        title='Income vs. Credit Score',
        labels={'Income': 'Annual Income', 'Credit_Score': 'Credit Score'},
        color='State',
        color_continuous_scale='Viridis'
    )

    st.plotly_chart(fig_income_credit, use_container_width=True)

    st.markdown("""
    This plot shows the relationship between income levels and credit scores, highlighting state-specific trends and clusters of high and low credit scores across different income levels.
    """)

    st.markdown("---")

#Age Distributions Interactive Plot
    st.header("🎂 Age Distribution")
    st.markdown("""
    This plot shows the distribution of customer ages within the dataset.
    """)
# Define age bins
    age_bins = [0, 18, 25, 35, 45, 55, 65, 75, 85, 100]
    age_labels = ['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '66-75', '76-85', '86-100']
    filtered_df['Age_Group'] = pd.cut(filtered_df['Age'], bins=age_bins, labels=age_labels, right=False)
    fig_age = px.histogram(
        filtered_df,
        x='Age_Group',
        title='Age Distribution of Customers',
        labels={'Age_Group': 'Age Group'},
        color='Age_Group',  # Use Age_Group for color differentiation
        color_discrete_sequence=px.colors.qualitative.Plotly  # Use Plotly's qualitative color sequence
    )
    st.plotly_chart(fig_age, use_container_width=True)
    st.markdown("""
    **Insights:**
    The plot is useful in understanding the diversity in age, helping in  age-targeted services and products.
    """)
    st.markdown("---")
# Credit Score Distribution Interactive Plot
    st.header("📈 Credit Score Distribution")
    st.markdown("""
    The plot shows the distribution of credit scores among customers.
    """)

    fig_credit = px.histogram(
        filtered_df,
        x='Credit_Score',
        nbins=20,
        title='Credit Score Distribution of Customers',
        labels={'Credit_Score': 'Credit Score'},
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    st.plotly_chart(fig_credit, use_container_width=True)
    st.markdown("""
    This plot shows the overall credit health of the customer base.
    """)

def household_detection_page():
    if df is None or household_df is None:
        st.error("Dataset not loaded. Please ensure 'customers.csv' is in the project directory.")
        return
    st.title("🏠 Household Detection")
    st.markdown("""
    Analyze the household structures within the customer base. Understand the distribution of households, their sizes, and financial metrics.
    """)
    st.markdown("---")  # Horizontal line for separation
    st.sidebar.header("🔧 Filter Options")
# State Filter
    selected_states = st.sidebar.multiselect(
        "Select States:",
        options=sorted(household_df['State'].unique()),
        default=sorted(household_df['State'].unique())
    )
# Apply initial filter: households with Number_of_People >1
    filtered_household_df = household_df[household_df['Number_of_People'] >1]
# Number of People in Household Filter
    min_people = 2
    max_people = int(filtered_household_df['Number_of_People'].max())
    selected_people = st.sidebar.slider(
        "Select Number of People in Household:",
        min_value=min_people,
        max_value=max_people,
        value=(min_people, max_people)
    )
# Total Income Range Filter
    min_income = int(filtered_household_df['Total_Income'].min())
    max_income = int(filtered_household_df['Total_Income'].max())
    selected_income = st.sidebar.slider(
        "Select Total Income Range:",
        min_value=min_income,
        max_value=max_income,
        value=(min_income, max_income)
    )
    filtered_household_df = filtered_household_df[
        (filtered_household_df['State'].isin(selected_states)) &
        (filtered_household_df['Number_of_People'] >= selected_people[0]) &
        (filtered_household_df['Number_of_People'] <= selected_people[1]) &
        (filtered_household_df['Total_Income'] >= selected_income[0]) &
        (filtered_household_df['Total_Income'] <= selected_income[1])
    ]

 # Display number of households after filtering
    st.sidebar.markdown(f"### Total Households: {filtered_household_df.shape[0]}")
    st.markdown(f"### 🔎 Displaying {filtered_household_df.shape[0]} out of {household_df.shape[0]} households based on selected filters.")
    st.markdown("---")

# Number of Households 
    st.header("📊 Total Number of Households")
    st.markdown("""
    The total number of households in the dataset provides an overview of the customer base's household distribution.
    """)
    total_households = filtered_household_df.shape[0]
    st.metric("Total Households", total_households)
    st.markdown("---")
    
# Household df
    st.header("📋 Household Data")
    st.markdown("""
    Below is the detailed household information, including last name, address, number of people, and total income.
    """)
    st.dataframe(filtered_household_df[['Last_Name', 'Address', 'Number_of_People', 'Total_Income']].reset_index(drop=True))
    st.markdown("---")
# Total Number of Households in Each State Plot
    st.header("📍 Total Number of Households in Each State")
    st.markdown("""
    This bar chart displays the distribution of households across different states.
    """)
    household_state_counts = filtered_household_df['State'].value_counts().reset_index()
    household_state_counts.columns = ['State', 'Number_of_Households']

    fig_household_state = px.bar(
        household_state_counts.sort_values('Number_of_Households', ascending=False),
        x='Number_of_Households',
        y='State',
        orientation='h',
        title='Number of Households in Each State',
        labels={'Number_of_Households': 'Number of Households', 'State': 'State'},
        color='Number_of_Households',
        color_continuous_scale='Viridis'
    )

    st.plotly_chart(fig_household_state, use_container_width=True)

    st.markdown("""
    This plot shows the distribution of households across different states.
    """)

    st.markdown("---")
#  Plot for average Income of a Household vs. Number of People in Household
    st.header("💸 Average Income vs. Number of People in Household")
    st.markdown("""
    Analyze the relationship between the size of a household and its total income.
    """)
    fig_income_people = px.scatter(
        filtered_household_df,
        x='Number_of_People',
        y='Total_Income',
        size='Total_Income',
        color='State',
        hover_data=['Last_Name', 'Address'],
        title='Total Income vs. Number of People in Household',
        labels={'Number_of_People': 'Number of People', 'Total_Income': 'Total Household Income'},
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_income_people, use_container_width=True)
    st.markdown("**Insights:** Understand how income scales with household size and identify households with higher income relative to the number of members.")
    st.markdown("---")

def future_studies_page():
    st.title("🔮 Future Studies")
    st.markdown("""
    ## **Conclusion**
    
    The **Bank Household Analysis Dashboard** has provided key insights into customer demographics and household structures. Key findings include:
    
    - **Demographic Diversity:** Customers vary widely in age, income, and credit scores.
    - **Geographical Focus:** Many customers are concentrated in specific states.
    - **Household Patterns:** Higher-income households tend to have better credit scores.
    - **Positive Correlations:** Income and credit scores are positively correlated.
    
    These insights can help banks improve customer segmentation, tailor financial products, and enhance risk assessments.
    
    ## **Future Work**
    
    To expand on this analysis, consider the following:
    
    1. **Temporal Analysis:**
       - **Trends:** Track changes in customer demographics and financial metrics over time.
       - **Seasonal Patterns:** Identify seasonal changes in customer behavior.
    
    2. **Geospatial Mapping:**
       - **Maps:** Use maps to visualize customer and household distributions.
       - **Regional Analysis:** Study financial metrics across different regions.
    
    3. **Predictive Modeling:**
       - **Credit Risk:** Develop models to predict credit risk.
       - **Customer Value:** Estimate customer lifetime value to prioritize high-value customers.
    
    4. **Personalized Recommendations:**
       - **Financial Products:** Recommend financial products based on customer profiles.
       - **Customer Engagement:** Develop targeted marketing strategies.
    """)

# Page Routing
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
