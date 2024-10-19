# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
# 2. Load and Clean the Dataset
# ---------------------------

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
    # Check if the file exists
    if not os.path.exists(filepath):
        st.error(f"The file '{filepath}' does not exist in the specified directory.")
        return None, None

    # Load the dataset with proper NA handling
    df = pd.read_csv(filepath, na_values=['', ' ', 'NA', 'NaN'])

    # ----------------------------------------
    # 1. Data Type Conversions
    # ----------------------------------------

    # a. Convert 'Created_At' to datetime if it exists
    if 'Created_At' in df.columns:
        df['Created_At'] = pd.to_datetime(df['Created_At'], errors='coerce')

    # b. Convert specified columns to 'category' dtype
    categorical_cols = ['Last_Name', 'State']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
        else:
            st.warning(f"Warning: Column '{col}' not found in the dataset.")

    # ----------------------------------------
    # 2. Handling Missing Values
    # ----------------------------------------

    # a. Identify numerical and categorical columns
    numerical_cols = ['Income', 'Credit_Score', 'Age']

    # b. Impute missing numerical values with the mean
    imputer = SimpleImputer(strategy='mean')
    for col in numerical_cols:
        if col in df.columns:
            df[col] = imputer.fit_transform(df[[col]])
        else:
            st.warning(f"Warning: Numerical column '{col}' not found in the dataset.")

    # c. Handle missing categorical values by filling with 'Unknown'
    for col in categorical_cols:
        if col in df.columns:
            # Add 'Unknown' category if it's not already present
            if 'Unknown' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(['Unknown'])
            # Fill NaN with 'Unknown'
            df[col] = df[col].fillna('Unknown')
        else:
            st.warning(f"Warning: Categorical column '{col}' not found in the dataset.")

    # ----------------------------------------
    # 3. Feature Engineering
    # ----------------------------------------

    # a. Create Household DataFrame
    # Assuming households are determined by shared Last_Name and Address
    household_df = df.groupby(['Last_Name', 'Address', 'State'], observed=True).agg({
        'Income': 'sum',
        'Credit_Score': 'mean',
        'Age': 'mean',
        'First_Name': 'count'  # Using 'First_Name' count as number of people
    }).reset_index()

    household_df.rename(columns={
        'Income': 'Total_Income',
        'Credit_Score': 'Average_Credit_Score',
        'Age': 'Average_Age',
        'First_Name': 'Number_of_People'
    }, inplace=True)

    # ----------------------------------------
    # 4. Additional Data Cleaning
    # ----------------------------------------

    # a. Trim whitespace from string columns to ensure consistency
    string_cols = ['First_Name', 'Last_Name', 'Address']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            st.warning(f"Warning: String column '{col}' not found in the dataset.")

    # b. Capitalize categorical variables for consistency
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.title()
        else:
            st.warning(f"Warning: Categorical column '{col}' not found in the dataset.")

    # ----------------------------------------
    # 5. Final Checks and Balances
    # ----------------------------------------

    # a. Verify no remaining missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        st.warning("There are still missing values in the dataset:")
        st.write(missing_values[missing_values > 0])
    else:
        st.success("No missing values remain in the dataset.")

    # b. Return the cleaned DataFrames
    return df, household_df

# Load the data
data_filepath = 'customers.csv'
df, household_df = load_and_clean_data(data_filepath)

# ---------------------------
# 3. Sidebar - Page Navigation
# ---------------------------

st.sidebar.title("📑 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Customer Insights", "Household Detection", "Future Studies"])

# ---------------------------
# 4. Define Pages
# ---------------------------

def home_page():
    st.title("🏦 Bank Household Analysis Dashboard")
    st.markdown("""
    ## **Introduction**
    
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
    Dive into the customer demographics and financial metrics to understand the distribution and relationships within the dataset. Use the interactive filters on the sidebar to customize the data view.
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
        color='Number_of_Customers',
        color_continuous_scale='Viridis'
    )

    st.plotly_chart(fig_state, use_container_width=True)

    st.markdown("""
    **Insights:**
    - **High Concentration States:** Identify states with a high number of customers, informing targeted marketing and service strategies.
    - **Low Concentration States:** Recognize states with fewer customers, presenting opportunities for growth and expansion.
    """)

    st.markdown("---")

    # ---------------------------
    # 2. Income vs. State Interactive Plot
    # ---------------------------

    st.header("💰 Income vs. State")
    st.markdown("""
    Explore the relationship between customer income and their residing state.
    """)

    fig_income_state = px.box(
        filtered_df,
        x='State',
        y='Income',
        title='Income Distribution by State',
        labels={'Income': 'Annual Income', 'State': 'State'},
        color='State',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    st.plotly_chart(fig_income_state, use_container_width=True)

    st.markdown("""
    **Insights:**
    - **Income Variability:** Assess how income levels vary across different states.
    - **High-Earning States:** Identify states with significantly higher incomes, aiding in region-specific financial strategies.
    """)

    st.markdown("---")

    # ---------------------------
    # 3. Age Distributions Interactive Plot
    # ---------------------------

    st.header("🎂 Age Distribution")
    st.markdown("""
    Analyze the age distribution of customers.
    """)

    fig_age = px.histogram(
        filtered_df,
        x='Age',
        nbins=20,
        title='Age Distribution of Customers',
        labels={'Age': 'Age'},
        color_discrete_sequence=['#636EFA']
    )

    st.plotly_chart(fig_age, use_container_width=True)

    st.markdown("""
    **Insights:**
    - **Peak Age Groups:** Identify the most common age ranges among customers.
    - **Age Spread:** Understand the diversity in age, informing age-targeted services and products.
    """)

    st.markdown("---")

    # ---------------------------
    # 4. Credit Score Distribution Interactive Plot
    # ---------------------------

    st.header("📈 Credit Score Distribution")
    st.markdown("""
    Examine the distribution of credit scores among customers.
    """)

    fig_credit = px.histogram(
        filtered_df,
        x='Credit_Score',
        nbins=20,
        title='Credit Score Distribution of Customers',
        labels={'Credit_Score': 'Credit Score'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    st.plotly_chart(fig_credit, use_container_width=True)

    st.markdown("""
    **Insights:**
    - **Credit Health:** Assess the overall credit health of the customer base.
    - **Risk Assessment:** Identify potential high-risk segments based on credit scores.
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

    # Apply Additional Filters
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

    # ---------------------------
    # 1. Total Number of Households
    # ---------------------------

    st.header("📊 Total Number of Households")
    st.markdown("""
    The total number of households in the dataset provides an overview of the customer base's household distribution.
    """)

    total_households = filtered_household_df.shape[0]
    st.metric("Total Households", total_households)

    st.markdown("---")

    # ---------------------------
    # 2. Household DataFrame
    # ---------------------------

    st.header("📋 Household Data")
    st.markdown("""
    Below is the detailed household information, including last name, address, number of people, and total income.
    """)

    st.dataframe(filtered_household_df[['Last_Name', 'Address', 'Number_of_People', 'Total_Income']].reset_index(drop=True))

    st.markdown("---")

    # ---------------------------
    # 3. Total Number of Households in Each State Plot
    # ---------------------------

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
    **Insights:**
    - **Household Concentration:** Identify states with a high number of households, aiding in targeted household-based strategies.
    - **Growth Opportunities:** Recognize states with emerging household clusters for potential service expansion.
    """)

    st.markdown("---")

    # ---------------------------
    # 4. Average Income of a Household vs. Number of People Plot
    # ---------------------------

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

    st.markdown("""
    **Insights:**
    - **Income Per Capita:** Understand how income scales with household size.
    - **Household Efficiency:** Identify households with higher income relative to the number of members.
    """)

    st.markdown("---")

def future_studies_page():
    st.title("🔮 Future Studies")
    st.markdown("""
    ## **Conclusion**
    
    The **Bank Household Analysis Dashboard** has provided valuable insights into customer demographics and household structures within the banking dataset. Key findings include:
    
    - **Demographic Distribution:** The customer base spans a wide age range with varying income levels and credit scores, indicating diverse financial profiles.
    - **Geographical Concentration:** A significant number of customers are concentrated in specific states, highlighting potential regions for targeted services.
    - **Household Structures:** Analysis of households reveals patterns in total income and average credit scores, suggesting that higher-income households tend to have better credit health.
    - **Correlation Insights:** Positive correlations between income and credit scores emphasize the importance of financial stability in creditworthiness assessments.
    
    These insights can assist banks in refining their customer segmentation strategies, tailoring financial products, and enhancing risk assessment models.
    
    ## **Future Work**
    
    To build upon the current analysis, the following areas are recommended for future exploration:
    
    1. **Temporal Analysis:**
       - **Trend Identification:** Examine how customer demographics and financial metrics evolve over time.
       - **Seasonal Patterns:** Identify any seasonal fluctuations in customer behavior or financial performance.
    
    2. **Geospatial Mapping:**
       - **Interactive Maps:** Incorporate geographical maps to visualize customer and household distributions spatially.
       - **Regional Analysis:** Analyze financial metrics across different regions to identify regional trends and needs.
    
    3. **Advanced Predictive Modeling:**
       - **Credit Risk Prediction:** Develop machine learning models to predict credit risk based on customer and household attributes.
       - **Customer Lifetime Value (CLV):** Estimate the CLV to prioritize high-value customers and tailor retention strategies.
    
    4. **Enhanced Data Enrichment:**
       - **Behavioral Data:** Integrate transactional or behavioral data to gain deeper insights into customer activities.
       - **External Data Sources:** Incorporate external economic indicators or demographic data to enrich the analysis.
    
    5. **Personalized Recommendations:**
       - **Financial Products:** Utilize the insights to recommend personalized financial products to customers based on their profiles.
       - **Customer Engagement:** Develop strategies for targeted marketing and customer engagement initiatives.
    
    6. **User Experience Enhancements:**
       - **Multi-Page Navigation:** Further refine the dashboard with intuitive navigation and user-friendly interfaces.
       - **Responsive Design:** Ensure the dashboard is optimized for various devices and screen sizes.
    
    7. **Reporting and Exporting:**
       - **Reporting Tools:** Integrate reporting tools to generate automated reports based on the analysis.
       - **Export Options:** Provide options to export visualizations and data tables for external use.
    
    By pursuing these areas, the dashboard can evolve into a more comprehensive tool, offering deeper insights and actionable strategies to enhance banking operations and customer satisfaction.
    """)

    st.markdown("---")  # Horizontal line for separation

    # ---------------------------
    # 3. Summary of Findings
    # ---------------------------

    st.header("🔍 Summary of Findings")
    st.markdown("""
    - **Age and Income Diversity:** The customer base exhibits a broad range of ages and incomes, indicating a varied clientele.
    - **Geographical Insights:** Certain states show a higher concentration of customers, which can inform region-specific strategies.
    - **Household Financial Health:** Households with higher total incomes tend to have better average credit scores.
    - **Positive Correlations:** There are notable positive correlations between income and credit scores, emphasizing the role of financial stability in creditworthiness.
    """)

    st.markdown("---")

    # ---------------------------
    # 4. Future Work Details
    # ---------------------------

    st.header("🚀 Future Work")
    st.markdown("""
    To further enhance the dashboard's capabilities and provide more actionable insights, the following initiatives are proposed:
    
    1. **Temporal Analysis:**
       - **Objective:** Understand how customer demographics and financial behaviors change over time.
       - **Approach:** Incorporate time-series data to track trends and identify patterns.
    
    2. **Geospatial Mapping:**
       - **Objective:** Visualize customer distributions geographically to identify regional strengths and opportunities.
       - **Approach:** Utilize Plotly's map visualizations or integrate with geospatial libraries like `folium`.
    
    3. **Predictive Modeling:**
       - **Objective:** Forecast customer behaviors and financial outcomes to inform strategic decisions.
       - **Approach:** Develop machine learning models using features such as income, age, and credit scores to predict outcomes like default rates or product uptake.
    
    4. **Data Enrichment:**
       - **Objective:** Enhance the dataset with additional attributes for a more comprehensive analysis.
       - **Approach:** Integrate external data sources such as economic indicators, housing data, or consumer behavior metrics.
    
    5. **Personalized Recommendations:**
       - **Objective:** Utilize the insights to recommend personalized financial products to customers based on their profiles.
       - **Approach:** Develop strategies for targeted marketing and customer engagement initiatives.
    
    6. **User Experience Improvements:**
       - **Objective:** Enhance the dashboard's usability and interactivity.
       - **Approach:** Implement features like dynamic tooltips, drill-down capabilities, and customizable layouts to cater to diverse user needs.
    
    7. **Reporting and Exporting:**
       - **Objective:** Facilitate easy sharing and reporting of insights.
       - **Approach:** Add functionalities to export visualizations and summaries as PDFs or interactive reports.
    
    By undertaking these initiatives, the dashboard can transform into a robust tool that not only analyzes current data but also anticipates future trends and supports strategic decision-making.
    """)

def future_studies_page():
    st.title("🔮 Future Studies")
    st.markdown("""
    ## **Conclusion**
    
    The **Bank Household Analysis Dashboard** has provided valuable insights into customer demographics and household structures within the banking dataset. Key findings include:
    
    - **Demographic Distribution:** The customer base spans a wide age range with varying income levels and credit scores, indicating diverse financial profiles.
    - **Geographical Concentration:** A significant number of customers are concentrated in specific states, highlighting potential regions for targeted services.
    - **Household Structures:** Analysis of households reveals patterns in total income and average credit scores, suggesting that higher-income households tend to have better credit health.
    - **Correlation Insights:** Positive correlations between income and credit scores emphasize the importance of financial stability in creditworthiness assessments.
    
    These insights can assist banks in refining their customer segmentation strategies, tailoring financial products, and enhancing risk assessment models.
    
    ## **Future Work**
    
    To build upon the current analysis, the following areas are recommended for future exploration:
    
    1. **Temporal Analysis:**
       - **Trend Identification:** Examine how customer demographics and financial metrics evolve over time.
       - **Seasonal Patterns:** Identify any seasonal fluctuations in customer behavior or financial performance.
    
    2. **Geospatial Mapping:**
       - **Interactive Maps:** Incorporate geographical maps to visualize customer and household distributions spatially.
       - **Regional Analysis:** Analyze financial metrics across different regions to identify regional trends and needs.
    
    3. **Advanced Predictive Modeling:**
       - **Credit Risk Prediction:** Develop machine learning models to predict credit risk based on customer and household attributes.
       - **Customer Lifetime Value (CLV):** Estimate the CLV to prioritize high-value customers and tailor retention strategies.
    
    4. **Enhanced Data Enrichment:**
       - **Behavioral Data:** Integrate transactional or behavioral data to gain deeper insights into customer activities.
       - **External Data Sources:** Incorporate external economic indicators or demographic data to enrich the analysis.
    
    5. **Personalized Recommendations:**
       - **Financial Products:** Utilize the insights to recommend personalized financial products to customers based on their profiles.
       - **Customer Engagement:** Develop strategies for targeted marketing and customer engagement initiatives.
    
    6. **User Experience Enhancements:**
       - **Multi-Page Navigation:** Further refine the dashboard with intuitive navigation and user-friendly interfaces.
       - **Responsive Design:** Ensure the dashboard is optimized for various devices and screen sizes.
    
    7. **Reporting and Exporting:**
       - **Reporting Tools:** Integrate reporting tools to generate automated reports based on the analysis.
       - **Export Options:** Provide options to export visualizations and data tables for external use.
    
    By pursuing these areas, the dashboard can evolve into a more comprehensive tool, offering deeper insights and actionable strategies to enhance banking operations and customer satisfaction.
    """)

# ---------------------------
# 5. Page Routing
# ---------------------------

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
