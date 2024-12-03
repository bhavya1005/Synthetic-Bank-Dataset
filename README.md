# **Synthetic Bank Dataset: Bank Household Analysis Dashboard**

## **Introduction**
The **Bank Household Analysis Dashboard** is a powerful tool designed to assist banks and financial institutions in understanding their customer base through interactive and data-driven insights. By analyzing demographic and financial metrics such as income, credit scores, and household structures, this dashboard empowers banks to make informed decisions tailored to their clients' needs.

Whether it’s identifying high-income households, analyzing creditworthiness, or detecting households from customer data, this dashboard offers a comprehensive solution to modern banking challenges.

---

## **Project Overview**
This project demonstrates the real-world application of data analytics in banking through an intuitive and interactive dashboard. The key objectives are:
- **Customer Distribution Across States**: Pinpoint where the majority of customers are located for better regional targeting.
- **Income and Credit Score Analysis**: Analyze financial stability and creditworthiness across customer demographics.
- **Household Detection**: Group customers into households based on shared addresses and last names, revealing their collective financial health.

Developed as part of my midterm project, the **Bank Household Analysis Dashboard** uses advanced data analysis techniques and visualizations to address real-world banking problems. This project highlights my ability to process, analyze, and visualize large datasets while building interactive web applications.

Explore the live dashboard here: **[Bank Household Analysis Dashboard](https://bankhouseholdresearchbhavyachawla.streamlit.app/Analysis_and_Results)**

---

## **Key Features**
1. **Interactive Dashboard**:
   - User-friendly Streamlit interface with intuitive navigation.
   - Sidebar menu for seamless exploration of all sections.

2. **Comprehensive Data Analysis**:
   - Visualize customer distribution across states.
   - Analyze income trends and credit score distributions.
   - Explore aggregated household-level data.

3. **Advanced Machine Learning Models**:
   - **Regression**: Predict household income using demographic and financial features.  
   - **Classification**: Identify households with high creditworthiness (credit score ≥ 700).  
   - Implemented models include **Linear Regression**, **Logistic Regression**, and **XGBoost**.  

4. **Dynamic Visualizations**:
   - Scatter plots, bar charts, and heatmaps to illustrate trends and relationships.
   - Feature importance plots to identify key predictors.

5. **Household Detection**:
   - Detect households based on shared addresses and last names.
   - Aggregate financial metrics like total income and average credit score at the household level.

---

## **Technologies Used**
### **Programming Languages and Libraries**
- **Python**: Core programming language for the project.
- **Pandas**: Data manipulation and aggregation.
- **NumPy**: Efficient numerical computations.
- **Matplotlib & Seaborn**: Static visualizations and heatmaps.
- **Plotly**: Interactive visualizations.
- **Scikit-learn**: Machine learning for regression and classification.
- **XGBoost**: Advanced modeling with feature importance.

### **Framework**
- **Streamlit**: Interactive web application framework for seamless user experience.

---

## **How to Run the Dashboard**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/bhavya1005/rep7.git
   cd rep7
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

5. **Access the Dashboard**:
   - Local URL: `http://localhost:8501`
   - Network URL: `http://<your-network-ip>:8501`

---

## **Project Structure**

```
rep7/
├── app.py               # Main dashboard entry point
├── pages/
│   ├── introduction.py  # Project introduction
│   ├── dataset.py       # Dataset overview
│   ├── household.py     # Household-level analysis
│   └── results.py       # Regression and classification results
├── customerdataset.csv  # Input dataset
├── datageneration.py    # Synthetic data generation script
├── requirements.txt     # Dependencies
├── LICENSE              # License information
└── README.md            # Project documentation
```

---

## **Insights**
- **Regression Analysis**:
  - Strong predictors of household income include household size, average credit score, and age.
  - **Key Finding**: Larger households tend to have higher total incomes but lower income per member.

- **Classification Analysis**:
  - Households with high creditworthiness (credit score ≥ 700) are more financially stable.
  - Misclassification in the model highlights areas for further feature engineering.

- **Regional Trends**:
  - Certain states exhibit higher household incomes and concentrations of high-creditworthy customers, ideal for premium financial products.

---

## **Future Enhancements**
1. **Real-Time Data Integration**:
   - Incorporate live customer data for dynamic analysis.
2. **Enhanced Modeling**:
   - Introduce advanced models like Random Forest or Gradient Boosting.
3. **Improved User Experience**:
   - Add more filters and interactivity in the dashboard.

---
