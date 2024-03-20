import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

def load_data_from_db(db_file, table_name):
    engine = create_engine(f'sqlite:///{db_file}')
    query = f'SELECT * FROM {table_name};'
    data = pd.read_sql(query, engine, parse_dates=['time'])
    return data

def train_xgboost_model(X_train, y_train):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


import seaborn as sns


def eda_tab(data):
    st.subheader('Exploratory Data Analysis (EDA)')

    # Display the head of the first 10 rows
    st.write("Head of the first 10 rows:")
    st.write(data.head(10))

    # Generate box plots for the numeric columns
    st.write("Box plots for numeric columns:")
    numeric_columns = data.select_dtypes(include=['int', 'float']).columns
    for col in numeric_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data[col])
        plt.title(f'Box Plot of {col}')
        st.pyplot()


def visualization_tab(data):
    st.subheader('Data Visualizations')

    # Line plot of total monthly load
    plt.figure(figsize=(10, 6))
    data['month'] = data['time'].dt.month
    monthly_load = data.groupby('month')['total load actual'].sum()
    plt.plot(monthly_load.index, monthly_load.values, marker='o', linestyle='-')
    plt.xlabel('Month')
    plt.ylabel('Total Monthly Load')
    plt.title('Total Monthly Load over Time')
    st.pyplot()

    # Histogram of price actual
    plt.figure(figsize=(10, 6))
    sns.histplot(data['price actual'], kde=True, color='skyblue')
    plt.xlabel('Price Actual')
    plt.ylabel('Frequency')
    plt.title('Histogram of Price Actual')
    st.pyplot()

    # Average price actual per month (line plot)
    plt.figure(figsize=(10, 6))
    avg_price_monthly = data.groupby('month')['price actual'].mean()
    plt.plot(avg_price_monthly.index, avg_price_monthly.values, marker='o', linestyle='-')
    plt.xlabel('Month')
    plt.ylabel('Average Price Actual')
    plt.title('Average Price Actual per Month')
    st.pyplot()

def xgboost_results_tab(X_train, X_test, y_train, y_test):
    st.subheader('XGBoost Results')
    model = train_xgboost_model(X_train, y_train)
    mse, r2 = evaluate_model(model, X_test, y_test)
    st.write(f'Mean Squared Error (MSE): {mse}')
    st.write(f'R-squared (R²) Score: {r2}')

def main():
    # Page title
    st.title('Energy Price Prediction Web App')

    # Database configuration
    db_file = 'cleaned_data.db'
    table_name = 'energy_final_data'

    # Load data
    data = load_data_from_db(db_file, table_name)

    # Split data into training and testing sets
    X = data.drop(columns=['time', 'price day ahead', 'price actual'])
    y = data['price actual']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create tabs
    tabs = ['Exploratory Data Analysis', 'Data Visualizations', 'XGBoost Results']
    selected_tab = st.sidebar.selectbox('Select Tab', tabs)

    # Display selected tab
    if selected_tab == 'Exploratory Data Analysis':
        eda_tab(data)
    elif selected_tab == 'Data Visualizations':
        visualization_tab(data)
    elif selected_tab == 'XGBoost Results':
        xgboost_results_tab(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
