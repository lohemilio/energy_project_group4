import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sqlalchemy import create_engine
from joblib import load
from sklearn.metrics import mean_squared_error
import numpy as np

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
    data = pd.read_csv(file_path)
    return data

def load_data_from_db(db_file, table_name):
    """
    Load data from a SQLite database.
    """
    engine = create_engine(f'sqlite:///{db_file}')
    query = f'SELECT * FROM {table_name};'
    data = pd.read_sql(query, engine, parse_dates=['time'])
    return data

def calculate_percentage_of_right_predictions(actual, predicted):
    """
    Calculate the percentage of 'right' predictions (within ±5% of the actual price).
    """
    error = np.abs(predicted - actual) / actual
    right_predictions = np.sum(error <= 0.05)
    total_predictions = len(error)
    percentage_right = (right_predictions / total_predictions) * 100
    return percentage_right

def eda_tab():
    st.subheader('Exploratory Data Analysis (EDA)')

    # Load energy dataset
    energy_file_path = '../data/energy_dataset.csv'
    energy_data = load_data(energy_file_path)
    if energy_data is not None:
        # Display head of energy dataset
        st.write("Head of Energy Dataset (First 10 rows):")
        st.write(energy_data.head(10))

        # Descriptive statistics of energy dataset
        st.write("Descriptive Statistics of Energy Dataset:")
        st.write(energy_data.describe())

        # Plot of 'total load actual' by day for the first two weeks of 2015
        energy_data['time'] = pd.to_datetime(energy_data['time'], utc=True)
        st.write("Total Load Actual by Day (First two weeks of 2015):")
        sns.set()
        fig, ax = plt.subplots(figsize=(30, 12))
        ax.set_xlabel('Time', fontsize=16)
        ax.plot(energy_data['total load actual'][0:24*7*2], label=None)
        ax.set_ylabel('Total Load (MWh)', fontsize=16)
        ax.plot(pd.Series([]), label= None)
        ax.set_ylabel('Total Load (MWh)', fontsize=16)
        ax.legend(fontsize=16)
        ax.set_title('Actual Total Load (First 2 weeks - Original)', fontsize=24)
        st.pyplot()

    # Load weather dataset
    weather_file_path = '../data/weather_features.csv'
    weather_data = load_data(weather_file_path)
    if weather_data is not None:
        # Display head of energy dataset
        st.write("Head of Weather Features Dataset (First 10 rows):")
        st.write(weather_data.head(10))

        # Descriptive statistics of energy dataset
        st.write("Descriptive Statistics of Weather Features Dataset:")
        st.write(weather_data.describe())
        # Number of observations per city in weather dataset
        st.write("Number of Observations per City in Weather Dataset:")
        city_counts = weather_data['city_name'].value_counts()
        st.write(city_counts)

        # Box plots for 'pressure' and 'wind_speed' columns in weather dataset
        st.write("Box Plots for 'Pressure' and 'Wind Speed' in Weather Dataset:")
        plt.figure(figsize=(12,16))
        plt.subplot(2, 1, 1)
        sns.boxplot(x=weather_data['pressure'])
        plt.title('Box Plot of Pressure')
        plt.subplot(2, 1, 2)
        sns.boxplot(x=weather_data['wind_speed'])
        plt.title('Box Plot of Wind Speed')
        plt.subplots_adjust(hspace=0.5)
        st.pyplot()

def data_visualization_tab(data):
    """
    Generate visualizations for the data.
    """
    st.subheader('Data Visualization')

    # Display the head of the first 10 rows
    st.write("Head of the first 10 rows:")
    st.write(data.head(10))

    # Display the monthly total load graph
    monthly_total_load_reordered = [
        79480216.5, 82325942.0, 83466690.0, 89079493.0, 85044320.0,
        81583790.5, 82364932.5, 83861399.0, 85193113.0, 88662713.0,
        80033157.0, 85180772.0
    ]
    desired_order = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'January', 'February', 'March' ]

    plt.figure(figsize=(12, 6))
    plt.plot(desired_order, monthly_total_load_reordered, marker='o', color='b')
    plt.title('Monthly Total Load')
    plt.xlabel('Month')
    plt.ylabel('Total Load')
    plt.grid(True)
    plt.xticks(desired_order)
    st.pyplot()

    # Plot histogram of 'price actual'
    plt.figure(figsize=(10, 6))
    sns.histplot(data['price actual'], kde=True, color='skyblue')
    plt.xlabel('Price Actual')
    plt.ylabel('Frequency')
    plt.title('Histogram of Price Actual')
    st.pyplot()

    # Plot autocorrelation and partial autocorrelation plots for 'price actual'
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(data['price actual'], ax=ax[0], lags=50)
    ax[0].set_title('Autocorrelation of Price Actual')
    plot_pacf(data['price actual'], ax=ax[1], lags=50)
    ax[1].set_title('Partial Autocorrelation of Price Actual')
    st.pyplot(fig)

    # Find correlations between 'price actual' and other features
    correlations = data.corr()['price actual'].sort_values(ascending=False)
    st.write("Correlations with 'price actual':")
    st.write(correlations.head(10))

def model_results_tab(data):
    """
    Display results of trained models.
    """
    st.subheader('Model Results')

    # Load energy dataset
    db_file = 'cleaned_data.db'
    table_name = 'energy_final_data'
    data = load_data_from_db(db_file, table_name)

    # Split data into features and target
    X = data.drop(columns=['time', 'price day ahead', 'price actual'])
    y = data['price actual']

    # Load Random Forest model
    rf_model = load("random_forest_model.joblib")

    # Load XGBoost model
    xgb_model = load("xgboost_model.joblib")

    # Predictions
    rf_predictions = rf_model.predict(X)
    xgb_predictions = xgb_model.predict(X)

    # Mean Squared Error
    mse_rf = mean_squared_error(y, rf_predictions)
    mse_xgb = mean_squared_error(y, xgb_predictions)

    # Percentage of right predictions for XGBoost
    percentage_right_xgb = calculate_percentage_of_right_predictions(y, xgb_predictions)

    # Display Random Forest results
    st.write("Random Forest Model Results:")
    st.write(f"Mean Squared Error: {mse_rf}")
    st.write(f"Random Forest Model Score: {rf_model.score(X, y)}")

    # Load actual vs predicted prices for Random Forest
    plt.figure(figsize=(10, 6))
    plt.scatter(y, rf_predictions, label='Predicted Price (Random Forest)')
    plt.plot(y, y, color='red', linestyle='--', label='Trend')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices (Random Forest)')
    plt.legend()
    st.pyplot()

    # Display Random Forest prediction results in table
    st.write("Random Forest Prediction Results:")
    rf_predictions_df = pd.DataFrame({
        'Actual Price': y,
        'Predicted Price': rf_predictions
    })
    st.write(rf_predictions_df)

    # Display XGBoost results
    st.write("XGBoost Model Results:")
    st.write(f"Mean Squared Error: {mse_xgb}")
    st.write(f"XGBoost Model Score: {xgb_model.score(X, y)}")
    st.write(f"Percentage of right predictions (within ±5% of actual price): {percentage_right_xgb}%")

    # Load actual vs predicted prices for XGBoost
    plt.figure(figsize=(10, 6))
    plt.scatter(y, xgb_predictions, label='Predicted Price (XGBoost)')
    plt.plot(y, y, color='red', linestyle='--', label='Trend')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices (XGBoost)')
    plt.legend()
    st.pyplot()

    # Display XGBoost prediction results in table
    st.write("XGBoost Prediction Results (Sample :")
    xgb_predictions_df = pd.DataFrame({
        'Actual Price': y,
        'Predicted Price': xgb_predictions
    })
    st.write(xgb_predictions_df)

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Page title
    st.title('Electricity Consumption Consulting')
    st.image('energy.png', width=50)

    # Database configuration
    db_file = 'cleaned_data.db'
    table_name = 'energy_final_data'

    # Load data
    data = load_data_from_db(db_file, table_name)

    # Sidebar
    st.sidebar.title('Navigation')

    # Create tabs
    tabs = ['Exploratory Data Analysis', 'Data Visualization','Model Results']
    selected_tab = st.sidebar.selectbox('Select Tab', tabs)

    # Display selected tab
    if selected_tab == 'Exploratory Data Analysis':
        eda_tab()
    elif selected_tab == 'Data Visualization':
        data_visualization_tab(data)
    elif selected_tab == 'Model Results':
        model_results_tab(data)

if __name__ == "__main__":
    main()
