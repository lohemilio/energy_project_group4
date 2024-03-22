import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np


def load_data_from_db(db_file, table_name):
    """
    Load data from a SQLite database.
    """
    engine = create_engine(f'sqlite:///{db_file}')
    query = f'SELECT * FROM {table_name};'
    data = pd.read_sql(query, engine, parse_dates=['time'])
    return data


def train_test_split_data(data):
    """
    Split the data into training and testing sets by using a manual time-ordered split based on the number of instances.
    """
    X = data.drop(columns=['time', 'price day ahead', 'price actual'])
    y = data['price actual']

    # Calculate the split index based on the number of observations in the data set
    split_index = int(len(data) - 24)

    # Split into train and test based on the split index
    X_train = X.iloc[:split_index, :]
    X_test = X.iloc[split_index:, :]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test



def train_model(X_train, y_train, model_type='random_forest'):
    """
    Train a machine learning model.
    """
    if model_type == 'random_forest':
        model = RandomForestRegressor()
    elif model_type == 'xgboost':
        model = XGBRegressor()
    else:
        raise ValueError("Invalid model type. Supported types are 'random_forest' and 'xgboost'.")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate percentage of predictions within a threshold of 5%
    error_percentage = np.abs((y_pred - y_test) / y_test) * 100
    threshold = 5
    right_predictions_percentage = np.mean(error_percentage <= threshold) * 100

    return mse, r2, right_predictions_percentage


def save_model(model, filename):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, filename)


def main():
    # Database configuration
    db_file = 'cleaned_data.db'
    table_name = 'energy_final_data'

    # Load data
    data = load_data_from_db(db_file, table_name)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split_data(data)

    # Train the RandomForestRegressor model
    rf_model = train_model(X_train, y_train, model_type='random_forest')

    # Evaluate the RandomForestRegressor model
    rf_mse, rf_r2, rf_percentage_within_threshold = evaluate_model(rf_model, X_test, y_test)
    print(f"RandomForestRegressor Mean Squared Error: {rf_mse}")
    print(f"RandomForestRegressor R-squared (R²) Score: {rf_r2}")
    print(f"RandomForestRegressor Percentage of 'right' predictions (within ±5% of the actual price): {rf_percentage_within_threshold:.2f}%")

    # Train the XGBoost model
    xgb_model = train_model(X_train, y_train, model_type='xgboost')

    # Evaluate the XGBoost model
    xgb_mse, xgb_r2, xgb_percentage_within_threshold = evaluate_model(xgb_model, X_test, y_test)
    print(f"XGBoost Mean Squared Error: {xgb_mse}")
    print(f"XGBoost R-squared (R²) Score: {xgb_r2}")
    print(f"XGBoost Percentage of 'right' predictions (within ±5% of the actual price): {xgb_percentage_within_threshold:.2f}%")

    # Save the models to files
    rf_model_filename = 'random_forest_model.joblib'
    xgb_model_filename = 'xgboost_model.joblib'
    save_model(rf_model, rf_model_filename)
    save_model(xgb_model, xgb_model_filename)
    print(f"RandomForestRegressor model saved to {rf_model_filename}")
    print(f"XGBoost model saved to {xgb_model_filename}")


if __name__ == "__main__":
    main()
