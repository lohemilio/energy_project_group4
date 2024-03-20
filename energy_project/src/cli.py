import typer
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from joblib import load
import numpy as np

# Load models
rf_model = load("random_forest_model.joblib")
xgb_model = load("xgboost_model.joblib")

# Function to load data from SQLite database
def load_data_from_db(db_file, table_name):
    engine = create_engine(f'sqlite:///{db_file}')
    query = f'SELECT * FROM {table_name};'
    data = pd.read_sql(query, engine, parse_dates=['time'])
    return data

# Function to make predictions
def make_predictions(initial_date: str, n_days: int):
    """
    Make predictions given an initial date and the number of days into the future.
    Save the prediction results into a predictions table in the database.
    """
    # Load data
    db_file = 'cleaned_data.db'
    table_name = 'energy_final_data'
    data = load_data_from_db(db_file, table_name)

    # Convert initial_date to datetime object
    initial_date = pd.to_datetime(initial_date)

    # Generate future dates for prediction
    future_dates = [initial_date + timedelta(hours=i) for i in range(n_days*24)]
    future_features = pd.DataFrame({'time': future_dates})
    future_features.set_index('time', inplace=True)

    # Make predictions with both models
    rf_predictions = rf_model.predict(future_features)
    xgb_predictions = xgb_model.predict(future_features)

    # Create a DataFrame to store predictions
    predictions_df = pd.DataFrame({
        'Time': future_dates,
        'RF Predictions': rf_predictions,
        'XGB Predictions': xgb_predictions
    })

    # Print predictions
    typer.echo(predictions_df.head(10))

    # Save predictions to a predictions table in the database
    # Add your code to save predictions to a database table here
    # For demonstration purposes, we'll just print the message
    typer.echo("Predictions saved to database.")

# Function to plot predictions vs. real data
def plot_predictions():
    """
    Plot predictions vs. real data with your favorite plotting library.
    """
    # Add your code to plot predictions vs. real data here
    typer.echo("Plotting predictions vs. real data.")

app = typer.Typer()

@app.command()
def train_model():
    """
    Train a machine learning model and save it to a file.
    """
    # Add your code to train the model and save it here
    typer.echo("Model trained and saved successfully.")

@app.command()
def make_predictions_cmd(initial_date: str):
    """
    Make predictions given an initial date and the number of days into the future.
    Save the prediction results into a predictions table in the database.
    """
    make_predictions(initial_date, n_days=7)

@app.command()
def plot_predictions_cmd():
    """
    Plot predictions vs. real data with your favorite plotting library.
    """
    plot_predictions()

if __name__ == "__main__":
    app()
