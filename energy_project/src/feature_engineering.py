import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data_from_db(db_file, table_name):
    """
    Load data from a SQLite database.
    """
    engine = create_engine(f'sqlite:///{db_file}')
    query = f'SELECT * FROM {table_name};'
    data = pd.read_sql(query, engine, parse_dates=['time'])
    return data

def feature_engineering(df_energy):
    """
    Perform feature engineering on energy data.
    """
    # Extract date-time components
    df_energy['year'] = df_energy['time'].dt.year
    df_energy['month'] = df_energy['time'].dt.month
    df_energy['day'] = df_energy['time'].dt.day
    df_energy['hour'] = df_energy['time'].dt.hour
    df_energy['weekday'] = df_energy['time'].dt.weekday

    # Create cyclical features for 'month' and 'hour' to capture their cyclical nature
    df_energy['month_sin'] = np.sin(2 * np.pi * df_energy['month']/12)
    df_energy['month_cos'] = np.cos(2 * np.pi * df_energy['month']/12)
    df_energy['hour_sin'] = np.sin(2 * np.pi * df_energy['hour']/24)
    df_energy['hour_cos'] = np.cos(2 * np.pi * df_energy['hour']/24)

    return df_energy

def create_sqlite_table(df, db_file, table_name):
    """
    Create SQLite table and load dataframe data.
    """
    engine = create_engine(f'sqlite:///{db_file}')
    df.to_sql(table_name, engine, if_exists='replace', index=False)

def main():
    # Database configuration
    db_file = 'cleaned_data.db'

    # Load data
    df_energy = load_data_from_db(db_file, 'energy_cleaned')

    # Perform feature engineering
    df_energy = feature_engineering(df_energy)

    # Save final features to database
    create_sqlite_table(df_energy, db_file, 'energy_final_data')
    print("Feature engineering completed successfully!")

if __name__ == "__main__":
    main()
