import pandas as pd
from sqlalchemy import create_engine
import numpy as np


def load_data(file_path):
    """
    Load data from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data


def clean_energy_data(df_energy):
    """
    Clean energy dataset.
    """
    # Dropping irrelevant columns
    df_energy = df_energy.drop(['generation fossil coal-derived gas', 'generation fossil oil shale',
                                'generation fossil peat', 'generation geothermal',
                                'generation hydro pumped storage aggregated', 'generation marine',
                                'generation wind offshore', 'forecast wind offshore eday ahead',
                                'total load forecast', 'forecast solar day ahead',
                                'forecast wind onshore day ahead'],
                               axis=1)

    # Adjusting data types
    df_energy['time'] = pd.to_datetime(df_energy['time'], utc=True)

    # Filling missing values with linear interpolation
    numerical_columns = df_energy.select_dtypes(include=['float64']).columns
    df_energy[numerical_columns] = df_energy[numerical_columns].interpolate(method='linear')

    # Setting the index to 'time'
    df_energy = df_energy.set_index('time')

    return df_energy


def clean_weather_data(df_weather):
    """
    Clean weather dataset.
    """
    # Converting int columns to float
    integer_columns = ['pressure', 'humidity', 'wind_speed', 'wind_deg', 'clouds_all', 'weather_id']
    df_weather[integer_columns] = df_weather[integer_columns].astype(float)

    # Convert dt_iso to datetime type and rename it
    df_weather['time'] = pd.to_datetime(df_weather['dt_iso'], utc=True)
    df_weather.drop(columns=['dt_iso'], inplace=True)

    # Dropping duplicate rows
    df_weather = df_weather.reset_index().drop_duplicates(subset=['time', 'city_name'], keep='first').set_index('time')

    # Dropping columns that will not be used
    df_weather = df_weather.drop(['weather_main', 'weather_id', 'weather_description', 'weather_icon'], axis=1)

    # Identifying outliers in pressure
    Q1 = df_weather['pressure'].quantile(0.25)
    Q3 = df_weather['pressure'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_weather.loc[(df_weather['pressure'] < lower_bound) | (df_weather['pressure'] > upper_bound), 'pressure'] = np.nan
    df_weather['pressure'].interpolate(method='linear', inplace=True)

    # Imputing outliers in wind_speed
    df_weather.loc[df_weather.wind_speed > 50, 'wind_speed'] = np.nan
    df_weather.interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)

    return df_weather


def merge_dataframes(df_energy, df_weather):
    """
    Merge energy and weather dataframes.
    """
    # Splitting the df_weather into 5 dataframes (one for each city)
    dfs = [x for _, x in df_weather.groupby('city_name')]

    # Merge all dataframes into the final dataframe
    df_combined = df_energy

    for df in dfs:
        city = df['city_name'].unique()
        city_str = str(city).replace("'", "").replace('[', '').replace(']', '').replace(' ', '')
        df = df.add_suffix('_{}'.format(city_str))
        df_combined = df_combined.merge(df, on=['time'], how='outer')
        df_combined = df_combined.drop('city_name_{}'.format(city_str), axis=1)

    return df_combined


def create_sqlite_table(df, db_file, table_name):
    """
    Create SQLite table and load dataframe data.
    """
    engine = create_engine(f'sqlite:///{db_file}')
    df.to_sql(table_name, engine, if_exists='replace', index=True)


def main():
    # File paths
    energy_file_path = '../data/energy_dataset.csv'
    weather_file_path = '../data/weather_features.csv'

    # Database configuration
    db_file = 'cleaned_data.db'

    # Load data
    df_energy = load_data(energy_file_path)
    df_weather = load_data(weather_file_path)

    # Clean data
    df_energy = clean_energy_data(df_energy)
    df_weather = clean_weather_data(df_weather)

    # Merge dataframes
    df_combined = merge_dataframes(df_energy, df_weather)

    # Create SQLite table and load preprocessed data
    create_sqlite_table(df_combined, db_file, 'energy_cleaned')
    print("Data ingestion completed successfully!")


if __name__ == "__main__":
    main()
