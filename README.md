# Energy Price Prediction Project

## Overview

The Energy Price Prediction Project aims to forecast energy prices based on various features such as weather conditions, time of day, and historical energy consumption. The project utilizes a dataset containing information about energy generation, weather features, and actual energy prices. The goal is to develop machine learning models that can accurately predict energy prices, which can be valuable for energy market analysis and decision-making.

### Dataset

The dataset used in this project consists of two main files:
- `energy_dataset.csv`: Contains information about energy generation and actual energy prices.
- `weather_features.csv`: Contains weather-related features such as temperature, humidity, and wind speed.

### Authors

This project is authored by:
* Emilio López
* Bendix Sibel
* Sebastian Hirsch
* Veronica Jardon
* Benedita Bacelar.

## Machine Learning Model

The machine learning model used in this project is based on ensemble methods, specifically Random Forest and XGBoost regressors. These models are trained on the historical data to predict future energy prices. 

### Data Preprocessing

The data preprocessing involves several steps:
1. Loading the data from the CSV files and merging them based on a common key (e.g., timestamp).
2. Cleaning the data by handling missing values and removing outliers.
3. Feature engineering to create additional features that may be useful for prediction.
4. Splitting the data into training and testing sets for model evaluation.

## Project Structure

The project is organized as follows:
- **data**: Contains the dataset files (`energy_dataset.csv` and `weather_features.csv`).
- **src**: Contains the source code for data ingestion, feature engineering, model training, and prediction.
  - `data_ingestion.py`: Script to load and merge the dataset.
  - `feature_engineering.py`: Script for feature engineering tasks.
  - `forecasting.py`: Script to train machine learning models and make predictions.
  - `main.py`: Main script to run the project.
- **poetry.lock** and **pyproject.toml**: Dependency management files.

## Installation

To install the project dependencies, follow these steps:
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Run the following command to install dependencies using Poetry:

## Running the code
To run the code and train the machine learning models, follow the steps below:
`python src/main.py`

### Using the CLI

To use the command-line interface (CLI) provided with this project, follow the instructions below:

#### Prerequisites
- Python installed on your machine.
- Required Python packages installed (install using `pip install -r requirements.txt`).

#### Commands

1. **Train a Model**
   - Description: Train a machine learning model and save it to a file.
   - Command:
     ```bash
     python cli.py train-model
     ```

2. **Make Predictions**
   - Description: Make predictions for the next 7 days given an initial date.
   - Command:
     ```bash
     python cli.py make-predictions-cmd --initial-date "2024-03-01"
     ```
   - Replace `"2024-03-01"` with your desired initial date.

3. **Plot Predictions**
   - Description: Plot predictions vs. real data with your favorite plotting library.
   - Command:
     ```bash
     python cli.py plot-predictions-cmd
     ```

### CLI Code Explanation

The provided CLI script allows you to perform the following tasks:

- **Load Models**: Load the pre-trained Random Forest and XGBoost models from the saved joblib files.
- **Load Data**: Load data from the SQLite database.
- **Make Predictions**: Generate predictions for the next 7 days based on the provided initial date using the loaded models.
- **Plot Predictions**: Plot the generated predictions vs. real data using a plotting library.

To use the CLI, simply run the script `cli.py` with the desired command-line arguments as described above.


## Running the Streamlit App
The Streamlit app provides an interactive interface for exploring the project results and predictions. To run the app, use the following command:
`streamlit run src/main.py`

Once the app is running, you can navigate through the different tabs to view exploratory data analysis, visualizations, and model results.
