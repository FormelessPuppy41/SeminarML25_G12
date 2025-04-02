import pandas as pd

from data.data_loader import DataLoader
from data.data_processing import explore_data
from configuration import FileNames

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, Dropout, Conv1D, TimeDistributed, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

file_names = FileNames()
data_loader = DataLoader()

##
## TO RUN THIS FILE:
## - OPEN THE TERMINAL (ctrl + ` [see the left bottom corner of your keyboard])
## - WRITE: python -m kaggle.forecasts.wessel [or the name of your file]
## - PRESS ENTER TO RUN
## - RESULTS APPEAR IN THE TERMINAL
##



# -------------------------------
# STEP 1: Data Preparation Functions
# -------------------------------
def prepare_data(energy_data: pd.DataFrame, weather_data: pd.DataFrame, feature_set: str = "lag_features"):
    """
    Prepare the data based on the selected feature set.
    
    feature_set options:
      - "lag_features": Use base time features + lags for solar, wind, total load and price plus weather features.
      - "generation_features": Use base time features + lags for solar, wind, total load, price and add lags for additional generation features.
      - "combined": All of the above (weather + generation features).
    """
    # Convert time and set index
    df = energy_data.copy()
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time')
    df.index = df.index.tz_convert(None)
    
    # Always include these columns for forecasting (target is generation solar)
    required_cols = [
        'generation solar', 'generation wind onshore', 
        'forecast solar day ahead', 'forecast wind onshore day ahead',
        'total load actual', 'price actual'
    ]
    if feature_set in ['generation_features', 'combined']:
        # Additional generation features (note: ensure column names match those in your data)
        generation_cols = [
            'generation biomass', 'generation fossil brown coal/lignite', 'generation fossil gas',
            'generation fossil hard coal',  # mapping "fossil coal" to the available "fossil hard coal"
            'generation fossil oil', 'generation hydro pumped storage consumption',
            'generation hydro run-of-river and poundage', 'generation hydro water reservoir',
            'generation nuclear', 'generation other', 'generation other renewable', 'generation waste'
        ]
        required_cols += generation_cols

    df = df[required_cols]

    # Add derived wind column if not already present
    if 'generation wind onshore' in df.columns and 'generation wind' not in df.columns:
        df['generation wind'] = df['generation wind onshore']

    # Add time features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # Set lags to be created
    solar_lags = [24, 48, 168]  # Daily, 2-day, weekly lags
    lags = [1, 2, 3, 6, 12, 24, 48, 168]
    # Create lag features for solar, wind, total load actual and price actual
    for lag in solar_lags:
        df[f'solar_lag_{lag}'] = df['generation solar'].shift(lag)
        df[f'wind_lag_{lag}'] = df['generation wind'].shift(lag)

    for lag in lags:
        for col in ['total load actual', 'price actual']:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # If the feature set includes additional generation features, create lag features for those as well.
    if feature_set in ['generation_features', 'combined']:
        generation_cols = [
            'generation biomass', 'generation fossil brown coal/lignite', 'generation fossil gas',
            'generation fossil hard coal', 'generation fossil oil', 'generation hydro pumped storage consumption',
            'generation hydro run-of-river and poundage', 'generation hydro water reservoir',
            'generation nuclear', 'generation other', 'generation other renewable', 'generation waste'
        ]
        for lag in lags:
            for col in generation_cols:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # If the feature set uses weather features, process and merge weather data.
    if feature_set in ['lag_features', 'combined']:
        weather_data['dt_iso'] = pd.to_datetime(weather_data['dt_iso'], utc=True)
        weather_data = weather_data.set_index('dt_iso')
        weather_data.index = weather_data.index.tz_convert(None)
        keep_weather_features = ['temp', 'pressure', 'humidity', 'wind_speed', 'rain_1h', 'rain_3h', 'snow_3h', 'clouds_all']
        weather_numeric = weather_data[keep_weather_features]
        weather_agg = weather_numeric.resample('h').mean()
        # Rename columns to avoid name collisions
        weather_agg.columns = [f'weather_{col}' for col in weather_agg.columns]
        df = df.merge(weather_agg, left_index=True, right_index=True, how='left')

        # Create lag features for weather variables
        weather_lags = [1, 3, 6, 12, 24]
        for lag in weather_lags:
            for col in weather_agg.columns:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        # Drop original weather columns
        df.drop(columns=weather_agg.columns, inplace=True)

    # Drop original non-lagged total load and price columns to avoid leakage
    df.drop(columns=['total load actual', 'price actual'], inplace=True)

    # Drop any rows with missing values
    df.dropna(inplace=True)

    # Define features X (remove target and forecast columns) and target y (generation solar)
    X = df.drop(columns=['generation solar', 
                         'generation wind', 
                         'generation wind onshore', 'forecast solar day ahead', 'forecast wind onshore day ahead'])
    y_solar = df['generation solar']

    print("\n🧠 Features used for training:")
    print(X.columns.tolist())

    return X, y_solar


# -------------------------------
# STEP 2: Forecasting Functions
# -------------------------------
def forecast_for_day(forecast_day, X, y, model_name, model_params):
    """
    Perform the expanding window forecast for one day.
    """
    # Training data: all data before the forecast day
    train_mask = X.index < forecast_day
    X_train = X[train_mask]
    y_train = y[train_mask]

    # Test data: data within the forecast day (24 hours)
    test_mask = (X.index >= forecast_day) & (X.index < forecast_day + pd.Timedelta(days=1))
    X_test = X[test_mask]
    y_test = y[test_mask]

    if X_train.empty or X_test.empty:
        return None

    # Select model based on model_name
    if model_name == "xgboost":
        model = xgb.XGBRegressor(**model_params)
    elif model_name == "random_forest":
        model = RandomForestRegressor(**model_params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)

    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)

    timestamps = X_test.index.tolist()
    return (forecast_day, train_mse, test_mse, y_test.tolist(), y_test_pred.tolist(), timestamps)


def expanding_window_forecaster(energy_data: pd.DataFrame, weather_data: pd.DataFrame,
                                model_name: str, model_params: dict, feature_set: str, plot_graphs: bool = True):
    """
    Run an expanding window forecast with a given model and feature set.
    """
    # Prepare data with the chosen feature set
    X, y_solar = prepare_data(energy_data, weather_data, feature_set=feature_set)
    
    # Define forecast days (e.g. starting after a given number of days)
    start_date = X.index.min().normalize() + pd.Timedelta(days=10)
    end_date = X.index.max().normalize()
    forecast_days = pd.date_range(start=start_date, end=end_date, freq='D')

    forecast_dates, train_mse_list, test_mse_list = [], [], []
    all_test_y, all_test_preds = [], []
    all_rows = []

    # Run forecasts in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(forecast_for_day, fd, X, y_solar, model_name, model_params)
            for fd in forecast_days
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Forecasting days"):
            res = future.result()
            if res is not None:
                results.append(res)

    # Sort results by forecast day and accumulate metrics
    results.sort(key=lambda x: x[0])
    for forecast_day, train_mse, test_mse, day_y, day_pred, timestamps in results:
        forecast_dates.append(forecast_day)
        train_mse_list.append(train_mse)
        test_mse_list.append(test_mse)
        all_test_y.extend(day_y)
        all_test_preds.extend(day_pred)
        for ts, actual, pred in zip(timestamps, day_y, day_pred):
            squared_error = (actual - pred) ** 2
            all_rows.append({
                'date&time': ts,
                'actual realized generation': actual,
                'forecast': pred,
                'squared error': squared_error
            })
        print(f"Forecast Day: {forecast_day.date()} | Train MSE: {train_mse:.3f} | Test MSE: {test_mse:.3f}")

    overall_test_mse = mean_squared_error(all_test_y, all_test_preds)
    print("\nOverall Test MSE over all forecast sessions:", overall_test_mse)
    for row in all_rows:
        row['total mse'] = overall_test_mse

    output_df = pd.DataFrame(all_rows)
    output_filename = f"{model_name}_{feature_set}_day_ahead_forecast_results.csv"
    output_df.to_csv(output_filename, index=False)
    print(f"✅ Forecast results exported to '{output_filename}'")

    # Plotting MSE evolution over time if enabled
    if plot_graphs:
        plt.figure(figsize=(10, 6))
        plt.plot(forecast_dates, train_mse_list, label="Train MSE", marker='o')
        plt.plot(forecast_dates, test_mse_list, label="Test MSE", marker='o')
        plt.xlabel("Forecast Day")
        plt.ylabel("MSE")
        plt.title(f"Expanding Window Forecast ({model_name} - {feature_set})")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show() 



def compute_forecast_correlation_matrix(file_dict: dict):
    """
    Compute a correlation matrix between forecasts from multiple CSV files.
    
    Parameters:
      - file_dict: Dictionary mapping a label (e.g., "xgboost_lag_features") to its CSV file path.
    
    Returns:
      - A pandas DataFrame containing the correlation matrix of forecasts.
    """
    forecast_dfs = {}
    for label, file_path in file_dict.items():
        df = pd.read_csv(file_path)
        # Keep only the date and forecast columns and rename forecast column to the label
        forecast_dfs[label] = df[['date&time', 'forecast']].rename(columns={'forecast': label})
    
    # Merge all dataframes on "date&time"
    merged_df = None
    for label, df in forecast_dfs.items():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='date&time', how='outer')
    merged_df.dropna(inplace=True)  # ensure alignment

    # Compute and return the correlation matrix on the forecast columns
    corr_matrix = merged_df.drop(columns='date&time').corr()
    return corr_matrix


def combine_forecasts_to_csv(forecast_files: dict, output_filename: str = "combined_forecasts.csv"):
    """
    Combines multiple forecast CSVs into one file with columns: Zeit, HR, A1, ..., A6.

    Parameters:
    - forecast_files: dict with keys A1 to A6 and values as file paths.
    - output_filename: path for the output CSV file.
    """
    import pandas as pd

    merged_df = None

    for i, (label, file_path) in enumerate(forecast_files.items()):
        df = pd.read_csv(file_path, parse_dates=['date&time'])
        df = df.rename(columns={'date&time': 'Zeit'})

        if i == 0:
            # First file: include Zeit, actual, forecast
            df = df[['Zeit', 'actual realized generation', 'forecast']]
            df = df.rename(columns={
                'actual realized generation': 'HR',
                'forecast': label  # A1
            })
        else:
            # Subsequent files: keep Zeit and forecast only
            df = df[['Zeit', 'forecast']]
            df = df.rename(columns={'forecast': label})  # A2–A6

        # Merge with existing
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='Zeit', how='outer')

    # Ensure column order: Zeit, HR, A1...A6
    final_columns = ['Zeit', 'HR'] + list(forecast_files.keys())
    merged_df = merged_df[final_columns]
    merged_df = merged_df.sort_values('Zeit').dropna()

    merged_df.to_csv(output_filename, sep=';', index=False)
    print(f"✅ Combined forecast file saved to: {output_filename}")




if __name__ == "__main__":
    use_gpu = False

    if use_gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print("✅ Using GPU:", gpus[0])
            except RuntimeError as e:
                print("❌ GPU setup error:", e)
        else:
            print("⚠️ No GPU detected. Running on CPU.")
    else:
        # Force TensorFlow to use only CPU
        tf.config.set_visible_devices([], 'GPU')
        print("🚫 Forcing TensorFlow to run on CPU only.")

    energy_df = data_loader.load_kaggle_data(file_names.kaggle_files.energy_data_file)
    weather_df = data_loader.load_kaggle_data(file_names.kaggle_files.weather_data_file)

    # COMMENT THIS OUT IF YOU NO LONGER WANT TO EXPLORE THE DATA
    # Explore the data 
    explore_data(energy_df, "Energy Data")
    explore_data(weather_df, "Weather Data")

    # Define model configurations
    model_configs = {
        "xgboost": {
            "params": {
                'objective': 'reg:squarederror',
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 7,
                'subsample': 0.8,
                'colsample_bytree': 1.0,
                'n_jobs': -1,
                'verbosity': 0
            }
        },
        "random_forest": {
            "params": {
                'n_estimators': 10,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'n_jobs': -1,
                'random_state': 42
            }
        }
    }

    # Define which feature sets to run. Options: "lag_features", "generation_features", "combined"
    feature_sets_to_run = ["lag_features", 
                           "generation_features", 
                           "combined"
                           ]
    
    # Dictionary to hold forecast CSV file paths for later correlation analysis.
    forecast_files = {}

    # Loop over each combination of model and feature set
    for model_name, config in model_configs.items():
        for feature_set in feature_sets_to_run:
            print(f"\n========== Running {model_name} with feature set: {feature_set} ==========")
            # You can disable plotting here by setting plot_graphs=False if needed.
            expanding_window_forecaster(energy_df, weather_df, model_name, config["params"], feature_set, plot_graphs=False)
            # Store the forecast CSV file path using a key combining model and feature set.
            key = f"{model_name}_{feature_set}"
            forecast_files[key] = f"{model_name}_{feature_set}_day_ahead_forecast_results.csv"

    # -------------------------------
    # Compute and display the correlation matrix
    # -------------------------------
    corr_matrix = compute_forecast_correlation_matrix(forecast_files)
    print("\nCorrelation Matrix between forecasts:")
    print(corr_matrix)


    forecast_files = {
    "A1": "xgboost_lag_features_day_ahead_forecast_results.csv",
    "A2": "xgboost_generation_features_day_ahead_forecast_results.csv",
    "A3": "xgboost_combined_day_ahead_forecast_results.csv",
    "A4": "random_forest_lag_features_day_ahead_forecast_results.csv",
    "A5": "random_forest_generation_features_day_ahead_forecast_results.csv",
    "A6": "random_forest_combined_day_ahead_forecast_results.csv"
    }

    # Combine all forecasts into a single CSV file
    combine_forecasts_to_csv(forecast_files, output_filename="all_combined_forecasts.csv")

    # Optionally, visualize the correlation matrix as a heatmap
    # plt.figure(figsize=(8, 6))
    # plt.imshow(corr_matrix, cmap='viridis', interpolation='none')
    # plt.colorbar()
    # plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
    # plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
    # plt.title("Forecast Correlation Matrix")
    # plt.tight_layout()
    # plt.show()