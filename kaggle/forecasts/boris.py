import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


from data.data_loader import DataLoader
from data.data_processing import explore_data
from configuration import FileNames


file_names = FileNames()
data_loader = DataLoader()

##
## TO RUN THIS FILE:
## - OPEN THE TERMINAL (ctrl + ` [see the left bottom corner of your keyboard])
## - WRITE: python -m kaggle.forecasts.boris [or the name of your file]
## - PRESS ENTER TO RUN
## - RESULTS APPEAR IN THE TERMINAL
##
def xgboost_forecast(df: pd.DataFrame, initial_train_days: int = 10, forecast_horizon: int = 24) -> pd.DataFrame:
    """
    Perform day-ahead forecasts using an expanding window approach with XGBoost.
    
    Parameters:
      df (pd.DataFrame): The merged DataFrame with preprocessed features and target.
      initial_train_days (int): Number of days to use for the initial training window.
      forecast_horizon (int): Number of hours to forecast (24 for a day-ahead forecast).
    
    Returns:
      pd.DataFrame: A DataFrame containing timestamps, predicted values, actual values, and the day's MSE.
    """
    # Define the target column and list of feature columns for the model.
    target_col = 'generation solar'
    feature_cols = [
        'hour', 'day_of_week', 'lag_24', 'roll_mean_24', 'lag_48',
        'temp', 'temp_min', 'temp_max', 'pressure', 'humidity',
        'wind_speed', 'wind_deg', 'rain_1h', 'rain_3h', 'snow_3h', 'clouds_all', 'weather_severity_score'
    ]
    
    # Ensure data is sorted by time
    df = df.sort_values('time').reset_index(drop=True)
    
    # Determine the total number of days (assuming hourly data: 24 observations per day)
    total_hours = len(df)
    total_days = total_hours // 24

    # This list will store forecast results for each day.
    forecast_results = []

    # Loop over days starting from initial_train_days until the last full day available.
    for day in range(initial_train_days, total_days):
        # Calculate indices for training and testing.
        train_end_index = day * 24  # all data up to the day to forecast
        test_start_index = train_end_index
        test_end_index = test_start_index + forecast_horizon

        # If there's not a full day left, break out of the loop.
        if test_end_index > total_hours:
            break

        # Slice the data for training and testing.
        train_data = df.iloc[:train_end_index].copy()
        test_data = df.iloc[test_start_index:test_end_index].copy()

        # Drop rows with NaN values in the features or target.
        train_data = train_data.dropna(subset=feature_cols + [target_col])
        test_data = test_data.dropna(subset=feature_cols + [target_col])
        
        # Check if test_data is empty after dropna; if so, skip this iteration.
        if test_data.empty:
            print(f"Day {day+1} forecast skipped due to insufficient test samples after dropna.")
            continue
        
        # Prepare the training and testing matrices.
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]

        # Again check if X_test or y_test are empty.
        if X_test.empty or y_test.empty:
            print(f"Day {day+1} forecast skipped due to empty feature/target arrays in test data.")
            continue

        # Initialize the XGBoost regressor with default parameters.
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        
        # Train the model on the training data.
        model.fit(X_train, y_train)
        
        # Predict for the next day (24 hours ahead).
        y_pred = model.predict(X_test)
        
        # Calculate the Mean Squared Error for this day.
        mse_day = mean_squared_error(y_test, y_pred)
        
        # Build a DataFrame for this day's forecast.
        day_forecast = pd.DataFrame({
            'time': test_data['time'],
            'predicted_generation_solar': y_pred,
            'actual_generation_solar': y_test
        })
        day_forecast['day_mse'] = mse_day
        
        # Append the day's results.
        forecast_results.append(day_forecast)
        #print(f"Day {day+1} forecast completed. MSE: {mse_day:.4f}")

    # Combine all day forecasts into one DataFrame.
    if forecast_results:
        all_forecasts = pd.concat(forecast_results, ignore_index=True)
        # Compute overall MSE across all forecasted hours.
        overall_mse = mean_squared_error(all_forecasts['actual_generation_solar'], all_forecasts['predicted_generation_solar'])
        print(f"Overall MSE across all forecast days: {overall_mse:.4f}")
    else:
        all_forecasts = pd.DataFrame()
        print("No forecasts were produced.")
    
    return all_forecasts

# Modify the forecaster function to integrate the new forecasting code.
def forecaster(energy_data: pd.DataFrame, weather_data: pd.DataFrame):
    """
    Forecast 'generation solar' using an expanding window approach with XGBoost.
    
    Steps:
      1. Preprocess and merge energy and weather data (including creating features).
      2. Save the merged data for reference.
      3. Apply an expanding-window forecast to predict the next day's (24 hours) generation.
      4. Save the forecast results to CSV and print the overall MSE.
    """
    # Merge and create features.
    df = create_features(energy_data, weather_data)
    
    # Save the merged preprocessed data (optional but useful for inspection).
    df.to_csv("merged_preprocessed_data.csv", index=False)
    print("Merged preprocessed data saved to 'merged_preprocessed_data.csv'")
    
    # Run the expanding-window forecasting procedure.
    forecast_df = xgboost_forecast(df, initial_train_days=10, forecast_horizon=24)
    
    # Save the forecast results to CSV.
    forecast_df.to_csv("xgboost_expanding_forecasts.csv", index=False)
    print("Forecasts saved to 'xgboost_expanding_forecasts.csv'")
    

def preprocess_energy_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the energy data:
    - Convert 'time' to datetime.
    - Sort by time.
    - Extract time features (hour and day_of_week).
    - Create lag features for 'generation solar'.
    """
    # Convert the 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert(None)
    
    # Sort the data by time to ensure proper ordering for lag features
    df = df.sort_values('time').reset_index(drop=True)
    
    # Extract time-based features
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek  # Monday=0, Sunday=6
    
    # Create lag features (assuming the data is hourly):
    # - Lag of 24 hours (same hour previous day)
    df['lag_24'] = df['generation solar'].shift(24)
    # - Rolling average over the past 24 hours (shifted to avoid lookahead bias)
    df['roll_mean_24'] = df['generation solar'].rolling(window=24).mean().shift(1)
    # - Lag of 48 hours as an additional feature
    df['lag_48'] = df['generation solar'].shift(48)
    
    return df

def preprocess_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the weather data:
    - Convert 'dt_iso' to datetime.
    - Average numerical weather features by timestamp.
    - Determine the most common 'weather_main' for each timestamp.
    - Merge the averages with the common weather condition.
    - Add a 'weather severity' feature.
    """
    # Convert 'dt_iso' to datetime
    df['dt_iso'] = pd.to_datetime(df['dt_iso'], utc=True).dt.tz_convert(None)
    
    # Average numeric weather data by time (in case of multiple entries per timestamp)
    avg_weather = df.groupby('dt_iso').mean(numeric_only=True).reset_index()
    
    # For the categorical weather condition, get the most common value for each timestamp
    common_weather = df.groupby('dt_iso')['weather_main'].agg(lambda x: x.mode().iloc[0]).reset_index()
    
    # Merge numeric averages with the most common weather condition
    weather_processed = pd.merge(avg_weather, common_weather, on='dt_iso', how='left')
    
    # Add weather severity feature:
    # Define base severity for each weather condition (convert to lowercase for consistency)
    severity_map = {
        'clear': 0,
        'clouds': 1,
        'mist': 2,
        'fog': 3,
        'drizzle': 4,
        'rain': 5,
        'snow': 6,
        'dust': 6,
        'haze': 6,
        'smoke': 6,
        'squall': 7,
        'thunderstorm': 8
    }
    # Create a lowercase version of weather_main to match keys in severity_map
    weather_processed['weather_main_lower'] = weather_processed['weather_main'].str.lower()
    weather_processed['weather_base_severity'] = weather_processed['weather_main_lower'].map(severity_map)
    
    # Extract detail from 'weather_id' (last two digits)
    weather_processed['weather_detail_raw'] = weather_processed['weather_id'] % 100
    # Rank the detail within each weather condition group
    weather_processed['detail_rank'] = weather_processed.groupby('weather_main_lower')['weather_detail_raw'].rank(method='dense').astype(int)
    
    # Normalize the detail rank to a [0, 1] range for each weather condition
    def normalize(series):
        if series.max() > 1:
            return (series - 1) / (series.max() - 1)
        else:
            return 0.5
    weather_processed['normalized_detail'] = weather_processed.groupby('weather_main_lower')['detail_rank'].transform(normalize)
    
    # Compute the weather severity score as the sum of base severity and normalized detail
    weather_processed['weather_severity_score'] = weather_processed['weather_base_severity'] + weather_processed['normalized_detail']
    
    # Clean up intermediate columns
    weather_processed.drop(columns=['weather_main_lower', 'weather_base_severity', 'weather_detail_raw', 'detail_rank', 'normalized_detail'], inplace=True)
    
    return weather_processed

def merge_energy_weather(energy_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge energy and weather data on the timestamp columns.
    """
    # Create copies to avoid modifying the original DataFrames
    energy_df = energy_df.copy()
    weather_df = weather_df.copy()
    
    # Merge on energy_data['time'] and weather_data['dt_iso']
    merged_df = pd.merge(energy_df, weather_df, left_on='time', right_on='dt_iso', how='inner')
    # Drop the redundant 'dt_iso' column after merging
    merged_df.drop(columns=['dt_iso'], inplace=True)
    
    # Ensure the merged DataFrame is sorted by time
    merged_df = merged_df.sort_values('time').reset_index(drop=True)
    return merged_df

def create_features(energy_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and merge energy and weather data, creating additional features.
    """
    # Process the energy and weather data individually
    energy_processed = preprocess_energy_data(energy_df)
    weather_processed = preprocess_weather_data(weather_df)
    
    # Merge the processed data on the timestamp
    merged_df = merge_energy_weather(energy_processed, weather_processed)
    return merged_df


if __name__ == "__main__":
    energy_df = data_loader.load_kaggle_data(file_names.kaggle_files.energy_data_file)
    weather_df = data_loader.load_kaggle_data(file_names.kaggle_files.weather_data_file)

    # Explore the data (optional)
    #explore_data(energy_df, "Energy Data")
    #explore_data(weather_df, "Weather Data")

    start_time = time.time()
    forecaster(energy_df, weather_df)
    end_time = time.time()
    print("Total runtime: {:.2f} seconds".format(end_time - start_time))