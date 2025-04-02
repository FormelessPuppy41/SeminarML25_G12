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


def expanding_window_forecaster(energy_data: pd.DataFrame, weather_data: pd.DataFrame):
    """
    Uses an expanding window to train an XGBoost model for day-ahead forecasting.
    
    - The initial training window is the first 10 days.
    - For each subsequent day, the model is trained on all prior data and the next 24 hours are forecast.
    - Computes both training and test MSE per day and overall MSE over all forecast sessions.
    - Plots the evolution of training and test MSE over time.
    """
    # === Step 1: Preprocess the energy data (same as your original code) ===
    df = energy_data.copy()
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time')
    df.index = df.index.tz_convert(None)
    
    # Target variables (for example, we forecast solar generation)
    df['generation wind'] = df['generation wind onshore']  # optional
    df = df[['generation solar', 'generation wind', 
             'forecast solar day ahead', 'forecast wind onshore day ahead',
             'total load actual', 'price actual'
             ]]
    
    # Add time features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Add lag features
    lags = [1, 2, 3, 6, 12, 24, 48, 168]  # hourly lags up to 7 days
    for lag in lags:
        df[f'solar_lag_{lag}'] = df['generation solar'].shift(lag)
        df[f'wind_lag_{lag}'] = df['generation wind'].shift(lag)
    
    # === Step 2: Process weather data and merge ===
    weather_data['dt_iso'] = pd.to_datetime(weather_data['dt_iso'], utc=True)
    weather_data = weather_data.set_index('dt_iso')

    # Keep only numeric and useful weather features
    keep_weather_features = ['temp', 'pressure', 'humidity', 'wind_speed',
                            'rain_1h', 'rain_3h', 'snow_3h', 'clouds_all']
    weather_numeric = weather_data[keep_weather_features]

    # Resample to hourly and rename
    weather_agg = weather_numeric.resample('h').mean()
    weather_agg.index = weather_agg.index.tz_convert(None)
    weather_agg.columns = [f'weather_{col}' for col in weather_agg.columns]

    # Merge with energy data
    df = df.merge(weather_agg, left_index=True, right_index=True, how='left')

    # Create lagged weather features
    weather_lags = [1, 3, 6, 12, 24]
    for lag in weather_lags:
        for col in weather_agg.columns:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Drop original (non-lagged) weather columns to prevent leakage
    df.drop(columns=weather_agg.columns.tolist(), inplace=True)

    # === Step 3: Add lagged versions of 'total load actual' and 'price actual' ===
    load_price_lag_features = ['total load actual', 'price actual']
    load_price_lags = [1, 3, 6, 12, 24]

    for lag in load_price_lags:
        for col in load_price_lag_features:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Drop original columns (they leak future information)
    df.drop(columns=load_price_lag_features, inplace=True)

    # Drop missing values (from lagging, merging, etc.)
    df.dropna(inplace=True)

    # Old wrong weather features saved
    # # === Step 2: Process weather data and merge ===
    # weather_data['dt_iso'] = pd.to_datetime(weather_data['dt_iso'], utc=True)
    # weather_data = weather_data.set_index('dt_iso')
    # weather_numeric = weather_data.select_dtypes(include=[np.number])
    # weather_agg = weather_numeric.groupby('dt_iso').mean().resample('h').mean()
    # weather_agg.index = weather_agg.index.tz_convert(None)
    # weather_agg.columns = [f'weather_{col}' for col in weather_agg.columns]
    
    # df = df.merge(weather_agg, left_index=True, right_index=True, how='left')
    # df.dropna(inplace=True)
    
    # === Step 3: Define features and target ===
    # We drop the columns we want to predict from X
    X = df.drop(columns=['generation solar', 'generation wind', 
                         'forecast solar day ahead', 'forecast wind onshore day ahead'])
    y_solar = df['generation solar']
    # Optionally, you can do the same for wind: y_wind = df['generation wind']
    
    print("\n🧠 Features used for training:")
    print(X.columns.tolist())
    
    # === Step 4: Set up expanding window forecasting ===
    # Define forecast days starting after the first 10 days.
    start_date = X.index.min().normalize() + pd.Timedelta(days=10)
    end_date = X.index.max().normalize()
    forecast_days = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # To accumulate per-day metrics and overall predictions
    forecast_dates = []
    train_mse_list = []
    test_mse_list = []
    all_test_y = []
    all_test_preds = []
    
    # Define XGBoost parameters (using your solar model as an example)
    xgb_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 1.0,
        #'gamma': 0,
        #'min_child_weight': 10,
        #'reg_alpha': 0,
        #'reg_lambda': 2.0,
        'n_jobs': -1,
        'verbosity': 0
    }

    rf_params = {
        'n_estimators': 10,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'n_jobs': -1,
        'random_state': 42
    }

    
    model_name = "random_forest"
    #model_name = "xgboost"
    model_params = rf_params 
    #model_params = xgb_params
    
    # === Step 6: Run forecasts in parallel ===
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all forecast tasks in parallel
        futures = [executor.submit(forecast_for_day, fd, X, y_solar, model_name, model_params) for fd in forecast_days]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Forecasting days"):
            res = future.result()
            if res is not None:
                results.append(res)
    
    # Sort results by forecast day
    results.sort(key=lambda x: x[0])
    
    # Unpack results
    forecast_dates = []
    train_mse_list = []
    test_mse_list = []
    all_test_y = []
    all_test_preds = []
    all_rows = []
    
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
    
    # === Step 5: Compute overall test MSE over all forecast sessions ===
    overall_test_mse = mean_squared_error(all_test_y, all_test_preds)
    print("\nOverall Test MSE over all forecast sessions:", overall_test_mse)

    for row in all_rows:
        row['total mse'] = overall_test_mse

    output_df = pd.DataFrame(all_rows)
    output_df.to_csv(f"{model_name}_day_ahead_forecast_results.csv", index=False)
    print("✅ Forecast results exported to 'day_ahead_forecast_results.csv'")

    # === Step 6: Plot the evolution of MSE ===
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_dates, train_mse_list, label="Train MSE", marker='o')
    plt.plot(forecast_dates, test_mse_list, label="Test MSE", marker='o')
    plt.xlabel("Forecast Day")
    plt.ylabel("MSE")
    plt.title("Expanding Window Forecast: Train vs Test MSE")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()    



def forecast_for_day(forecast_day, X, y, model_name, model_params):
    # Training data
    train_mask = X.index < forecast_day
    X_train = X[train_mask]
    y_train = y[train_mask]

    # Test data: 24-hour forecast
    test_mask = (X.index >= forecast_day) & (X.index < forecast_day + pd.Timedelta(days=1))
    X_test = X[test_mask]
    y_test = y[test_mask]

    if X_train.empty or X_test.empty:
        return None

    # Select model
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


def create_multistep_sequences(X_df, y_df, window_size=24, horizon=24):
    """
    X_df: input features
    y_df: target variable (e.g., solar gen)
    window_size: number of timesteps used for input
    horizon: number of timesteps to predict
    """
    X_seq, y_seq = [], []
    for i in range(window_size, len(X_df) - horizon + 1):
        X_seq.append(X_df.iloc[i - window_size:i].values)
        y_seq.append(y_df.iloc[i:i + horizon].values)
    return np.array(X_seq), np.array(y_seq)




# ========= Helper function to build NN models =========
def build_nn_model(model_type, input_shape):
    """
    Builds and compiles a NN model given the model type and input shape.
    
    model_type: one of ['lstm', 'stacked_lstm', 'cnn', 'cnn_lstm', 'time_distributed_mlp', 'encoder_decoder']
    input_shape: tuple (timesteps, features)
    """
    tf.keras.backend.clear_session()
    if model_type == "lstm":
        model = Sequential([
            tf.keras.Input(shape=input_shape),
            LSTM(100, return_sequences=True),
            Flatten(),
            Dense(200, activation='relu'),
            Dropout(0.1),
            Dense(24)
        ])
        optimizer = Adam(learning_rate=6e-3, amsgrad=True)
        
    elif model_type == "stacked_lstm":
        model = Sequential([
            tf.keras.Input(shape=input_shape),
            LSTM(250, return_sequences=True),
            LSTM(150, return_sequences=True),
            Flatten(),
            Dense(150, activation='relu'),
            Dropout(0.1),
            Dense(24)
        ])
        optimizer = Adam(learning_rate=3e-3, amsgrad=True)
        
    elif model_type == "cnn":
        # If sequence length is 1, use kernel_size=1 (otherwise kernel_size=2)
        ks = 1 if input_shape[0] < 2 else 2
        model = Sequential([
            tf.keras.Input(shape=input_shape),
            Conv1D(filters=48, kernel_size=ks, strides=1, padding='causal', activation='relu'),
            Flatten(),
            Dense(48, activation='relu'),
            Dense(24)
        ])
        optimizer = Adam(learning_rate=6e-3, amsgrad=True)
        
    elif model_type == "cnn_lstm":
        ks = 1 if input_shape[0] < 2 else 2
        model = Sequential([
            tf.keras.Input(shape=input_shape),
            Conv1D(filters=100, kernel_size=ks, strides=1, padding='causal', activation='relu'),
            LSTM(100, return_sequences=True),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(24)
        ])
        optimizer = Adam(learning_rate=4e-3, amsgrad=True)
        
    elif model_type == "time_distributed_mlp":
        model = Sequential([
            tf.keras.Input(shape=input_shape),
            TimeDistributed(Dense(200, activation='relu')),
            TimeDistributed(Dense(150, activation='relu')),
            TimeDistributed(Dense(100, activation='relu')),
            TimeDistributed(Dense(50, activation='relu')),
            Flatten(),
            Dense(150, activation='relu'),
            Dropout(0.1),
            Dense(24)
        ])
        optimizer = Adam(learning_rate=2e-3, amsgrad=True)
        
    elif model_type == "encoder_decoder":
        # Here we use the same lookback length as the number of timesteps in the input_shape.
        model = Sequential([
            tf.keras.Input(shape=input_shape),
            LSTM(50, activation='relu'),
            RepeatVector(input_shape[0]),
            LSTM(50, activation='relu', return_sequences=True),
            TimeDistributed(Dense(50, activation='relu')),
            Flatten(),
            Dense(25, activation='relu'),
            Dense(24)
        ])
        optimizer = Adam(learning_rate=1e-3, amsgrad=True)
        
    else:
        raise ValueError(f"Unsupported NN model type: {model_type}")
    
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(), 
        optimizer=optimizer, 
        metrics=[tf.keras.metrics.MeanSquaredError()]
    )

    return model


# # ========= Forecasting function for one day using NN =========
# def forecast_for_day_nn(forecast_day, X, y, model_type, epochs=50, batch_size=32, window_size=24, horizon=24):
#     """
#     For a given forecast day, trains the specified NN model on all data before forecast_day 
#     and predicts the next 24 hours.
    
#     For simplicity, we use the existing engineered features (including lags) and simply add 
#     a dummy time dimension so that the NN input shape becomes (1, num_features).
    
#     Returns:
#       (forecast_day, train_mse, test_mse, day_y, day_pred, timestamps)
#     """
#     # Split training and test data as before
#     train_mask = X.index < forecast_day
#     X_train = X[train_mask]
#     y_train_nn = y[train_mask]
    
#     test_mask = (X.index >= forecast_day) & (X.index < forecast_day + pd.Timedelta(days=1))
#     X_test = X[test_mask]
#     y_test_nn = y[test_mask]
    
#     if X_train.empty or X_test.empty or len(X_test) < window_size:
#         return None

#     # Reshape the data: add a time-dimension of 1.
#     # (This is a simple trick to allow reuse of the NN architectures.
#     # Note: for CNNs a sequence length of 1 is not ideal so we adjust kernel_size in build_nn_model.)
#     #X_train_nn = np.expand_dims(X_train.values, axis=1)  # shape: (n_train, 1, num_features)
#     #X_test_nn = np.expand_dims(X_test.values, axis=1)    # shape: (n_test, 1, num_features)


#     X_train_nn, y_train_nn = create_sliding_windows(X_train, y_train_nn, window_size)
#     X_test_nn, y_test_nn = create_sliding_windows(X_test, y_test_nn, window_size)
    
#     # Define input shape for NN model (timesteps, features)
#     input_shape = (X_train_nn.shape[1], X_train_nn.shape[2])
    
#     # Build model
#     model = build_nn_model(model_type, input_shape)
    
#     # Set up callbacks (using early stopping; we mimic model checkpointing by saving a temporary file)
#     checkpoint_path = f"{model_type}_temp_{forecast_day.strftime('%Y%m%d')}.keras"
#     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#     model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
    
#     # For validation, you could use a split of the training set
#     history = model.fit(X_train_nn, y_train_nn, epochs=epochs, batch_size=batch_size,
#                         validation_split=0.1, callbacks=[early_stopping, model_checkpoint],
#                         verbose=0)
    
#     # Optionally load the best model from checkpoint (if you want to strictly follow the snippet)
#     if os.path.exists(checkpoint_path):
#         model = tf.keras.models.load_model(checkpoint_path)
#         os.remove(checkpoint_path)  # clean up
    
#     # Compute training MSE
#     y_train_pred = model.predict(X_train_nn)
#     train_mse = mean_squared_error(y_train_nn, y_train_pred)
    
#     # Forecast on test set
#     y_test_pred = model.predict(X_test_nn)
#     test_mse = mean_squared_error(y_test_nn, y_test_pred)
    
#     # Get corresponding timestamps (align to y_test_nn)
#     timestamps = X_test.index[window_size:]

#     return (forecast_day, train_mse, test_mse, y_test_nn.tolist(), y_test_pred.flatten().tolist(), timestamps)
    

def forecast_for_day_nn(forecast_day, X, y, model_type, epochs=50, batch_size=32, window_size=24, horizon=24):
    # Training data: all data before forecast_day
    train_mask = X.index < forecast_day
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    # Ensure there's enough data for training sequences
    if len(X_train) < window_size + horizon:
        return None
    
    # Create training sequences (multi-output)
    X_train_nn, y_train_nn = create_multistep_sequences(X_train, y_train, window_size, horizon)
    
    # For testing, use the last window_size hours right before forecast_day as input
    test_input_start = forecast_day - pd.Timedelta(hours=window_size)
    test_input_end = forecast_day
    X_test_sample = X.loc[test_input_start:test_input_end].values
    X_test_nn = np.expand_dims(X_test_sample, axis=0)
    
    # Get true values for the forecast day (if available)
    y_test_sample = y.loc[forecast_day: forecast_day + pd.Timedelta(hours=horizon-1)].values
    y_test_nn = np.expand_dims(y_test_sample, axis=0)
    
    # Define input shape for NN model (timesteps, features)
    input_shape = (X_train_nn.shape[1], X_train_nn.shape[2])
    
    # Build model with modified output layer for multi-output
    model = build_nn_model(model_type, input_shape)
    
    # Set up callbacks
    checkpoint_path = f"{model_type}_temp_{forecast_day.strftime('%Y%m%d')}.keras"
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
    
    history = model.fit(X_train_nn, y_train_nn, epochs=epochs, batch_size=batch_size,
                        validation_split=0.1, callbacks=[early_stopping, model_checkpoint],
                        verbose=0)
    
    if os.path.exists(checkpoint_path):
        model = tf.keras.models.load_model(checkpoint_path)
        os.remove(checkpoint_path)
    
    # Compute training MSE
    y_train_pred = model.predict(X_train_nn)
    train_mse = mean_squared_error(y_train_nn, y_train_pred)
    
    # Forecast on the test sample (one forward pass for 24 outputs)
    y_test_pred = model.predict(X_test_nn)
    test_mse = mean_squared_error(y_test_nn, y_test_pred)
    
    # Timestamps for the forecast day (24 hourly timestamps)
    timestamps = y.loc[forecast_day: forecast_day + pd.Timedelta(hours=horizon-1)].index.tolist()

    return (forecast_day, train_mse, test_mse, y_test_nn.flatten().tolist(), y_test_pred.flatten().tolist(), timestamps)



# ========= Main expanding window forecaster for NN =========
def expanding_window_forecaster_nn(energy_data: pd.DataFrame, weather_data: pd.DataFrame, 
                                   nn_model_type="lstm", epochs=50, batch_size=32, use_gpu=True):
    """
    Uses an expanding window to train a neural network model for day-ahead forecasting.
    
    - Preprocesses the energy data and weather data.
    - Uses an expanding window: the model is trained on all prior data and then used to forecast the next 24 hours.
    - Computes training and test MSE per forecast day and overall test MSE.
    - Exports forecast details (timestamp, actual, forecast, squared error) to CSV.
    - Plots the evolution of training and test MSE over time.
    
    nn_model_type: one of ['lstm', 'stacked_lstm', 'cnn', 'cnn_lstm', 'time_distributed_mlp', 'encoder_decoder']
    """
    # === Step 1: Preprocess energy data (similar to your original code) ===
    df = energy_data.copy()
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time')
    df.index = df.index.tz_convert(None)
    
    # Target variables (example: forecasting generation solar)
    df['generation wind'] = df['generation wind onshore']  # optional
    df = df[['generation solar', 'generation wind', 
             'forecast solar day ahead', 'forecast wind onshore day ahead',
             'total load actual', 'price actual'
             ]]
    
    # Add time features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Add lag features
    lags = [1, 2, 3, 6, 12, 24, 48, 168]  # hourly lags up to 7 days
    for lag in lags:
        df[f'solar_lag_{lag}'] = df['generation solar'].shift(lag)
        df[f'wind_lag_{lag}'] = df['generation wind'].shift(lag)
    
    # === Step 2: Process weather data and merge ===
    weather_data['dt_iso'] = pd.to_datetime(weather_data['dt_iso'], utc=True)
    weather_data = weather_data.set_index('dt_iso')

    # Keep only numeric and useful weather features
    keep_weather_features = ['temp', 'pressure', 'humidity', 'wind_speed',
                            'rain_1h', 'rain_3h', 'snow_3h', 'clouds_all']
    weather_numeric = weather_data[keep_weather_features]

    # Resample to hourly and rename
    weather_agg = weather_numeric.resample('h').mean()
    weather_agg.index = weather_agg.index.tz_convert(None)
    weather_agg.columns = [f'weather_{col}' for col in weather_agg.columns]

    # Merge with energy data
    df = df.merge(weather_agg, left_index=True, right_index=True, how='left')

    # Create lagged weather features
    weather_lags = [1, 3, 6, 12, 24]
    for lag in weather_lags:
        for col in weather_agg.columns:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Drop original (non-lagged) weather columns to prevent leakage
    df.drop(columns=weather_agg.columns.tolist(), inplace=True)

    # === Step 3: Add lagged versions of 'total load actual' and 'price actual' ===
    load_price_lag_features = ['total load actual', 'price actual']
    load_price_lags = [1, 3, 6, 12, 24]

    for lag in load_price_lags:
        for col in load_price_lag_features:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Drop original columns (they leak future information)
    df.drop(columns=load_price_lag_features, inplace=True)

    # Drop missing values (from lagging, merging, etc.)
    df.dropna(inplace=True)
    
    # === Step 3: Define features and target ===
    X = df.drop(columns=['generation solar', 'generation wind', 
                         'forecast solar day ahead', 'forecast wind onshore day ahead'])
    y_solar = df['generation solar']
    
    print("\n🧠 Features used for training:")
    print(X.columns.tolist())
    
    # === Step 4: Set up expanding window forecasting ===
    # Define forecast days (starting after an initial window, e.g., 165 days, as in your code)
    start_date = X.index.min().normalize() + pd.Timedelta(days=10)
    end_date = X.index.max().normalize()
    forecast_days = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # To accumulate per-day metrics and overall predictions
    forecast_dates = []
    train_mse_list = []
    test_mse_list = []
    all_test_y = []
    all_test_preds = []
    all_rows = []
    

    if use_gpu:
        # === Run sequentially (GPU execution) ===
        for fd in tqdm(forecast_days, desc="Forecasting days"):
            res = forecast_for_day_nn(fd, X, y_solar, nn_model_type, epochs=epochs, batch_size=batch_size)
            if res is not None:
                forecast_day, train_mse, test_mse, day_y, day_pred, timestamps = res
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
    else:
        # === Run in parallel (CPU execution) ===
        print("⚙️ Running forecasts in parallel using CPU cores...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(forecast_for_day_nn, fd, X, y_solar, nn_model_type, epochs, batch_size)
                for fd in forecast_days
            ]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Forecasting days"):
                res = future.result()
                if res is not None:
                    forecast_day, train_mse, test_mse, day_y, day_pred, timestamps = res
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
    
    # === Step 5: Compute overall test MSE over all forecast sessions ===
    overall_test_mse = mean_squared_error(all_test_y, all_test_preds)
    print("\nOverall Test MSE over all forecast sessions:", overall_test_mse)
    
    for row in all_rows:
        row['total mse'] = overall_test_mse
    
    output_df = pd.DataFrame(all_rows)
    csv_filename = f"{nn_model_type}_day_ahead_forecast_results.csv"
    output_df.to_csv(csv_filename, index=False)
    print(f"✅ Forecast results exported to '{csv_filename}'")
    
    # === Step 6: Plot the evolution of MSE ===
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_dates, train_mse_list, label="Train MSE", marker='o')
    plt.plot(forecast_dates, test_mse_list, label="Test MSE", marker='o')
    plt.xlabel("Forecast Day")
    plt.ylabel("MSE")
    plt.title(f"Expanding Window Forecast ({nn_model_type}) - Train vs Test MSE")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


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

    # Run the forecaster
    #expanding_window_forecaster(energy_df, weather_df)

    # Choose one of the NN methods: e.g., "lstm", "stacked_lstm", "cnn", "cnn_lstm", "time_distributed_mlp", "encoder_decoder"
    #nn_model_type = "time_distributed_mlp"  # change as desired
    # Run the NN forecaster
    #expanding_window_forecaster_nn(energy_df, weather_df, nn_model_type=nn_model_type, epochs=50, batch_size=32, use_gpu=use_gpu)

    #nn_model_type = "cnn" 
    #expanding_window_forecaster_nn(energy_df, weather_df, nn_model_type=nn_model_type, epochs=50, batch_size=32, use_gpu=use_gpu)

    nn_model_type = "encoder_decoder" 
    expanding_window_forecaster_nn(energy_df, weather_df, nn_model_type=nn_model_type, epochs=50, batch_size=32, use_gpu=use_gpu)
