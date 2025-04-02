"""
XGBoost regression model
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

def data_interpolate_prev(
        raw_df: pd.DataFrame, 
        indicator_df: pd.DataFrame, 
        current_date: pd.Timestamp, 
        rolling_window_days: int = 165
    ) -> pd.DataFrame:
    """
    Returns the training data with interpolated values for the previous rolling window.

    Args:
        raw_df (pd.DataFrame): The raw input data -> df(time, HR, forecasts)
        indicator_df (pd.DataFrame): The indicator data, indicting which time/fcst combination has either 0 (complete), or other value.
        current_date (pd.Timestamp): The current date.

    Returns:
        pd.DataFrame: The training data with interpolated values for the previous rolling window.
    """
    # Define time window and slice once
    start_date = current_date - pd.Timedelta(days=rolling_window_days)
    end_date = current_date
    mask = (raw_df.index >= start_date) & (raw_df.index <= end_date)

    # cannot use index bcs of double 02:00 values for winter time.

    # Create a working copy of just the relevant slice
    working_df = raw_df.loc[mask].copy()
    
    # Initialize output with 'HR' column
    interpolated_df = pd.DataFrame(index=working_df.index)
    interpolated_df['HR'] = working_df['HR']

    for col in working_df.columns:
        if col == 'HR':
            continue

        if indicator_df.loc[current_date, col] == 0:
            interpolated_df[col] = working_df[col].interpolate(method='linear')
        else:
            # if not complete, fill with 0
            interpolated_df[col] = np.zeros(len(working_df))
            print(f"WARNING: No eligible data for {col} on {current_date}. Value is != 0 -> Skipping.")

    return interpolated_df

def data_interpolate_fut(
        raw_df: pd.DataFrame,
        indicator_df: pd.DataFrame,
        current_date: pd.Timestamp,
        forecast_horizon: int = 96
    ) -> pd.DataFrame:
    """
    Returns the training data with interpolated values for the future forecast horizon.

    Args:
        raw_df (pd.DataFrame): The raw input data -> df(time, HR, forecasts)
        indicator_df (pd.DataFrame): The indicator data, indicting which time/fcst combination has either 0 (complete), or other value.
        current_date (pd.Timestamp): The current date.  
        forecast_horizon (int, optional): _description_. Defaults to 24.
        freq (str, optional): _description_. Defaults to '15min'.

    Returns:
        pd.DataFrame: The training data with interpolated values for the future forecast horizon.
    """
    # Define time window and slice once
    start_date = current_date
    end_date = current_date + pd.Timedelta(minutes=(forecast_horizon - 1) * 15)
    mask = (raw_df.index >= start_date) & (raw_df.index <= end_date)

    # Create a working copy of just the relevant slice
    working_df = raw_df.loc[mask].copy()

    # Initialize output with 'HR' column
    interpolated_df = pd.DataFrame(index=working_df.index)
    interpolated_df['HR'] = working_df['HR']

    # Iterate over each column in the working DataFrame
    for col in working_df.columns:
        if col == 'HR':
            continue

        # Check if the column is complete
        if indicator_df.loc[current_date, col] == 0:
            # Interpolate the column using linear interpolation
            interpolated_df[col] = working_df[col].interpolate(method='linear')
        else:
            # If not complete, fill with zeros
            interpolated_df[col] = np.zeros(len(working_df))
            print(f"WARNING: No eligible data for {col} on {current_date}. Value is != 0 -> Skipping.")


def run_xgboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: List[str],
    params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Train and apply XGBoost regression model.

    Args:
        train: Training DataFrame
        test: Test DataFrame
        target: Name of target column
        features: List of feature column names
        params: XGBoost hyperparameters

    Returns:
        pd.DataFrame: Test set with predictions
    """
    # Set default parameters if not provided
    if 'n_estimators' not in params:
        params['n_estimators'] = 100
    if 'max_depth' not in params:
        params['max_depth'] = 3
    if 'learning_rate' not in params:
        params['learning_rate'] = 0.1
    if 'random_state' not in params:
        params['random_state'] = 42
    if 'objective' not in params:
        params['objective'] = 'reg:squarederror'

    model = XGBRegressor(**params)
    model.fit(train[features], train[target])

    test = test.copy()
    test['prediction'] = model.predict(test[features])

    return test

def run_day_ahead_xgboost(
    df: pd.DataFrame,
    indicator_df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    forecast_horizon: int = 96,
    rolling_window_days: int = 165,
    xgb_params: Dict[str, Any] = None,
    datetime_col: str = 'datetime',
    freq: str = '15min'
) -> pd.DataFrame:
    """
    Generate 96-step day-ahead forecasts using rolling XGBoost regression.

    Returns:
        pd.DataFrame with forecast results.
    """
    xgb_params = xgb_params or {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'random_state': 42, 'objective': 'reg:squarederror'}

    # Should be done in data preprocessing
    if datetime_col not in df.columns:
        raise ValueError(f"'{datetime_col}' not found in DataFrame.")
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col).sort_index()

    forecast_results = []
    unique_dates = df.index.normalize().unique()

    for forecast_date in unique_dates[rolling_window_days:]:
        train_df = data_interpolate_prev(df, indicator_df, forecast_date, rolling_window_days)
        test_df = data_interpolate_fut(df, indicator_df, forecast_date, forecast_horizon)

        if train_df.empty or test_df.empty or train_df[target_column].isnull().all() or test_df[target_column].isnull().all():
            print(f"WARNING: No data for forecast date: {forecast_date}")
            continue

        for ts in test_df.index:
            row_df = test_df.loc[[ts]]
            pred_df = run_xgboost(train_df, row_df, target_column, feature_columns, xgb_params)

            forecast_results.append({
                'target_time': ts,
                'prediction': pred_df['prediction'].values[0],
                'actual': pred_df[target_column].values[0]
            })

        """
        Maybe better:
        
        pred_df = run_xgboost(train_df, test_df, target_column, feature_columns, xgb_params)
        for ts in pred_df.index:
            forecast_results.append({
                'target_time': ts,
                'prediction': pred_df.loc[ts, 'prediction'],
                'actual': pred_df.loc[ts, target_column]
            })

        bcs:
        Problem: This refits the XGBoost model 96 times per day — that's highly inefficient and not how tree-based models are meant to be used.
        Fix: Train once per day, and predict for the whole test_df at once:
        """

    return pd.DataFrame(forecast_results)

# Example usage for local testing
if __name__ == '__main__':
    from ..utils import evaluate_forecast, generate_sample_data

    df_sample = generate_sample_data(start='2023-01-01', days=20)
    target_col = 'HR'
    feature_cols = [f'A{i}' for i in range(1, 8)]

    forecast_df = run_day_ahead_xgboost(
        df_sample,
        indicator_df=pd.DataFrame(0, index=pd.date_range('2023-01-01', periods=20, freq='D'), columns=feature_cols),
        target_column=target_col,
        feature_columns=feature_cols,
        rolling_window_days=5,
        xgb_params={'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05, 'random_state': 42}
    )

    print(forecast_df)

    if not forecast_df.empty:
        rmse = evaluate_forecast(forecast_df)
        print(f"\nRMSE on XGBoost forecast: {rmse:.2f}")
    else:
        print("No forecasts generated.")

