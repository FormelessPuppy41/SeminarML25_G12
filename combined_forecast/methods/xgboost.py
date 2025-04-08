"""
XGBoost regression model
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

from xgboost import XGBRegressor

from configuration import ModelSettings
from .utils import data_interpolate_prev, data_interpolate_fut


def get_xgb_model(params: Dict[str, Any]) -> XGBRegressor:
    """
    Create an XGBoost regression model using given parameters, with defaults where missing.

    Args:
        params (Dict[str, Any]): XGBoost parameters.

    Returns:
        XGBRegressor: Configured XGBRegressor instance.
    """
    return XGBRegressor(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 3),
        learning_rate=params.get('learning_rate', 0.1),
        random_state=params.get('random_state', 42),
        objective=params.get('objective', 'reg:squarederror'),
        verbosity=0
    )


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
        train: Training DataFrame.
        test: Test DataFrame.
        target: Name of target column.
        features: List of feature column names.
        params: XGBoost hyperparameters.

    Returns:
        pd.DataFrame: Test set with predictions.
    """
    model = get_xgb_model(params)
    model.fit(train[features], train[target])

    test = test.copy()
    test['prediction'] = model.predict(test[features])
    return test


def run_day_ahead_xgboost(
    df: pd.DataFrame,
    # flag_matrix_df: pd.DataFrame,
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

    Args:
        df: Full dataset with datetime, target, and features.
        target_column: Target variable name.
        feature_columns: Feature column names.
        forecast_horizon: Forecast steps (default=96).
        rolling_window_days: Days to use for training.
        xgb_params: XGBoost model parameters.
        datetime_col: Name of datetime column.
        freq: Data frequency.

    Returns:
        pd.DataFrame: Forecast results with ['target_time', 'prediction', 'actual'].
    """
    xgb_params = xgb_params or {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'random_state': 42}

    if datetime_col not in df.columns:
        raise ValueError(f"'{datetime_col}' not found in DataFrame.")

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    forecast_results = []
    unique_dates = df[datetime_col].unique()
    forecast_dates = [pd.Timestamp(d) for d in unique_dates if pd.Timestamp(d).hour == 9 and pd.Timestamp(d).minute == 0]

    current_date_interpolator_prev = None
    current_date_interpolator_fut = None
    last_forecast_date = None

    for forecast_date in forecast_dates[rolling_window_days:]:
        print(f"Forecast date (09:00): {forecast_date}")
        forecast_start = forecast_date.normalize() + pd.Timedelta(days=1)

        if last_forecast_date is None or forecast_date != last_forecast_date:
            last_forecast_date = forecast_date
            current_date_interpolator_prev = data_interpolate_prev(df, forecast_date, rolling_window_days)
            current_date_interpolator_fut = data_interpolate_fut(df, forecast_start, forecast_horizon, freq=freq)

        train_df = current_date_interpolator_prev
        test_df = current_date_interpolator_fut

        if train_df.isnull().values.any() or test_df.isnull().values.any():
            print("WARNING: Training or testing data contains NaN values.")
            continue

        if train_df.empty or test_df.empty or train_df[target_column].isnull().all() or test_df[target_column].isnull().all():
            print(f"WARNING: No data for forecast date: {forecast_date}")
            continue

        for ts in test_df[ModelSettings.datetime_col]:
            row_df = test_df[test_df[ModelSettings.datetime_col] == ts]

            if row_df.empty:
                print(f"WARNING: No test data for timestamp {ts}")
                continue

            pred_df = run_xgboost(train_df, row_df, target_column, feature_columns, xgb_params)
            forecast_results.append({
                'target_time': ts,
                'prediction': pred_df['prediction'].values[0],
                'actual': pred_df[target_column].values[0]
            })

    return pd.DataFrame(forecast_results)


if __name__ == '__main__':
    from ..utils import evaluate_forecast, generate_sample_data

    df_sample = generate_sample_data(start='2023-01-01', days=20)
    target_col = 'HR'
    feature_cols = [f'A{i}' for i in range(1, 8)]

    forecast_df = run_day_ahead_xgboost(
        df_sample,
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
