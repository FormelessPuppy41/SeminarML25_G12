"""
XGBoost regression model
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


from configuration import ModelSettings
from .utils import data_interpolate_prev, data_interpolate_fut

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


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
 ) -> Tuple[pd.DataFrame, int, int, float]:
    """
    Train and apply XGBoost regression model with optional GridSearchCV
    over n_estimators, max_depth, and learning_rate.

    Returns:
        pd.DataFrame: Test set with predictions in 'prediction' column.
        int: best n_estimators
        int: best max_depth
        float: best learning_rate
    """
    local_params = params.copy()
    n_estimators_grid = local_params.pop('n_estimators_grid', None)
    max_depth_grid = local_params.pop('max_depth_grid', None)
    learning_rate_grid = local_params.pop('learning_rate_grid', None)
    cv_folds = local_params.pop('cv', 5)

    # Base model with remaining fixed params
    base_model = get_xgb_model(local_params)

    param_grid = {}
    if n_estimators_grid is not None:
        param_grid['n_estimators'] = n_estimators_grid
    if max_depth_grid is not None:
        param_grid['max_depth'] = max_depth_grid
    if learning_rate_grid is not None:
        param_grid['learning_rate'] = learning_rate_grid

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    gs = GridSearchCV(base_model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
    gs.fit(train[features], train[target])
    best_model = gs.best_estimator_
    best_n_estimators = gs.best_params_.get('n_estimators')
    best_max_depth = gs.best_params_.get('max_depth')
    best_learning_rate = gs.best_params_.get('learning_rate')

    predictions = best_model.predict(test[features])
    test = test.copy()
    test['prediction'] = predictions
    return test, best_n_estimators, best_max_depth, best_learning_rate

def forecast_single_date_xgb(
    forecast_date,
    df,
    target_column,
    feature_columns,
    forecast_horizon,
    rolling_window_days,
    xgb_params,
    freq
):
    """
    Generate forecasts for a single date using XGBoost with optional grid search.
    """
    results = []
    forecast_start = forecast_date.normalize() + pd.Timedelta(days=1)

    train_df = data_interpolate_prev(df, forecast_date, rolling_window_days)
    test_df = data_interpolate_fut(df, forecast_start, forecast_horizon, freq=freq)

    if train_df.empty or test_df.empty:
        return results

    # Run XGBoost (with CV) on the train/test split
    pred_df, best_n_estimators, best_max_depth, best_learning_rate = run_xgboost(
        train_df, test_df, target_column, feature_columns, xgb_params
    )

    for _, row in pred_df.iterrows():
        results.append({
        'target_time': row[ModelSettings.datetime_col],
        'prediction': row['prediction'],
        'actual': row[target_column],
        'best_n_estimators': best_n_estimators,
        'best_max_depth': best_max_depth,
        'best_learning_rate': best_learning_rate
        })
    return results





"""
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
    ""
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
    ""
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
"""

def run_day_ahead_xgboost(
    df: pd.DataFrame,
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

    forecast_results = []
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                forecast_single_date_xgb,
                forecast_date,
                df,
                target_column,
                feature_columns,
                forecast_horizon,
                rolling_window_days,
                xgb_params,
                freq
            ): forecast_date for forecast_date in forecast_dates[rolling_window_days:]
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel forecasting (XGBoost)"):
            try:
                result = future.result()
                forecast_results.extend(result)
            except Exception as exc:
                print(f"Forecasting for {futures[future]} failed: {exc}")

    return pd.DataFrame(forecast_results)

