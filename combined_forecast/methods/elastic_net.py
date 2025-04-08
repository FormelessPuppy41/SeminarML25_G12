"""
Elastic Net regression model
"""
# imports:
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


from configuration import ModelSettings

from .utils import data_interpolate_prev, data_interpolate_fut, get_model_from_params


def run_elastic_net(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: List[str],
    params: Dict[str, Any]
):
    """
    Run an Elastic Net regression model on test data using training data.
    If an "alpha_grid" is provided in params, grid search is performed over the specified
    list of alpha values to select the best model.
    
    Args:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The testing data.
        target (str): The target column name.
        features (List[str]): List of feature column names.
        params (Dict[str, Any]): Parameters for the model. It can include 'alpha', 'l1_ratio',
                                 'random_state', and optionally 'alpha_grid' (a list of alphas).
    
    Returns:
        pd.DataFrame: Test set with predictions added in 'prediction' column.
    """
    if not params:
        raise ValueError("Elastic Net parameters must be provided.")
    # Check if grid search for alpha is requested
    if 'alpha_grid' in params:
        alpha_grid = params.pop('alpha_grid')
        l1_ratio_grid = params.pop('l1_ratio_grid', [params.get('l1_ratio', 0.5)])
        
        base_params = {
            'alpha': alpha_grid[0],
            'l1_ratio': l1_ratio_grid[0],
            'random_state': params.get('random_state', 42)
        }
        model = get_model_from_params(base_params)
        step_name = model.steps[-1][0]
        param_grid = {
            f"{step_name}__alpha": alpha_grid,
            f"{step_name}__l1_ratio": l1_ratio_grid
        }

        tscv = TimeSeriesSplit(n_splits=3)
        gs = GridSearchCV(model, param_grid, cv=tscv)
        gs.fit(train[features], train[target])

        best_model = gs.best_estimator_
        predictions = best_model.predict(test[features])
        best_alpha = gs.best_params_[f'{step_name}__alpha']
        best_l1_ratio = gs.best_params_[f'{step_name}__l1_ratio']
    else:
        fixed_params = {
            'alpha': params.get('alpha', 1.0),
            'l1_ratio': params.get('l1_ratio', 0.5),
            'random_state': params.get('random_state', 42)
        }
        best_alpha = fixed_params['alpha']
        best_l1_ratio = fixed_params['l1_ratio']
        model = get_model_from_params(fixed_params)
        model.fit(train[features], train[target])
        predictions = model.predict(test[features])

    test = test.copy()
    test['prediction'] = predictions
    return test, best_alpha, best_l1_ratio

def run_day_ahead_elastic_net(
        df: pd.DataFrame, 
        target_column: str, 
        feature_columns: List[str], 
        forecast_horizon: int = 96, 
        rolling_window_days: int = 165, 
        enet_params: Dict[str, Any] = None, 
        datetime_col: str = 'datetime', 
        freq: str = '15min'
    ) -> pd.DataFrame:
    """
    Generate forecasts at 09:00 AM each morning for the next day (24 hourly forecasts).
    
    Args:
        df: Full input data with datetime, target and features.
        #flag_matrix_df: DataFrame indicating which time/forecast combination is complete (0) or not.
        target_column: Column name of the target variable (e.g., 'HR').
        feature_columns: List of forecast provider feature columns.
        forecast_horizon: Number of forecast periods (default=24 for hourly forecasts).
        rolling_window_days: Number of days for the training window.
        enet_params: Elastic Net parameters (e.g. {'alpha': 1.0, 'l1_ratio': 0.5}).
        datetime_col: Name of the datetime column.
        freq: Sampling frequency (default='1H' for hourly).
        
    Returns:
        pd.DataFrame with columns ['target_time', 'prediction', 'actual'].
    """
    if not enet_params:
        raise ValueError("Elastic Net parameters must be provided.")
    
    if datetime_col not in df.columns:
        raise ValueError(f"'{datetime_col}' not found in DataFrame.")

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    forecast_results = []
    unique_dates = df[datetime_col].unique()
    forecast_dates = [pd.Timestamp(d) for d in unique_dates if (pd.Timestamp(d).hour == 9 and pd.Timestamp(d).minute == 0)]

    current_date_interpolator_prev = None
    current_date_interpolator_fut = None
    last_forecast_date = None

    for forecast_date in forecast_dates[rolling_window_days:]:
        print(f"Forecasting for date: {forecast_date}")
        forecast_start = forecast_date.normalize() + pd.Timedelta(days=1)
        if last_forecast_date is None or forecast_date != last_forecast_date:
            last_forecast_date = forecast_date
            current_date_interpolator_prev = data_interpolate_prev(df, forecast_date, rolling_window_days)
            current_date_interpolator_fut = data_interpolate_fut(df, forecast_start, forecast_horizon, freq=freq)

        train_df = current_date_interpolator_prev
        test_df = current_date_interpolator_fut

        if train_df.empty or test_df.empty:
            continue

        for ts in test_df[ModelSettings.datetime_col]:
            row_df = test_df[test_df[ModelSettings.datetime_col] == ts]
            if row_df.empty:
                continue
            pred_df, best_alpha, best_l1_ratio = run_elastic_net(train_df, row_df, target_column, feature_columns, enet_params)
            forecast_results.append({
                'target_time': ts,
                'prediction': pred_df['prediction'].values[0],
                'actual': pred_df[target_column].values[0],
                'best_alpha': best_alpha,
                'best_l1_ratio': best_l1_ratio
            })

    return pd.DataFrame(forecast_results)

if __name__ == '__main__':
    from ..utils import evaluate_forecast, generate_sample_data
    
    #df_sample = generate_sample_data(start='2023-01-01', days=20)
    df_sample = pd.read_csv('/Users/gebruiker/Documents/GitHub/SeminarML25_G12/data/kaggle_data/combined_forecasts.csv')
    target_col = 'HR'
    feature_cols = [f'A{i}' for i in range(1, 5)]

    # Load the flag matrix
    flag_matrix = pd.read_csv('/Users/gebruiker/Documents/GitHub/SeminarML25_G12/data/kaggle_data/flag_matrix.csv')

    forecast_df = run_day_ahead_elastic_net(
        df_sample,
        flag_matrix, 
        target_column=target_col,
        feature_columns=feature_cols,
        rolling_window_days=5,
        enet_params={'alpha': 1.0, 'l1_ratio': 0.5}
    )

    print(forecast_df)

    if not forecast_df.empty:
        rmse = evaluate_forecast(forecast_df)
        print(f"\nRMSE on ElNET forecast: {rmse:.2f}")
    else:
        print("No forecasts generated.")

# to run:
# cd /Users/gebruiker/Documents/GitHub/SeminarML25_G12
# python -m combined_forecast.methods.elastic_net
