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


from configuration import ModelSettings, ModelParameters

from .utils import data_interpolate_prev, data_interpolate_fut, get_model_from_params

from tqdm import tqdm  
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed



def run_elastic_net(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: List[str],
    params: Dict[str, Any],
    l1_grid: bool = False
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
    local_params = params.copy()

    # Pop the grid values
    alpha_grid = local_params.pop('alpha_grid')
    l1_ratio_grid = local_params.pop('l1_ratio_grid')
    
    # Build the model from the provided parameters (this determines model type)
    model = get_model_from_params(params)
    step_name = model.steps[-1][0]

    # Create the grid - add l1_ratio only if the estimator supports it.
    param_grid = {f"{step_name}__alpha": alpha_grid}
    if "l1_ratio" in model.named_steps[step_name].get_params():
        param_grid[f"{step_name}__l1_ratio"] = l1_ratio_grid

    tscv = TimeSeriesSplit(n_splits=5)
    gs = GridSearchCV(model, param_grid, cv=tscv)
    gs.fit(train[features], train[target])

    best_model = gs.best_estimator_
    predictions = best_model.predict(test[features])
    best_alpha = gs.best_params_[f'{step_name}__alpha']
    best_l1_ratio = gs.best_params_.get(f'{step_name}__l1_ratio')

    coefs = best_model.named_steps[step_name].coef_
    test = test.copy()
    test['prediction'] = predictions
    return test, best_alpha, best_l1_ratio, coefs

def run_elastic_net_adaptive(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: List[str],
    params: Dict[str, Any],
    l1_grid: bool = False
):
    """
    Run an adaptive elastic net regression using initial coefficient estimates
    to compute adaptive weights.
    
    This function computes initial estimates with a Ridge regression,
    calculates weights as: weights_j = 1/ (|β_j^{init}|**gamma + epsilon)
    (with a small epsilon to avoid division by zero), scales the features by these
    weights for the L1 part, and then runs the standard elastic net on the scaled data.
    The output coefficients are transformed back to the original scale.
    
    Args:
        train: Training DataFrame.
        test: Testing DataFrame.
        target: Name of the target column.
        features: List of feature column names.
        params: Dictionary of parameters; must include "adaptive": True to trigger this branch,
                and "gamma": <value> (default can be 1.0) for the weight computation.
        l1_grid: Whether to perform grid search on L1 penalty parameters.
    
    Returns:
        Tuple: (predicted test DataFrame, best_alpha, best_l1_ratio, coefficients)
    """
    # Set gamma (or default to 1.0)
    gamma = params.get("gamma", 1.0)
    # Of gridsearch?
    
    # 1. Compute initial estimates using a Ridge regressor
    model = get_model_from_params(ModelParameters.ridge_params)
    model.fit(train[features], train[target])
    beta_init = model.coef_
    
    # 2. Compute adaptive weights (small constant added to avoid division by zero)
    epsilon = 1e-6
    weights = 1.0 / (np.abs(beta_init)**gamma + epsilon)
    
    # 3. Scale the features for the L1 penalty; here, we create new DataFrames
    train_scaled = train.copy()
    test_scaled = test.copy()
    for i, col in enumerate(features):
        train_scaled[col] = train_scaled[col] / weights[i]
        test_scaled[col] = test_scaled[col] / weights[i]
    
    # 4. Run the standard elastic net on the scaled data.
    # (Reuse your existing run_elastic_net function)
    pred_df, best_alpha, best_l1_ratio, coefs_scaled = run_elastic_net(
        train_scaled, test_scaled, target, features, params, l1_grid
    )
    
    # 5. Adjust the coefficients back to their original scale.
    coefs = coefs_scaled / weights
    
    return pred_df, best_alpha, best_l1_ratio, coefs


# def run_day_ahead_elastic_net(
#         df: pd.DataFrame, 
#         target_column: str, 
#         feature_columns: List[str], 
#         forecast_horizon: int = 96, 
#         rolling_window_days: int = 165, 
#         enet_params: Dict[str, Any] = None, 
#         datetime_col: str = 'datetime', 
#         freq: str = '15min'
#     ) -> pd.DataFrame:
#     """
#     Generate forecasts at 09:00 AM each morning for the next day (24 hourly forecasts).
    
#     Args:
#         df: Full input data with datetime, target and features.
#         #flag_matrix_df: DataFrame indicating which time/forecast combination is complete (0) or not.
#         target_column: Column name of the target variable (e.g., 'HR').
#         feature_columns: List of forecast provider feature columns.
#         forecast_horizon: Number of forecast periods (default=24 for hourly forecasts).
#         rolling_window_days: Number of days for the training window.
#         enet_params: Elastic Net parameters (e.g. {'alpha': 1.0, 'l1_ratio': 0.5}).
#         datetime_col: Name of the datetime column.
#         freq: Sampling frequency (default='1H' for hourly).
        
#     Returns:
#         pd.DataFrame with columns ['target_time', 'prediction', 'actual'].
#     """
#     if not enet_params:
#         raise ValueError("Elastic Net parameters must be provided.")
    
#     if datetime_col not in df.columns:
#         raise ValueError(f"'{datetime_col}' not found in DataFrame.")

#     df = df.copy()
#     df[datetime_col] = pd.to_datetime(df[datetime_col])
#     forecast_results = []
#     unique_dates = df[datetime_col].unique()
#     forecast_dates = [pd.Timestamp(d) for d in unique_dates if (pd.Timestamp(d).hour == 9 and pd.Timestamp(d).minute == 0)]

#     current_date_interpolator_prev = None
#     current_date_interpolator_fut = None
#     last_forecast_date = None

#     for forecast_date in tqdm(forecast_dates[rolling_window_days:], desc="Forecasting dates"):
#         print(f"Forecasting for date: {forecast_date}")
#         forecast_start = forecast_date.normalize() + pd.Timedelta(days=1)
#         if last_forecast_date is None or forecast_date != last_forecast_date:
#             last_forecast_date = forecast_date
#             current_date_interpolator_prev = data_interpolate_prev(df, forecast_date, rolling_window_days)
#             current_date_interpolator_fut = data_interpolate_fut(df, forecast_start, forecast_horizon, freq=freq)

#         train_df = current_date_interpolator_prev
#         test_df = current_date_interpolator_fut

#         if train_df.empty or test_df.empty:
#             continue

#         for ts in test_df[ModelSettings.datetime_col]:
#             row_df = test_df[test_df[ModelSettings.datetime_col] == ts]
#             if row_df.empty:
#                 continue
#             pred_df, best_alpha, best_l1_ratio, coefs = run_elastic_net(train_df, row_df, target_column, feature_columns, enet_params)
#             forecast_results.append({
#                 'target_time': ts,
#                 'prediction': pred_df['prediction'].values[0],
#                 'actual': pred_df[target_column].values[0],
#                 'best_alpha': best_alpha,
#                 'best_l1_ratio': best_l1_ratio,
#                 'coefs': coefs
#             })

#     return pd.DataFrame(forecast_results)



def forecast_single_date(
    forecast_date: pd.Timestamp,
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    forecast_horizon: int,
    rolling_window_days: int,
    enet_params: dict,
    datetime_col: str,
    freq: str,
    l1_grid: bool = False
):
    """
    Processes the forecast for a single date.
    Computes training and testing data based on the forecast date,
    then loops over the test rows to generate predictions.
    """
    forecast_results_single = []
    # Calculate the forecast start date (e.g., next day)
    forecast_start = forecast_date.normalize() + pd.Timedelta(days=1)
    
    # Interpolate training and test data for this forecast date
    train_df = data_interpolate_prev(df, forecast_date, rolling_window_days)
    test_df = data_interpolate_fut(df, forecast_start, forecast_horizon, freq=freq)
    
    # Skip if there’s no data
    if train_df.empty or test_df.empty:
        return forecast_results_single

    # Loop over each test time in the test data using the ModelSettings datetime column
    for ts in test_df[ModelSettings.datetime_col]:
        row_df = test_df[test_df[ModelSettings.datetime_col] == ts]
        if row_df.empty:
            continue
        # Call run_elastic_net (which already handles grid search / fixed params)
        pred_df, best_alpha, best_l1_ratio, coefs = run_elastic_net(
            train_df, row_df, target_column, feature_columns, enet_params, l1_grid
        )
        forecast_results_single.append({
            'target_time': ts,
            'prediction': pred_df['prediction'].values[0],
            'actual': pred_df[target_column].values[0],
            'best_alpha': best_alpha,
            'best_l1_ratio': best_l1_ratio,
            'coefs': coefs
        })
    return forecast_results_single


def run_day_ahead_elastic_net(
        df: pd.DataFrame, 
        target_column: str, 
        feature_columns: List[str], 
        forecast_horizon: int = 96, 
        rolling_window_days: int = 165, 
        enet_params: Dict[str, Any] = None, 
        datetime_col: str = 'datetime', 
        freq: str = '15min',
        l1_grid: bool = False
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
    # Select forecast dates based on your criteria (here, dates with hour==9 and minute==0)
    forecast_dates = [
        pd.Timestamp(d) for d in unique_dates 
        if (pd.Timestamp(d).hour == 9 and pd.Timestamp(d).minute == 0)
    ]
    
    # Submit jobs for all forecast dates beyond the rolling window period
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                forecast_single_date,
                forecast_date,
                df,
                target_column,
                feature_columns,
                forecast_horizon,
                rolling_window_days,
                enet_params,
                datetime_col,
                freq,
                l1_grid
            ): forecast_date for forecast_date in forecast_dates[rolling_window_days:]
        }
        
        # Optionally, use tqdm to track progress as futures complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel forecasting"):
            try:
                result = future.result()  # each result is a list of forecasts for that date
                forecast_results.extend(result)
            except Exception as exc:
                print(f"Forecasting for date {futures[future]} raised an exception: {exc}")
        print('wauwzers, we are done!')
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
