"""
Elastic Net regression model
"""
# imports:
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from tqdm import tqdm  
from concurrent.futures import ProcessPoolExecutor, as_completed

from configuration import ModelParameters

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
    params: Dict[str, Any]
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
    final_model = model.named_steps[list(model.named_steps)[-1]]
    beta_init = final_model.coef_


    # 2. Compute adaptive weights (small constant added to avoid division by zero)
    epsilon = 1e-6
    weights = 1.0 / (np.abs(beta_init)**gamma) # + epsilon)
    
    # 3. Scale the features for the L1 penalty; here, we create new DataFrames
    train_scaled = train.copy()
    test_scaled = test.copy()
    for i, col in enumerate(features):
        train_scaled[col] = train_scaled[col] / weights[i]
        test_scaled[col] = test_scaled[col] / weights[i]
    
    # 4. Run the standard elastic net on the scaled data.
    # (Reuse your existing run_elastic_net function)
    pred_df, best_alpha, best_l1_ratio, coefs_scaled = run_elastic_net(
        train_scaled, test_scaled, target, features, params
    )
    
    # 5. Adjust the coefficients back to their original scale.
    coefs = coefs_scaled / weights
    
    return pred_df, best_alpha, best_l1_ratio, coefs


def forecast_single_date(
        forecast_date: pd.Timestamp,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        forecast_horizon: int,
        rolling_window_days: int,
        enet_params: dict,
        datetime_col: str,
        freq: str
    ):
    return forecast_single_date_generic(
        forecast_date,
        df,
        target_column,
        feature_columns,
        forecast_horizon,
        rolling_window_days,
        enet_params,
        datetime_col,
        freq,
        run_elastic_net
    )

def forecast_single_date_adaptive(
        forecast_date: pd.Timestamp,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: List[str],
        forecast_horizon: int,
        rolling_window_days: int,
        enet_params: dict,
        datetime_col: str,
        freq: str
    ):
    return forecast_single_date_generic(
        forecast_date,
        df,
        target_column,
        feature_columns,
        forecast_horizon,
        rolling_window_days,
        enet_params,
        datetime_col,
        freq,
        run_elastic_net_adaptive
    )

def forecast_single_date_generic(
    forecast_date: pd.Timestamp,
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    forecast_horizon: int,
    rolling_window_days: int,
    enet_params: dict,
    datetime_col: str,
    freq: str,
    model_runner: Callable
):
    """
    Generic function for forecasting a single date using a model runner.

    Args:
        model_runner: Function to train and apply the model (e.g., run_elastic_net).

    Returns:
        List of dictionaries with forecast results.
    """
    forecast_results_single = []
    forecast_start = forecast_date.normalize() + pd.Timedelta(days=1)

    train_df = data_interpolate_prev(df, forecast_date, rolling_window_days)
    test_df = data_interpolate_fut(df, forecast_start, forecast_horizon, freq=freq)

    if train_df.empty or test_df.empty:
        return forecast_results_single
    
    pred_df, best_alpha, best_l1_ratio, coefs = model_runner(
        train_df, test_df, target_column, feature_columns, enet_params
    )

    for _, row in pred_df.iterrows():
        forecast_results_single.append({
            'target_time': row[datetime_col],  # use passed-in datetime_col, not ModelSettings
            'prediction': row['prediction'],
            'actual': row[target_column],
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
        freq: str = '15min'
    ) -> pd.DataFrame:
    return run_day_ahead_forecasting(
        df,
        target_column,
        feature_columns,
        forecast_single_date,
        forecast_horizon,
        rolling_window_days,
        enet_params,
        datetime_col,
        freq
    )

def run_day_ahead_elastic_net_adaptive(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    forecast_horizon: int = 96,
    rolling_window_days: int = 165,
    enet_params: Dict[str, Any] = None,
    datetime_col: str = 'datetime',
    freq: str = '15min'
) -> pd.DataFrame:
    return run_day_ahead_forecasting(
        df,
        target_column,
        feature_columns,
        forecast_single_date_adaptive,
        forecast_horizon,
        rolling_window_days,
        enet_params,
        datetime_col,
        freq
    )

def run_day_ahead_forecasting(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    forecast_func: Callable,
    forecast_horizon: int = 96,
    rolling_window_days: int = 165,
    enet_params: Dict[str, Any] = None,
    datetime_col: str = 'datetime',
    freq: str = '15min',
    desc: str = "Forecasting"
) -> pd.DataFrame:
    if not enet_params:
        raise ValueError("Elastic Net parameters must be provided.")
    if datetime_col not in df.columns:
        raise ValueError(f"'{datetime_col}' not found in DataFrame.")

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    forecast_results = []
    forecast_dates = [
        d for d in df[datetime_col].unique()
        if pd.Timestamp(d).hour == 9 and pd.Timestamp(d).minute == 0
    ]
    
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                forecast_func,
                pd.Timestamp(forecast_date),
                df,
                target_column,
                feature_columns,
                forecast_horizon,
                rolling_window_days,
                enet_params,
                datetime_col,
                freq
            ): forecast_date for forecast_date in forecast_dates[rolling_window_days:]
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                result = future.result()
                forecast_results.extend(result)
            except Exception as exc:
                print(f"Forecasting failed for {futures[future]}: {exc}")
    
    print("Forecasting complete.")
    return pd.DataFrame(forecast_results)

