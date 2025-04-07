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

from .utils import data_interpolate_prev, data_interpolate_fut


def get_model_from_params(params: Dict[str, Any]):
    """
    Create a pipeline with an ElasticNet, Ridge, or Lasso model based on the parameters. 

    If l1_ratio is 0.0, Ridge regression is used.
    If l1_ratio is 1.0, Lasso regression is used.
    Otherwise, ElasticNet regression is used.

    Args:
        params (Dict[str, Any]): Parameters for the model. Such as 'alpha', 'l1_ratio', and 'random_state'.

    Returns:
        Pipeline: A pipeline with the selected model.
    """
    # Get the parameters
    alpha = params.get("alpha", 1.0)
    l1_ratio = params.get("l1_ratio", 0.5)
    random_state = params.get("random_state", 42)

    # Create a pipeline with the selected model
    if l1_ratio == 0.0:
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=random_state))
    elif l1_ratio == 1.0:   
        return make_pipeline(StandardScaler(), Lasso(alpha=alpha, random_state=random_state))
    else:
        return make_pipeline(StandardScaler(), ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=random_state))


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
    # Check if grid search for alpha is requested
    if 'alpha_grid' in params:
        # Extract the alpha grid values and remove from params to avoid conflicts
        alpha_grid = params.pop('alpha_grid')
        # Use the first value as a placeholder; grid search will tune this parameter.
        base_params = {
            'alpha': alpha_grid[0],
            'l1_ratio': params.get('l1_ratio', 0.5),
            'random_state': params.get('random_state', 42)
        }
        model = get_model_from_params(base_params)
        # The name of the estimator in the pipeline (e.g., 'elasticnet', 'ridge', or 'lasso')
        step_name = model.steps[-1][0]
        # Construct the parameter grid for grid search
        param_grid = {f"{step_name}__alpha": alpha_grid}
        
        tscv = TimeSeriesSplit(n_splits=3)
        gs = GridSearchCV(model, param_grid, cv=tscv)
        gs.fit(train[features], train[target])
        
        best_model = gs.best_estimator_
        predictions = best_model.predict(test[features])
        best_alpha = gs.best_params_[f'{step_name}__alpha']
    else:
        # Use fixed alpha value if no alpha_grid is provided.
        fixed_params = {
            'alpha': params.get('alpha', 1.0),
            'l1_ratio': params.get('l1_ratio', 0.5),
            'random_state': params.get('random_state', 42)
        }
        best_alpha = fixed_params['alpha']
        model = get_model_from_params(fixed_params)
        model.fit(train[features], train[target])
        predictions = model.predict(test[features])
        
    test = test.copy()
    test['prediction'] = predictions
    return test, best_alpha

def run_day_ahead_elastic_net(
    df: pd.DataFrame,
    #flag_matrix_df: pd.DataFrame,
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
    # Check if the parameters are provided, and set defaults if not
    enet_params = enet_params or {'alpha': 1.0, 'l1_ratio': 0.5}

    # TODO: This should be done in the data preparation step.
    if datetime_col not in df.columns:
        #print(df, flag_matrix_df)
        raise ValueError(f"'{datetime_col}' not found in DataFrame.")
    
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    #df = df.set_index(datetime_col).sort_index()

    # Initialize list to store forecast results
    forecast_results = []

    # Get unique forecast dates
    unique_dates = df[datetime_col].unique()
    forecast_dates = [pd.Timestamp(d) for d in unique_dates if pd.Timestamp(d).hour == 9]


    # Iterate over each forecast date
    current_date_interpolator_prev: pd.DataFrame = None
    current_date_interpolator_fut: pd.DataFrame = None
    last_forecast_date: pd.Timestamp = None

    for forecast_date in forecast_dates[rolling_window_days:]:
        print(f"Forecast date (09:00): {forecast_date}")
        
        forecast_start = forecast_date.normalize() + pd.Timedelta(days=1)  # Next day at 00:00

        # Update the interpolators if we move to a new forecast date
        if last_forecast_date is None or forecast_date != last_forecast_date:
            last_forecast_date = forecast_date
            current_date_interpolator_prev = data_interpolate_prev(df, forecast_date, rolling_window_days)
            current_date_interpolator_fut = data_interpolate_fut(df, forecast_start, forecast_horizon, freq=freq)

        # Get the training data for the previous rolling window
        train_df = current_date_interpolator_prev
        test_df = current_date_interpolator_fut

        # Check NA:
        if train_df.isnull().values.any() or test_df.isnull().values.any():
            print("WARNING: Training or testing data contains NaN values. Specifically in the following columns:")
            if train_df.isnull().values.any():
                print("Training data NaN columns:", train_df.columns[train_df.isnull().any()].tolist())
            if test_df.isnull().values.any():
                print("Testing data NaN columns:", test_df.columns[test_df.isnull().any()].tolist())
            continue

        if train_df.empty or test_df.empty or train_df[target_column].isnull().all() or test_df[target_column].isnull().all():
            print(f"WARNING: No data for forecast date: {forecast_date}")
            continue
        
        # Run Elastic Net for each forecast time (hour) in the next day
        for ts in test_df[ModelSettings.datetime_col]:
            row_df = test_df[test_df[ModelSettings.datetime_col] == ts]
    
            if row_df.empty:
                print(f"WARNING: No test data for timestamp {ts}")
                continue
            
            pred_df, best_alpha = run_elastic_net(train_df, row_df, target_column, feature_columns, enet_params)
            forecast_results.append({
                'target_time': ts,
                'prediction': pred_df['prediction'].values[0],
                'actual': pred_df[target_column].values[0],
                'best_alpha': best_alpha
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
