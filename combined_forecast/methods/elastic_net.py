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

    Args:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The testing data.
        target (str): The target column name.
        features (List[str]): List of feature column names.
        params (Dict[str, Any]): Parameters for ElasticNet.

    Returns:
        pd.DataFrame: Test set with predictions added in 'prediction' column.
    """
    params = {
        'alpha': params.get('alpha', 1.0),
        'l1_ratio': params.get('l1_ratio', 0.5),
        'random_state': params.get('random_state', 42)
    }

    model = get_model_from_params(params)
    model.fit(train[features], train[target])

    test = test.copy()
    test['prediction'] = model.predict(test[features])
    return test


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
    Generate 96-step day-ahead forecasts using rolling Elastic Net regression.

    Args:
        df: Full input data with datetime, target and features.
        target_column: Column name of target variable (e.g., 'HR').
        feature_columns: List of forecast provider feature columns.
        forecast_horizon: Number of 15-min intervals to predict (default=96).
        rolling_window_days: Number of days in rolling training window (default=165).
        enet_params: Elastic Net parameters (e.g. {'alpha': 1.0, 'l1_ratio': 0.5}).
        datetime_col: Name of datetime column in df.
        freq: Sampling frequency (default='15min').

    Returns:
        pd.DataFrame with columns ['forecast_time', 'target_time', 'prediction', 'actual'].
    """
    # Check if the parameters are provided, and set defaults if not
    enet_params = enet_params or {'alpha': 1.0, 'l1_ratio': 0.5}

    # TODO: This should be done in the data preparation step.
    if datetime_col not in df.columns:
        raise ValueError(f"'{datetime_col}' not found in DataFrame.")

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col).sort_index()

    # Initialize list to store forecast results
    forecast_results = []

    # Get unique forecast dates
    unique_dates = df.index.normalize().unique()

    # Iterate over each forecast date
    for forecast_date in unique_dates[rolling_window_days:]:
        # Define training and testing periods
        # - Training: rolling window days before forecast date with forecast horizon length (e.g., 165 days before 2023-01-01 to 2023-06-15)
        # - Testing: forecast date with forecast horizon length of one day (e.g., 96 interfalls for each 15min period in the forecast day)
        train_end = forecast_date - pd.Timedelta(freq)
        train_start = train_end - pd.Timedelta(days=rolling_window_days)
        test_start = forecast_date
        test_end = forecast_date + pd.Timedelta(minutes=(forecast_horizon - 1) * 15)

        # Get training and testing data
        train_df = df[train_start:train_end]
        test_df = df[test_start:test_end]

        # Skip, and log if no data
        if train_df.empty or test_df.empty:
            print(f"WARNING: No data for forecast date: {forecast_date}")
            continue
        
        # Run Elastic Net model and store results for each target time
        for ts in test_df.index:
            row_df = test_df.loc[[ts]]
            pred_df = run_elastic_net(train_df, row_df, target_column, feature_columns, enet_params)

            # Append forecast results
            forecast_results.append({
                'target_time': ts,
                'prediction': pred_df['prediction'].values[0],
                'actual': pred_df[target_column].values[0]
            })

    # Return forecast results as DataFrame
    return pd.DataFrame(forecast_results)


if __name__ == '__main__':
    from ..utils import evaluate_forecast, generate_sample_data
    
    df_sample = generate_sample_data(start='2023-01-01', days=20)
    target_col = 'HR'
    feature_cols = [f'A{i}' for i in range(1, 8)]

    forecast_df = run_day_ahead_elastic_net(
        df_sample,
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
