"""
Lasso regression using ElasticNet with l1_ratio = 0.0
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from combined_forecast.methods.elastic_net import run_elastic_net, run_day_ahead_elastic_net

# hallo
# hoi terug 
def run_lasso(
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        features: List[str],
        params: Dict[str, Any]
    ):
    """
    This function runs the lasso regression model.

    Args:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The testing data.
        target (str): The target column.
        features (List[str]): The feature columns.
        params (Dict[str, Any]): The parameters for the model. Must contain 'alpha', 'normalize', and 'random_state'. 'alpha' is the regularization strength, 'normalize' is whether to normalize the data, and 'random_state' is the random seed. 'alpha' defaults to 1.0, 'normalize' defaults to False, and 'random_state' defaults to 42 if not provided.

    Returns:
        pd.DataFrame: The testing data with the predictions.
    """
    # Lasso is ElasticNet with no L2 penalty
    lasso_params = {
        'alpha': params.get('alpha', 1.0),
        'l1_ratio': 1.0,  # Lasso is ElasticNet with no L2 penalty
        'random_state': params.get('random_state', 42)
    }

    # Run the ElasticNet model with l1_ratio = 1.0
    return run_elastic_net(train, test, target, features, lasso_params)




def run_day_ahead_lasso(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    forecast_horizon: int = 96,
    rolling_window_days: int = 165,
    lasso_params: Dict[str, Any] = None,
    datetime_col: str = 'datetime',
    freq: str = '15min'
) -> pd.DataFrame:
    """
    Generates 96-step day-ahead forecasts using rolling lasso regression.

    Wrapper for ElasticNet-based day-ahead rolling forecast using lasso (l1_ratio=1.0).

    Args:
        df: DataFrame with datetime index or datetime_col, and feature/target data.
        target_column: The column to forecast.
        feature_columns: List of input feature names.
        forecast_horizon: Number of 15-min intervals (default=96 for one day).
        rolling_window_days: Number of days to use for training (default=165).
        lasso_params: Dict with lasso parameters (e.g., {'alpha': 1.0}).
        datetime_col: Name of datetime column (optional).
        freq: Frequency of the data (default='15min').

    Returns:
        pd.DataFrame: Forecast results with columns ['forecast_time', 'target_time', 'prediction'].
    """
    # Check if the parameters are provided, and set defaults if not. Enforce Lasso behavior.
    lasso_params = lasso_params or {'alpha': 1.0}
    lasso_params['l1_ratio'] = 1.0  # Enforce lasso behavior

    # Run the ElasticNet model
    return run_day_ahead_elastic_net(
        df=df,
        target_column=target_column,
        feature_columns=feature_columns,
        forecast_horizon=forecast_horizon,
        rolling_window_days=rolling_window_days,
        enet_params=lasso_params,
        datetime_col=datetime_col,
        freq=freq
    )



if __name__ == '__main__':
    from combined_forecast.utils import generate_sample_data, evaluate_forecast

    df_sample = generate_sample_data()
    target_col = 'HR'
    feature_cols = [f'A{i}' for i in range(1, 8)]

    forecast_df = run_day_ahead_lasso(
        df_sample,
        target_column=target_col,
        feature_columns=feature_cols,
        rolling_window_days=5
    )

    print(forecast_df)

    if not forecast_df.empty:
        rmse = evaluate_forecast(forecast_df)
        print(f"\nRMSE on forecast: {rmse:.2f}")
    else:
        print("No forecasts generated.")


# to run:
# cd /Users/gebruiker/Documents/GitHub/SeminarML25_G12
# python -m combined_forecast.methods.lasso