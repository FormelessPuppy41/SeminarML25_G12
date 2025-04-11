"""
Adaptive Elastic Net Regression using the shared Elastic Net model pipeline
"""
import pandas as pd
from typing import List, Dict, Any

from combined_forecast.methods.elastic_net import run_day_ahead_elastic_net

def run_day_ahead_adaptive_elastic_net(
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
    
    return run_day_ahead_elastic_net(
        df=df,
        target_column=target_column,
        feature_columns=feature_columns,
        forecast_horizon=forecast_horizon,
        rolling_window_days=rolling_window_days,
        enet_params=enet_params,
        datetime_col=datetime_col,
        freq=freq,
        l1_grid=True
    )



if __name__ == '__main__':
    from combined_forecast.utils import generate_sample_data, evaluate_forecast

    # Generate sample 15-min interval data for 20 days
    df_sample = generate_sample_data(start='2023-01-01', days=20)
    target_col = 'HR'
    feature_cols = [f'A{i}' for i in range(1, 8)]

    # Run Adaptive Elastic Net with rolling forecast and grid search
    forecast_df = run_day_ahead_adaptive_elastic_net(
        df=df_sample,
        target_column=target_col,
        feature_columns=feature_cols,
        rolling_window_days=5,
        param_grid={
            'elasticnet__alpha': [0.1, 1.0, 10.0],
            'elasticnet__l1_ratio': [0.0, 0.4, 1.0]
        },
        grid_params={
            'cv': 5,
            'n_jobs': -1,
            'verbose': 1
        }
    )

    print(forecast_df)

    if not forecast_df.empty:
        rmse = evaluate_forecast(forecast_df)
        print(f"\n RMSE on Adaptive Elastic Net forecast: {rmse:.2f}")
    else:
        print("No forecasts generated.")
