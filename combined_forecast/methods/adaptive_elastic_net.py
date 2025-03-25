"""
Adaptive Elastic Net Regression using the shared Elastic Net model pipeline
"""
import pandas as pd
from typing import List, Dict, Any
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from combined_forecast.methods.elastic_net import get_model_from_params

def run_day_ahead_adaptive_elastic_net(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    forecast_horizon: int = 96,
    rolling_window_days: int = 165,
    param_grid: Dict[str, List[Any]] = None,
    datetime_col: str = 'datetime',
    freq: str = '15min'
) -> pd.DataFrame:
    """
    Generate 96-step day-ahead forecasts using adaptive Elastic Net with rolling window & GridSearchCV.

    Args:
        - df (pd.Dataframe): DataFrame with datetime index or datetime_col, and feature/target data.
        - target_column (str): The column to forecast.
        - feature_columns (List[str]): List of feature columns.
        - forecast_horizon (int): Number of time steps to forecast. Default is 96 (24 hours).
        - rolling_window_days (int): Number of days to use for training. Default is 165 (6 months).
        - param_grid (Dict[str, List[Any]]): Grid of hyperparameters for GridSearchCV. Default is None.
        - datetime_col (str): Name of the datetime column. Default is 'datetime'.
        - freq (str): Frequency of the data. Default is '15min'.

    Returns:
        pd.DataFrame with columns ['target_time', 'prediction', 'actual'].
    """
    #TODO: This should be done in the data preprocessing.
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col).sort_index()

    forecast_results = []
    unique_dates = df.index.normalize().unique()

    for forecast_date in unique_dates[rolling_window_days:]:
        train_end = forecast_date - pd.Timedelta(freq)
        train_start = train_end - pd.Timedelta(days=rolling_window_days)
        test_start = forecast_date
        test_end = forecast_date + pd.Timedelta(minutes=(forecast_horizon - 1) * 15)

        train_df = df[train_start:train_end]
        test_df = df[test_start:test_end]

        if train_df.empty or test_df.empty:
            print(f"WARNING: No data for forecast date: {forecast_date}")
            continue

        for ts in test_df.index:
            row_df = test_df.loc[[ts]]

            # Get model pipeline
            base_params = {'alpha': 1.0, 'l1_ratio': 0.5, 'random_state': 42}
            pipeline = get_model_from_params(base_params)
            model_name = pipeline.steps[-1][0]
            pipeline.steps[-1] = (model_name, pipeline.steps[-1][1])

            # Tune ElasticNet
            grid = GridSearchCV(pipeline, param_grid=param_grid, **grid_params)
            try:
                grid.fit(train_df[feature_columns], train_df[target_column])
                prediction = grid.predict(row_df[feature_columns])[0]
                actual = row_df[target_column].values[0]
            except Exception as e:
                print(f"Skipping {ts} due to training error: {e}")
                continue

            forecast_results.append({
                'target_time': ts,
                'prediction': prediction,
                'actual': actual,
                'best_params': grid.best_params_  # Uncomment if you want to log these
            })

    return pd.DataFrame(forecast_results)

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
            'elasticnet__l1_ratio': [0.0, 0.5, 1.0]
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
