import pandas as pd
import numpy as np
from typing import List, Dict, Any


def run_simple_average(
        test: pd.DataFrame,
        target: str, 
        features: List[str]
    ) -> pd.DataFrame:
    """
    This function runs the simple average model.

    Args:
        test (pd.DataFrame): The data to forecast.
        target (str): The target column.
        features (List[str]): The feature columns.

    Returns:
        pd.DataFrame: The testing data with predictions.
    """
    test = test.copy()
    test['prediction'] = test[features].mean(axis=1)
    test['target_time'] = test.index
    test['actual'] = test[target]

    return test[['target_time', 'actual', 'prediction']]



def run_day_ahead_simple_average(
    df: pd.DataFrame,
    target_column: str,
    forecast_horizon: int = 96,
    datetime_col: str = 'datetime',
    features: List[str] = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
    freq: str = '15min'
) -> pd.DataFrame:
    """
    Generates 96-step day-ahead forecasts using a simple average over past days.

    Args:
        df: DataFrame with datetime index or a datetime column, and target data.
        target_column: The column to forecast.
        forecast_horizon: Number of 15-min intervals in the forecast horizon (default=96 for one day).
        datetime_col: Name of datetime column (default='datetime').
        freq: Frequency of the data (default='15min').

    Returns:
        pd.DataFrame: Forecast results with columns ['forecast_time', 'target_time', 'prediction'].
    """
    if datetime_col not in df.columns:
        df = df.reset_index()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col).sort_index()

    results = []
    df.reset_index(inplace=True)
    # TODO: This should be done in the data preparation step.
    if datetime_col not in df.columns:
        print(df)
        raise ValueError(f"'{datetime_col}' not found in DataFrame.")
    
    df = df.copy()

    df["prediction"] = df[["A1", "A2", "A3", "A4", "A5", "A6"]].mean(axis=1)
    df["target_time"] = df[datetime_col]
    df["actual"] = df[target_column]
    df = df[["target_time", "actual", "prediction"]]
    return df.fillna(0)

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col).sort_index()

    # Get unique forecast dates
    unique_dates = df.index.unique()
    forecast_dates = [pd.Timestamp(d) for d in unique_dates if (pd.Timestamp(d).hour == 9 and pd.Timestamp(d).minute == 0)]
    
    for forecast_date in forecast_dates[1:]:
        print(f"Forecast date (09:00): {forecast_date}")

        # Get the data to simple_average (that is, the data of the other forecasts for the next day)
        forecast_start = forecast_date + pd.Timedelta(hours=15) # 00:00 of the next day
        forecast_end = forecast_start + pd.Timedelta(minutes=(forecast_horizon - 1) * 15) # 24 hours later
    
        test = df.loc[forecast_start:forecast_end]
        test_result = run_simple_average(
            test=test,
            target=target_column,
            features=features
        )
        
        results.append(test_result)
    df = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=['target_time', 'actual', 'prediction'])
    return df.fillna(0)

        


if __name__ == '__main__':
    from combined_forecast.utils import generate_sample_data, evaluate_forecast

    df_sample = generate_sample_data()
    target_col = 'HR'

    forecast_df = run_day_ahead_simple_average(
        df_sample,
        target_column=target_col,
        rolling_window_days=5
    )

    print(forecast_df)

    if not forecast_df.empty:
        rmse = evaluate_forecast(forecast_df)
        print(f"\nRMSE on SIMPLE AVERAGE forecast: {rmse:.2f}")
    else:
        print("No forecasts generated.")
