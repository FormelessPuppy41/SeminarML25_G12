import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error



def generate_sample_data(start='2023-01-01', days=7) -> pd.DataFrame:
    """
    Generates sample 15-min interval data for testing.

    Returns:
        pd.DataFrame with datetime, HR (target), and A1–A7 (features).
    """
    periods = days * 96
    datetime_index = pd.date_range(start=start, periods=periods, freq='15min')

    np.random.seed(42)
    base = np.sin(np.linspace(0, 3 * np.pi, periods)) * 500 + 500

    data = {
        'datetime': datetime_index,
        'HR': base + np.random.normal(0, 30, size=periods),
    }

    for i in range(1, 8):
        data[f'A{i}'] = data['HR'] + np.random.normal(0, 50, size=periods)

    return pd.DataFrame(data)


def evaluate_forecast(df: pd.DataFrame) -> float:
    """
    Computes RMSE between actual and predicted.

    Returns:
        float: Root Mean Squared Error (RMSE).
    """
    return np.sqrt(mean_squared_error(df['actual'], df['prediction']))
