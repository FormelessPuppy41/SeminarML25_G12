import numpy as np
import pandas as pd

from configuration import ModelSettings


def apply_error_model(
        alpha: np.ndarray,
        k: np.ndarray,
        hr: np.ndarray,
        delta: np.ndarray,
        epsilon: np.ndarray
    ) -> np.ndarray:
    """
    Applies the base error model to produce forecasted values:
        forecast_i = HR_i + alpha_i * (K_i - HR_i) + delta_i + epsilon_i

    Where:
        - alpha_i: scaling factor (1D array)
        - K_i: raw forecast (1D array)
        - HR_i: actual solar yield (1D array)
        - delta_i: bias to be applied only when HR_i > 0
        - epsilon_i: noise added to the forecast

    All inputs must be the same shape.

    Returns:
        forecasted_value (np.ndarray): Forecasted HR values
    """
    if not (len(alpha) == len(k) == len(hr) == len(delta) == len(epsilon)):
        raise ValueError("All input arrays must have the same length.")

    # Base error component
    error = alpha * (k - hr)

    # Only add delta and noise when HR > 0 (i.e., sun is shining)
    sun_mask = hr > 0
    error[sun_mask] += delta[sun_mask] + epsilon[sun_mask]

    # Final forecast
    return hr + error


def run_forecasts1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forecasts HR using an error model based on:
        - alpha: seasonal cosine function
        - epsilon: noise from t-distribution
        - delta: constant bias added when HR > 0
    The structure and output match run_forecast2 for comparison.
    """
    df = df.copy()

    # Ensure datetime
    df[ModelSettings.datetime_col] = pd.to_datetime(df[ModelSettings.datetime_col])
    df['day_of_year'] = df[ModelSettings.datetime_col].dt.dayofyear

    # Seasonal alpha: varies with time of year
    df['alpha'] = 1 + 0.2 * np.cos(2 * np.pi * (df['day_of_year'] - 172) / 365)

    # Constant delta
    df['delta'] = 0.02

    # t-distribution noise
    df['epsilon'] = np.random.standard_t(df=3, size=len(df)) * 0.05

    # Only forecast where HR > 0
    hr = df[ModelSettings.target].values
    k = df['K'].values
    alpha = df['alpha'].values
    delta = df['delta'].values
    epsilon = df['epsilon'].values

    forecasted_value = np.full(len(df), np.nan)
    hr_nonzero_mask = hr != 0

    forecasted_value[hr_nonzero_mask] = apply_error_model(
        alpha=alpha[hr_nonzero_mask],
        k=k[hr_nonzero_mask],
        hr=hr[hr_nonzero_mask],
        delta=delta[hr_nonzero_mask],
        epsilon=epsilon[hr_nonzero_mask]
    )

    # Format output same as run_forecast2
    df['forecasted_value'] = forecasted_value
    result = df[[ModelSettings.datetime_col, 'forecasted_value']].reset_index(drop=True)
    return result.fillna(0)

def run_forecasts2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forecasts HR using an error model based on:
        - alpha: seasonal cosine function
        - epsilon: noise from t-distribution
        - delta: constant bias added when HR > 0
    The structure and output match run_forecast2 for comparison.
    """
    df = df.copy()

    # Ensure datetime
    df[ModelSettings.datetime_col] = pd.to_datetime(df[ModelSettings.datetime_col])
    df['day_of_year'] = df[ModelSettings.datetime_col].dt.dayofyear

    # Seasonal alpha: varies with time of year
    df['alpha'] = 1 + 0.2 * np.sin(2 * np.pi * (df['day_of_year'] - 172) / 365)

    # Constant delta
    df['delta'] = -0.02

    # t-distribution noise
    df['epsilon'] = np.random.standard_t(df=3, size=len(df)) * 0.05

    # Only forecast where HR > 0
    hr = df[ModelSettings.target].values
    k = df['K'].values
    alpha = df['alpha'].values
    delta = df['delta'].values
    epsilon = df['epsilon'].values

    forecasted_value = np.full(len(df), np.nan)
    hr_nonzero_mask = hr != 0

    forecasted_value[hr_nonzero_mask] = apply_error_model(
        alpha=alpha[hr_nonzero_mask],
        k=k[hr_nonzero_mask],
        hr=hr[hr_nonzero_mask],
        delta=delta[hr_nonzero_mask],
        epsilon=epsilon[hr_nonzero_mask]
    )

    # Format output same as run_forecast2
    df['forecasted_value'] = forecasted_value
    result = df[[ModelSettings.datetime_col, 'forecasted_value']].reset_index(drop=True)
    return result.fillna(0)

def run_forecast3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forecasts HR using an error model based on:
        - alpha: seasonal cosine function
        - epsilon: noise from t-distribution
        - delta: constant bias added when HR > 0
    The structure and output match run_forecast2 for comparison.
    """
    df = df.copy()

    # Ensure datetime
    df[ModelSettings.datetime_col] = pd.to_datetime(df[ModelSettings.datetime_col])
    df['day_of_year'] = df[ModelSettings.datetime_col].dt.dayofyear

    # Seasonal alpha: varies with time of year
    df['alpha'] = 1 + 0.2 * np.cos(2 * np.pi * (df['day_of_year'] - 172) / 365)

    # Constant delta
    df['delta'] = 0.02

    # t-distribution noise
    df['epsilon'] = np.random.standard_t(df=3, size=len(df)) * 0.05

    # Only forecast where HR > 0
    hr = df[ModelSettings.target].values
    k = df['K'].values
    alpha = df['alpha'].values
    delta = df['delta'].values
    epsilon = df['epsilon'].values
    forecasted_value = np.full(len(df), np.nan)
    hr_nonzero_mask = hr != 0
    forecasted_value[hr_nonzero_mask] = apply_error_model(
        alpha=alpha[hr_nonzero_mask],
        k=k[hr_nonzero_mask],
        hr=hr[hr_nonzero_mask],
        delta=delta[hr_nonzero_mask],
        epsilon=epsilon[hr_nonzero_mask]
    )
    # Format output same as run_forecast2
    df['forecasted_value'] = forecasted_value
    result = df[[ModelSettings.datetime_col, 'forecasted_value']].reset_index(drop=True)
    return result.fillna(0)

