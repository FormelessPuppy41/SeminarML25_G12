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
    SUMMER SPECIALIST!

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
    df['alpha'] = alpha = 1.2 - 0.4 * np.cos(2 * np.pi * (df['day_of_year'] - 172) / 365)


    # Constant delta
    df['delta'] = 0.03 * np.sin(2 * np.pi * df['day_of_year'] / 365)


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
    WINTER SPECIALIST!

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
    df['alpha'] = 0.8 + 0.4 * np.cos(2 * np.pi * (df['day_of_year'] - 172) / 365)


    # Constant delta
    df['delta'] = -0.04 * np.sin(2 * np.pi * df['day_of_year'] / 365)


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
    CLOUD ADAPTIVE!

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
    cloud_index = np.ones(len(df)) - df[ModelSettings.target].values / (df['K'].values + 1e-6)
    cloud_index = np.clip(cloud_index, -1, 1) 
    df['alpha'] = 1 + 0.5 * cloud_index
    
    # Constant delta
    df['delta'] = 0.02 + 0.05 * cloud_index

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

def run_forecast4(df: pd.DataFrame) -> pd.DataFrame:
    """
    VOLATILITY COMPENSATOR!

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

    # Rolling HR volatility (3 days ~ 288 steps of 15min)
    rolling_std = df[ModelSettings.target].rolling(window=288, min_periods=1).std().fillna(0)
    print(rolling_std)
    print(min(rolling_std), max(rolling_std))
    rolling_std = np.clip(rolling_std, 0, 2500)  # clip to avoid extreme values

    # Normalize std to keep alpha in reasonable range (assume std in [0, 0.3])
    df['alpha'] = 1.0 + 0.00025 * rolling_std  # scaling factor to exaggerate effect slightly
    
    # Constant delta
    df['delta'] = 0.01

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


def run_forecast5(df: pd.DataFrame) -> pd.DataFrame:
    """
    SYSTEMATICALLY BIASED FORECASTER!

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

    # Rolling HR volatility (3 days ~ 288 steps of 15min)
    rolling_std = df[ModelSettings.target].rolling(window=288, min_periods=1).std().fillna(0)

    # Constant alpha
    df['alpha'] = 1.0

    # Linearly drifting delta (e.g., from 0.05 to -0.05)
    df['delta'] = np.linspace(0.05, -0.05, len(df))

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

def run_forecast6(df: pd.DataFrame) -> pd.DataFrame:
    """
    ERROR-CORRECTING FORECASTER!

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

    # Compute previous day's error
    hr = df[ModelSettings.target].values
    k = df['K'].values
    prev_error = np.roll(k - hr, 1)  # yesterday's forecast error
    prev_error[0] = 0  # no previous error for first record

    # Alpha: reduce trust when previous forecast error was high
    df['alpha'] = 1.0 - 0.5 * prev_error  # could exceed 1.0 or drop below 0.5 depending on error
    df['alpha'] = np.clip(df['alpha'], 0.5, 1.5)

    # Delta: oppose the direction of previous forecast residual
    df['delta'] = -0.5 * prev_error

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
