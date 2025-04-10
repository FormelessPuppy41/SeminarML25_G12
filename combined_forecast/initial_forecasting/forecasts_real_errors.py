import numpy as np
import pandas as pd

from configuration import ModelSettings


# Params for random noise generation. 
mu = -0.2
sigma = 0.3
mu2 = -0.02
sigma2 = 0.01

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

    # Base error term from K vs HR
    error = alpha * (k - hr)

    # Only apply delta and epsilon when HR > 0
    sun_mask = hr > 0
    delta_hr = delta * hr
    epsilon_hr = epsilon * hr
    error[sun_mask] += delta_hr[sun_mask] + epsilon_hr[sun_mask]

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
    df['alpha'] = 1.2 - 0.4 * np.cos(2 * np.pi * (df['day_of_year'] - 172) / 365) + np.random.normal(-0.2, 0.3, len(df))


    # Constant delta
    df['delta'] = 0.02 * np.sin(2 * np.pi * df['day_of_year'] / 365) + np.random.normal(-0.02, 0.01, len(df))


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
    df['alpha'] = 0.8 + 0.4 * np.cos(2 * np.pi * (df['day_of_year'] - 172) / 365) + np.random.normal(-0.15, 0.25, len(df))


    # Constant delta
    df['delta'] = -0.03 * np.sin(2 * np.pi * df['day_of_year'] / 365) + np.random.normal(-0.04, 0.05, len(df))


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
    df['alpha'] = 1 + 0.5 * cloud_index + np.random.normal(-0.25, 0.35, len(df))
    
    # Constant delta
    df['delta'] = 0.01 + 0.05 * cloud_index + np.random.normal(0.02, 0.03, len(df))

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
    rolling_std = np.clip(rolling_std, 0, 2500)  # clip to avoid extreme values

    # Normalize std to keep alpha in reasonable range (assume std in [0, 0.3])
    df['alpha'] = 1.0 + 0.00010 * rolling_std + np.random.normal(-0.1, 0.2, len(df)) # scaling factor to exaggerate effect slightly
    
    # Constant delta
    df['delta'] = 0.00 + np.random.normal(-0.025, 0.02, len(df))

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
    df['alpha'] = 1.0 + np.random.normal(-0.05, 0.25, len(df))

    # Linearly drifting delta (e.g., from 0.05 to -0.05)
    df['delta'] = np.linspace(0.05, -0.05, len(df)) + np.random.normal(-0.02, 0.01, len(df))

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

    # Alpha: reduce trust when previous forecast error was high
    df['alpha'] = 1.0  + np.random.normal(-0.1, 0.3, len(df)) # could exceed 1.0 or drop below 0.5 depending on error
    
    # Delta: oppose the direction of previous forecast residual
    df['delta'] = -0.04  + np.random.normal(-0.01, 0.03, len(df))
    
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

def run_forecast7(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multiplicative Error Model for Solar Energy Forecasting.
    
    Forecast is computed as:
      forecast = HR * (1 + beta * ((K - HR) / (HR + eta))) + delta + epsilon
    
    Parameters:
      beta : scaling factor for the relative error (e.g., 0.5)
      eta  : small constant to avoid division by zero (e.g., 1e-6)
      delta: bias term, here based on a seasonal sine function derived from day-of-year
      epsilon: noise sampled from a t-distribution
      
    Only applied where HR > 0.
    """
    import numpy as np
    import pandas as pd

    df = df.copy()
    # Ensure datetime column is in datetime format
    df[ModelSettings.datetime_col] = pd.to_datetime(df[ModelSettings.datetime_col])
    # Derive day of year for seasonal bias (allowed since it is derived from the datetime)
    df['day_of_year'] = df[ModelSettings.datetime_col].dt.dayofyear

    # Set parameters
    beta = 0.5
    eta = 1e-6
    # Example seasonal bias: can be justified by known seasonal effects on production
    df['delta'] = 0.03 * np.sin(2 * np.pi * df['day_of_year'] / 365)
    # Noise from a t-distribution (with df=3) to capture heavier tails in error behavior
    df['epsilon'] = np.random.standard_t(df=3, size=len(df)) * 0.05

    hr = df[ModelSettings.target].values
    k = df['K'].values
    forecasted_value = np.full(len(df), np.nan)
    sun_mask = hr > 0

    # Compute relative error adjustment only when HR > 0
    relative_adjustment = np.zeros_like(hr)
    relative_adjustment[sun_mask] = beta * ((k[sun_mask] - hr[sun_mask]) / (hr[sun_mask] + eta))
    forecasted_value[sun_mask] = (
        hr[sun_mask] * (1 + relative_adjustment[sun_mask])
        + df['delta'].values[sun_mask]
        + df['epsilon'].values[sun_mask]
    )

    df['forecasted_value'] = forecasted_value
    result = df[[ModelSettings.datetime_col, 'forecasted_value']].reset_index(drop=True)
    return result.fillna(0)

def run_forecast8(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nonlinear Adjustment Model for Solar Energy Forecasting.
    
    Forecast is computed as:
      forecast = HR + alpha * sign(K - HR) * |K - HR|^gamma + delta + epsilon
    
    Parameters:
      alpha : scaling factor (set to 1.0 as a starting point)
      gamma : nonlinearity exponent (e.g., 0.8 to dampen large differences)
      delta : seasonal bias (here defined using a cosine function)
      epsilon: t-distributed noise
      
    Only applied when HR > 0.
    """
    import numpy as np
    import pandas as pd

    df = df.copy()
    df[ModelSettings.datetime_col] = pd.to_datetime(df[ModelSettings.datetime_col])
    df['day_of_year'] = df[ModelSettings.datetime_col].dt.dayofyear

    alpha = 1.0
    gamma = 0.8  # For example, damping differences for larger errors.
    df['delta'] = -0.02 * np.cos(2 * np.pi * (df['day_of_year'] - 1) / 365)
    df['epsilon'] = np.random.standard_t(df=3, size=len(df)) * 0.05

    hr = df[ModelSettings.target].values
    k = df['K'].values
    diff = k - hr
    adjustment = alpha * np.sign(diff) * np.abs(diff) ** gamma

    forecasted_value = np.full(len(df), np.nan)
    sun_mask = hr > 0
    forecasted_value[sun_mask] = hr[sun_mask] + adjustment[sun_mask] + df['delta'].values[sun_mask] + df['epsilon'].values[sun_mask]

    df['forecasted_value'] = forecasted_value
    result = df[[ModelSettings.datetime_col, 'forecasted_value']].reset_index(drop=True)
    return result.fillna(0)

def run_forecast9(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regime-Switching Model for Solar Energy Forecasting.
    
    The regime is determined by the ratio of HR to K. For instance, if HR/K >= 0.8,
    we assume clear-sky conditions (low adjustment), whereas if HR/K < 0.8, cloudy conditions 
    are assumed (requiring greater adjustment). The forecast is computed as:
      forecast = HR + alpha * (K - HR) + delta + epsilon
    with alpha and delta chosen per regime.
    Only applied when HR > 0.
    """
    import numpy as np
    import pandas as pd

    df = df.copy()
    df[ModelSettings.datetime_col] = pd.to_datetime(df[ModelSettings.datetime_col])
    
    # Define a small constant to prevent division by zero
    eta = 1e-6
    hr = df[ModelSettings.target].values
    k = df['K'].values
    ratio = np.where(k != 0, hr / (k + eta), 1.0)
    
    # Determine regime based on the ratio HR/K
    # Regime 1 (clear-sky): ratio >= 0.8; Regime 2 (cloudy): ratio < 0.8.
    clear_regime = ratio >= 0.8
    
    # For clear-sky, use a lower adjustment factor and small positive bias.
    # For cloudy conditions, use a higher adjustment factor and a negative bias.
    alpha = np.where(clear_regime, 0.9, 1.1)
    delta = np.where(clear_regime, 0.01, -0.02)
    
    # Noise term: still draw from a t-distribution
    epsilon = np.random.standard_t(df=3, size=len(df)) * 0.05

    forecasted_value = np.full(len(df), np.nan)
    sun_mask = hr > 0
    forecasted_value[sun_mask] = (
        hr[sun_mask]
        + alpha[sun_mask] * (k[sun_mask] - hr[sun_mask])
        + delta[sun_mask]
        + epsilon[sun_mask]
    )
    
    df['forecasted_value'] = forecasted_value
    result = df[[ModelSettings.datetime_col, 'forecasted_value']].reset_index(drop=True)
    return result.fillna(0)
