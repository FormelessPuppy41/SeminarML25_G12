import numpy as np
import pandas as pd

from configuration import ModelSettings


# Params for random noise generation. 
mu = -0.1
sigma = 1.0
mu2 = 0
sigma2 = 1.0
fourier_d = 100
fourier_d_sigma = 10
fourier_gamma = 20000.0
fourier_gamma_sigma = 20000.0

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
    
    k_hr = k - hr
    # Base error term from K vs HR
    error = alpha * (k_hr)

    # Only apply delta and epsilon when HR > 0
    sun_mask = hr > 0
    delta_hr = delta * k_hr 
    epsilon_hr = epsilon * k_hr
    error[sun_mask] += delta_hr[sun_mask] + epsilon_hr[sun_mask]
    
    return hr + error

def get_random_fourier_features(day_of_week: pd.Series, D: int = fourier_d, gamma: float = fourier_gamma) -> np.ndarray:
    """
    Generates and returns random Fourier features for a given one-dimensional input vector x.

    Parameters:
        x_norm (pd.Series): Input vector (day of the week) to transform.
        D (int): Number of random Fourier features (dimensions) to generate.
        gamma (float): Scale parameter that controls the variance of the random weights.
        
    Returns:
        np.ndarray: A matrix of shape (len(x), D) representing the Fourier features.
    """
    D = round(D)
    gamma = round(gamma)
    x_norm = day_of_week.values / 365.0
    # Generate D random weights from a normal distribution scaled by sqrt(2*gamma)
    w = np.random.normal(scale=np.sqrt(2 * gamma), size=D)
    # Generate D random phase shifts uniformly in [0, 2π)
    b = 2 * np.pi * np.random.rand(D)
    # Compute the Fourier features using the cosine transformation
    z = np.sqrt(2 / D) * np.cos(np.outer(x_norm, w) + b)
    #return np.zeros(day_of_week.shape)

    weights = np.random.uniform(-0.2, 0.2, size=z.shape[1])
    return z @ weights  # result is a (n_samples,) array


#TODO:  a) PCA naar elbow curve kijken, moet staps gewijs bewegen en niet 
#       b) Random foirier features -> hogere dimensionale data

# all betas gebruiken. 
# Coefficienten sommeren naar 1. 
# Transformers: zin voor eerste twee, maar logisch dat het completere model met decoder + autoregressive beter is. 

def run_forecast1(df: pd.DataFrame) -> pd.DataFrame:
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
    df['alpha'] = (
        1.2
        - 0.4 * np.cos(2 * np.pi * (df['day_of_year'] - 172) / 365)
        + 0.25 * get_random_fourier_features(df['day_of_year'], np.random.normal(fourier_d, fourier_d_sigma), np.random.normal(fourier_gamma, fourier_gamma_sigma))
        + np.random.normal(mu, sigma, len(df))
    )

    # Constant delta
    df['delta'] = 0.02 * np.sin(2 * np.pi * df['day_of_year'] / 365) + np.random.normal(mu2, sigma2, len(df))


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

def run_forecast2(df: pd.DataFrame) -> pd.DataFrame:
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
    df['alpha'] = (
        0.8 
        + 0.4 * np.cos(2 * np.pi * (df['day_of_year'] - 172) / 365) 
        + 0.25 * get_random_fourier_features(df['day_of_year'], np.random.normal(fourier_d, fourier_d_sigma), np.random.normal(fourier_gamma, fourier_gamma_sigma))
        + np.random.normal(mu, sigma, len(df))
    )


    # Constant delta
    df['delta'] = -0.03 * np.sin(2 * np.pi * df['day_of_year'] / 365) + np.random.normal(mu2, sigma2, len(df))


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
    df['alpha'] = (
        1 
        + 0.5 * cloud_index 
        + 0.25 * get_random_fourier_features(df['day_of_year'], np.random.normal(fourier_d, fourier_d_sigma), np.random.normal(fourier_gamma, fourier_gamma_sigma))
        + np.random.normal(mu, sigma, len(df))
    )
    
    # Constant delta
    df['delta'] = 0.01 + 0.05 * cloud_index + np.random.normal(mu2, sigma2, len(df))

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
    #rolling_std.hist()
    rolling_std = np.clip(rolling_std, 0, 2500)  # clip to avoid extreme values
    #rolling_std = np.clip(rolling_std, 0, 2500)  # clip to avoid extreme values

    # Normalize std to keep alpha in reasonable range (assume std in [0, 0.3])
    df['alpha'] = (
        1.0 
        + 0.00010 * rolling_std 
        + 0.25 * get_random_fourier_features(df['day_of_year'], np.random.normal(fourier_d, fourier_d_sigma), np.random.normal(fourier_gamma, fourier_gamma_sigma))
        + np.random.normal(mu, sigma, len(df))
    ) # scaling factor to exaggerate effect slightly
    
    # Constant delta
    df['delta'] = 0.00 + np.random.normal(mu2, sigma2, len(df))

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

    # Constant alpha
    df['alpha'] = (
        1.0 
        + 0.25 * get_random_fourier_features(df['day_of_year'], np.random.normal(fourier_d, fourier_d_sigma), np.random.normal(fourier_gamma, fourier_gamma_sigma))
        + np.random.normal(mu, sigma, len(df))
    )

    # Linearly drifting delta (e.g., from 0.05 to -0.05)
    df['delta'] = np.linspace(0.05, -0.05, len(df)) + np.random.normal(mu2, sigma2, len(df))

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
    df['alpha'] = (
        1.0 
        + 0.25 * get_random_fourier_features(df['day_of_year'], np.random.normal(fourier_d, fourier_d_sigma), np.random.normal(fourier_gamma, fourier_gamma_sigma))
        + np.random.normal(mu, sigma, len(df))
    ) # could exceed 1.0 or drop below 0.5 depending on error
    
    # Delta: oppose the direction of previous forecast residual
    df['delta'] = -0.04  + np.random.normal(mu2, sigma2, len(df))
    
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



# --- Updated Forecast Functions with Increased Variability ---

def run_forecasts1_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Updated 'SUMMER SPECIALIST' forecast with added sine component to diversify the seasonal pattern.
    """
    df = df.copy()
    df[ModelSettings.datetime_col] = pd.to_datetime(df[ModelSettings.datetime_col])
    df['day_of_year'] = df[ModelSettings.datetime_col].dt.dayofyear

    # Increase variability: add an extra sine term along with cosine for alpha.
    df['alpha'] = (
        1.2
        - 0.4 * np.cos(2 * np.pi * (df['day_of_year'] - 172) / 365)
        + 0.2 * np.sin(2 * np.pi * (df['day_of_year'] - 172) / 365)
        + np.random.normal(mu, sigma, len(df))
    )
    df['delta'] = (
        0.02 * np.sin(2 * np.pi * df['day_of_year'] / 365)
        + np.random.normal(mu2, sigma2, len(df))
    )
    df['epsilon'] = np.random.standard_t(df=3, size=len(df)) * 0.05

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
    df['forecasted_value'] = forecasted_value
    result = df[[ModelSettings.datetime_col, 'forecasted_value']].reset_index(drop=True)
    return result.fillna(0)


def run_forecasts2_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Updated 'WINTER SPECIALIST' forecast with a linear adjustment term added to the seasonal component,
    making the adjustment functionally different from the summer forecast.
    """
    df = df.copy()
    df[ModelSettings.datetime_col] = pd.to_datetime(df[ModelSettings.datetime_col])
    df['day_of_year'] = df[ModelSettings.datetime_col].dt.dayofyear

    # Here we subtract a linear trend from the cosine term for a different pattern.
    df['alpha'] = (
        0.8
        + 0.4 * np.cos(2 * np.pi * (df['day_of_year'] - 172) / 365)
        - 0.2 * ((df['day_of_year'] - 182) / 365)
        + np.random.normal(mu, sigma, len(df))
    )
    df['delta'] = (
        -0.03 * np.sin(2 * np.pi * df['day_of_year'] / 365)
        + np.random.normal(mu2, sigma2, len(df))
    )
    df['epsilon'] = np.random.standard_t(df=3, size=len(df)) * 0.05

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
    df['forecasted_value'] = forecasted_value
    result = df[[ModelSettings.datetime_col, 'forecasted_value']].reset_index(drop=True)
    return result.fillna(0)


def run_forecast_new(df: pd.DataFrame) -> pd.DataFrame:
    """
    NEW DIVERSIFIED FORECAST:
    Uses a polynomial-based adjustment for alpha and an exponential bias for delta,
    both of which are structured to add a new pattern that is different from the existing models.
    """
    df = df.copy()
    df[ModelSettings.datetime_col] = pd.to_datetime(df[ModelSettings.datetime_col])
    df['day_of_year'] = df[ModelSettings.datetime_col].dt.dayofyear

    # Alpha: Add a quadratic term based on day-of-year deviation (centered at mid-year)
    df['alpha'] = (
        1.0
        + 0.3 * ((df['day_of_year'] - 182) / 365) ** 2
        + np.random.normal(0, 0.15, len(df))
    )
    # Delta: Exponential decay adjustment with a random sign flip for diversity
    df['delta'] = (
        0.02 * np.exp(-((df['day_of_year'] - 182) / 365) ** 2)
        * np.where(np.random.rand(len(df)) > 0.5, 1, -1)
        + np.random.normal(mu2, sigma2, len(df))
    )
    # Increase the noise a bit more for this model to further differentiate it.
    df['epsilon'] = np.random.standard_t(df=3, size=len(df)) * 0.07

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
    df['forecasted_value'] = forecasted_value
    result = df[[ModelSettings.datetime_col, 'forecasted_value']].reset_index(drop=True)
    return result.fillna(0)
