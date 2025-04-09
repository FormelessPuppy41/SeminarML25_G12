
import pandas as pd
import numpy as np

from configuration import ModelSettings

# python -m combined_forecast.initial_forecasting.run_forecasts

# df has the following columns:
#     - datetime (datetime)
#     - HR (actual yield of solar energy)
#     - K (forecast of company)

def run_forecast1(df: pd.DataFrame, shift: int = 96):
    """
    This method forecasts the value HR by using a lagged value of HR. 

    So, it uses the actual yield (HR) of the solar energy two days ago (t = t - shift) to predict the yield of today (t = t).
    The lagged value is used as a simple forecast for the current value.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be forecasted.
        shift (int): The number of time steps to shift the HR value for forecasting. Default is 48 - should represent two days.
    """
    forecast_df = df.copy()
    forecast_df['forecasted_value'] = forecast_df[ModelSettings.target].shift(shift)
    
    # Drop rows with NaN forecast (i.e., first 48 rows)
    forecast_df = forecast_df[[ModelSettings.datetime_col, 'forecasted_value']].reset_index(drop=True)
    
    return forecast_df.fillna(0)  # Fill NaN values with 0 for forecasted_value


def run_forecast2(
        df: pd.DataFrame, 
        alpha: float = 0.9, 
        beta: float = 1.1,
        mu: float = 1, 
        sigma: float = 0.1
    ):
    """
    This method forecasts the value HR by using a random noise (N[mu, sigma]) multiplied by an indicator which scales positive and negative values differently (I[x<=0]* alpha + I[x>0]* beta). 
    However, this is only done if the HR value is not zero (0). 

    So, it uses the actual yield (HR) of the solar energy adjusted with a scaled random noise to predict the yield of today (t = t).
    The random noise is generated using a normal distribution with mean mu and standard deviation sigma.
    The scaling factors alpha and beta are used to adjust the noise for negative and positive values, respectively.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be forecasted.
        alpha (float): The scaling factor for negative values. 
        beta (float): The scaling factor for positive values.
        mu (float): The mean of the normal distribution. Default is 1 - scaling multiplier of 1 on avg.
        sigma (float): The standard deviation of the normal distribution. Default is 0.1 - scaling multiplier of 0.1 on avg. This means that 66% of obs will be scaled by [0.9, 1.1].
    """
    forecast_df = df.copy()

    # Create condition masks
    hr_nonzero = forecast_df[ModelSettings.target] != 0
    
    # Generate base noise (multiplicative factors ~ N(mu, sigma))
    base_noise = np.random.normal(loc=mu, scale=sigma, size=len(forecast_df))

    # Scale factors based on conditions
    scale_factors = np.where(base_noise > 0, beta, alpha)

    # Final noise is multiplicative: noise = N(mu, sigma) * scale
    final_noise = base_noise * scale_factors

    forecasted_value = np.full(len(forecast_df), np.nan)  # init with NaNs
    forecasted_value[hr_nonzero] = forecast_df[ModelSettings.target][hr_nonzero] * final_noise[hr_nonzero]

    forecast_df['forecasted_value'] = forecasted_value
    forecast_df = forecast_df[[ModelSettings.datetime_col, 'forecasted_value']].reset_index(drop=True)
    return forecast_df.fillna(0)  # Fill NaN values with 0 for forecasted_value


def run_forecast3(
        df: pd.DataFrame,
        gamma: float = 0.5,
        alpha: float = 0.5,
        beta: float = 1.0
    ):
    """
    This method forecasts the value HR by using a random noise (EXP[gamma] * U[alpha, beta]).
    However, this is only done if the HR value is not zero (0).

    So, it uses the actual yield (HR) of the solar energy adjusted with a scaled random noise to predict the yield of today (t = t). 
    The random noise is generated using an exponential distribution with parameter gamma and a uniform distribution with range [alpha, beta].

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be forecasted.
        gamma (float): The scale parameter for the exponential distribution. Default is 1.0. This means that the average value of the exponential distribution is 1.
        alpha (float): The lower bound of the uniform distribution. Default is 0.8.
        beta (float): The upper bound of the uniform distribution. Default is 1.2.
    """
    forecast_df = df.copy()

    # Mask for non-zero HR values
    hr_nonzero = forecast_df[ModelSettings.target] != 0
    n = hr_nonzero.sum()

    # Generate multiplicative noise
    exp_noise = np.random.exponential(scale=gamma, size=n)
    uniform_noise = np.random.uniform(low=alpha, high=beta, size=n)
    combined_noise = exp_noise * uniform_noise
    combined_noise = np.clip(combined_noise, 0.8, 1.2)


    forecasted_value = np.full(len(forecast_df), np.nan)
    forecasted_value[hr_nonzero] = forecast_df[ModelSettings.target][hr_nonzero].values * combined_noise

    forecast_df['forecasted_value'] = forecasted_value
    forecast_df = forecast_df[[ModelSettings.datetime_col, 'forecasted_value']].reset_index(drop=True)
    return forecast_df.fillna(0)  # Fill NaN values with 0 for forecasted_value



def run_forecast4(
    df: pd.DataFrame,
    alpha: float = 0.5,
    beta: float = 1,
    t_peak: int = 172,
    sigma_primary: float = 800,
    sigma_secondary: float = 600,
    secondary_weight: float = 0.3
) -> pd.DataFrame:
    """
    Forecasts the HR value using dual seasonal-scaled multiplicative Gaussian noise.

    Combines:
    - A primary bell curve (low noise near t_peak, high noise far from it).
    - A secondary inverted bell (adds mild noise in winter).

    Args:
        alpha: Mean of the Gaussian noise.
        beta: Standard deviation of the Gaussian noise.
        t_peak: Peak day of year with minimal noise (e.g. June 21).
        sigma_primary: Spread of main bell curve.
        sigma_secondary: Spread of inverted bell.
        secondary_weight: Scaling weight for the inverted seasonal component.
    """
    forecast_df = df.copy()

    if not np.issubdtype(forecast_df[ModelSettings.datetime_col].dtype, np.datetime64):
        forecast_df[ModelSettings.datetime_col] = pd.to_datetime(forecast_df[ModelSettings.datetime_col])

    day_of_year = forecast_df[ModelSettings.datetime_col].dt.dayofyear
    hr_nonzero = forecast_df[ModelSettings.target] != 0
    valid_days = day_of_year[hr_nonzero]

    # Symmetric seasonal distance from peak
    distance = np.minimum(
        np.abs(valid_days - t_peak),
        365 - np.abs(valid_days - t_peak)
    )

    # Main bell curve: minimal noise at t_peak
    scale_primary = np.exp(-(distance ** 2) / (2 * sigma_primary))

    # Inverted bell curve: added noise far from t_peak
    scale_secondary = 1 - np.exp(-(distance ** 2) / (2 * sigma_secondary))

    # Combined seasonal scaling
    seasonal_scale = scale_primary + secondary_weight * scale_secondary

    # Generate scaled noise
    base_noise = np.random.normal(loc=alpha, scale=beta, size=len(seasonal_scale))
    scaled_noise = base_noise * seasonal_scale

    forecasted_value = np.full(len(forecast_df), np.nan)
    forecasted_value[hr_nonzero] = forecast_df.loc[hr_nonzero, ModelSettings.target].values * (1 + scaled_noise)

    forecast_df['forecasted_value'] = forecasted_value
    forecast_df = forecast_df[[ModelSettings.datetime_col, 'forecasted_value']].reset_index(drop=True)
    return forecast_df.fillna(0)


def run_forecast5(
    df: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.8,
    delta: float = 1.2
) -> pd.DataFrame:
    """
    Forecasts the HR value using multiplicative noise drawn from:
        Gamma(alpha, beta) * Uniform(gamma, delta)

    Forecasting is only applied when HR ≠ 0.

    This models proportional variation, where the actual HR is scaled by
    a randomly generated factor from the above composite distribution.

    Args:
        df (pd.DataFrame): Input DataFrame containing ModelSettings.datetime_col and ModelSettings.target columns.
        alpha (float): Shape parameter of the gamma distribution. Default is 2.0.
        beta (float): Scale parameter of the gamma distribution. Default is 0.5.
        gamma (float): Lower bound of the uniform distribution. Default is 0.8.
        delta (float): Upper bound of the uniform distribution. Default is 1.2.

    Returns:
        pd.DataFrame: DataFrame with [ModelSettings.datetime_col, 'forecasted_value'] for HR ≠ 0 rows.
    """
    forecast_df = df.copy()

    # Mask for non-zero HR values
    hr_nonzero = forecast_df[ModelSettings.target] != 0
    n = hr_nonzero.sum()

    # Generate multiplicative noise
    gamma_noise = np.random.gamma(shape=alpha, scale=beta, size=n)
    uniform_noise = np.random.uniform(low=gamma, high=delta, size=n)
    combined_noise = gamma_noise * uniform_noise
    combined_noise = np.clip(combined_noise, 0.8, 1.2)


    forecasted_value = np.full(len(forecast_df), np.nan)
    forecasted_value[hr_nonzero] = forecast_df.loc[hr_nonzero, ModelSettings.target].values * combined_noise

    forecast_df['forecasted_value'] = forecasted_value
    forecast_df = forecast_df[[ModelSettings.datetime_col, 'forecasted_value']].reset_index(drop=True)
    return forecast_df.fillna(0)  # Fill NaN values with 0 for forecasted_value



