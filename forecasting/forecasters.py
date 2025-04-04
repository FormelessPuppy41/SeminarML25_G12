
import pandas as pd
import numpy as np


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
    forecast_df['forecasted_value'] = forecast_df['HR'].shift(shift)
    
    # Drop rows with NaN forecast (i.e., first 48 rows)
    forecast_df = forecast_df[['datetime', 'forecasted_value']].reset_index(drop=True)
    
    return forecast_df


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
    hr_nonzero = forecast_df['HR'] != 0
    hr_positive = forecast_df['HR'] > 0
    hr_negative = forecast_df['HR'] <= 0

    # Generate base noise (multiplicative factors ~ N(mu, sigma))
    base_noise = np.random.normal(loc=mu, scale=sigma, size=len(forecast_df))

    # Scaling based on HR polarity
    scale_factors = np.ones(len(forecast_df)) #FIXME: Should this not be zero?
    scale_factors[hr_positive] = beta
    scale_factors[hr_negative] = alpha

    # Final noise is multiplicative: noise = N(mu, sigma) * scale
    final_noise = base_noise * scale_factors

    forecasted_value = np.full(len(forecast_df), np.nan)  # init with NaNs
    forecasted_value[hr_nonzero] = forecast_df['HR'][hr_nonzero] * final_noise[hr_nonzero]

    forecast_df['forecasted_value'] = forecasted_value
    return forecast_df[['datetime', 'forecasted_value']].reset_index(drop=True)


def run_forecast3(
        df: pd.DataFrame,
        gamma: float = 1.0,
        alpha: float = 0.8,
        beta: float = 1.2
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
    hr_nonzero = forecast_df['HR'] != 0
    n = hr_nonzero.sum()

    # Generate multiplicative noise
    exp_noise = np.random.exponential(scale=gamma, size=n)
    uniform_noise = np.random.uniform(low=alpha, high=beta, size=n)
    combined_noise = exp_noise * uniform_noise

    forecasted_value = np.full(len(forecast_df), np.nan)
    forecasted_value[hr_nonzero] = forecast_df['HR'][hr_nonzero].values * combined_noise

    forecast_df['forecasted_value'] = forecasted_value
    return forecast_df[['datetime', 'forecasted_value']].reset_index(drop=True)



def run_forecast4(
    df: pd.DataFrame,
    alpha: float = 0,
    beta: float = 0.2,
    t_peak: int = 172,
    sigma: float = 400
) -> pd.DataFrame:
    """
    Forecasts the HR value using seasonally-scaled multiplicative Gaussian noise.
    
    The noise is defined as:
        noise ~ N(alpha, beta) * exp(-((day_of_year - t_peak)^2) / (2 * sigma))
    
    This noise acts as a multiplicative scaling factor applied to the HR value:
        forecast = HR * (1 + noise)
    
    Forecasting is only applied when HR ≠ 0.

    Args:
        df (pd.DataFrame): DataFrame with 'datetime' and 'HR' columns.
        alpha (float): Mean of the Gaussian noise (typically 0 for centered variation). Default is 0.
        beta (float): Standard deviation of the Gaussian noise. Default is 0.2.
        t_peak (int): Day of year with highest seasonal solar yield (e.g., 172 for June 21). Default is 172.
        sigma (float): Spread of the seasonal scaling envelope. Default is 400.

    Returns:
        pd.DataFrame: Forecasted DataFrame with ['datetime', 'forecasted_value'].
    """
    forecast_df = df.copy()

    # Ensure datetime column is datetime type
    if not np.issubdtype(forecast_df['datetime'].dtype, np.datetime64):
        forecast_df['datetime'] = pd.to_datetime(forecast_df['datetime'])

    # Get day of year (1–366)
    day_of_year = forecast_df['datetime'].dt.dayofyear

    # Mask for non-zero HR values
    hr_nonzero = forecast_df['HR'] != 0
    valid_days = day_of_year[hr_nonzero]

    # Seasonal scaling envelope (like a bell curve centered on t_peak)
    seasonal_scale = np.exp(-((valid_days - t_peak) ** 2) / (2 * sigma))

    # Generate scaled noise (multiplicative)
    base_noise = np.random.normal(loc=alpha, scale=beta, size=len(seasonal_scale))
    scaled_noise = base_noise * seasonal_scale

    forecasted_value = np.full(len(forecast_df), np.nan)
    forecasted_value[hr_nonzero] = forecast_df.loc[hr_nonzero, 'HR'].values * (1 + scaled_noise)

    forecast_df['forecasted_value'] = forecasted_value
    return forecast_df[['datetime', 'forecasted_value']].reset_index(drop=True)


def run_forecast5(
    df: pd.DataFrame,
    alpha: float = 2.0,
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
        df (pd.DataFrame): Input DataFrame containing 'datetime' and 'HR' columns.
        alpha (float): Shape parameter of the gamma distribution. Default is 2.0.
        beta (float): Scale parameter of the gamma distribution. Default is 0.5.
        gamma (float): Lower bound of the uniform distribution. Default is 0.8.
        delta (float): Upper bound of the uniform distribution. Default is 1.2.

    Returns:
        pd.DataFrame: DataFrame with ['datetime', 'forecasted_value'] for HR ≠ 0 rows.
    """
    forecast_df = df.copy()

    # Mask for non-zero HR values
    hr_nonzero = forecast_df['HR'] != 0
    n = hr_nonzero.sum()

    # Generate multiplicative noise
    gamma_noise = np.random.gamma(shape=alpha, scale=beta, size=n)
    uniform_noise = np.random.uniform(low=gamma, high=delta, size=n)
    combined_noise = gamma_noise * uniform_noise

    forecasted_value = np.full(len(forecast_df), np.nan)
    forecasted_value[hr_nonzero] = forecast_df.loc[hr_nonzero, 'HR'].values * combined_noise

    forecast_df['forecasted_value'] = forecasted_value
    forecast_df = forecast_df[['datetime', 'forecasted_value']].reset_index(drop=True)
    return forecast_df.fill_na



