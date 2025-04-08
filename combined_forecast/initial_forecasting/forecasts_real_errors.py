import numpy as np
import pandas as pd


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
