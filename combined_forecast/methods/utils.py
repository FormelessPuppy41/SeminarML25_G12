
import pandas as pd

from configuration import ModelSettings

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV

def tune_model_with_gridsearch(pipeline, param_grid, X_train, y_train, grid_params):
    grid = GridSearchCV(pipeline, param_grid=param_grid, **grid_params)
    grid.fit(X_train, y_train)
    return grid


def get_model_from_params(params: dict, adaptive: bool = False, fit_intercept: bool = True, standard_scaler_with_mean: bool = True):
    # Determine alpha
    if "alpha_grid" in params:
        alpha_grid = params["alpha_grid"]
        alpha = alpha_grid[0]  # This is used only as a starting value
    else:
        alpha = 1.0

    # Determine l1_ratio, and check if it's a grid with multiple values
    if "l1_ratio_grid" in params:
        l1_ratio_grid = params["l1_ratio_grid"]
        # If more than one value is provided, we choose ElasticNet.
        if len(l1_ratio_grid) > 1:
            # Set a default value that makes sense for ElasticNet (for example, 0.5)
            l1_ratio = 0.5
        else:
            l1_ratio = l1_ratio_grid[0]
    else:
        l1_ratio = 0.5

    random_state = 42 # params.get("random_state", 42)

    # Return the estimator based on the l1_ratio value.
    # When using grid search over a range of l1_ratio values, default to ElasticNet.
    if l1_ratio <= 1e-6: # Use a small threshold, bcs otherwise you get warnings when testing with 0.0 of convergence issues. 
        model = Ridge(alpha=alpha, random_state=random_state, max_iter=10000, fit_intercept=fit_intercept)
    elif l1_ratio == 1.0:
        model = Lasso(alpha=alpha, random_state=random_state, max_iter=10000, fit_intercept=fit_intercept)
    else:
        # For other cases (including when multiple values are provided) default to ElasticNet.
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=10000, fit_intercept=fit_intercept)
        
    if adaptive:
        return make_pipeline(model)
    #return make_pipeline(model)
    return make_pipeline(StandardScaler(with_mean=standard_scaler_with_mean), model)



def data_interpolate_prev(
        raw_df: pd.DataFrame, 
        current_date: pd.Timestamp, 
        rolling_window_days: int = 165
    ) -> pd.DataFrame:
    """
    Returns the training data with interpolated values for the previous rolling window.

    Args:
        raw_df (pd.DataFrame): The raw input data -> df(time, HR, forecasts)
        current_date (pd.Timestamp): The current date.

    Returns:
        pd.DataFrame: The training data with interpolated values for the previous rolling window.
    """
    if not isinstance(current_date, pd.Timestamp):
        current_date = pd.Timestamp(current_date)

    # Define time window and slice using the datetime column from raw_df
    start_date = current_date - pd.Timedelta(days=rolling_window_days)
    end_date = current_date
    mask = (raw_df[ModelSettings.datetime_col] >= start_date) & (raw_df[ModelSettings.datetime_col] <= end_date)
    working_df = raw_df.loc[mask].copy()

    # Ensure indicator_df datetime column is datetime type
    #indicator_df[ModelSettings.datetime_col] = pd.to_datetime(indicator_df[ModelSettings.datetime_col])
    
    # Initialize output with target column
    interpolated_df = pd.DataFrame()
    for col in working_df.columns:
        interpolated_df[col] = working_df[col].copy()
    
    # Loop over each column (skip target and datetime columns)
    for col in working_df.columns:
        if col == ModelSettings.datetime_col:
            continue
        
        interpolated_df[col] = interpolated_df[col].interpolate(method='linear')
        interpolated_df[col] = interpolated_df[col].fillna(0)

    return interpolated_df

def data_interpolate_fut(
        raw_df: pd.DataFrame,
        current_date: pd.Timestamp,
        forecast_horizon: int = 96,
        freq: str = '15min'
    ) -> pd.DataFrame:
    """
    Returns the training data with interpolated values for the future forecast horizon.

    Args:
        raw_df (pd.DataFrame): The raw input data -> df(time, HR, forecasts)
        current_date (pd.Timestamp): The current date.  
        forecast_horizon (int, optional): _description_. Defaults to 24.
        freq (str, optional): _description_. Defaults to '15min'.

    Returns:
        pd.DataFrame: The training data with interpolated values for the future forecast horizon.
    """
    if not isinstance(current_date, pd.Timestamp):
        current_date = pd.Timestamp(current_date)

    # Determine the time delta based on the frequency
    if freq == '1H':
        delta = pd.Timedelta(hours=1)
    else:
        # For example, for '15min' frequency:
        delta = pd.Timedelta(minutes=int(freq.replace('min', '')))

    start_date = current_date
    end_date = current_date + delta * (forecast_horizon - 1)
    mask = (raw_df[ModelSettings.datetime_col] >= start_date) & (raw_df[ModelSettings.datetime_col] <= end_date)
    working_df = raw_df.loc[mask].copy()

    # Initialize output with target column
    interpolated_df = pd.DataFrame()
    for col in working_df.columns:
        interpolated_df[col] = working_df[col].copy()
    
    # Loop over each forecast column
    for col in working_df.columns:
        if col == ModelSettings.datetime_col:
            continue

        # Interpolate the column using linear interpolation
        interpolated_df[col] = interpolated_df[col].interpolate(method='linear')
        interpolated_df[col] = interpolated_df[col].fillna(0)

    return interpolated_df


# def data_interpolate_prev(
#         raw_df: pd.DataFrame, 
#         #indicator_df: pd.DataFrame, 
#         current_date: pd.Timestamp, 
#         rolling_window_days: int = 165
#     ) -> pd.DataFrame:
#     """
#     Returns the training data for the previous rolling window, dropping rows with missing HR values.

#     Args:
#         raw_df (pd.DataFrame): The raw input data with columns for time, HR, and forecasts.
#         current_date (pd.Timestamp): The current date.

#     Returns:
#         pd.DataFrame: The subset of data for the rolling window with rows missing HR dropped.
#     """
#     if not isinstance(current_date, pd.Timestamp):
#         current_date = pd.Timestamp(current_date)

#     # Define time window using the datetime column from raw_df
#     start_date = current_date - pd.Timedelta(days=rolling_window_days)
#     end_date = current_date
#     mask = (raw_df[ModelSettings.datetime_col] >= start_date) & \
#            (raw_df[ModelSettings.datetime_col] <= end_date)
#     working_df = raw_df.loc[mask].copy()

#     # Drop rows with missing HR (target) values
#     working_df = working_df.dropna(subset=[ModelSettings.target])
    
#     return working_df


# def data_interpolate_fut(
#         raw_df: pd.DataFrame,
#         current_date: pd.Timestamp,
#         forecast_horizon: int = 96,
#         freq: str = '15min'
#     ) -> pd.DataFrame:
#     """
#     Returns the forecast horizon data by extracting the data between start_date and end_date,
#     and dropping rows with missing HR values rather than interpolating.

#     Args:
#         raw_df (pd.DataFrame): The raw input data with columns for time, HR, and forecasts.
#         current_date (pd.Timestamp): The starting date for the forecast horizon.
#         forecast_horizon (int, optional): Number of forecast periods (defaults to 96).
#         freq (str, optional): Data frequency ('15min' or '1H') (defaults to '15min').

#     Returns:
#         pd.DataFrame: The subset of future data with rows missing HR dropped.
#     """
#     if not isinstance(current_date, pd.Timestamp):
#         current_date = pd.Timestamp(current_date)

#     # Determine time delta based on frequency
#     if freq == '1H':
#         delta = pd.Timedelta(hours=1)
#     else:
#         delta = pd.Timedelta(minutes=int(freq.replace('min', '')))

#     start_date = current_date
#     end_date = current_date + delta * (forecast_horizon - 1)
#     mask = (raw_df[ModelSettings.datetime_col] >= start_date) & \
#            (raw_df[ModelSettings.datetime_col] <= end_date)
#     working_df = raw_df.loc[mask].copy()

#     # Drop rows with missing HR (target) values
#     working_df = working_df.dropna(subset=[ModelSettings.target])

#     return working_df


