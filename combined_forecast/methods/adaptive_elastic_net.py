"""
Adaptive Elastic Net Regression using the shared Elastic Net model pipeline
"""
import pandas as pd
from typing import List, Dict, Any

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet, Ridge, Lasso

from combined_forecast.methods.elastic_net import get_model_from_params

from configuration import ModelSettings

def data_interpolate_prev(
        raw_df: pd.DataFrame, 
        indicator_df: pd.DataFrame, 
        current_date: pd.Timestamp, 
        rolling_window_days: int = 165
    ) -> pd.DataFrame:
    """
    Returns the training data with interpolated values for the previous rolling window.

    Args:
        raw_df (pd.DataFrame): The raw input data -> df(time, HR, forecasts)
        indicator_df (pd.DataFrame): The indicator data, indicting which time/fcst combination has either 0 (complete), or other value.
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
    indicator_df[ModelSettings.datetime_col] = pd.to_datetime(indicator_df[ModelSettings.datetime_col])
    
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
        indicator_df (pd.DataFrame): The indicator data, indicting which time/fcst combination has either 0 (complete), or other value.
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


def get_model_from_params(params: Dict[str, Any]):
    """
    Create a pipeline with an ElasticNet, Ridge, or Lasso model based on the parameters. 

    If l1_ratio is 0.0, Ridge regression is used.
    If l1_ratio is 1.0, Lasso regression is used.
    Otherwise, ElasticNet regression is used.

    Args:
        params (Dict[str, Any]): Parameters for the model. Such as 'alpha', 'l1_ratio', and 'random_state'.

    Returns:
        Pipeline: A pipeline with the selected model.
    """
    # Get the parameters
    alpha = params.get("alpha", 1.0)
    l1_ratio = params.get("l1_ratio", 0.5)
    random_state = params.get("random_state", 42)

    # Create a pipeline with the selected model
    if l1_ratio == 0.0:
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=random_state))
    elif l1_ratio == 1.0:   
        return make_pipeline(StandardScaler(), Lasso(alpha=alpha, random_state=random_state))
    else:
        return make_pipeline(StandardScaler(), ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=random_state))


def run_elastic_net(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: List[str],
    params: Dict[str, Any]
):
    """
    Run an Elastic Net regression model on test data using training data.

    Args:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The testing data.
        target (str): The target column name.
        features (List[str]): List of feature column names.
        params (Dict[str, Any]): Parameters for ElasticNet.

    Returns:
        pd.DataFrame: Test set with predictions added in 'prediction' column.
    """
    params = {
        'alpha': params.get('alpha', 1.0),
        'l1_ratio': params.get('l1_ratio', 0.5),
        'random_state': params.get('random_state', 42)
    }

    # check if train has any NaN values
    if train.isnull().values.any():
        print("WARNING: Training data contains NaN values. Specifically in the following columns:")
        print(train.columns[train.isnull().any()].tolist())
        nan_cols = train.columns[train.isnull().any()].tolist()
        for col in nan_cols:
            # print the nan values:
            print(f"NaN values in column: {col}: {train[col].isnull().sum()}")
            print(train[col].tolist())
           
    model = get_model_from_params(params)
    model.fit(train[features], train[target])

    test = test.copy()
    test['prediction'] = model.predict(test[features])
    return test


def run_day_ahead_adaptive_elastic_net(
    df: pd.DataFrame,
    #flag_matrix_df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    forecast_horizon: int = 96,
    rolling_window_days: int = 165,
    param_grid: Dict[str, List[Any]] = None,
    datetime_col: str = 'datetime',
    freq: str = '15min'
) -> pd.DataFrame:
    """
    Generate forecasts at 09:00 AM each morning for the next day (24 hourly forecasts).
    
    Args:
        df: Full input data with datetime, target and features.
        #flag_matrix_df: DataFrame indicating which time/forecast combination is complete (0) or not.
        target_column: Column name of the target variable (e.g., 'HR').
        feature_columns: List of forecast provider feature columns.
        forecast_horizon: Number of forecast periods (default=24 for hourly forecasts).
        rolling_window_days: Number of days for the training window.
        enet_params: Elastic Net parameters (e.g. {'alpha': 1.0, 'l1_ratio': 0.5}).
        datetime_col: Name of the datetime column.
        freq: Sampling frequency (default='1H' for hourly).
        
    Returns:
        pd.DataFrame with columns ['target_time', 'prediction', 'actual'].
    """
    # Check if the parameters are provided, and set defaults if not
    enet_params = enet_params or {'alpha': 1.0, 'l1_ratio': 0.5}

    # TODO: This should be done in the data preparation step.
    if datetime_col not in df.columns:
        #print(df, flag_matrix_df)
        raise ValueError(f"'{datetime_col}' not found in DataFrame.")
    
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    #df = df.set_index(datetime_col).sort_index()

    # Initialize list to store forecast results
    forecast_results = []

    # Get unique forecast dates
    unique_dates = df[datetime_col].unique()
    forecast_dates = [pd.Timestamp(d) for d in unique_dates if (pd.Timestamp(d).hour == 9 and pd.Timestamp(d).minute == 0)]


    # Iterate over each forecast date
    current_date_interpolator_prev: pd.DataFrame = None
    current_date_interpolator_fut: pd.DataFrame = None
    last_forecast_date: pd.Timestamp = None

    for forecast_date in forecast_dates[rolling_window_days:]:
        print(f"Forecast date (09:00): {forecast_date}")

        forecast_start = forecast_date.normalize() + pd.Timedelta(days=1)  # Next day at 00:00

        # Update the interpolators if we move to a new forecast date
        if last_forecast_date is None or forecast_date != last_forecast_date:
            last_forecast_date = forecast_date
            current_date_interpolator_prev = data_interpolate_prev(df, forecast_date, rolling_window_days)
            current_date_interpolator_fut = data_interpolate_fut(df, forecast_start, forecast_horizon, freq=freq)

        # Get the training data for the previous rolling window
        train_df = current_date_interpolator_prev
        test_df = current_date_interpolator_fut

        # Check NA:
        if train_df.isnull().values.any() or test_df.isnull().values.any():
            print("WARNING: Training or testing data contains NaN values. Specifically in the following columns:")
            if train_df.isnull().values.any():
                print("Training data NaN columns:", train_df.columns[train_df.isnull().any()].tolist())
            if test_df.isnull().values.any():
                print("Testing data NaN columns:", test_df.columns[test_df.isnull().any()].tolist())
            continue

        if train_df.empty or test_df.empty or train_df[target_column].isnull().all() or test_df[target_column].isnull().all():
            print(f"WARNING: No data for forecast date: {forecast_date}")
            continue
        

        # Run Elastic Net for each forecast time (hour) in the next day
        for ts in test_df[ModelSettings.datetime_col]:
            row_df = test_df[test_df[ModelSettings.datetime_col] == ts]
    
            # Get model pipeline
            base_params = {'alpha': 1.0, 'l1_ratio': 0.5, 'random_state': 42}
            pipeline = get_model_from_params(base_params)
            model_name = pipeline.steps[-1][0]
            pipeline.steps[-1] = (model_name, pipeline.steps[-1][1])

            # Tune ElasticNet
            grid = GridSearchCV(pipeline, param_grid=param_grid, **param_grid)
            try:
                grid.fit(train_df[feature_columns], train_df[target_column])
                prediction = grid.predict(row_df[feature_columns])[0]
                actual = row_df[target_column].values[0]
            except Exception as e:
                print(f"Skipping {ts} due to training error: {e}")
                continue

            forecast_results.append({
                'target_time': ts,
                'prediction': prediction,
                'actual': actual,
                'best_params': grid.best_params_ 
            })

    return pd.DataFrame(forecast_results)



if __name__ == '__main__':
    from combined_forecast.utils import generate_sample_data, evaluate_forecast

    # Generate sample 15-min interval data for 20 days
    df_sample = generate_sample_data(start='2023-01-01', days=20)
    target_col = 'HR'
    feature_cols = [f'A{i}' for i in range(1, 8)]

    # Run Adaptive Elastic Net with rolling forecast and grid search
    forecast_df = run_day_ahead_adaptive_elastic_net(
        df=df_sample,
        target_column=target_col,
        feature_columns=feature_cols,
        rolling_window_days=5,
        param_grid={
            'elasticnet__alpha': [0.1, 1.0, 10.0],
            'elasticnet__l1_ratio': [0.0, 0.5, 1.0]
        },
        grid_params={
            'cv': 5,
            'n_jobs': -1,
            'verbose': 1
        }
    )

    print(forecast_df)

    if not forecast_df.empty:
        rmse = evaluate_forecast(forecast_df)
        print(f"\n RMSE on Adaptive Elastic Net forecast: {rmse:.2f}")
    else:
        print("No forecasts generated.")
