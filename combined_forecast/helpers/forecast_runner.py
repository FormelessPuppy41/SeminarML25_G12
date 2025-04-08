"""
Forecast combiner module
"""
# imports:
import pandas as pd
from typing import List, Dict, Any
from combined_forecast.methods import run_day_ahead_simple_average, run_day_ahead_adaptive_elastic_net, run_day_ahead_lasso, run_day_ahead_ridge, run_day_ahead_elastic_net, run_day_ahead_xgboost

class ForecastRunner:
    """
    Forecast runner class. Used to run the forecasting methods.

    Input:
        - df: pandas DataFrame
        - #flag_matrix_df: pandas DataFrame with the flag matrix
        - target: target column name
        - features: list of feature column names
        - forecast_horizon: forecast horizon
        - rolling_window_days: rolling window days
        - datetime_col: datetime column name
        - freq: frequency
        - ! params: parameters for the model are specified in the methods!

    Methods: 
        - run_ridge: run ridge regression model, returns testing data with predictions
        - run_lasso: run lasso regression model, returns testing data with predictions
        - run_elastic_net: run elastic net regression model, returns testing data with predictions
        - run_adaptive_elastic_net: run adaptive elastic net regression, returns testing data with predictions
    """
    def __init__(
            self, 
            df: pd.DataFrame, 
            #flag_matrix_df: pd.DataFrame,
            target: str, 
            features: List[str],
            forecast_horizon: int = 96,
            rolling_window_days: int = 165,
            datetime_col: str = 'datetime',
            freq: str = '15min'
        ):
        self._df = df
        #self._flag_matrix_df = flag_matrix_df
        self._target = target
        self._features = features
        self._forecast_horizon = forecast_horizon
        self._rolling_window_days = rolling_window_days
        self._datetime_col = datetime_col
        self._freq = freq

    def run_simple_average(self):
        """
        This function runs the simple average model.

        Returns:
            pd.DataFrame: The testing data with the predictions.
        """
        return run_day_ahead_simple_average(
            df=self._df, 
            target_column=self._target,  
            forecast_horizon=self._forecast_horizon,
            features=self._features,
            datetime_col=self._datetime_col,
            freq=self._freq
        )
    
    def run_ridge(self, input_params: Dict[str, Any]):
        """
        This function runs the ridge regression model.

        Args:
            input_params (Dict[str, Any]): The parameters for the model. 
                - alpha: regularization strength (default 1.0)

        Returns:
            pd.DataFrame: The testing data with the predictions.
        """
        params = {}
        params['alpha'] = input_params.get('alpha', 1.0)

        return run_day_ahead_ridge(
            df=self._df, 
            #flag_matrix_df=self._flag_matrix_df,
            target_column=self._target, 
            feature_columns=self._features, 
            forecast_horizon=self._forecast_horizon,
            rolling_window_days=self._rolling_window_days,
            ridge_params=params,
            datetime_col=self._datetime_col,
            freq=self._freq
        )

    def run_lasso (self, input_params: Dict[str, Any]):
        """
        This function runs the lasso regression model.

        Args:
            input_params (Dict[str, Any]): The parameters for the model.
                - alpha: regularization strength (default 1.0)

        Returns:
            pd.DataFrame: The testing data with the predictions.
        """
        params = {}
        params['alpha'] = input_params.get('alpha', 1.0)

        return run_day_ahead_lasso(
            df=self._df, 
            #flag_matrix_df=self._flag_matrix_df,
            target_column=self._target, 
            feature_columns=self._features, 
            forecast_horizon=self._forecast_horizon,
            rolling_window_days=self._rolling_window_days,
            lasso_params=params,
            datetime_col=self._datetime_col,
            freq=self._freq
        )
    
    def run_elastic_net(self, input_params: Dict[str, Any]):
        """
        This function runs the elastic net regression model.

        Args:
            input_params (Dict[str, Any]): The parameters for the model. 
                - alpha: regularization strength (default 1.0)
                - l1_ratio: L1 ratio (default 0.5)

        Returns:
            pd.DataFrame: The testing data with the predictions.
        """
        params = {}
        params['alpha'] = input_params.get('alpha', 1.0)
        params['l1_ratio'] = input_params.get('l1_ratio', 0.5)

        return run_day_ahead_elastic_net(
            df=self._df,
            #flag_matrix_df=self._flag_matrix_df,
            target_column=self._target,
            feature_columns=self._features,
            forecast_horizon=self._forecast_horizon,
            rolling_window_days=self._rolling_window_days,
            enet_params=params,
            datetime_col=self._datetime_col,
            freq=self._freq
        )
    
    def run_adaptive_elastic_net(self, input_params: Dict[str, Any]):
        """
        This function runs the adaptive elastic net regression model.

        Args:
            input_params (Dict[str, Any]): The parameters for the model.
                - elasticnet__alpha: regularization strength (default [0.1, 1.0, 10.0])
                - elasticnet__l1_ratio: L1 ratio (default [0.0, 0.5, 1.0])
                - cv: cross-validation folds (default 5)
                - n_jobs: number of jobs to run in parallel (default -1)
                - verbose: verbosity level (default 0)

        Returns:
            pd.DataFrame: The testing data with the predictions.
        """
        params = {}
        params['elasticnet__alpha'] = input_params.get('elasticnet__alpha', [0.1, 1.0, 10.0])
        params['elasticnet__l1_ratio'] = input_params.get('elasticnet__l1_ratio', [0.0, 0.5, 1.0])
        params['cv'] = input_params.get('cv', 5)
        params['n_jobs'] = input_params.get('n_jobs', -1)
        params['verbose'] = input_params.get('verbose', 0)

        return run_day_ahead_adaptive_elastic_net(
            df=self._df,
            target_column=self._target,
            feature_columns=self._features,
            forecast_horizon=self._forecast_horizon,
            rolling_window_days=self._rolling_window_days,
            datetime_col=self._datetime_col,
            freq=self._freq,
            enet_params=params
        )
    
    def run_xgboost(self, input_params: Dict[str, Any]):
        """
        This function runs the xgboost regression model.

        Args:
            input_params (Dict[str, Any]): The parameters for the model.
                - n_estimators: number of trees (default 100)
                - max_depth: maximum depth of trees (default 3)
                - learning_rate: learning rate (default 0.1)
                - random_state: random state (default 42)
                - objective: objective function (default 'reg:squarederror')

        Returns:
            pd.DataFrame: The testing data with the predictions.
        """
        params = {}
        params['n_estimators'] = input_params.get('n_estimators', 100)
        params['max_depth'] = input_params.get('max_depth', 3)
        params['learning_rate'] = input_params.get('learning_rate', 0.1)
        params['random_state'] = input_params.get('random_state', 42)
        params['objective'] = input_params.get('objective', 'reg:squarederror')

        return run_day_ahead_xgboost(
            df=self._df,
            #flag_matrix_df=self._flag_matrix_df,
            target_column=self._target,
            feature_columns=self._features,
            forecast_horizon=self._forecast_horizon,
            rolling_window_days=self._rolling_window_days,
            xgb_params=params,
            datetime_col=self._datetime_col,
            freq=self._freq
        )
    

    