"""
Forecast combiner module
"""
# imports:
import pandas as pd
from typing import List, Dict, Any, Callable
import time

from configuration import ModelParameters, FileNames
from combined_forecast.helpers.forecast_runner import ForecastRunner
from combined_forecast.helpers.forecast_writer import ForecastWriter

class ForecastController:
    """
    Forecast controller class. Used to forecast using the different methods.

    Input:
        - df: pandas DataFrame
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
            target: str, 
            features: List[str],
            forecast_horizon: int = 96,
            rolling_window_days: int = 165,
            datetime_col: str = 'datetime',
            freq: str = '15min'
        ):
        # Initialize input parameters
        self._df = df
        self._target = target
        self._features = features
        self._forecast_horizon = forecast_horizon
        self._rolling_window_days = rolling_window_days
        self._datetime_col = datetime_col
        self._freq = freq

        # Initialize model parameters
        self._model_parameters = ModelParameters()

        # Initialize forecast runner
        self._forecast_runner = ForecastRunner(
            df=self._df,
            target=self._target,
            features=self._features,
            forecast_horizon=self._forecast_horizon,
            rolling_window_days=self._rolling_window_days,
            datetime_col=self._datetime_col,
            freq=self._freq
        )

        # Initialize forecast writer
        self._forecast_writer = ForecastWriter()

        # Initialize file names
        self._file_names = FileNames().model_result_files

    def forecast_simple_average(self):
        """
        Run the simple average model and write the forecast to a CSV file.
        """
        simple_average_result, _ = self._time_forecaster(
                lambda: self._forecast_runner.run_simple_average(),
                forecast_name='Simple Average'
            )
        self._forecast_writer.write_forecast(forecast=simple_average_result, file_name=self._file_names.simple_average_forecast)

    def forecast_ridge(self):
        """
        Run the ridge regression model and write the forecast to a CSV file.
        """
        ridge_params = self._model_parameters.ridge_params
        ridge_result, _ = self._time_forecaster(
                lambda: self._forecast_runner.run_ridge(input_params=ridge_params),
                forecast_name='Ridge'
            )
        self._forecast_writer.write_forecast(forecast=ridge_result, file_name=self._file_names.ridge_forecast)

    
    def forecast_lasso(self):
        """
        Run the lasso regression model and write the forecast to a CSV file.
        """
        lasso_params = self._model_parameters.lasso_params
        lasso_result, _ = self._time_forecaster(
                lambda: self._forecast_runner.run_lasso(input_params=lasso_params),
                forecast_name='Lasso'
            )
        self._forecast_writer.write_forecast(forecast=lasso_result, file_name=self._file_names.lasso_forecast)

    
    def forecast_elastic_net(self):
        """
        Run the elastic net regression model and write the forecast to a CSV file
        """
        elastic_net_params = self._model_parameters.elastic_net_params
        elastic_net_result, _ = self._time_forecaster(
                lambda: self._forecast_runner.run_elastic_net(input_params=elastic_net_params),
                forecast_name='Elastic Net'
            )
        self._forecast_writer.write_forecast(forecast=elastic_net_result, file_name=self._file_names.elastic_net_forecast)

    
    def forecast_adaptive_elastic_net(self):
        """
        Run the adaptive elastic net regression model and write the forecast to a CSV file
        """
        adaptive_elastic_net_params = self._model_parameters.adaptive_elastic_net_params
        adaptive_elastic_net_result, _ = self._time_forecaster(
                lambda: self._forecast_runner.run_adaptive_elastic_net(input_params=adaptive_elastic_net_params),
                forecast_name='Adaptive Elastic Net'
            )
        self._forecast_writer.write_forecast(forecast=adaptive_elastic_net_result, file_name=self._file_names.adaptive_elastic_net_forecast)

    def forecast_xgboost(self):
        """
        Run the XGBoost regression model and write the forecast to a CSV file
        """
        xgboost_params = self._model_parameters.xgboost_params
        xgboost_result, _ = self._time_forecaster(
                lambda: self._forecast_runner.run_xgboost(input_params=xgboost_params),
                forecast_name='XGBoost'
            )
        self._forecast_writer.write_forecast(forecast=xgboost_result, file_name=self._file_names.xgboost_forecast)

    def _time_forecaster(self, forecast_func: Callable, forecast_name: str) -> pd.DataFrame:
        """
        Time the execution of a forecast method.

        Args:
            forecast_func (Callable): The forecasting method to execute.
            forecast_name (str): The name of the forecast

        Returns:
            pd.DataFrame: Forecast results with elapsed time printed.
        """
        print(f"Start running forecast {forecast_name}...")
        start_time = time.time()
        results = forecast_func()
        elapsed_time = time.time() - start_time
        print(f"\u23F1 Forecast ({forecast_name}) completed in {elapsed_time:.2f} seconds.\n\n")
        return results, elapsed_time