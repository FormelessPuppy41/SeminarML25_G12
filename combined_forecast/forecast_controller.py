"""
Forecast combiner module
"""
# imports:
import pandas as pd
from typing import List, Dict, Any, Callable
import time

from configuration import ModelParameters
from combined_forecast.forecast_runner import ForecastRunner
from combined_forecast.forecast_writer import ForecastWriter

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
        self.df = df
        self.target = target
        self.features = features
        self.forecast_horizon = forecast_horizon
        self.rolling_window_days = rolling_window_days
        self.datetime_col = datetime_col
        self.freq = freq

        # Initialize model parameters
        self.model_parameters = ModelParameters()

        # Initialize forecast runner
        self.forecast_runner = ForecastRunner(
            df=self.df,
            target=self.target,
            features=self.features,
            forecast_horizon=self.forecast_horizon,
            rolling_window_days=self.rolling_window_days,
            datetime_col=self.datetime_col,
            freq=self.freq
        )

        # Initialize forecast writer
        self.forecast_writer = ForecastWriter()


    def forecast_ridge(self):
        """
        Run the ridge regression model and write the forecast to a CSV file.
        """
        ridge_params = self.model_parameters.ridge_params
        ridge_result, _ = self._time_forecaster(
                lambda: self.forecast_runner.run_ridge(input_params=ridge_params),
                forecast_name='Ridge'
            )
        self.forecast_writer.write_forecast(forecast=ridge_result, file_name='ridge_forecast.csv')

    
    def forecast_lasso(self):
        """
        Run the lasso regression model and write the forecast to a CSV file.
        """
        lasso_params = self.model_parameters.lasso_params
        lasso_result, _ = self._time_forecaster(
                lambda: self.forecast_runner.run_lasso(input_params=lasso_params),
                forecast_name='Lasso'
            )
        self.forecast_writer.write_forecast(forecast=lasso_result, file_name='lasso_forecast.csv')

    
    def forecast_elastic_net(self):
        """
        Run the elastic net regression model and write the forecast to a CSV file
        """
        elastic_net_params = self.model_parameters.elastic_net_params
        elastic_net_result, _ = self._time_forecaster(
                lambda: self.forecast_runner.run_elastic_net(input_params=elastic_net_params),
                forecast_name='Elastic Net'
            )
        self.forecast_writer.write_forecast(forecast=elastic_net_result, file_name='elastic_net_forecast.csv')

    
    def forecast_adaptive_elastic_net(self):
        """
        Run the adaptive elastic net regression model and write the forecast to a CSV file
        """
        adaptive_elastic_net_params = self.model_parameters.adaptive_elastic_net_params
        adaptive_elastic_net_result, _ = self._time_forecaster(
                lambda: self.forecast_runner.run_adaptive_elastic_net(input_params=adaptive_elastic_net_params),
                forecast_name='Adaptive Elastic Net'
            )
        self.forecast_writer.write_forecast(forecast=adaptive_elastic_net_result, file_name='adaptive_elastic_net_forecast.csv')

    
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