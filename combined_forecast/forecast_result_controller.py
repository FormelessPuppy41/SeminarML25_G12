"""
This module is used to control the forecast results.
"""
#imports:
from icecream import ic

from combined_forecast.helpers.forecast_result_processor import ForecastResultProcessor
from data.data_loader import DataLoader
from configuration import FileNames

file_names = FileNames()

import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

def compute_yearly_rmse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute yearly RMSE between predicted and actual values.

    Parameters:
        df (pd.DataFrame): Must contain 'target_time', 'prediction', and 'actual'.

    Returns:
        pd.DataFrame: Yearly RMSE and number of data points.
    """
    df = df.copy()
    df['year'] = pd.to_datetime(df['target_time']).dt.year

    results = (
        df.groupby('year')
        .apply(lambda group: pd.Series({
            'RMSE': np.sqrt(mean_squared_error(group['actual'], group['prediction'])),
            'N': len(group)
        }))
        .reset_index()
    )

    return results

class ForecastResultController:
    """
    Forecast result controller. Used to control forecast results.
    """
    def __init__(self):
        pass

    def compute_metrics(self, file_name: str = file_names.model_result_files.ridge_forecast):
        """
        This method is used to compute the metrics for the forecast results.
        """
        result_df = DataLoader().load_model_results(file_name)
        ic(result_df.head())
        print(compute_yearly_rmse(result_df))
        forecast_result_processor = ForecastResultProcessor(result_df)
        ic(forecast_result_processor.compute_metrics())

    def visualise_ridge(self):
        """
        This method is used to visualise the results of the ridge forecast.
        """
        result_df = DataLoader().load_model_results(file_names.model_result_files.ridge_forecast)
        ic(result_df.head())
        forecast_result_processor = ForecastResultProcessor(result_df)

        forecast_result_processor.plot_forecast_vs_actual()
        forecast_result_processor.plot_error_over_time()
        forecast_result_processor.plot_error_distribution()
        forecast_result_processor.plot_mae_by_hour()

    def visualise_lasso(self):
        """
        This method is used to visualise the results of the lasso forecast.
        """
        result_df = DataLoader().load_model_results(file_names.model_result_files.lasso_forecast)
        ic(result_df.head())
        forecast_result_processor = ForecastResultProcessor(result_df)

        forecast_result_processor.plot_forecast_vs_actual()
        forecast_result_processor.plot_error_over_time()
        forecast_result_processor.plot_error_distribution()
        forecast_result_processor.plot_mae_by_hour()

    def visualise_elastic_net(self):
        """
        This method is used to visualise the results of the elastic net forecast.
        """
        result_df = DataLoader().load_model_results(file_names.model_result_files.elastic_net_forecast)
        ic(result_df.head())
        forecast_result_processor = ForecastResultProcessor(result_df)

        forecast_result_processor.plot_forecast_vs_actual()
        forecast_result_processor.plot_error_over_time()
        forecast_result_processor.plot_error_distribution()
        forecast_result_processor.plot_mae_by_hour()

    def visualise_adaptive_elnet(self):
        """
        This method is used to visualise the results of the adaptive elastic net forecast.
        """
        result_df = DataLoader().load_model_results(file_names.model_result_files.adaptive_elastic_net_forecast)
        ic(result_df.head())
        forecast_result_processor = ForecastResultProcessor(result_df)

        forecast_result_processor.plot_forecast_vs_actual()
        forecast_result_processor.plot_error_over_time()
        forecast_result_processor.plot_error_distribution()
        forecast_result_processor.plot_mae_by_hour()
        forecast_result_processor.analyze_best_params()



