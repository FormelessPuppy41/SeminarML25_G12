#
# This file is the main entry point for the application. 
# 
# It is responsible for: 
#   - setting up the data 
#   - running the forecasting methods
#   - running the evaluation methods 
#
import pandas as pd
import numpy as np

from combined_forecast.forecast_controller import ForecastController
from combined_forecast.forecast_result_controller import ForecastResultController
from data.data_loader import DataLoader

from configuration import ModelSettings, FileNames, ModelParameters

"""
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
"""



file_names = FileNames()

def run_models():
    model_settings = ModelSettings()

    df = DataLoader().load_input_data(file_names.input_files.real_error_data2)
    df[model_settings.datetime_col] = pd.to_datetime(df[model_settings.datetime_col])
    print(df)
    #df = df[df[model_settings.datetime_col] >= pd.to_datetime('01-01-2018')]
    #df = df[df[model_settings.datetime_col] < pd.to_datetime('12-31-2014')]
    #print(df)

    forecast_controller = ForecastController(
            df=df, 
            target=model_settings.target, 
            features=model_settings.features,
            forecast_horizon=model_settings.forecast_horizon, # 96 for 15min, 24 for 1H
            rolling_window_days=model_settings.rolling_window_days, # 165 own data, 30 or 61 for other paper.
            datetime_col=model_settings.datetime_col,
            freq=model_settings.freq # Change to '15min' or '1h' if needed.
        )
    
    #forecast_controller.forecast_elastic_net()
    #forecast_controller.forecast_ridge()
    #forecast_controller.forecast_lasso()
    forecast_controller.forecast_adaptive_elastic_net()



def run_results(file_name: str):
    forecast_result_processor = ForecastResultController()
    forecast_result_processor.compute_metrics(file_name)

if __name__ == "__main__":
    run_models() 
    
    print("RUNNING ELASTIC NET RESULTS")
    run_results(file_names.model_result_files.elastic_net_forecast)
    
    print("\n\nRUNNING ADAPTIVE ELASTIC NET RESULTS")
    run_results(file_names.model_result_files.adaptive_elastic_net_forecast)

    print("\n\nRUNNING LASSO RESULTS")
    run_results(file_names.model_result_files.lasso_forecast)

    print("\n\nRUNNING RIDGE RESULTS")
    run_results(file_names.model_result_files.ridge_forecast)

    print("\n\n\n!!!DO NOT FORGET TO RUN THE ELASTIC NET WITH FIXED ALPHA!!!\n\n\n")
