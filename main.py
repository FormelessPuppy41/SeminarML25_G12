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


file_names = FileNames()

def run_data_generation():
    from combined_forecast.initial_forecasting.run_forecasts_real_errors import run_error_model_forecasts, evaluate_and_plot_forecasts
    run_error_model_forecasts()
    #evaluate_and_plot_forecasts("data/data_files/input_files/data_ander_groepje.csv")

def run_models():
    model_settings = ModelSettings()

    # Either use 'real_error_data2' or 'data_different_group' for testing
    # Based on the data, you must modify the model_settings in configuration.py
    # to match the data.
    df = DataLoader().load_input_data(file_names.input_files.real_error_data2)
    df[model_settings.datetime_col] = pd.to_datetime(df[model_settings.datetime_col])

    # Uncomment the following lines to filter the data based on the datetime column
    #df = df[df[model_settings.datetime_col] >= pd.to_datetime('05-01-2018')]
    #df = df[df[model_settings.datetime_col] < pd.to_datetime('12-31-2014')]
    print(df)

    forecast_controller = ForecastController(
            df=df, 
            target=model_settings.target, 
            features=model_settings.features,
            forecast_horizon=model_settings.forecast_horizon, # 96 for 15min, 24 for 1H (hr vs price resp.)
            rolling_window_days=model_settings.rolling_window_days, 
            datetime_col=model_settings.datetime_col,
            freq=model_settings.freq # Change to '15min' or '1h' for hr and price forecasting resp.
        )
    forecast_controller.forecast_simple_average()
    # The following two methods write to different files, however, they use the same configuration input. 
    # Therefore, you cannot run them simultaneously. 
    # Please comment out one of them and adjust the elnet_params in configuration.py accordingly. 
    forecast_controller.forecast_elastic_net(bool_tune=True)
    forecast_controller.forecast_elastic_net(bool_tune=False)
    
    forecast_controller.forecast_ridge()
    forecast_controller.forecast_lasso()
    forecast_controller.forecast_adaptive_elastic_net()
    forecast_controller.forecast_xgboost()



def run_results(file_name: str):
    forecast_result_processor = ForecastResultController()
    forecast_result_processor.compute_metrics(file_name)
    # if file_name in [
    #     file_names.model_result_files.elastic_net_forecast, 
    #     file_names.model_result_files.tune_elastic_net_forecast, 
    #     file_names.model_result_files.adaptive_elastic_net_forecast
    #     ]:
    #     forecast_result_processor.visualise_alpha_l1(file_name)

    # plot grid of alpha and l1 ratio
    forecast_result_processor.visualise_alpha_l1_in_grid(
        list_file_names=[
            file_names.model_result_files.ridge_forecast, # alpha only
            file_names.model_result_files.lasso_forecast, # alpha ratio only
            file_names.model_result_files.elastic_net_forecast, # alpha only
            file_names.model_result_files.tune_elastic_net_forecast, # alpha and l1 ratio
            file_names.model_result_files.adaptive_elastic_net_forecast # alpha and l1 ratio
        ]
    )

if __name__ == "__main__":
    """run_models() 

    print("\n\nRUNNING SIMPLE AVERAGE RESULTS")
    run_results(file_names.model_result_files.simple_average_forecast)

    print("\n\nRUNNING XGBOOST RESULTS")
    run_results(file_names.model_result_files.xgboost_forecast)
    
    print("RUNNING ELASTIC NET RESULTS")
    run_results(file_names.model_result_files.elastic_net_forecast)

    print("RUNNING ELASTIC NET (TUNE) RESULTS")
    run_results(file_names.model_result_files.tune_elastic_net_forecast)
    
    print("\n\nRUNNING ADAPTIVE ELASTIC NET RESULTS")
    run_results(file_names.model_result_files.adaptive_elastic_net_forecast)

    print("\n\nRUNNING LASSO RESULTS")
    run_results(file_names.model_result_files.lasso_forecast)

    print("\n\nRUNNING RIDGE RESULTS")
    run_results(file_names.model_result_files.ridge_forecast)

    print("\n\n\n!!!DO NOT FORGET TO RUN THE ELASTIC NET WITH FIXED ALPHA!!!\n\n\n")"""

    run_data_generation()
