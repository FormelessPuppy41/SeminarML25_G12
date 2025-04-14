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
#from data.flagMatrix import run_flag_matrix

from configuration import ModelSettings, FileNames

"""
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
"""



file_names = FileNames()

def run_models():
    model_settings = ModelSettings()

    # flag_matrix_df = DataLoader().load_input_data(file_names.input_files.flag_matrix)
    # flag_matrix_df.rename(columns={'date': ModelSettings.datetime_col}, inplace=True)
    # print(flag_matrix_df.head())

    df = DataLoader().load_input_data(file_names.input_files.real_error_data)
    print(df)
    df[model_settings.datetime_col] = pd.to_datetime(df[model_settings.datetime_col])
    df = df[df[model_settings.datetime_col] >= pd.to_datetime('07-20-2013')]
    df = df[df[model_settings.datetime_col] < pd.to_datetime('12-31-2016')]
    #print(df.head())

    forecast_controller = ForecastController(
            df=df, 
            #flag_matrix_df=flag_matrix_df,
            target=model_settings.target, 
            features=model_settings.features,
            forecast_horizon=model_settings.forecast_horizon,
            rolling_window_days=model_settings.rolling_window_days,
            datetime_col=model_settings.datetime_col,
            freq=model_settings.freq
        )
    
    forecast_controller.forecast_adaptive_elastic_net()



def run_results(file_name: str):
    forecast_result_processor = ForecastResultController()
    forecast_result_processor.compute_metrics(file_name)

if __name__ == "__main__":
    #run_flag_matrix()
    run_models() 
    run_results(file_names.model_result_files.adaptive_elastic_net_forecast)
    #run_models()
    #run_results(file_names.model_result_files.ridge_forecast)

