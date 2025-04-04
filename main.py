#
# This file is the main entry point for the application. 
# 
# It is responsible for: 
#   - setting up the data 
#   - running the forecasting methods
#   - running the evaluation methods 
#

from combined_forecast.forecast_controller import ForecastController
from combined_forecast.forecast_result_controller import ForecastResultController
from data.data_loader import DataLoader

from configuration import ModelSettings, FileNames

file_names = FileNames()

def run_models():
    model_settings = ModelSettings()

    flag_matrix_df = DataLoader().load_kaggle_data(file_names.kaggle_files.flag_matrix_file)
    flag_matrix_df.rename(columns={'date': 'datetime'}, inplace=True)
    #df = DataLoader().load_kaggle_data(file_names.kaggle_files.combined_forecasts_file)
    df = DataLoader().load_input_data(file_names.input_files.forecast_data)
    print(df.head())

    forecast_controller = ForecastController(
            df=df, 
            flag_matrix_df=flag_matrix_df,
            target=model_settings.target, 
            features=model_settings.features,
            forecast_horizon=model_settings.forecast_horizon,
            rolling_window_days=model_settings.rolling_window_days,
            datetime_col=model_settings.datetime_col,
            freq=model_settings.freq
        )
    
    forecast_controller.forecast_ridge()


def run_results(file_name: str):
    forecast_result_processor = ForecastResultController()
    forecast_result_processor.compute_metrics(file_name)

if __name__ == "__main__":
    run_models()
    run_results(file_names.model_result_files.ridge_forecast)

