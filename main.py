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

    df = DataLoader().load_output_data(file_names.output_files.sample_complete_Ty)

    forecast_controller = ForecastController(
            df=df, 
            target=model_settings.target, 
            features=model_settings.features,
            forecast_horizon=model_settings.forecast_horizon,
            rolling_window_days=model_settings.rolling_window_days,
            datetime_col=model_settings.datetime_col,
            freq=model_settings.freq
        )
    
    forecast_controller.forecast_elastic_net()


def run_results():
    forecast_result_processor = ForecastResultController()
    forecast_result_processor.visualise_elastic_net()

if __name__ == "__main__":
    run_results()

