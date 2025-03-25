#
# This file is the main entry point for the application. 
# 
# It is responsible for: 
#   - setting up the data 
#   - running the forecasting methods
#   - running the evaluation methods 
#

from combined_forecast.forecast_controller import ForecastController
from data.data_loader import DataLoader

from configuration import ModelSettings, FileNames


def run_main():
    model_settings = ModelSettings()
    file_names = FileNames()

    df = DataLoader().load_output_data(file_names.input_files.data_file)

    forecast_controller = ForecastController(
            df=df, 
            target=model_settings.target, 
            features=model_settings.features,
            forecast_horizon=model_settings.forecast_horizon,
            rolling_window_days=model_settings.rolling_window_days,
            datetime_col=model_settings.datetime_col,
            freq=model_settings.freq
        )
    
    forecast_controller.forecast_ridge()

if __name__ == "__main__":
    run_main()