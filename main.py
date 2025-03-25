#
# This file is the main entry point for the application. 
# 
# It is responsible for: 
#   - setting up the data 
#   - running the forecasting methods
#   - running the evaluation methods 
#

from configuration import model_parameters
from combined_forecast.forecast_writer import ForecastWriter
from combined_forecast.forecast_controller import ForecastController
from data.data_loader import DataLoader



fcontroller = ForecastController()
fcontroller.run_ridge()
