#
# This file is the main entry point for the application. 
# 
# It is responsible for: 
#   - setting up the data 
#   - running the forecasting methods
#   - running the evaluation methods 
#

from forecast_writer.forecast_writer import ForecastWriter
from combined_forecast.forecast_combiner import ForecastCombiner
from data.data_loader import DataLoader


