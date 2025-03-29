import pandas as pd

from data.data_loader import DataLoader
from data.data_processing import explore_data
from configuration import FileNames


file_names = FileNames()
data_loader = DataLoader()

##
## TO RUN THIS FILE:
## - OPEN THE TERMINAL (ctrl + ` [see the left bottom corner of your keyboard])
## - WRITE: python -m kaggle.forecasts.boris [or the name of your file]
## - PRESS ENTER TO RUN
## - RESULTS APPEAR IN THE TERMINAL
##


def forecaster(energy_data: pd.DataFrame, weather_data: pd.DataFrame):
    """
    WRITE THE EXPLANATION HERE FOR YOUR SPECIFIC FORECASTER. 

    Args:
        energy_data (pd.DataFrame): df containing energy data
        weather_data (pd.DataFrame): df containing weather data

    """
    # Write your forecasting code here
    raise NotImplementedError("This function is not implemented yet.")

if __name__ == "__main__":
    energy_df = data_loader.load_kaggle_data(file_names.kaggle_files.energy_data_file)
    weather_df = data_loader.load_kaggle_data(file_names.kaggle_files.weather_data_file)

    # COMMENT THIS OUT IF YOU NO LONGER WANT TO EXPLORE THE DATA
    # Explore the data 
    explore_data(energy_df, "Energy Data")
    explore_data(weather_df, "Weather Data")

    # Run the forecaster
    forecaster(energy_df, weather_df)