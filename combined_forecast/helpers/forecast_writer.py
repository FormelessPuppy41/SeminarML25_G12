"""
Forecast Writer
"""
# imports:
import pandas as pd
from typing import List, Dict, Any


class ForecastWriter:
    """
    Forecast writer class. This class writes the forecast to a CSV file in the project.
    """
    def __init__(self):
        self.path = 'data/data_files/model_results'

    def write_forecast(
            self, 
            forecast: pd.DataFrame, 
            file_name: str
        ):
        """
        This function writes the forecast to a CSV file. It overwrites the file if it already exists.

        Args:
            forecast (pd.DataFrame): The forecast data.
            file_name (str): The file_name for the data to write the forecast to 'data/data_files/model_results'.
        """
        path = f'{self.path}/{file_name}'
        forecast.to_csv(path, index=False)