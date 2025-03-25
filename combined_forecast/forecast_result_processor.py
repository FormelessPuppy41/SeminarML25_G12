"""
Forecast result processor
"""
# imports:
import pandas as pd
from typing import List, Dict, Any
from combined_forecast.forecast_controller import ForecastCombiner

class ForecastResultProcessor:
    """
    Forecast result processor. Used to process and visualize forecast results.

    Input: 
        - dataframe: contains [datetimestamp, actual, prediction] columns
     
    """
    def __init__(self):
        pass
    

    def plot_forecast_results(self, forecast_results: pd.DataFrame, target_column: str, forecast_horizon: int = 96):
        """
        Plot forecast results for a given target column.
        
        """
        pass