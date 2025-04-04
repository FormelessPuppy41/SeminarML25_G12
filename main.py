#
# This file is the main entry point for the application. 
# 
# It is responsible for: 
#   - setting up the data 
#   - running the forecasting methods
#   - running the evaluation methods 
#
import pandas as pd

from combined_forecast.forecast_controller import ForecastController
from combined_forecast.forecast_result_controller import ForecastResultController
from data.data_loader import DataLoader

from configuration import ModelSettings, FileNames

from forecasting.forecasters import (
    run_forecast1,
    run_forecast2,
    run_forecast3,
    run_forecast4,
    run_forecast5
)

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

def combine_forecasts(df: pd.DataFrame, forecast_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Combines multiple forecast DataFrames into a single DataFrame with the structure:
        datetime | HR | A1 | A2 | ... | An

    Args:
        df (pd.DataFrame): Original DataFrame with 'datetime' and 'HR'
        forecast_dfs (list[pd.DataFrame]): List of forecast DataFrames,
                                           each with 'datetime' and 'forecasted_value'

    Returns:
        pd.DataFrame: Combined DataFrame with forecasts as columns A1, A2, ..., An
    """
    combined = df[['datetime', 'HR']].copy()

    for i, forecast_df in enumerate(forecast_dfs):
        forecast_df_renamed = forecast_df.rename(columns={'forecasted_value': f'A{i+1}'})
        combined = combined.merge(forecast_df_renamed, on='datetime', how='left')

    return combined

def run_combined_solar_data():
    df = pd.read_csv('SolarDataCombined.csv', sep=';', encoding='utf-8', engine='python')
    df['datetime'] = pd.to_datetime(df['Zeit'])
    df.drop(columns=['Zeit'], inplace=True)
    print(df)

    df1 = run_forecast1(df)
    print(df1)
    df2 = run_forecast2(df)
    print(df2)
    df3 = run_forecast3(df)
    print(df3)
    df4 = run_forecast4(df)
    print(df4)
    df5 = run_forecast5(df)
    print(df5)

    combined_forecast = combine_forecasts(df, [df1, df2, df3, df4, df5])
    print(combined_forecast)

    combined_forecast.to_csv('combined_forecast321.csv', index=False)



def run_results(file_name: str):
    forecast_result_processor = ForecastResultController()
    forecast_result_processor.compute_metrics(file_name)

if __name__ == "__main__":
    #run_models()
    #run_results(file_names.model_result_files.ridge_forecast)
    run_combined_solar_data()

