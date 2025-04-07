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

from configuration import ModelSettings, FileNames

from initial_forecasting.forecasters import (
    run_forecast1,
    run_forecast2,
    run_forecast3,
    run_forecast4,
    run_forecast5
)


file_names = FileNames()


def _combine_forecasts(df: pd.DataFrame, forecast_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Combines multiple forecast DataFrames into a single DataFrame with the structure:
        datetime | HR | A1 | A2 | ... | An

    Args:
        df (pd.DataFrame): Original DataFrame with ModelSettings.datetime_col and ModelSettings.target
        forecast_dfs (list[pd.DataFrame]): List of forecast DataFrames,
                                           each with ModelSettings.datetime_col and 'forecasted_value'

    Returns:
        pd.DataFrame: Combined DataFrame with forecasts as columns A1, A2, ..., An
    """
    combined = df[[ModelSettings.datetime_col, ModelSettings.target, 'K']].copy()

    for i, forecast_df in enumerate(forecast_dfs):
        forecast_df_renamed = forecast_df.rename(columns={'forecasted_value': f'A{i+1}'})
        combined = combined.merge(forecast_df_renamed, on=ModelSettings.datetime_col, how='left')

    return combined


def run_combined_solar_data():
    df = DataLoader().load_input_data(file_names.input_files.solar_combined_data)
    df[ModelSettings.datetime_col] = pd.to_datetime(df['Zeit'])
    df.drop(columns=['Zeit'], inplace=True)
    #print(df)

    #df = df[df[ModelSettings.datetime_col].dt.year >= 2010]

    np.random.seed(42)  # For reproducibility

    df1 = run_forecast1(df)
    #print(df1)
    df2 = run_forecast2(df, alpha=0.95, beta=1.05, mu=1, sigma=0.15)
    #print(df2)
    df3 = run_forecast3(df, gamma=0.6, alpha=0.98, beta=1.02)
    #print(df3)
    df4 = run_forecast4(df, alpha=0.4, beta=0.25, t_peak=172, sigma=300)
    #print(df4)
    df5 = run_forecast5(df, alpha=4.0, beta=0.25, gamma=0.99, delta=1.01)
    #print(df5)
    df6 = run_forecast4(df, alpha=2, beta=1, t_peak=344, sigma=300)

    combined_forecast = _combine_forecasts(df, [df1, df2, df3, df4, df5, df6])
    #print(combined_forecast)

    combined_forecast.to_csv('data/data_files/input_files/combined_forecasts.csv', index=False)
    evaluate_and_plot_forecasts("combined_forecast321.csv")


#TODO: Move this to the correct place.
def evaluate_and_plot_forecasts(filepath: str):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_squared_error

    # --- Load and preprocess data ---
    def load_data(filepath):
        df = pd.read_csv(filepath)
        df[ModelSettings.datetime_col] = pd.to_datetime(df[ModelSettings.datetime_col])
        df['year'] = df[ModelSettings.datetime_col].dt.year
        df.dropna(inplace=True)
        return df

    # --- Print overall and per-year MSE & Bias ---
    def print_error_metrics(df, forecast_cols):
        print("Overall Mean Squared Error and Average Bias per Forecast:")
        for col in forecast_cols:
            mse = mean_squared_error(df[ModelSettings.target], df[col])
            bias = np.mean(df[col] - df[ModelSettings.target])
            print(f"{col}: MSE = {mse:.2f}, Avg Bias = {bias:.2f}")

        print("\nMSE and Avg Bias per Forecast per Year:")
        for year in sorted(df['year'].unique()):
            df_year = df[df['year'] == year]
            print(f"\nYear: {year}")
            for col in forecast_cols:
                mse_year = mean_squared_error(df_year[ModelSettings.target], df_year[col])
                bias_year = np.mean(df_year[col] - df_year[ModelSettings.target])
                print(f"  {col}: MSE = {mse_year:.2f}, Avg Bias = {bias_year:.2f}")

    # --- Plot actual HR and forecasts per year ---
    def plot_forecasts_per_year(df, forecast_cols):
        for year in sorted(df['year'].unique()):
            yearly_df = df[df['year'] == year]
            plt.figure(figsize=(14, 6))
            plt.title(f'Forecasts vs HR in {year}', fontsize=16)

            max_val = max(yearly_df[[ModelSettings.target] + forecast_cols].max())
            y_limit = int(((max_val // 1000) + 1) * 1000)
            plt.ylim(0, y_limit)

            plt.plot(yearly_df[ModelSettings.datetime_col], yearly_df[ModelSettings.target], label='HR (Actual)', linewidth=2, color='black')
            for col in forecast_cols:
                plt.plot(yearly_df[ModelSettings.datetime_col], yearly_df[col], label=col, alpha=0.7)

            plt.xlabel('Time')
            plt.ylabel('Yield (HR)')
            plt.legend()
            plt.tight_layout()
            plt.grid(True)
            plt.show()

    # --- Correlation heatmap of forecast columns ---
    def plot_correlation_heatmap(df, forecast_cols):
        corr = df[forecast_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5, cbar_kws={"shrink": 0.75})
        plt.title("Correlation between forecast methods", fontsize=14)
        plt.tight_layout()
        plt.show()

    # --- MSE per forecast per year bar plot ---
    def plot_mse_per_year(df, forecast_cols):
        mse_data = []
        for year in sorted(df['year'].unique()):
            df_year = df[df['year'] == year]
            for col in forecast_cols:
                mse = mean_squared_error(df_year[ModelSettings.target], df_year[col])
                mse_data.append({'Year': year, 'Forecast': col, 'MSE': mse})
        mse_df = pd.DataFrame(mse_data)
        print(mse_df[mse_df['Forecast'] == 'K'])

        plt.figure(figsize=(12, 6))
        sns.barplot(data=mse_df, x='Year', y='MSE', hue='Forecast', palette='tab10')
        plt.title('Forecast MSE by Year', fontsize=16)
        plt.ylabel('Mean Squared Error')
        plt.xlabel('Year')
        plt.legend(title='Forecast', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()

    # --- Main workflow ---
    df = load_data(filepath)
    forecast_cols = [col for col in df.columns if col.startswith('A')]
    forecast_cols = forecast_cols + ['K']

    print_error_metrics(df, forecast_cols)
    plot_forecasts_per_year(df, forecast_cols)
    plot_correlation_heatmap(df, forecast_cols)
    plot_mse_per_year(df, forecast_cols)



if __name__ == "__main__":
    run_combined_solar_data()

