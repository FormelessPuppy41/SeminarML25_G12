#
# Main script for running error-based forecast models and evaluating them.
#
import pandas as pd
import numpy as np

from configuration import ModelSettings, FileNames
from data.data_loader import DataLoader
from combined_forecast.initial_forecasting.forecasts_real_errors import (
    run_forecasts1,
    run_forecasts2,
    run_forecast3,
    run_forecast4,
    run_forecast5,
    run_forecast6,
)

file_names = FileNames()


def _combine_forecasts(df: pd.DataFrame, forecast_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    combined = df[[ModelSettings.datetime_col, ModelSettings.target, 'K']].copy()
    for i, forecast_df in enumerate(forecast_dfs):
        forecast_df_renamed = forecast_df.rename(columns={'forecasted_value': f'A{i+1}'})
        combined = combined.merge(forecast_df_renamed, on=ModelSettings.datetime_col, how='left')
    return combined


def run_error_model_forecasts():
    df = DataLoader().load_input_data(file_names.input_files.solar_combined_data)
    df[ModelSettings.datetime_col] = pd.to_datetime(df['Zeit'])
    df.drop(columns=['Zeit'], inplace=True)
    df = df[df[ModelSettings.datetime_col] >= pd.to_datetime('01-01-2012')]
    np.random.seed(42)  # For reproducibility

    df1 = run_forecasts1(df)
    df2 = run_forecasts2(df)
    df3 = run_forecast3(df)
    df4 = run_forecast4(df)
    df5 = run_forecast5(df)
    df6 = run_forecast6(df)

    combined_forecast = _combine_forecasts(df, [df1, df2, df3, df4, df5, df6])
    combined_forecast.to_csv('data/data_files/input_files/error_model_combined_forecasts.csv', index=False)

    evaluate_and_plot_forecasts("data/data_files/input_files/error_model_combined_forecasts.csv")


def evaluate_and_plot_forecasts(filepath: str):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_squared_error

    def load_data(filepath):
        df = pd.read_csv(filepath)
        df[ModelSettings.datetime_col] = pd.to_datetime(df[ModelSettings.datetime_col])
        df['year'] = df[ModelSettings.datetime_col].dt.year
        df.dropna(inplace=True)
        return df

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

    def plot_correlation_heatmap(df, forecast_cols):
        corr = df[forecast_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5, cbar_kws={"shrink": 0.75})
        plt.title("Correlation between forecast methods", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_mse_per_year(df, forecast_cols):
        mse_data = []
        for year in sorted(df['year'].unique()):
            df_year = df[df['year'] == year]
            for col in forecast_cols:
                mse = mean_squared_error(df_year[ModelSettings.target], df_year[col])
                mse_data.append({'Year': year, 'Forecast': col, 'MSE': mse})
        mse_df = pd.DataFrame(mse_data)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=mse_df, x='Year', y='MSE', hue='Forecast', palette='tab10')
        plt.title('Forecast MSE by Year', fontsize=16)
        plt.ylabel('Mean Squared Error')
        plt.xlabel('Year')
        plt.legend(title='Forecast', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()

    df = load_data(filepath)
    forecast_cols = [col for col in df.columns if col.startswith('A')]
    forecast_cols = forecast_cols + ['K']

    print_error_metrics(df, forecast_cols)
    plot_forecasts_per_year(df, forecast_cols)
    plot_correlation_heatmap(df, forecast_cols)
    plot_mse_per_year(df, forecast_cols)


if __name__ == "__main__":
    run_error_model_forecasts()
