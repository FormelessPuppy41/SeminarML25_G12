"""
Forecast result processor
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Dict, Any, Optional


class ForecastResultProcessor:
    """
    Forecast result processor. Used to process and visualize forecast results.

    Input: 
        - dataframe: contains ['target_time', 'actual', 'prediction', 'best_params' (optional)] columns
    """
    def __init__(self, dataframe: Optional[pd.DataFrame] = None):
        self.df = dataframe.copy() if dataframe is not None else None

    def set_data(self, dataframe: pd.DataFrame):
        """
        Set or update the forecast dataframe.

        Args:
            dataframe (pd.DataFrame): contains ['target_time', 'actual', 'prediction', 'best_params' (optional)] columns
        """
        self.df = dataframe.copy()

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute RMSE, MAE, and R^2 for the forecast.

        Returns:
            Dict[str, float]: dictionary of metrics. Contains keys 'RMSE', 'MSE', 'MAE', 'R2'
        """
        if self.df is None:
            raise ValueError("No data set. Use set_data() first.")

        y_true = self.df['actual']
        y_pred = self.df['prediction']

        return {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MSE': mean_squared_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }

    def plot_forecast_vs_actual(self, title: str = "Forecast vs Actual"):
        """
        Plot forecast vs actual over time.

        Args:   
            title (str): plot title. Default is "Forecast vs Actual".

        Plots:
            - Actual & Forecast values on y-axis
            - Time on x-axis

        """
        if self.df is None:
            raise ValueError("No data set. Use set_data() first.")

        plt.figure(figsize=(14, 6))
        plt.plot(self.df['target_time'], self.df['actual'], label='Actual', linewidth=2)
        plt.plot(self.df['target_time'], self.df['prediction'], label='Prediction', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_error_over_time(self):
        """
        Plot prediction error over time.

        Plots:
            - Error on y-axis
            - Time on x-axis
            - Horizontal line at y=0
        """
        if self.df is None:
            raise ValueError("No data set. Use set_data() first.")

        self.df['error'] = self.df['prediction'] - self.df['actual']

        plt.figure(figsize=(14, 4))
        plt.plot(self.df['target_time'], self.df['error'], color='red', label='Error')
        plt.axhline(0, color='black', linestyle='--')
        plt.title("Prediction Error Over Time")
        plt.xlabel("Time")
        plt.ylabel("Error")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_error_distribution(self):
        """
        Plot histogram of forecast errors.

        Plots:
            - Error on x-axis
            - Frequency on y-axis
        """
        if self.df is None:
            raise ValueError("No data set. Use set_data() first.")

        self.df['error'] = self.df['prediction'] - self.df['actual']

        plt.figure(figsize=(6, 4))
        plt.hist(self.df['error'], bins=30, edgecolor='black')
        plt.title("Histogram of Forecast Errors")
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_mae_by_hour(self):
        #TODO: Shouldn't we use 15min?
        """
        Plot MAE by hour of day to see performance across time segments.

        Plots:
            - Hour on x-axis
            - MAE on y-axis
        """
        if self.df is None:
            raise ValueError("No data set. Use set_data() first.")

        df_temp = self.df.copy()
        df_temp['hour'] = pd.to_datetime(df_temp['target_time']).dt.hour
        df_temp['abs_error'] = np.abs(df_temp['prediction'] - df_temp['actual'])

        hourly_mae = df_temp.groupby('hour')['abs_error'].mean()

        plt.figure(figsize=(10, 5))
        plt.plot(hourly_mae.index, hourly_mae.values, marker='o')
        plt.title("Mean Absolute Error by Hour of Day")
        plt.xlabel("Hour")
        plt.ylabel("MAE")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def analyze_best_params(self) -> Optional[pd.DataFrame]:
        """
        Analyze and visualize best parameters from adaptive Elastic Net results.

        Returns:
            pd.DataFrame: DataFrame containing best parameters and target time.
        """
        if self.df is None:
            raise ValueError("No data set. Use set_data() first.")

        if 'best_params' not in self.df.columns:
            print("No best_params found in data. Skipping analysis.")
            return None

        param_df = pd.json_normalize(self.df['best_params'])
        param_df['target_time'] = self.df['target_time']

        print("\n Top 5 most frequent hyperparameter combinations:\n")
        print(param_df.drop(columns='target_time').value_counts().head())

        plt.figure(figsize=(12, 5))
        if 'elasticnet__alpha' in param_df.columns:
            plt.plot(param_df['target_time'], param_df['elasticnet__alpha'], label='alpha')
        if 'elasticnet__l1_ratio' in param_df.columns:
            plt.plot(param_df['target_time'], param_df['elasticnet__l1_ratio'], label='l1_ratio')

        plt.legend()
        plt.title("Best ElasticNet Parameters Over Time")
        plt.xlabel("Target Time")
        plt.ylabel("Parameter Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return param_df
