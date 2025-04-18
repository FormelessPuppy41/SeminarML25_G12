"""
Forecast result processor
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Dict, Any, Optional

import math


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
    
    def plot_model_grid(self, list_of_dfs: list[pd.DataFrame], model_names: list[str] = None):
        """
        Plots grids of best_alpha (log scale) and best_l1_ratio for multiple models.

        Parameters:
        list_of_dfs (list of pd.DataFrame): Each DataFrame must have 'target_time', 'best_alpha', and 'best_l1_ratio'.
        model_names (list of str, optional): Names for each model; if None, defaults to "Model 1", "Model 2", etc.
        """

        num_models = len(list_of_dfs)

        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(num_models)]

        # ----------------- Plot Best Alpha -----------------
        cols = 3
        rows = math.ceil(num_models / cols)

        plt.figure(figsize=(cols*5, rows*4))
        for i, (df, name) in enumerate(zip(list_of_dfs, model_names)):
            print(f"Plotting {name} Best Alpha... {i}")
            df = df.copy()
            df['target_time'] = pd.to_datetime(df['target_time'])
            df = df.set_index('target_time')
            df_daily = df[['best_alpha']].resample('D').mean()

            plt.subplot(rows, cols, i + 1)
            plt.plot(df_daily.index, df_daily['best_alpha'], marker='o', markersize=2, linestyle='-', linewidth=1)
            plt.yscale('log')
            plt.title(f'{name} - Best Alpha', fontsize=10)
            plt.xticks(rotation=45)
            plt.tight_layout()

        plt.suptitle('Daily Best Alpha (Log Scale)', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()

        # ----------------- Plot Best L1 Ratio -----------------
        valid_l1_models = []
        valid_model_names = []

        for df, name in zip(list_of_dfs, model_names):
            df = df.copy()
            df['target_time'] = pd.to_datetime(df['target_time'])
            df = df.set_index('target_time')
            df_daily = df[['best_l1_ratio']].resample('D').mean()

            unique_values = df_daily['best_l1_ratio'].dropna().unique()
            if len(unique_values) >= 2:
                valid_l1_models.append(df_daily)
                valid_model_names.append(name)
            else:
                print(f"Skipping {name} for L1 ratio plot (only {len(unique_values)} unique value(s)).")

        if valid_l1_models:
            num_valid = len(valid_l1_models)
            cols = 3
            rows = math.ceil(num_valid / cols)

            plt.figure(figsize=(cols*5, rows*4))
            for i, (df_daily, name) in enumerate(zip(valid_l1_models, valid_model_names)):
                plt.subplot(rows, cols, i + 1)
                plt.plot(df_daily.index, df_daily['best_l1_ratio'], marker='o', markersize=2, linestyle='-', linewidth=1)
                plt.title(f'{name} - Best L1 Ratio', fontsize=10)
                plt.xticks(rotation=45)
                plt.tight_layout()

            plt.suptitle('Daily Best L1 Ratio', y=1.02, fontsize=16)
            plt.tight_layout()
            plt.show()
        else:
            print("No models with varying 'best_l1_ratio' to plot.")

    

    def plot_daily_alpha_l1(self, df: pd.DataFrame):
        """
        Aggregates best_alpha and best_l1_ratio to daily frequency
        (taking the mean value each day) and plots them, with log scale for alpha.

        Parameters:
        df (pd.DataFrame): DataFrame with 'target_time', 'best_alpha', 'best_l1_ratio'.
                            'target_time' must be datetime or convertible to datetime.
        """

        # Ensure 'target_time' is datetime and set as index
        df['target_time'] = pd.to_datetime(df['target_time'])
        df = df.set_index('target_time')

        # Keep only numeric columns
        numeric_cols = ['best_alpha', 'best_l1_ratio']
        df_numeric = df[numeric_cols]

        # Resample to daily frequency, taking mean
        df_daily = df_numeric.resample('D').mean()

        plt.style.use('seaborn-v0_8-darkgrid')

        if 'best_alpha' in df_daily.columns:
            plt.figure(figsize=(14, 6))
            plt.plot(df_daily.index, df_daily['best_alpha'], marker='o', markersize=3, linestyle='-', linewidth=1)
            plt.yscale('log')  # <<<< MAKE Y-AXIS LOG SCALE
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Best Alpha (log scale)', fontsize=12)
            plt.title('Daily Best Alpha (2012–2018)', fontsize=14)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print('No "best_alpha" present in columns of df')

        if 'best_l1_ratio' in df_daily.columns:
            plt.figure(figsize=(14, 6))
            plt.plot(df_daily.index, df_daily['best_l1_ratio'], marker='o', markersize=3, linestyle='-', linewidth=1)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Best L1 Ratio', fontsize=12)
            plt.title('Daily Best L1 Ratio (2012–2018)', fontsize=14)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print('No "best_l1_ratio" present in columns of df')


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
