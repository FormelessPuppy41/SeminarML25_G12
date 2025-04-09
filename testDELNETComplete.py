import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import os

# ===================== CONFIGURATION =====================
INPUT_FILE = r"C:\Users\jfdou\Downloads\combined_forecast321 (2).csv"  # Updated input file path

datetime_col = "Zeit"
forecast_columns = ["A1", "A2", "A3", "A4", "A5", "A6"]
target_col = "HR"
baseline_col = "K"  # column used for baseline forecast

rolling_window_days = 165
points_per_day = 96
forecast_horizon = points_per_day
window_length = rolling_window_days * points_per_day
step = points_per_day  # slide one day forward

# Exclude alpha=0 by starting at 0.1; create grid from 0.1 to 1 with 10 values
alpha_grid = np.linspace(0.1, 1, 10)
enet_l1_ratio = 0.5
random_state = 42
# =========================================================

def get_daily_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure the index is datetime, then create a 'date' column with plain Python date objects.
    df.index = pd.to_datetime(df.index)
    df['date'] = df.index.to_series().dt.normalize().dt.date
    unique_dates = df['date'].unique()
    day_map = {d: i for i, d in enumerate(sorted(unique_dates))}
    df['day_idx'] = df['date'].map(day_map)
    return df

def main():
    print("=== Forecasting Script Started ===")

    # 1) Load raw forecast data
    print(f"Loading input data from {INPUT_FILE}...")
    data_df = pd.read_csv(INPUT_FILE, parse_dates=[datetime_col])
    data_df = data_df.sort_values(datetime_col).set_index(datetime_col)
    data_df = get_daily_data(data_df)

    total_observations = len(data_df)
    print(f"Total observations in dataset: {total_observations}")

    forecast_results = []
    mse_elnet_list = []
    rmse_elnet_list = []
    mse_K_list = []
    rmse_K_list = []

    # 2) Rolling forecast loop
    for start in range(window_length, total_observations - forecast_horizon + 1, step):
        train_window = data_df.iloc[start - window_length: start]
        test_window = data_df.iloc[start: start + forecast_horizon]

        if len(train_window) != window_length or len(test_window) != forecast_horizon:
            print(f"Skipping index {start}: Incomplete window.")
            continue

        # Drop rows with missing target values in both training and test windows
        train_window = train_window.dropna(subset=[target_col])
        test_window = test_window.dropna(subset=[target_col])
        if train_window.empty or test_window.empty:
            print(f"Skipping index {start}: No valid target data after dropping missing HR values.")
            continue

        forecast_date = test_window['date'].iloc[0]
        print(f"\n--- Forecasting for date: {forecast_date} (index {start}) ---")

        # Use the forecast columns directly from the input file
        cleaned_train = train_window[forecast_columns].copy()
        cleaned_test = test_window[forecast_columns].copy()

        # Set target variables
        y_train = train_window[target_col].copy()
        y_test = test_window[target_col].copy()

        # Set up TimeSeriesSplit for time series cross validation
        tscv = TimeSeriesSplit(n_splits=5)

        # Grid search for best alpha with ElasticNet using TimeSeriesSplit
        base_model = make_pipeline(
            StandardScaler(),
            ElasticNet(l1_ratio=enet_l1_ratio, max_iter=10000, random_state=random_state)
        )
        param_grid = {'elasticnet__alpha': alpha_grid}
        print("Performing grid search for alpha with TimeSeriesSplit...")
        grid_search = GridSearchCV(base_model, param_grid, scoring='neg_mean_squared_error', cv=tscv)
        grid_search.fit(cleaned_train, y_train)
        best_model = grid_search.best_estimator_
        best_alpha = grid_search.best_params_['elasticnet__alpha']
        print(f"Best alpha found: {best_alpha}")

        # Predict and evaluate ElasticNet forecast
        y_pred = best_model.predict(cleaned_test)
        mse_elnet = mean_squared_error(y_test, y_pred)
        rmse_elnet = np.sqrt(mse_elnet)
        mse_elnet_list.append(mse_elnet)
        rmse_elnet_list.append(rmse_elnet)
        print(f"ElasticNet forecast for {forecast_date}: MSE = {mse_elnet:.2f}, RMSE = {rmse_elnet:.2f}")

        # Evaluate baseline forecast using column 'K' (if available)
        if baseline_col in test_window.columns:
            y_baseline = test_window[baseline_col]
            mse_K = mean_squared_error(y_test, y_baseline)
            rmse_K = np.sqrt(mse_K)
            mse_K_list.append(mse_K)
            rmse_K_list.append(rmse_K)
            print(f"Baseline '{baseline_col}' forecast for {forecast_date}: MSE = {mse_K:.2f}, RMSE = {rmse_K:.2f}")
        else:
            print(f"Column '{baseline_col}' not found in the test window.")
            mse_K = None
            rmse_K = None

        forecast_results.append({
            'forecast_date': forecast_date,
            'mse_elnet': mse_elnet,
            'rmse_elnet': rmse_elnet,
            'mse_K': mse_K,
            'rmse_K': rmse_K,
            'y_pred': y_pred.tolist(),
            'y_actual': y_test.tolist(),
            'best_alpha': best_alpha
        })

    # 3) Save results
    results_df = pd.DataFrame(forecast_results)
    results_df.to_csv("forecast_results.csv", index=False)
    print("\n✅ Forecast results saved to 'forecast_results.csv'.")

    # Compute overall ElasticNet metrics regardless of baseline
    if mse_elnet_list:
        overall_mse_elnet = np.mean(mse_elnet_list)
        overall_rmse_elnet = np.mean(rmse_elnet_list)
        print(f"\n📈 Overall ElasticNet forecast: Average MSE = {overall_mse_elnet:.2f}, Average RMSE = {overall_rmse_elnet:.2f}")
    else:
        print("❌ No ElasticNet forecasts were computed.")

    # Compute overall baseline metrics if available.
    if mse_K_list:
        overall_mse_K = np.mean(mse_K_list)
        overall_rmse_K = np.mean(rmse_K_list)
        print(f"\n📈 Overall Baseline ('{baseline_col}') forecast: Average MSE = {overall_mse_K:.2f}, Average RMSE = {overall_rmse_K:.2f}")
    else:
        print("ℹ️ Baseline forecasts were not computed (column 'K' missing).")

    # --- Compute yearly metrics ---
    if not results_df.empty:
        results_df['forecast_date'] = pd.to_datetime(results_df['forecast_date'])
        results_df['year'] = results_df['forecast_date'].dt.year
        yearly_metrics_elnet = results_df.groupby('year').agg({'mse_elnet': 'mean', 'rmse_elnet': 'mean'}).reset_index()

        # --- Compute RMSE as a percentage of the total installed capacity per year ---
        installed_capacity = {
            2012: 1042563,
            2013: 2181227,
            2014: 2328977,
            2015: 2315274,
            2016: 2428164,
            2017: 2875506,
            2018: 2975598
        }
        yearly_metrics_elnet['installed_capacity'] = yearly_metrics_elnet['year'].map(installed_capacity)
        yearly_metrics_elnet['rmse_elnet_pct'] = (yearly_metrics_elnet['rmse_elnet'] / yearly_metrics_elnet['installed_capacity']) * 100

        print("\n📊 Yearly ElasticNet Metrics:")
        print(yearly_metrics_elnet[['year', 'mse_elnet', 'rmse_elnet', 'installed_capacity', 'rmse_elnet_pct']])

        # --- Build and print a summary DataFrame for overall metrics ---
        overall_metrics = {
            'Overall_MSE': overall_mse_elnet,
            'Overall_RMSE': overall_rmse_elnet
        }
        overall_df = pd.DataFrame([overall_metrics])
        print("\n📊 Overall ElasticNet Metrics (full sample):")
        print(overall_df)
    else:
        print("❌ No forecast results to compute yearly metrics.")

if __name__ == '__main__':
    main()
