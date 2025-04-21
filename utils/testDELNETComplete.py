import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import os

# ===================== CONFIGURATION =====================
INPUT_FILE = r"C:\Users\jfdou\Downloads\NewDataBig.csv"  # Updated input file path

datetime_col = "datetime"
forecast_columns = ["A1", "A2", "A3", "A4", "A5", "A6"]
target_col = "HR"
baseline_col = "K"  # column used for baseline forecast

rolling_window_days = 165
points_per_day = 96   # Change this to 96 if you are working with 15-minute data
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

    # This list will store one entry per forecast observation (not per day)
    forecast_results = []
    mse_elnet_list = []
    rmse_elnet_list = []
    mse_K_list = []
    rmse_K_list = []

    # Initialize accumulators for the overall RMSE computation
    total_squared_error_elnet = 0
    total_count_elnet = 0
    total_squared_error_K = 0
    total_count_K = 0

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

        # The forecast_date here is the date associated with the first observation in the test window.
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

        # Retrieve and print the intercept from the model
        intercept = best_model.named_steps['elasticnet'].intercept_
        print("Model intercept (from grid search):")
        print(intercept)

        # Predict and evaluate ElasticNet forecast for the current window
        y_pred = best_model.predict(cleaned_test)
        mse_elnet = mean_squared_error(y_test, y_pred)
        rmse_elnet = np.sqrt(mse_elnet)
        mse_elnet_list.append(mse_elnet)
        rmse_elnet_list.append(rmse_elnet)
        print(f"ElasticNet forecast for {forecast_date}: MSE = {mse_elnet:.2f}, RMSE = {rmse_elnet:.2f}")

        # Update the overall accumulators for ElasticNet
        n_pred = len(y_test)
        total_squared_error_elnet += mse_elnet * n_pred  # MSE * number of predictions = total SSE for the window
        total_count_elnet += n_pred

        # Evaluate baseline forecast using column 'K' (if available)
        if baseline_col in test_window.columns:
            y_baseline = test_window[baseline_col]
            mse_K = mean_squared_error(y_test, y_baseline)
            rmse_K = np.sqrt(mse_K)
            mse_K_list.append(mse_K)
            rmse_K_list.append(rmse_K)
            print(f"Baseline '{baseline_col}' forecast for {forecast_date}: MSE = {mse_K:.2f}, RMSE = {rmse_K:.2f}")

            # Update overall accumulators for the baseline forecast
            total_squared_error_K += mse_K * n_pred
            total_count_K += n_pred
        else:
            print(f"Column '{baseline_col}' not found in the test window.")
            mse_K = None
            rmse_K = None

        # Retrieve model coefficients from the ElasticNet step in the pipeline.
        coefs = best_model.named_steps['elasticnet'].coef_

        # Instead of one row per day, loop over each observation in the forecast horizon.
        for dt, pred, actual in zip(test_window.index, y_pred, y_test):
            # Get the raw feature values for the current observation
            x_values = cleaned_test.loc[[dt]]  # keeping as DataFrame
            print(f"\n>>> Forecast details for observation at datetime: {dt}")
            print("Raw feature values (x's):")
            print(x_values)

            # Transform the input values using the scaler from the pipeline
            x_scaled = best_model.named_steps['standardscaler'].transform(x_values)
            print("Standardized feature values (x's after scaling):")
            print(x_scaled)

            # Print model coefficients and the intercept (again, per observation)
            print("Model coefficients (betas):")
            print(coefs)
            print("Model intercept:")
            print(intercept)

            # Calculate the forecast manually using the standardized inputs and the coefficients
            manual_forecast = np.dot(x_scaled, coefs) + intercept
            print("Manually computed forecast (using dot product):")
            print(manual_forecast)

            # Also print the model forecast and actual value
            print(f"Model forecast: {pred}")
            print(f"Actual value: {actual}")

            # Append forecast result for record-keeping
            forecast_results.append({
                'datetime': dt,
                'forecast_date': forecast_date,
                'y_pred': pred,
                'y_actual': actual,
                'best_alpha': best_alpha,
                'mse_elnet': mse_elnet,
                'rmse_elnet': rmse_elnet,
                'mse_K': mse_K,
                'rmse_K': rmse_K
            })

    # 3) Save results for each observation
    results_df = pd.DataFrame(forecast_results)
    # Changed the output file to ForeCastsDelnet.csv in your Downloads folder:
    output_path = r"C:\Users\jfdou\Downloads\ForeCastsDelnet.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Forecast results saved to '{output_path}'.")

    # Compute overall ElasticNet metrics using the aggregated squared errors
    if total_count_elnet > 0:
        overall_mse_elnet = total_squared_error_elnet / total_count_elnet
        overall_rmse_elnet = np.sqrt(overall_mse_elnet)
        print(f"\n📈 Overall ElasticNet forecast: Total MSE = {overall_mse_elnet:.2f}, Total RMSE = {overall_rmse_elnet:.2f}")
    else:
        print("❌ No ElasticNet forecasts were computed.")

    # Compute overall baseline metrics if available.
    if total_count_K > 0:
        overall_mse_K = total_squared_error_K / total_count_K
        overall_rmse_K = np.sqrt(overall_mse_K)
        print(f"\n📈 Overall Baseline ('{baseline_col}') forecast: Total MSE = {overall_mse_K:.2f}, Total RMSE = {overall_rmse_K:.2f}")
    else:
        print("ℹ️ Baseline forecasts were not computed (column 'K' missing).")

    # --- Compute yearly metrics ---
    if not results_df.empty:
        results_df['forecast_date'] = pd.to_datetime(results_df['forecast_date'])
        results_df['year'] = results_df['forecast_date'].dt.year

        # We only need the MSE to compute a proper yearly RMSE = sqrt(MSE).
        yearly_metrics_elnet = results_df.groupby('year').agg({'mse_elnet': 'mean'}).reset_index()

        # Compute RMSE for each year as the square root of that year's MSE.
        yearly_metrics_elnet['rmse_elnet'] = np.sqrt(yearly_metrics_elnet['mse_elnet'])

        # --- Update installed capacities as given in the screenshot ---
        installed_capacity = {
            2012: 6740.51,
            2013: 7929.94,
            2014: 8380.68,
            2015: 8904.11,
            2016: 9726.56,
            2017: 10374.53,
            2018: 11239.32
        }
        yearly_metrics_elnet['installed_capacity'] = yearly_metrics_elnet['year'].map(installed_capacity)

        # Percentage of RMSE relative to installed capacity
        yearly_metrics_elnet['rmse_elnet_pct'] = (
            yearly_metrics_elnet['rmse_elnet'] / yearly_metrics_elnet['installed_capacity']
        ) * 100

        print("\n📊 Yearly ElasticNet Metrics (MSE, RMSE, capacity, % of capacity):")
        print(yearly_metrics_elnet[['year', 'mse_elnet', 'rmse_elnet', 'installed_capacity', 'rmse_elnet_pct']])

        # --- Build and print a summary DataFrame for overall metrics ---
        overall_metrics = {
            'Overall_MSE': overall_mse_elnet if total_count_elnet > 0 else None,
            'Overall_RMSE': overall_rmse_elnet if total_count_elnet > 0 else None
        }
        overall_df = pd.DataFrame([overall_metrics])
        print("\n📊 Overall ElasticNet Metrics (full sample):")
        print(overall_df)
    else:
        print("❌ No forecast results to compute yearly metrics.")

if __name__ == '__main__':
    main()