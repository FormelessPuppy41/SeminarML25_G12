import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# ---------- Step 0: Helper Methods ----------
def normalize_by_time_of_day(df: pd.DataFrame, columns):
    df = df.copy()
    df['time_bin'] = df.index.time

    for col in columns:
        grouped = df.groupby('time_bin')[col]
        mean = grouped.transform('mean')
        std = grouped.transform('std')

        # Avoid division by zero
        std = std.replace(0, 1e-6)

        df[f'{col}_normalized'] = (df[col] - mean) / std

    return df



# ---------- Step 1: Preprocess Data ----------
def preprocess(df: pd.DataFrame):
    df[ModelSettings.datetime_col] = pd.to_datetime(df['Zeit'])
    df.drop(columns=['Zeit'], inplace=True)
    df = df[[ModelSettings.datetime_col, ModelSettings.target, 'K']].dropna()
    df.set_index(ModelSettings.datetime_col, inplace=True)

    #scaler = StandardScaler()
    #df[[ModelSettings.target, 'K']] = scaler.fit_transform(df[[ModelSettings.target, 'K']])
    return df

# ---------- Step 3: Create Synthetic Forecasts ----------
def generate_deviations(length: np.ndarray, hr: np.ndarray, residuals: np.ndarray, daylight_mask: np.ndarray, scale=1, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, 2 * np.pi, length)

    # Define base deviation patterns (scaled unitless multipliers)
    dev1 = scale * 0.6 * np.sin(3 * t)                            # Diurnal bias
    dev2 = scale * 0.25 * np.random.normal(0, 1, length)           # Random noise
    dev3 = scale * 0.4 * np.cos(1.5 * t + np.pi / 3)              # Seasonal-ish bias
    dev4 = scale * 0.3 * residuals / (np.max(np.abs(residuals)) + 1e-6)  # Scaled error feedback
    dev5 = scale * 0.2 * residuals / (np.mean(hr[daylight_mask]) + 1e-6)  # Relative error signal

    # Apply only during daylight
    for dev in [dev1, dev2, dev3, dev4, dev5]:
        dev[~daylight_mask] = 0

    deviations = {
        "dev1_sin": dev1,
        "dev2_noise": dev2,
        "dev3_cos": dev3,
        "dev4_k_minus_hr_scaled": dev4,
        "dev5_relative_k_bias": dev5
    }

    # Log summaries
    print("\n--- Deviation Stats ---")
    for name, dev in deviations.items():
        print(f"{name}: mean={np.mean(dev):.4f}, std={np.std(dev):.4f}, min={np.min(dev):.4f}, max={np.max(dev):.4f}")

    return deviations

def apply_structural_bias(
    forecast: np.ndarray,
    index: int,
    t: np.ndarray,
    residuals: np.ndarray,
    bias_mask: np.ndarray
):
    if not bias_mask.any():
        return forecast

    biased = forecast.copy()
    
    if index == 0:
        return forecast
    elif index == 1:
        biased = np.roll(forecast, 2)
    elif index == 2:
        hour = (t % 24)
        bias = np.where((hour >= 6) & (hour <= 10), 1.1, 1.0)
        biased = forecast * bias
    elif index == 3:
        biased = forecast * 0.95
    elif index == 4:
        hour = (t % 24)
        spike = np.where((hour >= 11) & (hour <= 14), np.random.normal(1.1, 0.05, len(forecast)), 1.0)
        biased = forecast * spike
    elif index == 5:
        pattern = np.where(residuals > 0, 0.95, 1.05)
        biased = forecast * pattern
    else:
        return forecast

    # Only apply when the mask is active
    return np.where(bias_mask, biased, forecast)

def generate_bias_activation_masks(index: pd.DatetimeIndex, n_forecasts: int, block="W", seed=42):
    """
    For each forecast, randomly select time blocks (weeks, months, etc.) where structural bias is active.
    """
    np.random.seed(seed)
    df = pd.DataFrame(index=index)
    df["block"] = index.to_period(block)

    blocks = df["block"].unique()
    mask_dict = {}

    for i in range(n_forecasts):
        n_active = max(1, len(blocks) // 3)  # e.g., ~33% of weeks/months active
        active_blocks = np.random.choice(blocks, size=n_active, replace=False)
        df[f"bias_active_{i}"] = df["block"].isin(active_blocks).values

    return {i: df[f"bias_active_{i}"].values for i in range(n_forecasts)}


def generate_synthetic_forecasts_from_hr(df: pd.DataFrame, seed=42):
    np.random.seed(seed)
    hr = df[ModelSettings.target].values
    k = df['K'].values
    residuals = k - hr
    mid_hr_and_k = (hr + k) / 2
    daylight_mask = hr > 0

    # Estimate base error scale (or define manually)
    k_day = k[daylight_mask]
    hr_day = hr[daylight_mask]
    observed_mape = np.mean(np.abs(k_day - hr_day) / (hr_day + 1e-6))
    target_scale = observed_mape  # Or use fixed value, e.g., 0.15

    deviations = generate_deviations(length=len(df), hr=hr, residuals=residuals, daylight_mask=daylight_mask, scale=target_scale, seed=seed)

    # Weighted combinations to create forecasts
    weights_list = [
    {
        "dev1_sin": 0.20,
        "dev2_noise": 0.40,
        "dev3_cos": 0.20,
        "dev4_k_minus_hr_scaled": 0.10,
        "dev5_relative_k_bias": 0.10
    },
    {
        "dev1_sin": 0.20,
        "dev2_noise": 0.35,
        "dev3_cos": 0.15,
        "dev4_k_minus_hr_scaled": 0.15,
        "dev5_relative_k_bias": 0.15
    },
    {
        "dev1_sin": 0.10,
        "dev2_noise": 0.40,
        "dev3_cos": 0.25,
        "dev4_k_minus_hr_scaled": 0.10,
        "dev5_relative_k_bias": 0.15
    },
    {
        "dev1_sin": 0.30,
        "dev2_noise": 0.35,
        "dev3_cos": 0.15,
        "dev4_k_minus_hr_scaled": 0.15,
        "dev5_relative_k_bias": 0.05
    },
    {
        "dev1_sin": 0.20,
        "dev2_noise": 0.40,  # increase
        "dev3_cos": 0.15,
        "dev4_k_minus_hr_scaled": 0.15,
        "dev5_relative_k_bias": 0.10
    },
    {
        "dev1_sin": 0.15,
        "dev2_noise": 0.40,  # increase more
        "dev3_cos": 0.15,
        "dev4_k_minus_hr_scaled": 0.15,
        "dev5_relative_k_bias": 0.15
    }

    ]
    # Generate structural bias activation masks (e.g., weekly)
    bias_masks = generate_bias_activation_masks(df.index, n_forecasts=6, block="W", seed=seed)

    forecasts = []
    t = df.index.to_series().dt.hour.values

    scaler = {
        1: mid_hr_and_k,
        2: mid_hr_and_k,
        3: hr, # always best -> add more noise bcs currently the others automatically have more noise, bcs they are bias by not using HR. 
        4: hr, # always best -> add more noise bcs currently the others automatically have more noise, bcs they are bias by not using HR. 
        5: k,
        6: k
    }
    deviation_multiplier = {
        1: 0.8,
        2: 0.6,
        3: 2.5,
        4: 2.0,
        5: 0.4,
        6: 0.3
    }
    for i, weight_dict in enumerate(weights_list):
        total_deviation = deviation_multiplier.get(i+1) * sum(
            weight * deviations[name] for name, weight in weight_dict.items()
        )
        
        forecast = scaler.get(i+1) * (1 + total_deviation)
        forecast = apply_structural_bias(forecast, i, t, residuals, bias_masks[i])
        forecast = np.clip(forecast, 0, None)
        forecasts.append(forecast)

    forecasts = np.array(forecasts).T  # now shape is (time, 6)

    def forecast_evalutation():
        # Evaluation
        print("\n--- Forecast Evaluation ---")
        print(f"Observed MAPE from K: {observed_mape:.2%}\n")
        for i in range(6):
            mae = np.mean(np.abs((forecasts[:, i] - hr)[daylight_mask]))
            mape = np.mean(np.abs((forecasts[:, i] - hr)[daylight_mask] / (hr[daylight_mask] + 1e-6)))
            print(f"A{i+1}: MAE={mae:.2f}, MAPE={mape:.2%}")
    forecast_evalutation()
    
    return forecasts


# ---------- Step 4: Perform PCA ----------
def run_pca(df: pd.DataFrame, forecast_cols):
    df_norm = normalize_by_time_of_day(df, forecast_cols)
    data = df_norm[[f'{col}_normalized' for col in forecast_cols]].values

    pca = PCA()
    pca_components = pca.fit_transform(data)

    print("\n--- PCA Explained Variance Ratio ---")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"Component {i+1}: {var:.4f}")

    plt.figure()
    explained = np.cumsum(pca.explained_variance_ratio_)
    x_vals = np.arange(1, len(explained) + 1)  # 1-based indexing

    plt.plot(x_vals, explained, marker='o')
    plt.xticks(x_vals)  # explicitly set ticks to 1, 2, ..., n
    plt.xlabel('Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Scree Plot')
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    for i in range(min(3, pca_components.shape[1])):
        plt.plot(pca_components[:, i], label=f'PC{i+1}')
    plt.legend()
    plt.title('PCA Component Time Series')
    plt.xlabel('Time')
    plt.ylabel('Component Value')
    plt.tight_layout()

    return pca, pca_components

# ---------- Step 5: Perform ICA ----------
def run_ica(df: pd.DataFrame, forecast_cols, n_components=6):
    df_norm = normalize_by_time_of_day(df, forecast_cols)
    data = df_norm[[f'{col}_normalized' for col in forecast_cols]].values

    ica = FastICA(n_components=n_components, random_state=42)
    ica_components = ica.fit_transform(data)

    print("\n--- ICA Component Sample Statistics (First 3) ---")
    for i in range(min(3, ica_components.shape[1])):
        mean = np.mean(ica_components[:, i])
        std = np.std(ica_components[:, i])
        print(f"IC{i+1}: mean={mean:.4f}, std={std:.4f}")

    plt.figure()
    for i in range(min(3, ica_components.shape[1])):
        plt.plot(ica_components[:, i], label=f'IC{i+1}')
    plt.legend()
    plt.title('ICA Component Time Series')
    plt.xlabel('Time')
    plt.ylabel('Component Value')
    plt.tight_layout()

    return ica, ica_components

# ---------- Step 6: Evaluate Forecast Diversity Per Interval ----------
def evaluate_forecast_diversity(df: pd.DataFrame, interval='M'):
    df = df.copy()

    error_df = {}
    for i in range(6):
        forecast_col = f'A{i+1}'
        error_df[f'A{i+1}'] = df[ModelSettings.target] - df[forecast_col]
    
    error_df['K'] = df[ModelSettings.target] - df['K']

    error_df = pd.DataFrame(error_df, index=df.index)
    mae_by_interval = error_df.abs().resample(interval).mean()

    print("\n--- Mean Absolute Error (MAE) by Forecast and Month ---")
    print(mae_by_interval)

    # Plot
    plt.figure()
    for col in mae_by_interval.columns:
        plt.plot(mae_by_interval.index, mae_by_interval[col], label=col)
    plt.legend()
    plt.title(f'MAE per Forecast ({interval} intervals)')
    plt.xlabel('Time')
    plt.ylabel('MAE')
    plt.tight_layout()

    # Count per interval who is "best"
    best_forecast_counts = mae_by_interval.idxmin(axis=1).value_counts().sort_index()
    print("\n--- Count of Best Forecast per Month ---")
    print(best_forecast_counts)

    plt.figure()
    best_forecast_counts.plot(kind='bar')
    plt.title('Number of Months Each Forecast Was Best')
    plt.xlabel('Forecast')
    plt.ylabel('Count')
    plt.tight_layout()


def plot_error_direction(df, forecast_cols, target_col, resample_interval=None):
    """
    Plot the error direction (forecast minus actual) for each forecaster in separate subplots.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the ground truth and forecast columns.
        forecast_cols (list of str): Column names for the forecasts.
        target_col (str): Column name for the ground truth.
        resample_interval (str, optional): Pandas offset alias to resample the data 
            (e.g., 'D' for daily, 'M' for monthly). If provided, errors are aggregated by mean.
    """
    # If a resample interval is provided, aggregate errors accordingly.
    if resample_interval is not None:
        df_plot = df.copy()
        # Calculate error for each forecast column
        for col in forecast_cols:
            df_plot[f'{col}_error'] = df_plot[col] - df_plot[target_col]
        # Resample: use mean error in each interval
        df_plot = df_plot.resample(resample_interval).mean()
        error_cols = [f'{col}_error' for col in forecast_cols]
        time_index = df_plot.index
    else:
        # Use the raw errors
        error_cols = forecast_cols  # We'll compute error on the fly below
        time_index = df.index

    n_forecasts = len(forecast_cols)
    fig, axes = plt.subplots(n_forecasts, 1, figsize=(15, 3 * n_forecasts), sharex=True)
    
    # If there's only one forecast, wrap it in a list.
    if n_forecasts == 1:
        axes = [axes]
        
    # Loop through each forecaster and plot its error direction.
    for i, col in enumerate(forecast_cols):
        if resample_interval is not None:
            errors = df_plot[f'{col}_error']
        else:
            errors = df[col] - df[target_col]
            
        # Plot error as a bar chart.
        # We use different colors based on error direction: red for positive, green for negative.
        colors = ['red' if err > 0 else 'green' for err in errors]
        axes[i].bar(time_index, errors, color=colors, width=1, align='center')
        axes[i].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes[i].set_title(f'Error Direction for {col} (Forecast - Actual)')
        axes[i].set_ylabel('Error')
        # Improve readability when many time points are present.
        axes[i].tick_params(axis='x', rotation=45)
    
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.show()


# ---------- Step 7: Main Pipeline ----------
def generate_forecasts_and_decompose(df):
    df_clean = preprocess(df)
    n = len(df_clean)

    synthetic_forecasts = generate_synthetic_forecasts_from_hr(df_clean)

    # Add forecasts to the dataframe
    for i in range(6):
        df_clean[f'A{i+1}'] = synthetic_forecasts[:, i]


    print("\n--- Forecast Descriptive Statistics ---")
    print(df_clean[[f'A{i+1}' for i in range(6)]].describe())

    forecast_cols = [f'A{i+1}' for i in range(6)]
    pca, pca_components = run_pca(df_clean, forecast_cols)
    ica, ica_components = run_ica(df_clean, forecast_cols)

    evaluate_forecast_diversity(df_clean, interval='ME')
    plot_error_direction(df_clean, forecast_cols, ModelSettings.target, resample_interval='M')

    plt.show()
    return df_clean, pca, pca_components, ica, ica_components


if __name__ == "__main__":
    from data.data_loader import DataLoader
    from configuration import ModelSettings, FileNames
    file_names = FileNames()

    data_loader = DataLoader()
    df = data_loader.load_input_data(file_names.input_files.solar_combined_data, True)

    df_clean, pca, pca_components, ica, ica_components = generate_forecasts_and_decompose(df)
    print(df_clean[25:50])
    df_clean.to_csv('test_data.csv')