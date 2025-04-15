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

# ---------- Step 2: Generate Latent Signals ----------
def generate_latent_signals(length, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, 2 * np.pi * length / 96, length)

    base1 = np.sin(t)  # Diurnal cycle
    base2 = np.cos(t / 2)  # Slower trend
    base3 = np.random.normal(0, 0.5, length)  # Random noise / anomalies

    return base1, base2, base3

# ---------- Step 3: Create Synthetic Forecasts ----------
def generate_synthetic_forecasts_from_hr(df: pd.DataFrame, seed=42):
    np.random.seed(seed)
    hr = df[ModelSettings.target].values
    length = len(hr)

    daylight_mask = hr > 0

    # Time index for structured patterns
    t = np.linspace(0, 2 * np.pi, length)

    # Generate structured deviations
    dev1 = 0.05 * np.sin(3 * t)                          # daily pattern bias
    dev2 = 0.1 * np.random.normal(0, 1, length)          # random noise
    dev3 = 0.15 * np.cos(1.5 * t + np.pi / 3)            # phase-shifted seasonal-ish trend
    
    print(dev1, dev2, dev3)
    raise ValueError
    # Only apply deviations where HR > 0 (daylight hours)
    dev1[~daylight_mask] = 0
    dev2[~daylight_mask] = 0
    dev3[~daylight_mask] = 0

    forecasts = []
    weights = [
        (0.6, 0.3, 0.1),
        (0.5, 0.4, 0.1),
        (0.3, 0.5, 0.2),
        (0.2, 0.6, 0.2),
        (0.4, 0.2, 0.4),
        (0.5, 0.0, 0.5)
    ]

    for w1, w2, w3 in weights:
        total_deviation = w1 * dev1 + w2 * dev2 + w3 * dev3
        forecast = hr * (1 + total_deviation)
        forecast = np.clip(forecast, 0, None)
        forecasts.append(forecast)

    return np.array(forecasts).T  # shape: (time, 6)


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
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
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

# ---------- Step 7: Main Pipeline ----------
def generate_forecasts_and_decompose(df):
    df_clean = preprocess(df)
    n = len(df_clean)

    base1, base2, base3 = generate_latent_signals(n)
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

    plt.show()
    return df_clean, pca, pca_components, ica, ica_components


if __name__ == "__main__":
    from data.data_loader import DataLoader
    from configuration import ModelSettings, FileNames
    file_names = FileNames()

    data_loader = DataLoader()
    df = data_loader.load_input_data(file_names.input_files.solar_combined_data, True)

    df_clean, pca, pca_components, ica, ica_components = generate_forecasts_and_decompose(df)
    print(df_clean)
    print(df_clean[25:50])
    df_clean.to_csv('test_data.csv')