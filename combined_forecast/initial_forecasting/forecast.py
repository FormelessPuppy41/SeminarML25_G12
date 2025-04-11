import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -----------------------------------------------------------------------------
# Forecast functions using different transformations to enhance PCA variability
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np

def forecast1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forecast 1: Forecast HR using a rolling mean.
    
    Uses a rolling mean (window of 4 periods, ~1 hour) on the HR column.
    This simple smoothing is used as the forecast of HR.
    """
    hr = df['HR']
    forecast = hr.rolling(window=4, min_periods=1).mean()
    return pd.DataFrame({'A1': forecast})

def forecast2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forecast 2: Forecast HR using an exponentially weighted moving average.
    
    Uses an exponentially weighted moving average (span=12, weighting recent
    values more heavily) as the forecast of HR.
    """
    hr = df['HR']
    forecast = hr.ewm(span=12, adjust=False).mean()
    return pd.DataFrame({'A2': forecast})

def forecast3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forecast 3: Forecast K using a rolling median.
    
    Uses a rolling median (window of 4 periods) on the K column to provide a robust 
    forecast of the company forecast values.
    """
    K = df['K']
    forecast = K.rolling(window=4, min_periods=1).median()
    return pd.DataFrame({'A3': forecast})

def forecast4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forecast 4: Forecast K using the previous value.
    
    Uses a one-period lag of the K column as the forecast of K.
    """
    K = df['K']
    forecast = K.shift(1).fillna(method='bfill')
    return pd.DataFrame({'A4': forecast})

def forecast5(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forecast 5: Forecast HR based on the difference between HR and K.
    
    Computes the difference (HR - K), obtains its rolling mean (window=4), 
    and then adds it to the current K value. This yields a forecast for HR.
    """
    diff = df['HR'] - df['K']
    diff_mean = diff.rolling(window=4, min_periods=1).mean()
    forecast = df['K'] + diff_mean
    return pd.DataFrame({'A5': forecast})

def forecast6(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forecast 6: Forecast HR using a non-linear transformation of the difference (HR - K).
    
    Computes a sign-preserving square-root transformation on the difference 
    (i.e. sqrt(abs(diff)) * sign(diff)), then smooths it with a rolling mean (window=4),
    and finally adds the result to K to produce the HR forecast.
    """
    diff = df['HR'] - df['K']
    diff_sqrt = np.sqrt(diff.abs()) * np.sign(diff)
    diff_sqrt_mean = diff_sqrt.rolling(window=4, min_periods=1).mean()
    forecast = df['K'] + diff_sqrt_mean
    return pd.DataFrame({'A6': forecast})

# ----------------------------
# Combine forecast features and perform PCA
# ----------------------------

def compute_and_plot_pca(df: pd.DataFrame):
    """
    Combines forecast features from six methods, applies PCA,
    and plots the explained variance (scree plot) and a scatter
    plot of the first two principal components.
    """
    # Combine the forecasts into one DataFrame (each forecast returns 2 columns)
    forecast_features = pd.concat([
        forecast1(df),
        forecast2(df),
        forecast3(df),
        forecast4(df),
        forecast5(df),
        forecast6(df)
    ], axis=1)
    
    # Optionally, fill any remaining missing values
    forecast_features = forecast_features.fillna(method='bfill').fillna(method='ffill')
    print(forecast_features)
    # Apply PCA. Using n_components equal to the total number of features (12 here)
    pca = PCA(n_components=forecast_features.shape[1])
    pca.fit(forecast_features)
    pca_features = pca.transform(forecast_features)
    
    # ----------------------------
    # Scree plot: Explained variance ratio for each component
    # ----------------------------
    plt.figure(figsize=(8, 5))
    components = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    plt.bar(components, pca.explained_variance_ratio_, alpha=0.7)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot of Forecast Features')
    plt.xticks(components)
    plt.show()

    # ----------------------------
    # Scatter plot of the first two principal components
    # ----------------------------
    plt.figure(figsize=(8, 5))
    plt.scatter(pca_features[:, 0], pca_features[:, 1], alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Scatter Plot of PC1 vs PC2')
    plt.show()
    
    # Optionally, print the explained variance ratios to see how much variance each component explains
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"Component {i+1}: {ratio:.4f} of total variance")


if __name__ == "__main__":
    from data.data_loader import DataLoader
    from configuration import FileNames
    
    df = DataLoader().load_input_data(file_name=FileNames().input_files.solar_combined_data)
    compute_and_plot_pca(df)