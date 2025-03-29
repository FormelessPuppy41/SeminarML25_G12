import kagglehub
import pandas as pd
import os


# Download latest version
path = kagglehub.dataset_download("nicholasjhana/energy-consumption-generation-prices-and-weather")

print("Path to dataset files:", path)

files = os.listdir(path)
print(files)

# Example: Read energy dataset
energy_df = pd.read_csv(os.path.join(path, "energy_dataset.csv"))
weather_df = pd.read_csv(os.path.join(path, "weather_features.csv"))

# Optional: Show first few rows
print(energy_df.head())
print(weather_df.head())


write_path_energy = "data/kaggle_data/energy.csv"
write_path_weather = "data/kaggle_data/weather.csv"

# Write DataFrames to CSV
energy_df.to_csv(write_path_energy, index=False)
weather_df.to_csv(write_path_weather, index=False)

print(f"Energy data written to: {write_path_energy}")
print(f"Weather data written to: {write_path_weather}")


