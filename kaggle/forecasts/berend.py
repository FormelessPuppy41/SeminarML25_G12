
from data.data_loader import DataLoader
from data.data_processing import explore_data
from configuration import FileNames


file_names = FileNames()
data_loader = DataLoader()


if __name__ == "__main__":
    energy_df = data_loader.load_kaggle_data(file_names.kaggle_files.energy_data_file)
    weather_df = data_loader.load_kaggle_data(file_names.kaggle_files.weather_data_file)

    # Explore the data
    explore_data(energy_df, "Energy Data")
    explore_data(weather_df, "Weather Data")