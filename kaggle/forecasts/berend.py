
from ...data.data_loader import DataLoader
from configuration import FileNames


file_names = FileNames()
data_loader = DataLoader()


if __name__ == "__main__":
    energy_df = data_loader.load_input_data(file_names.kaggle_files.energy_data_file)
    weather_df = data_loader.load_input_data(file_names.kaggle_files.weather_data_file)

    # Display the first few rows of the data
    print("Energy Data:")
    print(energy_df.head())

    print("\nWeather Data:")
    print(weather_df.head())