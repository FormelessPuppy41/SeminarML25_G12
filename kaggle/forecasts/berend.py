import pandas as pd
import datetime

from data.data_loader import DataLoader
from data.data_processing import explore_data
from configuration import FileNames


file_names = FileNames()
data_loader = DataLoader()

##
## TO RUN THIS FILE:
## - OPEN THE TERMINAL (ctrl + ` [see the left bottom corner of your keyboard])
## - WRITE: python -m kaggle.forecasts.berend [or the name of your file]
## - PRESS ENTER TO RUN
## - RESULTS APPEAR IN THE TERMINAL
##


def forecaster(energy_data: pd.DataFrame, weather_data: pd.DataFrame):
    """
    WRITE THE EXPLANATION HERE FOR YOUR SPECIFIC FORECASTER. 

    Args:
        energy_data (pd.DataFrame): df containing energy data
        weather_data (pd.DataFrame): df containing weather data

    """
    # Write your forecasting code here
    raise NotImplementedError("This function is not implemented yet.")


def _data_preprocessor(energy_data: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data for forecasting.

    Args:
        energy_data (pd.DataFrame): df containing energy data
        weather_data (pd.DataFrame): df containing weather data

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: preprocessed energy and weather data
    """
    def preprocess_energy_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the energy data.

        Args:
            df (pd.DataFrame): df containing energy data

        Returns:
            pd.DataFrame: preprocessed energy data
        """
        return df
    
    def preprocess_weather_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the weather data.

        Args:
            df (pd.DataFrame): df containing weather data"

        Returns:
            pd.DataFrame: preprocessed weather data
        """
        # Convert 'dt_iso' to datetime
        df['dt_iso'] = df['dt_iso'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        # Convert 'weather_id' to int
        
        # Average the numerical weather data by time -> the geographic location (city) is averaged out
        avg_weather_by_time = df.groupby('dt_iso').mean(numeric_only=True)
        #print('head of the weather data:', avg_weather_by_time.head())

        # Get the most common weather condition in each city for each time
        most_common_weather = df.groupby('dt_iso')['weather_main'].agg(lambda x: x.mode().iloc[0])  # In case of tie, take first mode
        #print('head of the most common weather:', most_common_weather.head())

        # Join the average weather data with the most common weather
        avg_weather_by_time_with_weather_main = avg_weather_by_time.join(most_common_weather)
        avg_weather_by_time_with_weather_main.reset_index(inplace=True)
        #print('head of the weather data with weather main:', avg_weather_by_time_with_weather_main.head())

        def analyse_weather_mix():
            """
            Analyse the weather data to understand the mix of weather conditions.
            This function prints the unique weather conditions and their counts.
            """
            # Find all unique pairs of weather_id and weather_main
            weather_mapping = df[['weather_id', 'weather_main']].drop_duplicates().sort_values(by='weather_id')

            # See how many weather_main values exist per weather_id
            id_to_main_counts = df.groupby('weather_id')['weather_main'].nunique().reset_index().sort_values(by='weather_main', ascending=False)

            # See how many weather_id values exist per weather_main
            main_to_id_counts = df.groupby('weather_main')['weather_id'].nunique().reset_index().sort_values(by='weather_id', ascending=False)

            # Print the results
            print("Weather ID to Weather Main mapping:")
            print(weather_mapping)
            print("\nWeather ID counts per Weather Main:")
            print(id_to_main_counts)
            print("\nWeather Main counts per Weather ID:")
            print(main_to_id_counts)
        #analyse_weather_mix()

        def add_weather_severity_feature(df: pd.DataFrame) -> pd.DataFrame:
            """
            Add a weather severity feature to the weather data.
            This function assigns a severity score based on the weather condition.
            The score consists of:
            - A base severity derived from `weather_main`
            - A normalized detail component from the last two digits of `weather_id`,
            uniformly distributed within each `weather_main` group.
            
            Returns:
                pd.DataFrame: The input DataFrame with an added `weather_severity_score` column.
            """

            # Step 1: Define base severity mapping for each weather_main
            severity_map = {
                'clear': 0,
                'clouds': 1,
                'mist': 2,
                'fog': 3,
                'drizzle': 4,
                'rain': 5,
                'snow': 6,
                'dust': 6,
                'haze': 6,
                'smoke': 6,
                'squall': 7,
                'thunderstorm': 8
            }

            # Step 2: Map base severity to DataFrame
            df['weather_base_severity'] = df['weather_main'].map(severity_map)

            # Step 3: Extract last two digits of weather_id
            df['weather_detail_raw'] = df['weather_id'] % 100

            # Step 4: Create a normalized detail value within each weather_main
            detail_df = df[['weather_main', 'weather_detail_raw']].drop_duplicates()
            detail_df['detail_rank'] = detail_df.groupby('weather_main')['weather_detail_raw'] \
                                                .rank(method='dense').astype(int)

            # Normalize the detail rank to range [0, 1]
            detail_df['normalized_detail'] = detail_df.groupby('weather_main')['detail_rank'] \
                .transform(lambda x: (x - 1) / (x.max() - 1) if x.max() > 1 else 0.5)

            # Step 5: Merge normalized detail back to the main DataFrame
            df = df.merge(detail_df, on=['weather_main', 'weather_detail_raw'], how='left')

            # Step 6: Combine base severity with normalized detail
            df['weather_severity_score'] = df['weather_base_severity'] + df['normalized_detail']

            # Step 7: Drop intermediate columns
            df.drop(columns=['weather_base_severity', 'weather_detail_raw', 'detail_rank', 'normalized_detail'], inplace=True)
            return df
        weather_data = add_weather_severity_feature(avg_weather_by_time_with_weather_main)
        #print('head of the weather data with weather severity:', weather_data.head(), '\n Unique severity codes are: ',weather_data['weather_severity_score'].unique())

    #TODO: the time column is not in datetime format, but in string format with +01:00:00 at the end. Fix this.
    # Preprocess the energy data
    energy_data = preprocess_energy_data(energy_data)
    print('head of the preprocessed energy data:', energy_data.head())
    # Preprocess the weather data
    weather_data = preprocess_weather_data(weather_data)
    print('head of the preprocessed weather data:', weather_data.head())

    return energy_data, weather_data
    

if __name__ == "__main__":
    energy_df = data_loader.load_kaggle_data(file_names.kaggle_files.energy_data_file)
    weather_df = data_loader.load_kaggle_data(file_names.kaggle_files.weather_data_file)

    # COMMENT THIS OUT IF YOU NO LONGER WANT TO EXPLORE THE DATA
    # Explore the data 
    #explore_data(energy_df, "Energy Data")
    #explore_data(weather_df, "Weather Data")

    # Preprocess the data
    energy_df, weather_df = _data_preprocessor(energy_df, weather_df)
    
    # Run the forecaster
    forecaster(energy_df, weather_df)