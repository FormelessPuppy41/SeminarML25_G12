import pandas as pd
import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


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



def forecast_next_day_wind_opt_ridge(df: pd.DataFrame):
    """
    Forecast next-day hourly solar power generation using an optimized walk-forward Ridge regression.
    Forecast is made every day at 09:00 for the next day (24 hourly values).

    Args:
        df (pd.DataFrame): Merged dataframe with weather + energy features.

    Returns:
        pd.DataFrame: DataFrame with forecast timestamps and predictions.
    """
    df = df.copy().dropna()
    df.set_index('time', inplace=True)

    features = ['temp', 'clouds_all', 'weather_severity_score', 'sun_intensity', 'temp_min',
                'temp_max', 'pressure', 'humidity', 'rain_1h', 'rain_3h', 'snow_3h']
    target_col = 'generation solar'

    df = df.sort_index()
    lag_hours = 24 * 14
    forecast_horizon = 24
    forecast_offset = 15
    rolling_hours = 168 * 6

    forecasts = []
    feature_selection_counter = defaultdict(int)
    summary_stats = ['mean', 'std', 'min', 'max']
    expanded_feature_names = [f"{f}_{s}" for f in features for s in summary_stats]

    all_9am_points = df[df.index.hour == 9].index
    valid_9am_points = [t for t in all_9am_points if t > df.index[lag_hours]]

    print('Starting forecast for {} 09:00 points'.format(len(valid_9am_points)))

    for current_time in valid_9am_points:
        lag_start = current_time - pd.Timedelta(hours=rolling_hours)
        lag_end = current_time - pd.Timedelta(hours=1)
        future_start = current_time + pd.Timedelta(hours=forecast_offset)
        future_end = future_start + pd.Timedelta(hours=forecast_horizon - 1)

        if lag_start not in df.index or future_end not in df.index:
            #print('lag_start or future_end not in df.index')
            #continue
            pass

        history = df.loc[lag_start:lag_end]
        if len(history) < rolling_hours:
            #print('Possible date issue at time: ', current_time)
            #continue
            pass

        X_train = []
        y_train = []

        past_9ams = [t for t in all_9am_points if lag_start <= t < current_time]
        for t in past_9ams:
            x_start = t - pd.Timedelta(hours=lag_hours)
            x_end = t - pd.Timedelta(hours=1)
            y_start = t + pd.Timedelta(hours=forecast_offset)
            y_end = y_start + pd.Timedelta(hours=forecast_horizon - 1)

            if x_start not in df.index or y_end not in df.index:
                #print('x_start or y_end not in df.index')
                #continue
                pass

            x_window = df.loc[x_start:x_end][features]
            y_window = df.loc[y_start:y_end][target_col]
            if len(y_window) != forecast_horizon:
                #print('y_window length issue at time: ', t)
                continue
                pass

            x_features = []
            for feature in features:
                x_features.extend([
                    x_window[feature].mean(),
                    x_window[feature].std(),
                    x_window[feature].min(),
                    x_window[feature].max(),
                ])
            X_train.append(x_features)
            y_train.append(y_window.values)

        if not X_train:
            continue

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Feature selection using RFE
        base_model = Ridge(alpha=1.0)
        rfe_model = RFE(base_model, n_features_to_select=1)
        rfe_model.fit(X_train_scaled, y_train)

        selected_features_mask = rfe_model.support_
        selected_feature_names = np.array(expanded_feature_names)[selected_features_mask]

        for feat in selected_feature_names:
            feature_selection_counter[feat] += 1

        X_train_selected = X_train_scaled[:, selected_features_mask]

        model = MultiOutputRegressor(Ridge(alpha=1.0))
        model.fit(X_train_selected, y_train)

        # Latest input (same feature engineering)
        latest_window = df.loc[current_time - pd.Timedelta(hours=lag_hours): current_time - pd.Timedelta(hours=1)]
        x_latest = []
        for feature in features:
            x_latest.extend([
                latest_window[feature].mean(),
                latest_window[feature].std(),
                latest_window[feature].min(),
                latest_window[feature].max(),
            ])

        x_latest_scaled = scaler.transform([x_latest])
        x_latest_selected = x_latest_scaled[:, selected_features_mask]
        prediction = model.predict(x_latest_selected)[0]
        prediction = np.clip(prediction, 0, None)

        forecast_index = pd.date_range(start=future_start, periods=forecast_horizon, freq='h')
        forecast_df = pd.DataFrame({
            'forecast_time': forecast_index,
            'predicted_solar_generation': prediction,
            'forecast_made_at': current_time
        })
        forecasts.append(forecast_df)

    if forecasts:
        all_forecasts_df = pd.concat(forecasts).reset_index(drop=True)
    else:
        all_forecasts_df = pd.DataFrame(columns=['forecast_time', 'predicted_solar_generation', 'forecast_made_at'])

    merged = all_forecasts_df.merge(df[[target_col]], left_on='forecast_time', right_index=True)
    mse = mean_squared_error(merged[target_col], merged['predicted_solar_generation'])
    print('Mean Squared Error:', mse)

    merged.to_csv('data/data_files/model_results/solar_forecast_ridge_berend.csv', index=False)
    print('Forecasted values:')
    print(merged)

    print("\nFeature selection frequency across all 09:00 forecasts:")
    sorted_counts = sorted(feature_selection_counter.items(), key=lambda x: x[1], reverse=True)
    for feat, count in sorted_counts:
        print(f"{feat:30} → {count} times selected")

    return all_forecasts_df

# 2018-07-11 23:00:00 -> 2018-08-24 00:00
# 2018-03-25 23:00:00 -> 2018-05-07 00:00
# 2017-03-26 23:00:00 -> 2017-05-08 00:00
# 2016-07-10 23:00:00 -> 2016-08-22 00:00
# 2016-03-27 23:00:00 -> 2016-05-09 00:00

def forecast_next_day_wind_opt_elnet(df: pd.DataFrame):
    """
    Forecast next-day hourly solar power generation using an optimized walk-forward Ridge regression.
    Forecast is made every day at 09:00 for the next day (24 hourly values).

    Args:
        df (pd.DataFrame): Merged dataframe with weather + energy features.

    Returns:
        pd.DataFrame: DataFrame with forecast timestamps and predictions.
    """
    df = df.copy().dropna()
    df.set_index('time', inplace=True)

    features = ['temp', 'clouds_all', 'weather_severity_score', 'sun_intensity', 'temp_min',
                'temp_max', 'pressure', 'humidity', 'rain_1h', 'rain_3h', 'snow_3h']
    target_col = 'generation solar'

    df = df.sort_index()
    lag_hours = 24 * 14
    forecast_horizon = 24
    forecast_offset = 15
    rolling_hours = 168 * 6

    forecasts = []
    feature_selection_counter = defaultdict(int)
    summary_stats = ['mean', 'std', 'min', 'max']
    expanded_feature_names = [f"{f}_{s}" for f in features for s in summary_stats]

    all_9am_points = df[df.index.hour == 9].index
    valid_9am_points = [t for t in all_9am_points if t > df.index[lag_hours]]

    print('Starting forecast for {} 09:00 points'.format(len(valid_9am_points)))

    for current_time in valid_9am_points:
        lag_start = current_time - pd.Timedelta(hours=rolling_hours)
        lag_end = current_time - pd.Timedelta(hours=1)
        future_start = current_time + pd.Timedelta(hours=forecast_offset)
        future_end = future_start + pd.Timedelta(hours=forecast_horizon - 1)

        if lag_start not in df.index or future_end not in df.index:
            #print('lag_start or future_end not in df.index')
            #continue
            pass

        history = df.loc[lag_start:lag_end]
        if len(history) < rolling_hours:
            #print('possible issue for date: ', current_time)
            #continue
            pass

        X_train = []
        y_train = []

        past_9ams = [t for t in all_9am_points if lag_start <= t < current_time]
        for t in past_9ams:
            x_start = t - pd.Timedelta(hours=lag_hours)
            x_end = t - pd.Timedelta(hours=1)
            y_start = t + pd.Timedelta(hours=forecast_offset)
            y_end = y_start + pd.Timedelta(hours=forecast_horizon - 1)

            if x_start not in df.index or y_end not in df.index:
                #print('x_start or y_end not in df.index')
                #continue
                pass

            x_window = df.loc[x_start:x_end][features]
            y_window = df.loc[y_start:y_end][target_col]
            if len(y_window) != forecast_horizon:
                #print('y_window length issue at time: ', t)
                continue
                pass

            x_features = []
            for feature in features:
                x_features.extend([
                    x_window[feature].mean(),
                    x_window[feature].std(),
                    x_window[feature].min(),
                    x_window[feature].max(),
                ])
            X_train.append(x_features)
            y_train.append(y_window.values)

        if not X_train:
            continue

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Feature selection using RFE
        base_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
        rfe_model = RFE(base_model, n_features_to_select=3)
        rfe_model.fit(X_train_scaled, y_train)

        selected_features_mask = rfe_model.support_
        selected_feature_names = np.array(expanded_feature_names)[selected_features_mask]

        for feat in selected_feature_names:
            feature_selection_counter[feat] += 1

        X_train_selected = X_train_scaled[:, selected_features_mask]

        model = MultiOutputRegressor(ElasticNet(alpha=1.0, l1_ratio=0.5))
        model.fit(X_train_selected, y_train)

        # Latest input (same feature engineering)
        latest_window = df.loc[current_time - pd.Timedelta(hours=lag_hours): current_time - pd.Timedelta(hours=1)]
        x_latest = []
        for feature in features:
            x_latest.extend([
                latest_window[feature].mean(),
                latest_window[feature].std(),
                latest_window[feature].min(),
                latest_window[feature].max(),
            ])

        x_latest_scaled = scaler.transform([x_latest])
        x_latest_selected = x_latest_scaled[:, selected_features_mask]
        prediction = model.predict(x_latest_selected)[0]
        prediction = np.clip(prediction, 0, None)

        forecast_index = pd.date_range(start=future_start, periods=forecast_horizon, freq='h')
        forecast_df = pd.DataFrame({
            'forecast_time': forecast_index,
            'predicted_solar_generation': prediction,
            'forecast_made_at': current_time
        })
        forecasts.append(forecast_df)

    if forecasts:
        all_forecasts_df = pd.concat(forecasts).reset_index(drop=True)
    else:
        all_forecasts_df = pd.DataFrame(columns=['forecast_time', 'predicted_solar_generation', 'forecast_made_at'])

    merged = all_forecasts_df.merge(df[[target_col]], left_on='forecast_time', right_index=True)
    mse = mean_squared_error(merged[target_col], merged['predicted_solar_generation'])
    print('Mean Squared Error:', mse)

    merged.to_csv('data/data_files/model_results/solar_forecast_elnet_berend.csv', index=False)
    print('Forecasted values:')
    print(merged)

    print("\nFeature selection frequency across all 09:00 forecasts:")
    sorted_counts = sorted(feature_selection_counter.items(), key=lambda x: x[1], reverse=True)
    for feat, count in sorted_counts:
        print(f"{feat:30} → {count} times selected")

    return all_forecasts_df



def forecaster(merged_df: pd.DataFrame):
    """
    WRITE THE EXPLANATION HERE FOR YOUR SPECIFIC FORECASTER. 

    Args:
        merged_df (pd.DataFrame): df containing merged energy and weather data

    """
    # Write your forecasting code here
    forecast_next_day_wind_opt_ridge(merged_df)
    forecast_next_day_wind_opt_elnet(merged_df)


def _data_preprocessor(energy_data: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data for forecasting.

    Args:
        energy_data (pd.DataFrame): df containing energy data
        weather_data (pd.DataFrame): df containing weather data

    Returns:
        pd.DataFrame: preprocessed data
    """
    def preprocess_energy_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the energy data by converting the time column to datetime and selecting relevant columns.
        This function also removes the timezone information from the 'dt_iso' column.
        Furthermore, it selects only the relevant columns for the analysis which are: [generation solar, generation wind onshore, forecast solar day ahead, forecast wind onshore day ahead]

        Args:
            df (pd.DataFrame): df containing energy data

        Returns:
            pd.DataFrame: preprocessed energy data
        """
        # Convert 'dt_iso' to datetime
        df['time'] = df['time'].str.split('+').str[0]  # Remove the timezone information
        df['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        
        keep_cols = ['generation solar', 'generation wind onshore', 'forecast solar day ahead', 'forecast wind onshore day ahead']
        # Select only the relevant columns
        df = df[['time'] + keep_cols]
        
        return df
    
    def preprocess_weather_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the weather data by converting the time column to datetime and adds relevant columns.
        This function also removes the timezone information from the 'dt_iso' column.
        Furthermore, it creates a weather severity score based on the weather condition.
        The score consists of:
        - A base severity derived from `weather_main`
        - A normalized detail component from the last two digits of `weather_id`, uniformly distributed within each `weather_main` group.

        Args:
            df (pd.DataFrame): df containing weather data"

        Returns:
            pd.DataFrame: preprocessed weather data
        """
        # Convert 'dt_iso' to datetime
        df['dt_iso'] = df['dt_iso'].str.split('+').str[0]  # Remove the timezone information
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

        def sun_intensity(hour: int, date: datetime, min_daylight: float = 8.0, max_daylight: float = 16.0,
                        min_strength: float = 0.4, max_strength: float = 1.0, peak_hour: int = 13) -> float:
            """
            Estimate seasonally-adjusted sun intensity based on hour and date.

            Args:
                hour (int): Hour of the day (0–23).
                date (datetime): Date of the year.
                min_daylight (float): Minimum daylight duration (e.g. winter).
                max_daylight (float): Maximum daylight duration (e.g. summer).
                min_strength (float): Relative sun strength in winter.
                max_strength (float): Relative sun strength in summer.
                peak_hour (int): Hour of peak intensity (solar noon).

            Returns:
                float: Adjusted sun intensity (0–1)
            """
            day_of_year = date.timetuple().tm_yday

            # DAYLIGHT DURATION (cosine seasonal variation)
            daylight_range = max_daylight - min_daylight
            daylight_hours = min_daylight + (daylight_range / 2) * (1 + np.cos(2 * np.pi * (day_of_year - 172) / 365))

            # SEASONAL STRENGTH FACTOR (more intense sun in summer)
            strength_range = max_strength - min_strength
            strength = min_strength + (strength_range / 2) * (1 + np.cos(2 * np.pi * (day_of_year - 172) / 365))

            # Sunrise & sunset centered around solar noon
            half_daylight = daylight_hours / 2
            sunrise = peak_hour - half_daylight
            sunset = peak_hour + half_daylight

            if hour < sunrise or hour > sunset:
                return 0.0

            # Sun follows sine curve during daylight hours
            angle = np.pi * (hour - sunrise) / daylight_hours
            intensity = np.sin(angle) * strength

            return round(float(intensity), 4)
        # Add the sun intensity feature
        weather_data['hour'] = weather_data['dt_iso'].dt.hour
        weather_data['sun_intensity'] = weather_data.apply(lambda row: sun_intensity(row['hour'], row['dt_iso']), axis=1)
        # Drop the 'hour' column as it's no longer needed
        weather_data.drop(columns=['hour'], inplace=True)

        # Drop unnecessary columns
        weather_data.drop(columns=['weather_id', 'weather_main'], inplace=True)

        return weather_data

    # Preprocess the energy data
    energy_data = preprocess_energy_data(energy_data)
    #print('\n\nhead of the preprocessed energy data:', energy_data.head())
    # Preprocess the weather data
    weather_data = preprocess_weather_data(weather_data)
    #print('\n\nhead of the preprocessed weather data:', weather_data.head())

    # Merge the energy and weather data on time 
    merged_df = pd.merge(energy_data, weather_data, left_on='time', right_on='dt_iso', how='outer')
    merged_df.drop(columns=['dt_iso'], inplace=True)  # Drop the dt_iso column as it's redundant after merging


    # Step 1: Load the time_data_fcst CSV
    time_data_fcst = pd.read_csv('data/data_files/model_results/merged_and_forecastsCVS.csv')
    #print('timedatefcs', time_data_fcst.shape)
    #print('mergeddf', merged_df.shape)
    # Ensure 'time' is in datetime format
    time_data_fcst['time'] = pd.to_datetime(time_data_fcst['time'])
    merged_df['time'] = pd.to_datetime(merged_df['time'])

    # Step 2: Extract unique time values
    all_times = pd.DataFrame({'time': time_data_fcst['time'].dropna().unique()})

    # Step 3: Identify missing times not in merged_df
    missing_times = all_times[~all_times['time'].isin(merged_df['time'])]
    #print('Missing times:', missing_times)

    # Step 4: Create a DataFrame with these missing times and NaNs for other columns
    missing_rows = pd.DataFrame(columns=merged_df.columns)
    missing_rows['time'] = missing_times['time']
    # Fill the other columns with NaNs
    for col in merged_df.columns:
        if col != 'time':
            missing_rows[col] = pd.NA

    # Step 5: Append the missing rows to merged_df
    merged_df = pd.concat([merged_df, missing_rows], ignore_index=True)

    # (Optional) Sort by time for readability
    merged_df.sort_values(by='time', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)


    # Step 1: Create a mask of rows that contain at least one NaN
    na_rows = merged_df[merged_df.isna().any(axis=1)]

    # Step 2: Print those rows (before filling)
    #print("Rows with NaN values before filling:")
    #print(na_rows)

    merged_df.fillna(0, inplace=True)  # Fill NaN values with 0
    #print('\nShape of the energy data:', energy_data.shape, 'and the shape of the weather data:', weather_data.shape, 'Shape of the merged data:', merged_df.shape)
    #print('\n\nhead of the merged data:', merged_df.head(), '\n', explore_data(merged_df, 'Merged Data'))

    return merged_df


if __name__ == "__main__":
    energy_df = data_loader.load_kaggle_data(file_names.kaggle_files.energy_data_file)
    weather_df = data_loader.load_kaggle_data(file_names.kaggle_files.weather_data_file)

    # COMMENT THIS OUT IF YOU NO LONGER WANT TO EXPLORE THE DATA
    # Explore the data 
    #explore_data(energy_df, "Energy Data")
    #explore_data(weather_df, "Weather Data")

    # Preprocess the data
    merged_df = _data_preprocessor(energy_df, weather_df)
    
    # Run the forecaster
    #forecaster(merged_df)

    ridge_forecast = data_loader.load_model_results(file_names.model_result_files.ridge_forecast_berend)
    elnet_forecast = data_loader.load_model_results(file_names.model_result_files.elnet_forecast_berend)
    
    merged_df['time'] = pd.to_datetime(merged_df['time'])
    ridge_forecast['forecast_time'] = pd.to_datetime(ridge_forecast['forecast_time'])
    elnet_forecast['forecast_time'] = pd.to_datetime(elnet_forecast['forecast_time'])

    # list of times not in forecasts ('forecast_time') but present in merged_df ('time')
    missing_times_ridge = set(merged_df['time']) - set(ridge_forecast['forecast_time'])
    #print('Missing times in the ridge forecast:', len(missing_times_ridge), 'Merged has: ', len(merged_df), 'Ridge has:', len(ridge_forecast))
    #print('misssing values are: ', missing_times)
    
    #print('\n\n\n\n\n')
    # list of times not in forecasts ('forecast_time') but present in merged_df ('time')
    missing_times_elnet = set(merged_df['time']) - set(elnet_forecast['forecast_time'])
    #print('Missing times in the elastic net forecast:', len(missing_times_elnet), 'Merged has: ', len(merged_df), 'Elastic net has:', len(elnet_forecast))
    #print('misssing values are: ', missing_times)
    

    # Ensure forecast_time is the index in original forecasts
    ridge_forecast.set_index('forecast_time', inplace=True)
    elnet_forecast.set_index('forecast_time', inplace=True)

    # Convert missing times to proper datetime index
    missing_times_ridge = pd.DatetimeIndex(sorted(missing_times_ridge), name='forecast_time')
    missing_times_elnet = pd.DatetimeIndex(sorted(missing_times_elnet), name='forecast_time')

    # Define default values for missing rows
    default_row = {
        'predicted_solar_generation': 0.0,
        'forecast_made_at': pd.NaT,
        'generation solar': 0.0
    }

    ridge_missing_df = pd.DataFrame(default_row, index=missing_times_ridge)
    elnet_missing_df = pd.DataFrame(default_row, index=missing_times_elnet)

    # Concatenate and sort
    ridge_forecast = pd.concat([ridge_forecast, ridge_missing_df])
    ridge_forecast = ridge_forecast[~ridge_forecast.index.duplicated(keep='first')]
    ridge_forecast = ridge_forecast.sort_index()

    elnet_forecast = pd.concat([elnet_forecast, elnet_missing_df])
    elnet_forecast = elnet_forecast[~elnet_forecast.index.duplicated(keep='first')]
    elnet_forecast = elnet_forecast.sort_index()

    ambiguous_times = pd.to_datetime([
        '2015-10-25 02:00',
        '2016-10-30 02:00',
        '2017-10-29 02:00',
        '2018-10-28 02:00'
    ], format='%Y-%m-%d %H:%M')
    ridge_forecast.reset_index(drop=False, inplace=True)
    elnet_forecast.reset_index(drop=False, inplace=True)

    extra_ridge = []
    extra_elnet = []
    def duplicate_and_boost(df: pd.DataFrame, ambiguous_times: list) -> pd.DataFrame:
        for dt in ambiguous_times:
            # Select original row (first occurrence of 02:00)
            original_row = df[df['forecast_time'] == dt].copy() if dt in df['forecast_time'].values else None
            if original_row is None:
                print(f"Missing original row for {dt}")
            if original_row is not None and not original_row.empty:
                print(f'Original row for {dt}: \n', original_row)
                # Create new row with +10%
                boosted_row = original_row.copy().reset_index(drop=True)

                # Only apply +10% to numeric columns
                boosted_row['predicted_solar_generation'] = boosted_row['predicted_solar_generation'] * 1.1

                # Set the forecast_time of the new row (since reset_index dropped it)
                boosted_row['forecast_time'] = dt

                # Append the boosted row
                df = pd.concat([df, boosted_row], ignore_index=True)
                print('new df shape:', df.shape, '\n', boosted_row)

        # Sort by time again
        df.sort_values(by='forecast_time', inplace=True)
        return df.reset_index(drop=True)

    ridge_forecast = duplicate_and_boost(ridge_forecast, ambiguous_times)
    elnet_forecast = duplicate_and_boost(elnet_forecast, ambiguous_times)


    # Done!
    print("Final Ridge Forecast:", ridge_forecast.shape)
    print("Final Elastic Net Forecast:", elnet_forecast.shape)


    ridge_forecast.to_csv('data/data_files/model_results/solar_forecast_ridge_berend_filled.csv', index=False)
    elnet_forecast.to_csv('data/data_files/model_results/solar_forecast_elnet_berend_filled.csv', index=False)

