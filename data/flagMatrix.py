#!/usr/bin/env python3
"""
Dynamic Data Preprocessing Module – Rolling Window Flagging

This module implements the dynamic data preprocessing (DDP) as described in Nikodinoska et al. (2022),
translated into Python with a focus on quality checking only (i.e. without modifying the raw data).
The goal is to produce a flag matrix (a DataFrame) where each row corresponds to one day (starting from the day after the rolling window)
and each column corresponds to a forecast provider (e.g., A1, A2, …). For each day, the module:
  • Extracts the new day‐ahead forecast for that day (Step 1) and checks it against quality criteria.
  • Forms a historical forecast series by aggregating all observations in the rolling window (Step 2)
    preceding that day and checks it against its own quality criteria.
  • Immediately upon the first violation in each step, a corresponding violation code is recorded.
    (Codes 1–5 are for new day checks, and codes 6–10 are for historical checks.)
  • A flag of 0 indicates that the provider passed both checks and its forecast can be used for that day.
  
The module is parameterized:
  - You can set the resolution (default 24 observations per day; can be changed to 96 for quarter-hourly data).
  - You can set the rolling window length (in days).
  - Threshold values for new forecasts and historical forecasts can be passed as parameters.

This flag matrix can later be used in a separate function to perform the actual data transformations
(e.g. interpolation or zeroing out) before model estimation.
"""

import numpy as np
import pandas as pd

from data.data_loader import DataLoader
from configuration import FileNames, ModelSettings

# -----------------------------------------------------------------------------
# Helper functions for quality checking (do not modify data)
# -----------------------------------------------------------------------------

def _check_new_forecast_flags(B, max_install, thresholds):
    """
    Check a new day forecast (B) for one provider against quality criteria (Step 1)
    without modifying the data. It iterates through the forecast array and, upon the first
    violation of any criterion, returns a violation code.
    
    The criteria (with example threshold names and violation codes) are:
      1. Count of impossible/unfeasible values (value < 0 or > max_install): if exceeded, return 1.
      2. Maximum consecutive zeros: if exceeded, return 2.
      3. Maximum consecutive identical nonzero values: if exceeded, return 3.
      4. Total missing values (NAs) count: if exceeded, return 4.
      5. Maximum consecutive missing values: if exceeded, return 5.
      
    If no violation is found, the function returns 0.
    
    Parameters:
      B : np.array
          Array of forecast values for one day (length equals resolution, e.g. 24 or 96).
      max_install : float
          Maximum installed capacity.
      thresholds : dict
          Dictionary with keys:
            - 'unfeasible': maximum allowed count of impossible values.
            - 'zero_reps': maximum allowed consecutive zeros.
            - 'nonzero_reps': maximum allowed consecutive identical nonzero values.
            - 'total_na': maximum allowed total missing values.
            - 'consecutive_na': maximum allowed consecutive missing values.
    
    Returns:
      int: Violation code (0 if no violation; otherwise one of 1, 2, 3, 4, or 5).
    """
    n = len(B)
    count_unfeasible = 0
    consecutive_zeros = 0
    consecutive_nonzero = 0
    total_na = 0
    consecutive_na = 0
    
    for i in range(n):
        value = B[i]
        
        # Check for missing (NA) values first:
        if np.isnan(value):
            total_na += 1
            consecutive_na += 1
            if consecutive_na > thresholds['consecutive_na']:
                return 5
            if total_na > thresholds['total_na']:
                return 4
            continue
        else:
            consecutive_na = 0
        
        # Check for impossible/unfeasible values:
        if value < 0 or value > max_install:
            count_unfeasible += 1
            if count_unfeasible > thresholds['unfeasible']:
                return 1
        
        # Check for consecutive zeros:
        if value == 0:
            consecutive_zeros += 1
            if consecutive_zeros > thresholds['zero_reps']:
                return 2
        else:
            consecutive_zeros = 0
        
        # Check for repeated identical nonzero values:
        if i > 0 and (not np.isnan(B[i-1])) and B[i-1] == value and value != 0:
            consecutive_nonzero += 1
            if consecutive_nonzero > thresholds['nonzero_reps']:
                return 3
        else:
            consecutive_nonzero = 1  # reset counter (start at 1)
    
    return 0

def _check_hist_forecast_flags(B_hist, max_install, thresholds):
    """
    Check a historical forecast (B_hist) for one provider (Step 2) without modifying it.
    The criteria (with example threshold names and violation codes) are:
      6. Count of impossible values: if exceeded, return 6.
      7. Maximum consecutive zeros: if exceeded, return 7.
      8. Maximum consecutive identical nonzero values: if exceeded, return 8.
      9. Total missing values count: if exceeded, return 9.
      10. Maximum consecutive missing values: if exceeded, return 10.
      
    If no violation is found, the function returns 0.
    
    Parameters:
      B_hist : np.array
          Historical forecast array for one provider (rolling window).
      max_install : float
          Maximum installed capacity.
      thresholds : dict
          Dictionary with keys:
            - 'unfeasible': maximum allowed count of impossible values.
            - 'zero_reps': maximum allowed consecutive zeros.
            - 'nonzero_reps': maximum allowed consecutive identical nonzero values.
            - 'total_na': maximum allowed total missing values.
            - 'consecutive_na': maximum allowed consecutive missing values.
    
    Returns:
      int: Violation code (0 if no violation; otherwise one of 6, 7, 8, 9, or 10).
    """
    n = len(B_hist)
    count_unfeasible = 0
    consecutive_zeros = 0
    consecutive_nonzero = 0
    total_na = 0
    consecutive_na = 0
    
    for i in range(n):
        value = B_hist[i]
        if np.isnan(value):
            total_na += 1
            consecutive_na += 1
            if consecutive_na > thresholds['consecutive_na']:
                return 10
            if total_na > thresholds['total_na']:
                return 9
            continue
        else:
            consecutive_na = 0
        
        if value < 0 or value > max_install:
            count_unfeasible += 1
            if count_unfeasible > thresholds['unfeasible']:
                return 6
        
        if value == 0:
            consecutive_zeros += 1
            if consecutive_zeros > thresholds['zero_reps']:
                return 7
        else:
            consecutive_zeros = 0
        
        if i > 0 and (not np.isnan(B_hist[i-1])) and B_hist[i-1] == value and value != 0:
            consecutive_nonzero += 1
            if consecutive_nonzero > thresholds['nonzero_reps']:
                return 8
        else:
            consecutive_nonzero = 1
    
    return 0

def _process_provider_flags(B, B_hist, max_install, thresholds_new, thresholds_hist):
    """
    Process a single forecast provider for one day by checking both the new day forecast (B)
    and the historical forecast (B_hist) against their respective criteria.
    
    The procedure is as follows:
      - First, check the new day forecast (Step 1) using _check_new_forecast_flags(). If any violation
        is found, immediately return that violation code (1–5).
      - If the new day forecast passes (returns 0), then check the historical forecast (Step 2)
        using _check_hist_forecast_flags(). If a violation is found, return that violation code (6–10).
      - If both checks return 0, then the provider is valid (return 0).
    
    Parameters:
      B : np.array
          New day forecast for one provider.
      B_hist : np.array
          Historical forecast for the same provider (aggregated over the rolling window).
      max_install : float
          Maximum installed capacity.
      thresholds_new : dict
          Thresholds for new day forecasts.
      thresholds_hist : dict
          Thresholds for historical forecasts.
    
    Returns:
      int: Final flag code for the provider on that day (0 if valid; nonzero if any violation).
    """
    flag_new = _check_new_forecast_flags(B, max_install, thresholds_new)
    if flag_new != 0:
        return flag_new
    else:
        flag_hist = _check_hist_forecast_flags(B_hist, max_install, thresholds_hist)
        return flag_hist

# -----------------------------------------------------------------------------
# Rolling Window Processing over a Full Dataset
# -----------------------------------------------------------------------------

def _process_full_dataset(df, rolling_window_days, resolution, max_install,
                         thresholds_new, thresholds_hist):
    """
    Process an entire CSV dataset (spanning multiple years) using a rolling window.
    
    The input DataFrame must have:
      - A ModelSettings.datetime_col column (with date and time).
      - Forecast columns (e.g., 'A1', 'A2', ..., 'A5').
      - (Other columns, like 'generation solar', are ignored here.)
    
    The procedure is as follows:
      - The DataFrame is sorted by datetime.
      - The data is grouped by day (each day should have exactly 'resolution' rows).
      - For each day starting from day (rolling_window_days + 1):
          * The "new day" forecasts are those from the current day.
          * The historical data for each provider is formed by concatenating the forecasts
            from the previous 'rolling_window_days' days (i.e., resolution * rolling_window_days values).
          * For each forecast provider, process the new day forecast (using _check_new_forecast_flags)
            and the historical forecast (using _check_hist_forecast_flags) via _process_provider_flags().
          * Record the resulting flag for that provider on that day.
      - The output is a DataFrame where each row corresponds to one day (the new day) and each column
        (besides 'date') corresponds to a forecast provider. A 0 means the provider is valid for that day;
        any nonzero value is the violation code indicating why it is not valid.
    
    Parameters:
      df : pd.DataFrame
          The full dataset containing a ModelSettings.datetime_col column and forecast columns.
      rolling_window_days : int
          The number of days in the rolling window (e.g., 165).
      resolution : int
          Number of observations per day (e.g., 24 for hourly, 96 for quarter-hourly).
      max_install : float
          Maximum installed capacity.
      thresholds_new : dict
          Thresholds for new day forecasts.
      thresholds_hist : dict
          Thresholds for historical forecasts.
    
    Returns:
      pd.DataFrame: A DataFrame with one row per day (starting from day (rolling_window_days + 1))
                    and columns for each forecast provider containing the flag codes.
    """
    # Convert ModelSettings.datetime_col to pandas datetime and add a 'date' column (date only)
    df[ModelSettings.datetime_col] = pd.to_datetime(df[ModelSettings.datetime_col])
    df['date'] = df[ModelSettings.datetime_col].dt.date
    # optional: Sort the DataFrame by datetime (if not already sorted) by removing commented line below.
    #df = df.sort_values(by=ModelSettings.datetime_col).reset_index(drop=True)
    
    # Identify forecast provider columns. Assume all columns except ModelSettings.datetime_col, 'date', and 'generation solar' are forecast columns.
    exclude_cols = {ModelSettings.datetime_col, 'date', ModelSettings.target}
    forecast_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Group data by day; each group should have exactly 'resolution' rows.
    grouped = df.groupby('date')
    # Build a dictionary mapping each day (date) to a DataFrame of that day’s data.
    day_dict = {day: group for day, group in grouped if len(group) == resolution}
    
    # Get a sorted list of valid days
    sorted_days = sorted(day_dict.keys())
    
    # Prepare a list to collect the flag results per day.
    flag_rows = []
    
    # Loop over days starting from index = rolling_window_days (i.e., day number rolling_window_days + 1)
    for idx in range(rolling_window_days, len(sorted_days)):
        current_day = sorted_days[idx]
        # New day forecasts for current_day: create a dictionary mapping each provider to its np.array of forecasts.
        new_day_data = {}
        day_df = day_dict[current_day]
        for prov in forecast_cols:
            new_day_data[prov] = day_df[prov].to_numpy()
        
        # Build historical data for each provider by concatenating forecasts from the previous 'rolling_window_days' days.
        hist_data = {}
        # Iterate over the days in the rolling window: from index idx - rolling_window_days to idx - 1
        hist_arrays = {prov: [] for prov in forecast_cols}
        for j in range(idx - rolling_window_days, idx):
            day_j = sorted_days[j]
            day_j_df = day_dict[day_j]
            for prov in forecast_cols:
                hist_arrays[prov].append(day_j_df[prov].to_numpy())
        # Concatenate the arrays along the time axis for each provider.
        for prov in forecast_cols:
            # Flatten into one 1D array (length = resolution * rolling_window_days)
            hist_data[prov] = np.concatenate(hist_arrays[prov])
        
        # Now for each provider, get the flag from processing the new day forecast and historical forecast.
        day_flags = {}
        for prov in forecast_cols:
            B = new_day_data[prov]
            B_hist = hist_data[prov]
            flag = _process_provider_flags(B, B_hist, max_install, thresholds_new, thresholds_hist)
            day_flags[prov] = flag
        # Record the date and the flags as one row.
        day_flags['date'] = current_day
        flag_rows.append(day_flags)
    
    # Create a DataFrame from the flag rows.
    flag_df = pd.DataFrame(flag_rows)
    print('flag_df: ',flag_df.head())
    # Ensure 'date' is the first column.
    cols = flag_df.columns.tolist()
    print(cols)
    cols.remove('date')
    flag_df = flag_df[['date'] + cols]
    
    return flag_df

# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
def run_flag_matrix():
    # Load your CSV file
    df = DataLoader().load_input_data(FileNames.input_files.combined_forecasts)
    print(df.head())
    # Convert datetime using ISO format (since your CSV uses "2015-01-13 00:00:00")
    df[ModelSettings.datetime_col] = pd.to_datetime(df[ModelSettings.datetime_col], format="%Y-%m-%d %H:%M:%S")
    
    # Define the parameters
    rolling_window_days = 165
    resolution = 24  # Or 96 if your data is quarter-hourly
    max_install = 1000000000.0

    thresholds_new = {
        'unfeasible': 5,
        'zero_reps': 10,
        'nonzero_reps': 10,
        'total_na': 10,
        'consecutive_na': 5
    }

    thresholds_hist = {
        'unfeasible': 100,
        'zero_reps': 960,
        'nonzero_reps': 960,
        'total_na': 960,
        'consecutive_na': 96
    }

    # Run the preprocessing to generate the flag matrix (one row per day, one column per forecast provider)
    flag_matrix = _process_full_dataset(df, rolling_window_days, resolution, max_install, thresholds_new, thresholds_hist)

    # View the flag matrix on screen
    print(flag_matrix)
    
    # Save the resulting flag matrix to a CSV file
    output_file = "/Users/gebruiker/Documents/GitHub/SeminarML25_G12/data/input_files/flag_matrix.csv"
    flag_matrix.to_csv(output_file, index=False)
    print(f"Flag matrix saved to {output_file}")




if __name__ == "__main__":
    # Load your CSV file
    df = DataLoader().load_input_data(FileNames.input_files.combined_forecasts)
    
    # Convert datetime using ISO format (since your CSV uses "2015-01-13 00:00:00")
    df[ModelSettings.datetime_col] = pd.to_datetime(df[ModelSettings.datetime_col], format="%Y-%m-%d %H:%M:%S")
    
    # Define the parameters
    rolling_window_days = 165
    resolution = 24  # Or 96 if your data is quarter-hourly
    max_install = 1000000000.0

    thresholds_new = {
        'unfeasible': 5,
        'zero_reps': 10,
        'nonzero_reps': 10,
        'total_na': 10,
        'consecutive_na': 5
    }

    thresholds_hist = {
        'unfeasible': 100,
        'zero_reps': 960,
        'nonzero_reps': 960,
        'total_na': 960,
        'consecutive_na': 96
    }

    # Run the preprocessing to generate the flag matrix (one row per day, one column per forecast provider)
    flag_matrix = _process_full_dataset(df, rolling_window_days, resolution, max_install, thresholds_new, thresholds_hist)

    # View the flag matrix on screen
    print(flag_matrix)
    
    # Save the resulting flag matrix to a CSV file
    output_file = "/Users/gebruiker/Documents/GitHub/SeminarML25_G12/data/input_files/flag_matrix.csv"
    flag_matrix.to_csv(output_file, index=False)
    print(f"Flag matrix saved to {output_file}")
