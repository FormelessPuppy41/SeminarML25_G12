import pandas as pd
import os
import csv

def load_solar_csv(file_path, value_name):
    """
    Reads a solar CSV file, normalizing header differences between forecast and actual files.
    
    For forecasts, the header is like:
      "Datum;""Von"";""bis"";""MW"";"
    For actuals, the header is like:
      Datum;von;bis;MW
    
    After cleaning, the required columns become: 'datum', 'von', 'mw'.
    
    The function then:
      - Selects these columns,
      - Combines 'datum' and 'von' into a single datetime (using day-first format),
      - Converts 'mw' (which may use a comma as decimal separator) to a numeric value,
      - Renames the 'mw' column to the given value_name (e.g., 'forecast' or 'actual'),
      - Returns a DataFrame with columns: [Datetime, value_name].
    """
    # Read the CSV file with quoting disabled so extra quotes are not interpreted.
    df = pd.read_csv(
        file_path,
        sep=';',
        header=0,
        quoting=csv.QUOTE_NONE,
        dtype=str
    )
    
    # Clean the header: remove quotes, strip whitespace, and convert to lowercase.
    df.columns = [col.replace('"', '').strip().lower() for col in df.columns]
    
    # Drop an extra column if it exists (e.g. due to trailing semicolon).
    if df.columns[-1] == '':
        df = df.iloc[:, :-1]
    
    # Check that the required columns exist.
    required_cols = ['datum', 'von', 'mw']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Not all required columns found in {file_path}. Found columns: {df.columns.tolist()}")
    
    # Select only the required columns.
    df = df[required_cols].copy()
    
    # Combine 'datum' and 'von' into a single datetime column.
    df['datetime'] = pd.to_datetime(
        df['datum'] + ' ' + df['von'],
        dayfirst=True,
        errors='coerce'
    )
    
    # Convert 'mw' to numeric, replacing commas with dots.
    df['mw'] = df['mw'].str.replace(',', '.')
    df['mw'] = pd.to_numeric(df['mw'], errors='coerce')
    
    # Drop rows where the datetime or mw could not be parsed.
    df.dropna(subset=['datetime', 'mw'], inplace=True)
    
    # Keep only the 'datetime' and 'mw' columns, and rename them.
    df = df[['datetime', 'mw']].copy()
    df.rename(columns={'datetime': 'Datetime', 'mw': value_name}, inplace=True)
    
    return df

# ------------------- MAIN SCRIPT -------------------

# Adjust the years range if you have multiple years.
years = range(2011, 2025) 
forecast_frames = []
actual_frames = []

for year in years:
    forecast_file = f"50Hertz/Solarenergie_Prognose_{year}.csv"
    actual_file   = f"50Hertz/Solarenergie_Hochrechnung_{year}.csv"
    
    # Process forecast file if it exists.
    if os.path.exists(forecast_file):
        try:
            df_forecast = load_solar_csv(forecast_file, 'forecast')
            forecast_frames.append(df_forecast)
            print(f"Processed forecast file: {forecast_file}")
        except Exception as e:
            print(f"Error processing forecast file {forecast_file}: {e}")
    else:
        print(f"Forecast file not found: {forecast_file}")
    
    # Process actual file if it exists.
    if os.path.exists(actual_file):
        try:
            df_actual = load_solar_csv(actual_file, 'actual')
            actual_frames.append(df_actual)
            print(f"Processed actual file: {actual_file}")
        except Exception as e:
            print(f"Error processing actual file {actual_file}: {e}")
    else:
        print(f"Actual file not found: {actual_file}")

# Combine all forecast and actual dataframes.
df_forecast_all = pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame(columns=['Datetime','forecast'])
df_actual_all   = pd.concat(actual_frames, ignore_index=True)   if actual_frames   else pd.DataFrame(columns=['Datetime','actual'])

# Merge on the Datetime column using an outer join.
df_merged = pd.merge(df_forecast_all, df_actual_all, on='Datetime', how='outer')

# Sort by Datetime.
df_merged.sort_values('Datetime', inplace=True)
df_merged.reset_index(drop=True, inplace=True)

# Save the combined data to a new CSV file.
output_file = "Solar_Combined.csv"
df_merged.to_csv(output_file, index=False)
print(f"Combined CSV written to {output_file}")
