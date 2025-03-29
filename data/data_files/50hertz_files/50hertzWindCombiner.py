import pandas as pd
import os
import csv

def load_wind_csv(file_path, value_name):
    """
    Reads a semicolon-delimited wind CSV with a header line like:
       "Datum;""Von"";""bis"";""MW"";""Onshore MW"";""Offshore MW"";"
    and data rows like:
       01.01.2011;00:00;00:15;4209,13;0;0;
       
    This function disables normal quoting, then cleans the column names,
    selects only the columns "Datum", "Von", and "MW", combines Datum and Von
    into a datetime, and converts MW to numeric.
    
    :param file_path: Path to the CSV file.
    :param value_name: A string ("forecast" or "actual") used to rename the MW column.
    :return: A DataFrame with columns [Datetime, value_name].
    """
    # Read the CSV file with quoting disabled so that the extra quotes are treated as literal
    df = pd.read_csv(
        file_path,
        sep=';',
        header=0,
        quoting=csv.QUOTE_NONE,
        dtype=str
    )
    
    # Clean the column names by removing all double quotes and stripping whitespace.
    df.columns = [col.replace('"', '').strip() for col in df.columns]
    
    # If an extra empty column is created because of a trailing semicolon, drop it.
    if df.columns[-1] == '':
        df = df.iloc[:, :-1]
    
    # At this point, we expect the columns to be:
    #   Datum, Von, bis, MW, Onshore MW, Offshore MW
    # We only need Datum, Von, and MW.
    df = df[['Datum', 'Von', 'MW']]
    
    # Combine Datum and Von into a single datetime column.
    df['Datetime'] = pd.to_datetime(
        df['Datum'] + ' ' + df['Von'],
        dayfirst=True,
        errors='coerce'
    )
    
    # Convert MW to numeric, replacing comma decimals with dot.
    df['MW'] = df['MW'].str.replace(',', '.')
    df['MW'] = pd.to_numeric(df['MW'], errors='coerce')
    
    # Drop rows that couldn't be parsed.
    df.dropna(subset=['Datetime', 'MW'], inplace=True)
    
    # Keep only the Datetime and MW columns, renaming MW as specified.
    df = df[['Datetime', 'MW']].copy()
    df.rename(columns={'MW': value_name}, inplace=True)
    return df

# -------------------- MAIN SCRIPT --------------------

years = range(2011, 2025)  # For years 2011 through 2024
forecast_frames = []
actual_frames = []

for year in years:
    forecast_file = f"50Hertz/Windenergie_Prognose_{year}.csv"
    actual_file   = f"50Hertz/Windenergie_Hochrechnung_{year}.csv"
    
    # Process forecast file if available
    if os.path.exists(forecast_file):
        try:
            df_forecast = load_wind_csv(forecast_file, 'forecast')
            forecast_frames.append(df_forecast)
            print(f"Processed forecast file: {forecast_file}")
        except Exception as e:
            print(f"Error processing forecast file {forecast_file}: {e}")
    else:
        print(f"Forecast file not found: {forecast_file}")
    
    # Process actual file if available
    if os.path.exists(actual_file):
        try:
            df_actual = load_wind_csv(actual_file, 'actual')
            actual_frames.append(df_actual)
            print(f"Processed actual file: {actual_file}")
        except Exception as e:
            print(f"Error processing actual file {actual_file}: {e}")
    else:
        print(f"Actual file not found: {actual_file}")

# Combine all the forecast and actual dataframes.
df_forecast_all = pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame(columns=['Datetime','forecast'])
df_actual_all   = pd.concat(actual_frames,   ignore_index=True) if actual_frames   else pd.DataFrame(columns=['Datetime','actual'])

# Merge both on the Datetime column (outer join to include all time points).
df_merged = pd.merge(df_forecast_all, df_actual_all, on='Datetime', how='outer')

# Sort by Datetime.
df_merged.sort_values('Datetime', inplace=True)
df_merged.reset_index(drop=True, inplace=True)

# Save the combined data to a new CSV file.
output_file = "Windenergie_Combined.csv"
df_merged.to_csv(output_file, index=False)
print(f"Combined CSV written to {output_file}")
