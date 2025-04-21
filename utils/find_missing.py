

import pandas as pd
from datetime import datetime, timedelta

import pandas as pd
from datetime import datetime, timedelta

def check_missing_intervals(file_path):
    # Load only needed columns (ignore extra/missing ones)
    df = pd.read_csv(file_path, sep=';', encoding='utf-16')
    df.columns = df.columns.str.strip()  # Clean up header names
    print(df)
    print(df.columns)

    # Combine date and time into datetime objects
    df['timestamp'] = pd.to_datetime(df['Datum'] + ' ' + df['von'], format='%d.%m.%Y %H:%M')

    # Detect the year from the first timestamp
    year = df['timestamp'].dt.year.min()
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)

    # Generate all 15-minute intervals for the entire year
    full_range = pd.date_range(start=start, end=end, freq='15min')

    # Identify missing timestamps
    actual_set = set(df['timestamp'])
    expected_set = set(full_range)
    missing = sorted(expected_set - actual_set)

    # Output results
    if missing:
        print(f"Missing intervals: {len(missing)}")
        for ts in missing:
            print(ts.strftime('%d.%m.%Y %H:%M'))
    else:
        print("No missing intervals. All 15-minute slots for the year are present.")

# Example usage
check_missing_intervals("Solarenergie_Hochrechnung_2016.csv")
