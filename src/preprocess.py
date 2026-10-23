"""Module 2: Data Preprocessing for Geospatial Ride-Share Demand Forecasting.

This script performs the following steps:
1. Loads raw taxi trip data from the data/ directory.
2. Identifies and standardizes column names (pickup_datetime, latitude, longitude).
3. Parses pickup_datetime into pandas datetime objects.
4. Drops rows with missing coordinates or timestamps.
5. Filters out coordinates outside valid NYC geographic boundaries.
6. Removes duplicate ride entries.
7. Samples up to 100,000 records with random_state=42 if the dataset is large.
8. Saves the clean dataset to data/cleaned_taxi.csv.
9. Prints comprehensive summary statistics.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATA_DIR,
    RAW_DATA_FILE,
    CLEANED_DATA_FILE,
    NYC_LAT_MIN,
    NYC_LAT_MAX,
    NYC_LON_MIN,
    NYC_LON_MAX,
    SAMPLE_SIZE,
    RANDOM_STATE,
)
from src.download_data import download_or_prepare_dataset


def find_raw_dataset_path() -> Path:
    """Finds an existing raw dataset in the data/ directory.

    If none is found, triggers automatic data acquisition.
    """
    if RAW_DATA_FILE.exists():
        return RAW_DATA_FILE

    # Check for any other CSV in data/ that is not the cleaned output
    csv_candidates = [
        f for f in DATA_DIR.glob("*.csv") if f.name != CLEANED_DATA_FILE.name
    ]
    if csv_candidates:
        return csv_candidates[0]

    # If no CSV exists, prepare or download it
    print("[*] No raw CSV found in data/. Triggering dataset acquisition...")
    return download_or_prepare_dataset()


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Maps varying dataset column names to standardized names:

    - pickup_datetime
    - latitude
    - longitude
    """
    rename_mapping = {}

    # Datetime mappings
    datetime_candidates = [
        "tpep_pickup_datetime",
        "trip_pickup_datetime",
        "pickup_datetime",
        "Date/Time",
        "datetime",
        "pickup_time",
    ]
    for col in df.columns:
        if col.strip() in datetime_candidates:
            rename_mapping[col] = "pickup_datetime"
            break

    # Latitude mappings
    latitude_candidates = [
        "pickup_latitude",
        "latitude",
        "lat",
        "Pickup_Latitude",
        "Pickup_lat",
    ]
    for col in df.columns:
        if col.strip() in latitude_candidates:
            rename_mapping[col] = "latitude"
            break

    # Longitude mappings
    longitude_candidates = [
        "pickup_longitude",
        "longitude",
        "lon",
        "lng",
        "Pickup_Longitude",
        "Pickup_lon",
    ]
    for col in df.columns:
        if col.strip() in longitude_candidates:
            rename_mapping[col] = "longitude"
            break

    df = df.rename(columns=rename_mapping)

    # Validate that all required columns are present
    required_cols = ["pickup_datetime", "latitude", "longitude"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Could not identify required columns: {missing}. Available columns: {list(df.columns)}"
        )

    return df


def preprocess_taxi_data(
    input_path: Path = None,
    output_path: Path = CLEANED_DATA_FILE,
    max_records: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Cleans and standardizes raw taxi records."""
    if input_path is None:
        input_path = find_raw_dataset_path()

    print("=" * 70)
    print(" MODULE 2: DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"Loading raw dataset from: {input_path}")

    # Step 1: Load raw data
    df = pd.read_csv(input_path)
    original_count = len(df)
    print(f"[1] Original number of records: {original_count:,}")

    # Step 2: Standardize column names
    df = standardize_column_names(df)
    print(f"[2] Standardized columns identified: ['pickup_datetime', 'latitude', 'longitude']")

    # Step 3: Parse pickup_datetime
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

    # Step 4: Ensure latitude and longitude are numeric
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Step 5: Remove missing values
    before_missing = len(df)
    df = df.dropna(subset=["pickup_datetime", "latitude", "longitude"]).copy()
    missing_dropped = before_missing - len(df)
    print(f"[3] Removed missing values: {missing_dropped:,}")

    # Step 6: Filter invalid NYC geographic coordinates
    before_geo = len(df)
    geo_mask = (
        (df["latitude"] >= NYC_LAT_MIN)
        & (df["latitude"] <= NYC_LAT_MAX)
        & (df["longitude"] >= NYC_LON_MIN)
        & (df["longitude"] <= NYC_LON_MAX)
    )
    df = df[geo_mask].copy()
    geo_dropped = before_geo - len(df)
    print(
        f"[4] Removed out-of-boundary coordinates: {geo_dropped:,} "
        f"(Bounds: Lat [{NYC_LAT_MIN}, {NYC_LAT_MAX}], Lon [{NYC_LON_MIN}, {NYC_LON_MAX}])"
    )

    # Step 7: Remove duplicate records
    before_dups = len(df)
    df = df.drop_duplicates(subset=["pickup_datetime", "latitude", "longitude"]).copy()
    dups_dropped = before_dups - len(df)
    print(f"[5] Removed duplicate records: {dups_dropped:,}")

    # Step 8: Sample up to max_records if dataset is very large
    if len(df) > max_records:
        print(f"[6] Sampling dataset from {len(df):,} to {max_records:,} records (random_state={random_state})...")
        df = df.sample(n=max_records, random_state=random_state).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Sort chronologically
    df = df.sort_values(by="pickup_datetime").reset_index(drop=True)

    # Step 9: Save cleaned dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[7] Cleaned dataset saved to: {output_path}")

    # Step 10: Print final summary statistics
    cleaned_count = len(df)
    removed_count = original_count - cleaned_count

    print("-" * 70)
    print(" PREPROCESSING SUMMARY & VALIDATION")
    print("-" * 70)
    print(f"  - Original number of records       : {original_count:,}")
    print(f"  - Number of records after cleaning : {cleaned_count:,}")
    print(f"  - Total removed records            : {removed_count:,} ({(removed_count / original_count) * 100:.2f}%)")
    print(f"  - Latitude range                   : [{df['latitude'].min():.6f}, {df['latitude'].max():.6f}]")
    print(f"  - Longitude range                  : [{df['longitude'].min():.6f}, {df['longitude'].max():.6f}]")
    print(f"  - Datetime range                   : {df['pickup_datetime'].min()} to {df['pickup_datetime'].max()}")
    print("=" * 70)

    return df


if __name__ == "__main__":
    preprocess_taxi_data()
