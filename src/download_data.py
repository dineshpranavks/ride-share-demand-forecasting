"""Data acquisition script for Geospatial Ride-Share Demand Forecasting.

Downloads the official Kaggle Taxi Demand & Fare Prediction dataset:
Dataset ID: sohaibanwaar1203/taxidemandfarepredictiondataset

Uses kagglehub to acquire and verify the real NYC yellow taxi trip data:
- tpep_pickup_datetime (pickup timestamp)
- pickup_latitude (pickup latitude)
- pickup_longitude (pickup longitude)

Processes with chunked CSV reading to extract a reproducible sample of up to 100,000 trips
without excessive memory consumption.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import kagglehub

# Ensure package root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    RAW_DATA_FILE,
    SAMPLE_SIZE,
    RANDOM_STATE,
)

KAGGLE_DATASET_ID = "sohaibanwaar1203/taxidemandfarepredictiondataset"


def download_kaggle_dataset() -> Path:
    """Downloads or locates the Kaggle taxi dataset using kagglehub."""
    print("=" * 70)
    print(" [Step 1] Kaggle Dataset Acquisition via kagglehub")
    print(f" Dataset ID: {KAGGLE_DATASET_ID}")
    print("=" * 70)

    try:
        path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
        dataset_path = Path(path)
    except Exception as e:
        # Fallback check in default kagglehub cache directory
        cache_base = Path.home() / ".cache" / "kagglehub" / "datasets" / KAGGLE_DATASET_ID
        found_versions = list(cache_base.glob("versions/*"))
        if found_versions:
            dataset_path = found_versions[-1]
            print(f"[*] Found cached Kaggle dataset at: {dataset_path}")
        else:
            raise RuntimeError(f"Failed to acquire Kaggle dataset: {e}")

    print(f"[+] Path to dataset files: {dataset_path}")
    return dataset_path


def inspect_dataset_files(dataset_path: Path) -> list[Path]:
    """Inspects downloaded files and displays actual columns and coordinates."""
    csv_files = sorted(list(dataset_path.glob("*.csv")))
    if not csv_files:
        # Check subdirectories
        csv_files = sorted(list(dataset_path.rglob("*.csv")))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in dataset path: {dataset_path}")

    print(f"\n[*] Discovered {len(csv_files)} dataset file(s):")
    for f in csv_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    - {f.name} ({size_mb:.1f} MB)")

    # Inspect schema of first available file
    sample_file = csv_files[0]
    preview_df = pd.read_csv(sample_file, nrows=3)
    cols = list(preview_df.columns)
    print(f"\n[*] Inspecting schema for '{sample_file.name}':")
    print(f"    Columns ({len(cols)}): {cols}")

    # Requirement 3: Identify pickup timestamps, pickup latitude, and pickup longitude
    has_pickup_time = any("pickup_datetime" in c for c in cols)
    has_pickup_lat = any("pickup_latitude" in c for c in cols)
    has_pickup_lon = any("pickup_longitude" in c for c in cols)

    print(f"    - Pickup timestamp identified : {has_pickup_time} ('tpep_pickup_datetime')")
    print(f"    - Pickup latitude identified  : {has_pickup_lat} ('pickup_latitude')")
    print(f"    - Pickup longitude identified : {has_pickup_lon} ('pickup_longitude')")

    if not (has_pickup_time and has_pickup_lat and has_pickup_lon):
        raise ValueError(
            "Limitation: Dataset lacks required spatial coordinates or pickup timestamps. "
            f"Available columns: {cols}"
        )

    return csv_files


def extract_reproducible_sample(
    csv_files: list[Path],
    target_records: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Reads required columns via chunked reading and creates a reproducible sample."""
    print(f"\n[*] Extracting reproducible sample of {target_records:,} trips via chunked reading...")
    primary_file = csv_files[0]
    print(f"    Source file: {primary_file.name}")

    req_cols = ["tpep_pickup_datetime", "pickup_latitude", "pickup_longitude"]
    # Check if optional passenger_count or trip_distance are present
    header = pd.read_csv(primary_file, nrows=1).columns.tolist()
    use_cols = [c for c in req_cols if c in header]
    for opt in ["passenger_count", "trip_distance"]:
        if opt in header:
            use_cols.append(opt)

    print(f"    Selected columns for memory-efficient ingestion: {use_cols}")

    chunks = []
    total_loaded = 0
    buffer_target = int(target_records * 1.5)  # buffer to ensure 100k valid rows after filtering

    chunk_size = 50_000
    for chunk in pd.read_csv(primary_file, usecols=use_cols, chunksize=chunk_size):
        # Drop rows with null coordinates or invalid 0.0 values early
        chunk = chunk.dropna(subset=["pickup_latitude", "pickup_longitude", "tpep_pickup_datetime"])
        chunk = chunk[(chunk["pickup_latitude"] != 0) & (chunk["pickup_longitude"] != 0)]
        chunks.append(chunk)
        total_loaded += len(chunk)
        if total_loaded >= buffer_target:
            break

    combined = pd.concat(chunks, ignore_index=True)
    print(f"    Buffered {len(combined):,} valid candidates.")

    if len(combined) > target_records:
        sampled_df = combined.sample(n=target_records, random_state=random_state).reset_index(drop=True)
    else:
        sampled_df = combined.reset_index(drop=True)

    print(f"    Sampled {len(sampled_df):,} trips with random_state={random_state}.")
    return sampled_df


def download_or_prepare_dataset(force_download: bool = False) -> Path:
    """Acquires Kaggle dataset and prepares raw_taxi_data.csv."""
    if RAW_DATA_FILE.exists() and not force_download:
        print(f"[*] Raw dataset already prepared at: {RAW_DATA_FILE}", flush=True)
        return RAW_DATA_FILE

    dataset_path = download_kaggle_dataset()
    csv_files = inspect_dataset_files(dataset_path)
    df = extract_reproducible_sample(csv_files, SAMPLE_SIZE, RANDOM_STATE)

    RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_FILE, index=False)
    print(f"[+] Saved raw taxi dataset with {len(df):,} records to: {RAW_DATA_FILE}\n", flush=True)
    return RAW_DATA_FILE


if __name__ == "__main__":
    download_or_prepare_dataset(force_download=True)
