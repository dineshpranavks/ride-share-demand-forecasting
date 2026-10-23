"""Module 5: Time-Series Feature Engineering.

Generates historical demand features for each geographical zone so the
forecasting model can learn from past demand patterns without leaking
future information.

Features created (per zone, sorted chronologically):
  Lag features:
    lag_1    - demand 1 hour ago
    lag_2    - demand 2 hours ago
    lag_3    - demand 3 hours ago
    lag_24   - demand 24 hours ago (same hour yesterday)
    lag_168  - demand 168 hours ago (same hour last week)

  Rolling window features (computed on already-shifted series to prevent leakage):
    rolling_mean_3   - mean demand over the previous 3 hours
    rolling_mean_24  - mean demand over the previous 24 hours

Data leakage prevention:
  All rolling windows are applied on df['demand'].shift(1), so the
  computation window ends strictly BEFORE the current hour.

Input:  data/hourly_demand.csv
Output: data/model_data.csv
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to sys.path so src imports work
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import HOURLY_DEMAND_FILE, MODEL_DATA_FILE


# ---------------------------------------------------------------------------
# LAG OFFSETS AND ROLLING WINDOWS
# ---------------------------------------------------------------------------
LAG_OFFSETS = [1, 2, 3, 24, 168]
ROLLING_WINDOWS = [3, 24]


def load_hourly_demand(input_path: Path = HOURLY_DEMAND_FILE) -> pd.DataFrame:
    """Loads the hourly demand aggregation produced by Module 4."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Hourly demand dataset not found at: {input_path}. "
            "Please run Module 4 (src/feature_aggregation.py) first."
        )
    print(f"[1] Loading hourly demand data from: {input_path}")
    df = pd.read_csv(input_path, parse_dates=["hour"])
    print(f"    Loaded {len(df):,} rows across {df['zone'].nunique()} zones.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates lag and rolling window features per zone.

    Steps
    -----
    1. Sort by zone then hour to ensure chronological ordering.
    2. For each zone group:
       a. Create lag_N features via shift(N).
       b. Create rolling_mean_K features using shift(1) before rolling so
          the window [t-K, t-1] is used, never the current observation t.
    3. Drop any rows that still contain NaN in the feature columns.
    """
    print("[2] Sorting dataset by zone and hour...")
    df = df.sort_values(by=["zone", "hour"]).reset_index(drop=True)

    rows_before = len(df)
    print(f"    Rows before feature engineering: {rows_before:,}")

    print("[3] Generating lag features per zone...")
    for lag in LAG_OFFSETS:
        col_name = f"lag_{lag}"
        df[col_name] = (
            df.groupby("zone")["demand"]
            .shift(lag)
        )
        print(f"    Created: {col_name}")

    print("[4] Generating rolling mean features per zone (no data leakage)...")
    for window in ROLLING_WINDOWS:
        col_name = f"rolling_mean_{window}"
        # shift(1) ensures we look at [t-window .. t-1], NOT including t
        df[col_name] = (
            df.groupby("zone")["demand"]
            .transform(lambda s: s.shift(1).rolling(window=window).mean())
        )
        print(f"    Created: {col_name}")

    # -----------------------------------------------------------------------
    # Drop rows where any lag / rolling feature is NaN.
    # These appear at the beginning of each zone's time series where there
    # isn't enough history to fill the largest window (lag_168 = 168 rows).
    # -----------------------------------------------------------------------
    feature_cols = [f"lag_{l}" for l in LAG_OFFSETS] + [
        f"rolling_mean_{w}" for w in ROLLING_WINDOWS
    ]

    print("[5] Dropping rows with missing lag/rolling values...")
    rows_with_nulls = df[feature_cols].isnull().any(axis=1).sum()
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    rows_after = len(df)

    print(f"    Rows dropped (insufficient history): {rows_with_nulls:,}")
    print(f"    Rows after feature engineering    : {rows_after:,}")

    return df, feature_cols


def print_feature_summary(df: pd.DataFrame, feature_cols: list, rows_before: int) -> None:
    """Prints a clear, structured summary of the engineered feature set."""
    all_model_features = [
        "zone", "hour_of_day", "day_of_week", "day_of_month", "month", "is_weekend"
    ] + feature_cols

    print("\n" + "=" * 70)
    print(" MODULE 5: TIME-SERIES FEATURE ENGINEERING SUMMARY")
    print("=" * 70)
    print(f"  Rows before feature engineering : {rows_before:,}")
    print(f"  Rows after feature engineering  : {len(df):,}")
    print(f"  Rows removed (NaN from lag)     : {rows_before - len(df):,}")
    print(f"  Unique zones retained           : {df['zone'].nunique()}")
    print(f"  Temporal span                   : {df['hour'].min()} to {df['hour'].max()}")
    print("-" * 70)
    print("  All feature columns created:")
    for col in feature_cols:
        print(f"    - {col}")
    print("-" * 70)
    print("  Complete feature set for model training:")
    for col in all_model_features:
        print(f"    - {col}")
    print("-" * 70)
    print("  Sample rows (first 5):")
    sample_display_cols = ["zone", "hour", "demand"] + feature_cols
    print(df[sample_display_cols].head(5).to_string(index=False))
    print("=" * 70 + "\n")


def run_feature_engineering(
    input_file: Path = HOURLY_DEMAND_FILE,
    output_file: Path = MODEL_DATA_FILE,
) -> pd.DataFrame:
    """Executes the complete Module 5 feature engineering pipeline."""
    print("=" * 70)
    print(" MODULE 5: TIME-SERIES FEATURE ENGINEERING")
    print("=" * 70)

    # 1. Load hourly demand
    df = load_hourly_demand(input_file)
    rows_before = len(df)

    # 2. Engineer lag and rolling features
    df, feature_cols = engineer_features(df)

    # 3. Save engineered dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"[6] Saved model-ready dataset to: {output_file}")

    # 4. Print summary
    print_feature_summary(df, feature_cols, rows_before)

    print("=" * 70)
    print(" [OK] MODULE 5 EXECUTION COMPLETED SUCCESSFULLY")
    print(f"      - Output File : {output_file}")
    print(f"      - Total Rows  : {len(df):,}")
    print(f"      - Features    : {len(feature_cols)} lag/rolling + 6 calendar = "
          f"{len(feature_cols) + 6} total predictors")
    print("=" * 70 + "\n")

    return df


if __name__ == "__main__":
    run_feature_engineering()
