"""Module 4: Spatio-Temporal Demand Aggregation for Ride-Share Demand Forecasting.

This module aggregates individual ride transactions into regular spatio-temporal grids:
1. Loads taxi data with assigned geographical cluster zones (data/taxi_with_zones.csv).
2. Converts pickup_datetime into timestamp and floors to hourly intervals ('hour').
3. Aggregates ride demand per (zone, hour) pair.
4. Generates a complete Cartesian grid across all zones and continuous hourly intervals,
   handling zero-demand (unobserved) intervals by filling with 0.
5. Derives calendar and temporal features:
   - hour_of_day (0-23)
   - day_of_week (0-6, where 0=Monday, 6=Sunday)
   - day_of_month (1-31)
   - month (1-12)
   - is_weekend (1 if Saturday or Sunday, else 0)
6. Saves the aggregated dataset to data/hourly_demand.csv.
7. Prints comprehensive summary statistics and displays a dataset sample.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATA_DIR,
    TAXI_WITH_ZONES_FILE,
    HOURLY_DEMAND_FILE,
)


def load_zoned_data(input_path: Path = TAXI_WITH_ZONES_FILE) -> pd.DataFrame:
    """Loads taxi dataset with pre-assigned zone clusters."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Zoned dataset not found at: {input_path}. "
            f"Please run Module 3 (src/clustering.py) first."
        )
    print(f"[1] Loading zoned taxi data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"    Loaded {len(df):,} ride records.")
    return df


def aggregate_hourly_demand(
    df: pd.DataFrame,
    fill_missing_intervals: bool = True,
) -> pd.DataFrame:
    """Aggregates ride-level records into hourly demand counts per zone.

    Args:
        df: Input DataFrame with 'pickup_datetime' and 'zone'.
        fill_missing_intervals: Whether to create a complete spatio-temporal grid
            covering every zone and hour, filling missing demand with 0.

    Returns:
        DataFrame containing hourly demand per zone with calendar features.
    """
    print("[2] Processing timestamps and rounding to hourly intervals...")
    # Step 1: Convert pickup_datetime to datetime
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    
    # Step 2: Floor pickup_datetime to hourly intervals ('hour')
    df["hour"] = df["pickup_datetime"].dt.floor("h")

    # Step 3 & 4: Group by zone and hour, count demand
    print("[3] Grouping records by (zone, hour) and calculating demand...")
    aggregated = (
        df.groupby(["zone", "hour"], as_index=False)
        .size()
        .rename(columns={"size": "demand"})
    )

    if fill_missing_intervals:
        print("[4] Generating complete spatio-temporal grid to handle unobserved zone-hours...")
        unique_zones = sorted(df["zone"].dropna().unique())
        min_hour = df["hour"].min()
        max_hour = df["hour"].max()

        # Build continuous hourly time range
        full_time_range = pd.date_range(start=min_hour, end=max_hour, freq="h")
        
        # Cartesian product: all zones x all hours
        grid_index = pd.MultiIndex.from_product(
            [unique_zones, full_time_range],
            names=["zone", "hour"],
        )
        grid_df = pd.DataFrame(index=grid_index).reset_index()

        # Merge observed demand with full grid
        hourly_df = pd.merge(grid_df, aggregated, on=["zone", "hour"], how="left")
        
        # Unobserved combinations had 0 rides during that hour
        hourly_df["demand"] = hourly_df["demand"].fillna(0).astype(int)
    else:
        hourly_df = aggregated

    # Ensure integer zone type
    hourly_df["zone"] = hourly_df["zone"].astype(int)

    # Step 5: Extract temporal features
    print("[5] Generating calendar and temporal features...")
    hourly_df["hour_of_day"] = hourly_df["hour"].dt.hour
    hourly_df["day_of_week"] = hourly_df["hour"].dt.dayofweek
    hourly_df["day_of_month"] = hourly_df["hour"].dt.day
    hourly_df["month"] = hourly_df["hour"].dt.month
    hourly_df["is_weekend"] = (hourly_df["day_of_week"] >= 5).astype(int)

    # Sort chronologically by zone and time
    hourly_df = hourly_df.sort_values(by=["zone", "hour"]).reset_index(drop=True)

    # Define standardized column structure
    ordered_columns = [
        "zone",
        "hour",
        "demand",
        "hour_of_day",
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend",
    ]
    hourly_df = hourly_df[ordered_columns]

    return hourly_df


def print_aggregation_summary(hourly_df: pd.DataFrame) -> None:
    """Prints statistics and a sample of the hourly demand dataset."""
    num_zones = hourly_df["zone"].nunique()
    total_records = len(hourly_df)
    min_demand = int(hourly_df["demand"].min())
    max_demand = int(hourly_df["demand"].max())
    avg_demand = float(hourly_df["demand"].mean())

    print("\n" + "=" * 70)
    print(" MODULE 4: SPATIO-TEMPORAL DEMAND AGGREGATION SUMMARY")
    print("=" * 70)
    print(f"  - Number of Zones            : {num_zones}")
    print(f"  - Number of Hourly Records   : {total_records:,}")
    print(f"  - Minimum Hourly Demand      : {min_demand}")
    print(f"  - Maximum Hourly Demand      : {max_demand}")
    print(f"  - Average Hourly Demand      : {avg_demand:.4f} rides/hour")
    print(f"  - Temporal Span              : {hourly_df['hour'].min()} to {hourly_df['hour'].max()}")
    print("-" * 70)
    print(" Sample of resulting dataset (head):")
    print("-" * 70)
    sample_cols = ["zone", "hour", "demand", "hour_of_day", "day_of_week", "is_weekend"]
    print(hourly_df[sample_cols].head(10).to_string(index=False))
    print("=" * 70 + "\n")


def run_spatiotemporal_aggregation(
    input_file: Path = TAXI_WITH_ZONES_FILE,
    output_file: Path = HOURLY_DEMAND_FILE,
) -> pd.DataFrame:
    """Executes the complete Module 4 aggregation workflow."""
    print("=" * 70)
    print(" MODULE 4: SPATIO-TEMPORAL DEMAND AGGREGATION")
    print("=" * 70)

    # 1. Load zoned data
    df = load_zoned_data(input_file)

    # 2. Aggregate demand and extract time features
    hourly_df = aggregate_hourly_demand(df, fill_missing_intervals=True)

    # 3. Save resulting dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    hourly_df.to_csv(output_file, index=False)
    print(f"[6] Saved aggregated hourly demand dataset to: {output_file}")

    # 4. Print summary metrics and sample
    print_aggregation_summary(hourly_df)

    print("================================================================================")
    print(" [OK] MODULE 4 EXECUTION COMPLETED SUCCESSFULLY")
    print(f"      - Output File : {output_file}")
    print(f"      - Total Rows  : {len(hourly_df):,}")
    print("================================================================================\n")

    return hourly_df


if __name__ == "__main__":
    run_spatiotemporal_aggregation()
