"""Module 9: Future Demand Prediction.

Provides a simple command-line interface (CLI) to predict expected ride demand
for a specified zone and date/time using the trained XGBoost model.

Workflow:
1. Load the trained XGBoost model (models/xgboost_demand_model.pkl).
2. Load historical records (data/model_data.csv).
3. Extract calendar features from the target date/time:
   - hour_of_day, day_of_week, day_of_month, month, is_weekend
4. Retrieve the most representative / recent historical lag and rolling values:
   - Matches zone, hour_of_day, day_of_week if available, otherwise zone & hour_of_day,
     falling back to the latest observed record for that zone.
5. Predict ride demand with XGBoost (clamped at 0 since demand is non-negative).
6. Format output clearly for demonstrations and presentations.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import XGBOOST_MODEL_FILE, MODEL_DATA_FILE
from src.train import FEATURE_COLUMNS


def load_model_and_data(
    model_path: Path = XGBOOST_MODEL_FILE,
    data_path: Path = MODEL_DATA_FILE,
):
    """Loads the trained model and historical baseline data."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Please run Module 6 (src/train.py) first."
        )
    if not data_path.exists():
        raise FileNotFoundError(
            f"Historical model data not found at {data_path}. "
            "Please run Module 5 (src/features.py) first."
        )

    model = joblib.load(model_path)
    df = pd.read_csv(data_path, parse_dates=["hour"])
    return model, df


def get_lag_and_rolling_features(
    df: pd.DataFrame,
    zone: int,
    target_dt: datetime,
) -> dict[str, float]:
    """Retrieves representative historical lag and rolling features for the given zone and time.

    Strategy:
    1. Filter historical records for the specified zone.
    2. Try to find historical observations with matching (hour_of_day, day_of_week).
    3. If found, average the lag and rolling values to provide a stable diurnal baseline.
    4. Fall back to matching hour_of_day, or the zone's latest available records.
    """
    zone_df = df[df["zone"] == zone]
    if zone_df.empty:
        raise ValueError(f"Zone {zone} not found in historical data. Available zones: 0 to {df['zone'].max()}.")

    h = target_dt.hour
    dow = target_dt.weekday()

    lag_cols = [
        "lag_1", "lag_2", "lag_3", "lag_24", "lag_168",
        "rolling_mean_3", "rolling_mean_24"
    ]

    # Level 1: Match same hour of day and day of week
    matched = zone_df[(zone_df["hour_of_day"] == h) & (zone_df["day_of_week"] == dow)]

    # Level 2: Match same hour of day
    if matched.empty:
        matched = zone_df[zone_df["hour_of_day"] == h]

    # Level 3: Fall back to latest records for this zone
    if matched.empty:
        matched = zone_df.tail(10)

    # Compute mean for smooth historical baseline
    features_dict = {}
    for col in lag_cols:
        features_dict[col] = float(matched[col].mean())

    return features_dict


def predict_demand(
    model,
    historical_df: pd.DataFrame,
    zone: int,
    date_str: str,
) -> tuple[float, dict]:
    """Computes features and executes prediction for a given zone and datetime string."""
    # Parse target datetime
    target_dt = pd.to_datetime(date_str)

    # Calculate calendar features
    hour_of_day = int(target_dt.hour)
    day_of_week = int(target_dt.weekday())
    day_of_month = int(target_dt.day)
    month = int(target_dt.month)
    is_weekend = 1 if day_of_week >= 5 else 0

    # Retrieve historical lag and rolling features
    lags = get_lag_and_rolling_features(historical_df, zone, target_dt)

    feature_record = {
        "zone": int(zone),
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "day_of_month": day_of_month,
        "month": month,
        "is_weekend": is_weekend,
        **lags,
    }

    # Construct DataFrame aligned with model training schema
    X_input = pd.DataFrame([feature_record])[FEATURE_COLUMNS]

    # Predict
    raw_pred = float(model.predict(X_input)[0])
    # Ride demand cannot be negative
    clamped_pred = max(0.0, raw_pred)

    return clamped_pred, feature_record


def format_prediction_output(
    zone: int,
    date_str: str,
    predicted_rides: float,
) -> None:
    """Formats prediction output cleanly matching the required output format."""
    print("\nPredicted Ride Demand")
    print(f"Zone: {zone}")
    print(f"Time: {date_str}")
    print(f"Predicted rides: {predicted_rides:.2f} (approx. {int(round(predicted_rides))} rides)\n")


def interactive_cli():
    """Simple interactive command-line interface."""
    print("=" * 60)
    print("  RIDE-SHARE DEMAND FORECASTING - PREDICTION CLI")
    print("=" * 60)
    print("Loading model and historical features...")
    model, historical_df = load_model_and_data()
    print("System ready. Available zones: 0 to 19.\n")

    try:
        zone_input = input("Enter zone: ").strip()
        if not zone_input:
            print("No zone provided. Exiting.")
            return
        zone = int(zone_input)

        dt_input = input("Enter date and time: ").strip()
        if not dt_input:
            print("No date/time provided. Exiting.")
            return

        predicted_rides, _ = predict_demand(model, historical_df, zone, dt_input)
        format_prediction_output(zone, dt_input, predicted_rides)

    except KeyboardInterrupt:
        print("\nExiting CLI.")
    except Exception as e:
        print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(description="Predict ride demand for a zone and time.")
    parser.add_argument("--zone", type=int, help="Geographical zone ID (0 to 19)")
    parser.add_argument("--datetime", type=str, help="Target date and time (YYYY-MM-DD HH:MM)")

    args = parser.parse_args()

    # If parameters provided via CLI flags, run single prediction directly
    if args.zone is not None and args.datetime is not None:
        model, historical_df = load_model_and_data()
        predicted_rides, _ = predict_demand(model, historical_df, args.zone, args.datetime)
        format_prediction_output(args.zone, args.datetime, predicted_rides)
    else:
        # Otherwise start interactive prompt
        interactive_cli()


if __name__ == "__main__":
    main()
