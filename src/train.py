"""Module 6: Demand Forecasting Using XGBoost.

Trains an XGBRegressor to predict hourly ride demand per geographical zone.

Key design decisions:
- This is a time-series problem. Data is sorted chronologically by 'hour'
  and split 80/20 WITHOUT shuffling to prevent future data leaking into
  the training set.
- The model is trained once on all zones jointly; 'zone' is an integer
  feature that lets the model learn per-zone demand patterns.
- The trained model is persisted with joblib for inference in Module 7.

Input:  data/model_data.csv
Output: models/xgboost_demand_model.pkl
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_DATA_FILE, XGBOOST_MODEL_FILE, MODELS_DIR


# ---------------------------------------------------------------------------
# FEATURE / TARGET DEFINITION
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "zone",
    "hour_of_day",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_168",
    "rolling_mean_3",
    "rolling_mean_24",
]

TARGET_COLUMN = "demand"

# XGBoost hyper-parameters (fixed for reproducibility)
XGBOOST_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror",
)


def load_model_data(input_path: Path = MODEL_DATA_FILE) -> pd.DataFrame:
    """Loads the feature-engineered model dataset from Module 5."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Model data file not found at: {input_path}. "
            "Please run Module 5 (src/features.py) first."
        )
    print(f"[1] Loading model data from: {input_path}")
    df = pd.read_csv(input_path, parse_dates=["hour"])
    print(f"    Loaded {len(df):,} rows | {df['zone'].nunique()} zones | "
          f"{df.columns.tolist().__len__()} columns")
    return df


def chronological_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data chronologically into train and test sets.

    The dataset is sorted by 'hour' globally (across all zones) so the
    boundary falls at the same calendar timestamp for every zone — the
    final 20% of chronological time is used as the hold-out test set.
    """
    print("[2] Performing chronological train/test split (80% / 20%)...")

    # Sort strictly by time (not by zone, to maintain temporal integrity)
    df_sorted = df.sort_values(by="hour").reset_index(drop=True)

    split_idx = int(len(df_sorted) * train_ratio)
    split_timestamp = df_sorted.iloc[split_idx]["hour"]

    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    print(f"    Split timestamp      : {split_timestamp}")
    print(f"    Training rows        : {len(train_df):,}  "
          f"({train_df['hour'].min().date()} to {train_df['hour'].max().date()})")
    print(f"    Testing rows         : {len(test_df):,}  "
          f"({test_df['hour'].min().date()} to {test_df['hour'].max().date()})")

    return train_df, test_df


def build_model() -> XGBRegressor:
    """Instantiates the XGBRegressor with fixed hyper-parameters."""
    print("[3] Initialising XGBRegressor with parameters:")
    for key, value in XGBOOST_PARAMS.items():
        print(f"    {key:<20}: {value}")
    return XGBRegressor(**XGBOOST_PARAMS)


def train_model(
    model: XGBRegressor,
    train_df: pd.DataFrame,
) -> XGBRegressor:
    """Fits the XGBRegressor on the training split."""
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    print("[4] Training XGBoost model...")
    print(f"    Feature columns ({len(FEATURE_COLUMNS)}):")
    for col in FEATURE_COLUMNS:
        print(f"      - {col}")

    model.fit(X_train, y_train)
    print("    Model training completed successfully.")
    return model


def save_model(model: XGBRegressor, output_path: Path = XGBOOST_MODEL_FILE) -> None:
    """Persists the trained model to disk using joblib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"[5] Trained model saved to: {output_path}")


def print_training_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model: XGBRegressor,
    model_path: Path,
) -> None:
    """Prints a clear training summary."""
    print("\n" + "=" * 70)
    print(" MODULE 6: XGBOOST TRAINING SUMMARY")
    print("=" * 70)
    print(f"  Training rows        : {len(train_df):,}")
    print(f"  Testing rows         : {len(test_df):,}")
    print(f"  Train date range     : {train_df['hour'].min().date()} to "
          f"{train_df['hour'].max().date()}")
    print(f"  Test date range      : {test_df['hour'].min().date()} to "
          f"{test_df['hour'].max().date()}")
    print(f"  Split strategy       : Chronological (no shuffling — time-series safe)")
    print("-" * 70)
    print(f"  Feature columns ({len(FEATURE_COLUMNS)}):")
    for col in FEATURE_COLUMNS:
        print(f"    - {col}")
    print("-" * 70)
    print(f"  Model type           : XGBRegressor")
    print(f"  n_estimators         : {XGBOOST_PARAMS['n_estimators']}")
    print(f"  max_depth            : {XGBOOST_PARAMS['max_depth']}")
    print(f"  learning_rate        : {XGBOOST_PARAMS['learning_rate']}")
    print(f"  subsample            : {XGBOOST_PARAMS['subsample']}")
    print(f"  colsample_bytree     : {XGBOOST_PARAMS['colsample_bytree']}")
    print(f"  objective            : {XGBOOST_PARAMS['objective']}")
    print(f"  Saved model path     : {model_path}")
    print("=" * 70 + "\n")


def run_model_training(
    input_file: Path = MODEL_DATA_FILE,
    model_file: Path = XGBOOST_MODEL_FILE,
    train_ratio: float = 0.80,
) -> tuple[XGBRegressor, pd.DataFrame, pd.DataFrame]:
    """Executes the complete Module 6 training pipeline.

    Returns
    -------
    model     : Fitted XGBRegressor
    train_df  : Training split DataFrame
    test_df   : Test split DataFrame (for use in Module 7 evaluation)
    """
    print("=" * 70)
    print(" MODULE 6: DEMAND FORECASTING USING XGBOOST")
    print("=" * 70)

    # 1. Load feature-engineered data
    df = load_model_data(input_file)

    # 2. Chronological split
    train_df, test_df = chronological_train_test_split(df, train_ratio)

    # 3. Build model
    model = build_model()

    # 4. Train
    model = train_model(model, train_df)

    # 5. Save
    save_model(model, model_file)

    # 6. Print summary
    print_training_summary(train_df, test_df, model, model_file)

    print("=" * 70)
    print(" [OK] MODULE 6 EXECUTION COMPLETED SUCCESSFULLY")
    print(f"      - Model saved to  : {model_file}")
    print(f"      - Training rows   : {len(train_df):,}")
    print(f"      - Testing rows    : {len(test_df):,}")
    print("=" * 70 + "\n")

    return model, train_df, test_df


if __name__ == "__main__":
    run_model_training()
