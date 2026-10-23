"""Module 7: Model Evaluation.

Evaluates the trained XGBoost demand forecasting model on the hold-out
test dataset (20% chronological split) using regression metrics:
  1. MAE (Mean Absolute Error)
  2. RMSE (Root Mean Squared Error)
  3. R2 Score (Coefficient of Determination)

Outputs:
  - outputs/evaluation.txt
  - outputs/metrics.csv
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    MODEL_DATA_FILE,
    XGBOOST_MODEL_FILE,
    EVALUATION_TXT,
    METRICS_CSV,
    OUTPUTS_DIR,
)
from src.train import FEATURE_COLUMNS, TARGET_COLUMN, chronological_train_test_split


def load_model_and_test_data(
    model_path: Path = XGBOOST_MODEL_FILE,
    data_path: Path = MODEL_DATA_FILE,
    train_ratio: float = 0.80,
) -> tuple[object, pd.DataFrame]:
    """Loads the trained model and extracts the chronological test set."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Please run Module 6 (src/train.py) first."
        )
    if not data_path.exists():
        raise FileNotFoundError(
            f"Model dataset not found at {data_path}. "
            "Please run Module 5 (src/features.py) first."
        )

    print(f"[1] Loading model from: {model_path}")
    model = joblib.load(model_path)

    print(f"[2] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path, parse_dates=["hour"])

    print(f"[3] Extracting chronological hold-out test set ({int((1 - train_ratio)*100)}%)...")
    _, test_df = chronological_train_test_split(df, train_ratio=train_ratio)
    return model, test_df


def evaluate_model(
    model,
    test_df: pd.DataFrame,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Generates predictions on test_df and calculates MAE, RMSE, and R2."""
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].values

    print("[4] Generating predictions on hold-out test set...")
    y_pred = model.predict(X_test)

    # Compute metrics using sklearn.metrics
    mae = mean_absolute_error(y_test, y_pred)
    # Using root_mean_squared_error or sqrt(mean_squared_error) for sklearn compatibility
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)

    return float(mae), float(rmse), float(r2), y_test, y_pred


def save_evaluation_results(
    mae: float,
    rmse: float,
    r2: float,
    txt_path: Path = EVALUATION_TXT,
    csv_path: Path = METRICS_CSV,
) -> None:
    """Saves evaluation results to evaluation.txt and metrics.csv."""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save evaluation.txt exactly formatted
    txt_content = (
        "Model Evaluation\n"
        "----------------\n"
        f"MAE: {mae:.4f}\n"
        f"RMSE: {rmse:.4f}\n"
        f"R2 Score: {r2:.4f}\n"
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)
    print(f"[5] Saved evaluation summary to: {txt_path}")

    # 2. Save metrics.csv
    metrics_df = pd.DataFrame([
        {
            "Metric": "MAE",
            "Value": mae,
            "Description": "Mean Absolute Error",
        },
        {
            "Metric": "RMSE",
            "Value": rmse,
            "Description": "Root Mean Squared Error",
        },
        {
            "Metric": "R2 Score",
            "Value": r2,
            "Description": "Coefficient of Determination",
        },
    ])
    metrics_df.to_csv(csv_path, index=False)
    print(f"[6] Saved metrics table to: {csv_path}")


def print_evaluation_summary(
    mae: float,
    rmse: float,
    r2: float,
    test_len: int,
) -> None:
    """Prints results and explains each metric clearly."""
    print("\n" + "=" * 70)
    print(" MODULE 7: MODEL EVALUATION RESULTS")
    print("=" * 70)
    print("Model Evaluation")
    print("----------------")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print("-" * 70)
    print(f"Evaluated on {test_len:,} test samples.")
    print("=" * 70)
    print("\nMetric Explanations:")
    print("--------------------")
    print(
        "1. MAE (Mean Absolute Error):\n"
        "   Measures the average magnitude of prediction errors in the same units as\n"
        "   demand (number of rides per hour). It treats all errors equally without\n"
        f"   disproportionately penalizing large outliers. (MAE = {mae:.4f} rides/hr)\n"
    )
    print(
        "2. RMSE (Root Mean Squared Error):\n"
        "   Takes the square root of average squared errors. Because errors are squared\n"
        "   before averaging, RMSE heavily penalizes larger forecasting errors or spike\n"
        f"   misses, also expressed in rides per hour. (RMSE = {rmse:.4f} rides/hr)\n"
    )
    print(
        "3. R2 Score (Coefficient of Determination):\n"
        "   Indicates the proportion of variance in ride demand that is explained by\n"
        "   the model features. A score of 1.0 means perfect predictions, 0.0 means the\n"
        f"   model performs no better than the mean demand baseline. (R2 = {r2:.4f})\n"
    )
    print("=" * 70 + "\n")


def run_evaluation(
    model_path: Path = XGBOOST_MODEL_FILE,
    data_path: Path = MODEL_DATA_FILE,
    txt_path: Path = EVALUATION_TXT,
    csv_path: Path = METRICS_CSV,
) -> dict[str, float]:
    """Executes the full Module 7 evaluation pipeline."""
    print("=" * 70)
    print(" MODULE 7: MODEL EVALUATION")
    print("=" * 70)

    model, test_df = load_model_and_test_data(model_path, data_path)
    mae, rmse, r2, y_test, y_pred = evaluate_model(model, test_df)

    save_evaluation_results(mae, rmse, r2, txt_path, csv_path)
    print_evaluation_summary(mae, rmse, r2, len(test_df))

    print("=" * 70)
    print(" [OK] MODULE 7 EXECUTION COMPLETED SUCCESSFULLY")
    print(f"      - Text Summary : {txt_path}")
    print(f"      - Metrics CSV  : {csv_path}")
    print("=" * 70 + "\n")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


if __name__ == "__main__":
    run_evaluation()
