"""Smart Cities & Logistics: Geospatial Ride-Share Demand Forecasting.

Main pipeline runner.
Initial Phase: Environment verification, directory structure setup,
and raw dataset acquisition.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import (
    BASE_DIR,
    DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    NOTEBOOKS_DIR,
    RAW_DATA_FILE,
)
from src.download_data import download_or_prepare_dataset
from src.preprocess import preprocess_taxi_data
from src.clustering import generate_demand_zones
from src.feature_aggregation import run_spatiotemporal_aggregation
from src.features import run_feature_engineering
from src.train import run_model_training
from src.evaluate import run_evaluation
from src.visualize import run_all_visualizations
import pandas as pd


def print_banner():
    banner = """
================================================================================
   SMART CITIES & LOGISTICS: GEOSPATIAL RIDE-SHARE DEMAND FORECASTING
================================================================================
 Pipeline Architecture:
   [1] Raw taxi/ride dataset preparation        [Completed]
   [2] Data preprocessing & coordinate cleaning  [Completed]
   [3] K-Means geographical clustering           [Completed]
   [4] Spatio-temporal demand aggregation        [Completed]
   [5] Feature engineering (Time, Lag, Rolling)  [Completed]
   [6] XGBoost demand forecasting model          [Completed]
   [7] Evaluation (MAE, RMSE, R2)                [Completed]
   [8] Visualization & Results                   [Completed]
   [9] Future Demand Prediction (CLI)            [Completed]
   [10] Interactive Web Presentation Dashboard   [Completed]
================================================================================
"""
    print(banner)


def check_directories():
    print("[+] Checking Project Directory Structure:")
    dirs = [
        ("Base Directory", BASE_DIR),
        ("Data Directory", DATA_DIR),
        ("Notebooks Directory", NOTEBOOKS_DIR),
        ("Models Directory", MODELS_DIR),
        ("Outputs Directory", OUTPUTS_DIR),
    ]
    for name, path in dirs:
        status = "EXISTS" if path.exists() else "CREATED"
        path.mkdir(parents=True, exist_ok=True)
        print(f"    - {name:<22}: {path} [{status}]")


def main():
    print_banner()
    check_directories()

    # Module 1: Dataset acquisition
    print("\n--- [Step 1] Dataset Ingestion ---")
    download_or_prepare_dataset(force_download=False)

    # Module 2: Data preprocessing
    print("\n--- [Step 2] Data Preprocessing ---")
    cleaned_df = preprocess_taxi_data()

    # Module 3: Geographical Demand Zone Generation using K-Means
    print("\n--- [Step 3] Geographical Demand Zone Generation (K-Means) ---")
    zoned_df = generate_demand_zones()

    # Module 4: Spatio-Temporal Demand Aggregation
    print("\n--- [Step 4] Spatio-Temporal Demand Aggregation ---")
    hourly_df = run_spatiotemporal_aggregation()

    # Module 5: Time-Series Feature Engineering
    print("\n--- [Step 5] Time-Series Feature Engineering ---")
    model_df = run_feature_engineering()

    # Module 6: XGBoost Demand Forecasting
    print("\n--- [Step 6] XGBoost Demand Forecasting Model ---")
    model, train_df, test_df = run_model_training()

    # Module 7: Model Evaluation
    print("\n--- [Step 7] Model Evaluation ---")
    metrics = run_evaluation()

    # Module 8: Visualization and Results
    print("\n--- [Step 8] Visualization and Results ---")
    plot_files = run_all_visualizations()

    print("\n================================================================================")
    print(" [OK] Pipeline Completed Successfully Through Module 8!")
    print(f" Total Zoned Records     : {len(zoned_df):,}")
    print(f" Hourly Aggregated Rows  : {len(hourly_df):,}")
    print(f" Model-Ready Rows        : {len(model_df):,}")
    print(f" Training Samples        : {len(train_df):,}")
    print(f" Testing Samples         : {len(test_df):,}")
    print(f" Test MAE                : {metrics['MAE']:.4f}")
    print(f" Test RMSE               : {metrics['RMSE']:.4f}")
    print(f" Test R2 Score           : {metrics['R2']:.4f}")
    print(" Evaluation saved to     : outputs/evaluation.txt & outputs/metrics.csv")
    print(f" Presentation Plots      : {len(plot_files)} PNG charts saved in outputs/")
    print(" Prediction CLI          : python src/predict.py (Ready for live demo)")
    print(" Web Dashboard           : python src/app.py (http://127.0.0.1:5000)")
    print("================================================================================")


if __name__ == "__main__":
    main()