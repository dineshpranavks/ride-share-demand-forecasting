from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Data File Paths
RAW_DATA_FILE = DATA_DIR / "raw_taxi_data.csv"
CLEANED_DATA_FILE = DATA_DIR / "cleaned_taxi.csv"
TAXI_WITH_ZONES_FILE = DATA_DIR / "taxi_with_zones.csv"
CLUSTERED_DATA_FILE = TAXI_WITH_ZONES_FILE  # Alias for backward compatibility
HOURLY_DEMAND_FILE = DATA_DIR / "hourly_demand.csv"
MODEL_DATA_FILE = DATA_DIR / "model_data.csv"

# Model File Paths
KMEANS_MODEL_FILE = MODELS_DIR / "kmeans_zones.pkl"
XGBOOST_MODEL_FILE = MODELS_DIR / "xgboost_demand_model.pkl"

# Visualization and Evaluation File Paths
DEMAND_ZONES_PLOT = OUTPUTS_DIR / "demand_zones.png"
GEOGRAPHICAL_ZONES_PLOT = OUTPUTS_DIR / "geographical_zones.png"
ACTUAL_VS_PREDICTED_PLOT = OUTPUTS_DIR / "actual_vs_predicted.png"
DEMAND_OVER_TIME_PLOT = OUTPUTS_DIR / "demand_over_time.png"
FEATURE_IMPORTANCE_PLOT = OUTPUTS_DIR / "feature_importance.png"
HOURLY_DEMAND_PATTERN_PLOT = OUTPUTS_DIR / "hourly_demand_pattern.png"
ZONE_DEMAND_PLOT = OUTPUTS_DIR / "zone_demand.png"
EVALUATION_TXT = OUTPUTS_DIR / "evaluation.txt"
METRICS_CSV = OUTPUTS_DIR / "metrics.csv"

# Geographic Bounding Box for NYC
NYC_LAT_MIN = 40.55
NYC_LAT_MAX = 40.95
NYC_LON_MIN = -74.10
NYC_LON_MAX = -73.70

# Project Hyperparameters
SAMPLE_SIZE = 100_000
N_CLUSTERS = 20
RANDOM_STATE = 42

# Ensure core output directories exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
