# Smart Cities & Logistics: Geospatial Ride-Share Demand Forecasting

A Machine Learning system that predicts ride/taxi demand across geographical zones and time periods in urban environments.

---

## 📌 Project Overview & Pipeline

The system models urban ride-hailing demand by partitioning geographical coordinates into spatial clusters and formulating demand forecasting as a multi-zone time-series problem.

```text
Raw Taxi Dataset (NYC)
       │
       ▼
Data Preprocessing (Coordinate & Outlier Filtering)
       │
       ▼
K-Means Geographical Clustering (NYC Demand Zones)
       │
       ▼
Dynamic Demand Zones & Hourly Aggregation
       │
       ▼
Feature Engineering (Time, Lag & Rolling Features)
       │
       ▼
XGBoost Demand Forecasting Engine
       │
       ▼
Model Evaluation & Visualizations (Plots, Heatmaps)
       │
       ▼
Future Demand Prediction Interface
```

---

## 📁 Project Directory Structure

```text
ride-share-demand-forecasting/
│
├── data/                  # Raw and processed datasets (CSV / Parquet)
├── notebooks/             # Jupyter notebooks for exploratory analysis and prototyping
├── src/                   # Production modular Python source code
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Central configuration, file paths, and hyperparameters
│   └── download_data.py   # Dataset acquisition and ingestion script
├── models/                # Trained model artifacts (K-Means, XGBoost)
├── outputs/               # Evaluation charts, maps, forecasts, and metrics
├── requirements.txt       # Project dependencies
├── README.md              # Documentation and instructions
└── main.py                # Pipeline execution entrypoint
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.9+ (Python 3.10 to 3.12 recommended)
- Git (optional)

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

To execute the project and verify the environment and dataset preparation:

```bash
python main.py
```

### Running the Data Acquisition Script Separately:
```bash
python src/download_data.py
```

---

## 📊 Dataset Specification

The project uses NYC Taxi/Ride-Share trip data containing:
- `pickup_datetime`: Timestamp of the ride request/pickup.
- `pickup_latitude`: Latitude coordinate of the pickup location.
- `pickup_longitude`: Longitude coordinate of the pickup location.
- *(Additional fields: `passenger_count`, `trip_distance`)*

The dataset is sampled to ~100,000 records to ensure fast iterations and efficient training while maintaining statistical validity.

---

## 🛣️ Implementation Roadmap
- [x] **Module 1**: Project Scaffold & Requirements Setup
- [x] **Module 2**: Data Preprocessing & Coordinate Cleaning
- [ ] **Module 3**: K-Means Geographical Clustering
- [ ] **Module 4**: Hourly Aggregation & Dynamic Demand Zones
- [ ] **Module 5**: Feature Engineering (Time, Lags, Rolling Windows)
- [ ] **Module 6**: XGBoost Model Training & Hyperparameter Tuning
- [ ] **Module 7**: Model Evaluation & Demand Forecasting Visualizations
- [ ] **Module 8**: Future Demand Prediction Interface
