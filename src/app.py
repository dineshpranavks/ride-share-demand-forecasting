"""Flask Web Application for Geospatial Ride-Share Demand Forecasting.

Features:
- Page A: /          – User-facing Demand Forecast with interactive map, zone selector,
                       date/time picker, and real XGBoost predictions.
- Page B: /insights  – ML Insights with dataset stats, algorithm explanations,
                       evaluation metrics, and visualisation gallery.
- API:    /api/predict   – POST endpoint for real predictions
         /api/zones      – GET endpoint with zone metadata (centers, avg demand)
"""

import sys
from pathlib import Path

import pandas as pd
import joblib

from flask import Flask, render_template, request, jsonify, send_from_directory

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.config import OUTPUTS_DIR, XGBOOST_MODEL_FILE, MODEL_DATA_FILE, KMEANS_MODEL_FILE, HOURLY_DEMAND_FILE
from src.predict import load_model_and_data, predict_demand

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# ---------------------------------------------------------------------------
# Load ML artefacts once at startup
# ---------------------------------------------------------------------------
print("[*] Loading ML model and historical data…")
model, historical_df = load_model_and_data(XGBOOST_MODEL_FILE, MODEL_DATA_FILE)
kmeans = joblib.load(KMEANS_MODEL_FILE)
hourly_df = pd.read_csv(HOURLY_DEMAND_FILE, parse_dates=["hour"])
print("[*] ML engine ready.")

# ---------------------------------------------------------------------------
# Pre-compute zone metadata (cluster centres + average historical demand)
# ---------------------------------------------------------------------------
ZONE_CENTERS = {}
centers = kmeans.cluster_centers_          # shape (20, 2): [lat, lon]
zone_avg_demand = hourly_df.groupby("zone")["demand"].mean().to_dict()

# Human-readable area names derived from the real cluster centroids (NYC boroughs)
ZONE_NAMES = {
    0:  "Upper West Side / Lincoln Sq",
    1:  "Murray Hill / Midtown South",
    2:  "LaGuardia Airport (LGA)",
    3:  "JFK International Airport",
    4:  "Gramercy / Union Square",
    5:  "Midtown East / Grand Central",
    6:  "Financial District / Downtown",
    7:  "Upper East Side (South) / Lenox Hill",
    8:  "Chelsea / Meatpacking District",
    9:  "Harlem / East Harlem",
    10: "Williamsburg / Greenpoint",
    11: "Times Square / Midtown West",
    12: "Theater District / Columbus Circle",
    13: "Flatbush / Prospect Park",
    14: "Downtown Brooklyn / Boerum Hill",
    15: "Elmhurst / Corona (Queens)",
    16: "SoHo / Greenwich Village",
    17: "Manhattan Valley / Morningside",
    18: "Yorkville / Upper East Side (North)",
    19: "Long Island City / Sunnyside",
}

for zone_id, (lat, lon) in enumerate(centers):
    ZONE_CENTERS[zone_id] = {
        "zone": zone_id,
        "lat": round(float(lat), 6),
        "lon": round(float(lon), 6),
        "name": ZONE_NAMES.get(zone_id, f"Zone {zone_id}"),
        "avg_demand": round(float(zone_avg_demand.get(zone_id, 0)), 3),
    }

# Demand-level thresholds (based on actual percentiles: p33≈2.0, p66≈7.0)
DEMAND_LOW_THRESHOLD = 3.0
DEMAND_HIGH_THRESHOLD = 10.0


def classify_demand(predicted: float) -> str:
    """Classify predicted demand into Low / Medium / High."""
    if predicted < DEMAND_LOW_THRESHOLD:
        return "Low"
    elif predicted < DEMAND_HIGH_THRESHOLD:
        return "Medium"
    else:
        return "High"


def build_explanation(zone: int, predicted: float, level: str, features: dict) -> str:
    """Generate a plain-English explanation of the prediction."""
    zone_name = ZONE_NAMES.get(zone, f"Zone {zone}")
    day = features.get("day_name", "")
    hour_str = features.get("hour_of_day", "")
    weekend = features.get("is_weekend", "No (Weekday)")
    rolling = features.get("rolling_mean_24", 0)

    period = "night" if int(hour_str[:2]) < 6 else (
        "morning" if int(hour_str[:2]) < 12 else (
            "afternoon" if int(hour_str[:2]) < 17 else (
                "evening" if int(hour_str[:2]) < 21 else "night"
            )
        )
    )

    base = (
        f"The model forecasts {level.lower()} demand for {zone_name} on {day} {period}. "
        f"The 24-hour rolling average for this zone is {rolling:.2f} rides/hr, "
    )
    if "Yes" in weekend:
        base += "and weekend patterns typically show different demand than weekdays. "
    else:
        base += "consistent with typical weekday traffic in this area. "

    base += (
        f"The XGBoost model used lag features (previous 1-hour, 24-hour, and weekly demand) "
        f"along with calendar signals to generate this forecast."
    )
    return base


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Page A – Demand Forecast."""
    zones_list = sorted(ZONE_CENTERS.values(), key=lambda z: -z["avg_demand"])
    return render_template("index.html", zones=zones_list)


@app.route("/insights")
def insights():
    """Page B – ML Insights."""
    # Read evaluation metrics from file
    metrics = {"MAE": "3.9076", "RMSE": "6.6978", "R2": "0.4419"}
    eval_file = OUTPUTS_DIR / "evaluation.txt"
    if eval_file.exists():
        try:
            for line in open(eval_file):
                if "MAE:" in line:
                    metrics["MAE"] = line.split(":")[-1].strip()
                elif "RMSE:" in line:
                    metrics["RMSE"] = line.split(":")[-1].strip()
                elif "R2 Score:" in line:
                    metrics["R2"] = line.split(":")[-1].strip()
        except Exception:
            pass

    # Dataset statistics from real Kaggle dataset
    dataset_stats = {
        "raw_records": "100,000",
        "cleaned_records": "99,946",
        "zones": 20,
        "date_range": "Jan 1, 2015 – Jan 31, 2015",
        "hourly_rows": "14,880",
        "train_samples": "9,216",
        "test_samples": "2,304",
    }

    # Zone cluster centres table
    zone_table = sorted(ZONE_CENTERS.values(), key=lambda z: z["zone"])

    return render_template(
        "insights.html",
        metrics=metrics,
        dataset_stats=dataset_stats,
        zone_table=zone_table,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Real-time demand prediction endpoint."""
    try:
        data = request.get_json() or request.form
        zone = int(data.get("zone", 0))
        date_str = str(data.get("datetime", "")).strip()

        if not date_str:
            return jsonify({"status": "error", "message": "Please provide a date and time."}), 400

        if zone not in ZONE_CENTERS:
            return jsonify({"status": "error", "message": f"Zone {zone} is not valid. Choose 0–19."}), 400

        predicted_rides, feature_record = predict_demand(model, historical_df, zone, date_str)
        level = classify_demand(predicted_rides)

        features_out = {
            "hour_of_day": f"{feature_record['hour_of_day']:02d}:00",
            "day_name": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][feature_record["day_of_week"]],
            "is_weekend": "Yes (Weekend)" if feature_record["is_weekend"] == 1 else "No (Weekday)",
            "lag_1": round(feature_record["lag_1"], 3),
            "lag_24": round(feature_record["lag_24"], 3),
            "rolling_mean_24": round(feature_record["rolling_mean_24"], 3),
        }

        explanation = build_explanation(zone, predicted_rides, level, features_out)
        zone_info = ZONE_CENTERS[zone]

        return jsonify({
            "status": "success",
            "zone": zone,
            "zone_name": zone_info["name"],
            "datetime": date_str,
            "predicted_rides": round(predicted_rides, 3),
            "approx_rides": int(round(predicted_rides)),
            "demand_level": level,
            "explanation": explanation,
            "features": features_out,
        })

    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/zones")
def api_zones():
    """Returns zone metadata for map initialization."""
    return jsonify({"zones": list(ZONE_CENTERS.values())})


@app.route("/outputs/<path:filename>")
def serve_output_file(filename):
    return send_from_directory(str(OUTPUTS_DIR), filename)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[*] Starting RidePulse at http://127.0.0.1:5000 …")
    app.run(host="127.0.0.1", port=5000, debug=False)
