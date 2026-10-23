"""Module 8: Visualizations and Results.

Generates high-quality, presentation-grade visualizations for the ride-share
demand forecasting project:
  1. Geographical demand zones (Scatter plot: lon vs lat colored by zone)
     -> outputs/geographical_zones.png
  2. Actual vs predicted demand (Representative sample of test set)
     -> outputs/actual_vs_predicted.png
  3. Demand over time (Total hourly demand time-series)
     -> outputs/demand_over_time.png
  4. Feature importance (XGBoost model feature importances)
     -> outputs/feature_importance.png
  5. Average demand by hour (hour_of_day vs average demand pattern)
     -> outputs/hourly_demand_pattern.png
  6. Average demand by zone (zone vs average demand bar plot)
     -> outputs/zone_demand.png
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    TAXI_WITH_ZONES_FILE,
    HOURLY_DEMAND_FILE,
    MODEL_DATA_FILE,
    XGBOOST_MODEL_FILE,
    OUTPUTS_DIR,
    GEOGRAPHICAL_ZONES_PLOT,
    ACTUAL_VS_PREDICTED_PLOT,
    DEMAND_OVER_TIME_PLOT,
    FEATURE_IMPORTANCE_PLOT,
    HOURLY_DEMAND_PATTERN_PLOT,
    ZONE_DEMAND_PLOT,
)
from src.train import FEATURE_COLUMNS, TARGET_COLUMN, chronological_train_test_split

# Setup matplotlib styling for clean, professional college presentation graphics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_geographical_zones(
    zones_file: Path = TAXI_WITH_ZONES_FILE,
    output_path: Path = GEOGRAPHICAL_ZONES_PLOT,
    sample_size: int = 50_000,
) -> None:
    """1. Geographical demand zones scatter plot (longitude vs latitude colored by zone)."""
    print("[1] Generating Geographical Demand Zones plot...")
    df = pd.read_csv(zones_file)

    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    cmap = plt.get_cmap("tab20", df["zone"].nunique())

    scatter = ax.scatter(
        df_sample["longitude"],
        df_sample["latitude"],
        c=df_sample["zone"],
        cmap=cmap,
        s=4,
        alpha=0.6,
        edgecolors="none",
    )

    # Compute and plot zone centroids
    centroids = df.groupby("zone")[["longitude", "latitude"]].mean().reset_index()
    ax.scatter(
        centroids["longitude"],
        centroids["latitude"],
        c="black",
        s=50,
        marker="X",
        edgecolors="white",
        linewidths=1.2,
        label="Zone Centroid",
        zorder=5,
    )

    for _, row in centroids.iterrows():
        ax.annotate(
            f"Z{int(row['zone'])}",
            (row["longitude"], row["latitude"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
            zorder=6,
        )

    cbar = plt.colorbar(scatter, ax=ax, ticks=range(0, df["zone"].nunique(), 2))
    cbar.set_label("Cluster Zone ID", fontsize=11, fontweight="bold")

    ax.set_title("Geographical Demand Zones (K-Means Clustering, K=20)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude (°W)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"    Saved: {output_path}")


def plot_actual_vs_predicted(
    model_file: Path = XGBOOST_MODEL_FILE,
    data_file: Path = MODEL_DATA_FILE,
    output_path: Path = ACTUAL_VS_PREDICTED_PLOT,
) -> None:
    """2. Actual vs predicted demand for a representative portion of the test dataset."""
    print("[2] Generating Actual vs Predicted Demand plot...")
    model = joblib.load(model_file)
    df = pd.read_csv(data_file, parse_dates=["hour"])

    _, test_df = chronological_train_test_split(df, train_ratio=0.80)
    test_df = test_df.copy()
    test_df["predicted_demand"] = model.predict(test_df[FEATURE_COLUMNS])

    # Select the most active zone and a 14-day time window for clear presentation visualization
    top_zone = test_df.groupby("zone")["demand"].sum().idxmax()
    zone_data = test_df[test_df["zone"] == top_zone].sort_values("hour")
    # Take a representative 14-day window (336 hours)
    representative_window = zone_data.head(336)

    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
    ax.plot(
        representative_window["hour"],
        representative_window["demand"],
        label="Actual Demand",
        color="#1f77b4",
        linewidth=1.8,
        alpha=0.9,
    )
    ax.plot(
        representative_window["hour"],
        representative_window["predicted_demand"],
        label="Predicted Demand (XGBoost)",
        color="#d62728",
        linewidth=1.8,
        linestyle="--",
        alpha=0.85,
    )

    ax.set_title(
        f"Hourly Actual vs. Predicted Demand (Zone {top_zone} - 14-Day Test Horizon)",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Date & Time", fontsize=11, fontweight="bold")
    ax.set_ylabel("Rides per Hour (Demand)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, fontsize=10)
    plt.xticks(rotation=20)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"    Saved: {output_path}")


def plot_demand_over_time(
    hourly_file: Path = HOURLY_DEMAND_FILE,
    output_path: Path = DEMAND_OVER_TIME_PLOT,
) -> None:
    """3. Demand over time (Total hourly demand across all zones over time)."""
    print("[3] Generating Demand Over Time plot...")
    df = pd.read_csv(hourly_file, parse_dates=["hour"])

    # Aggregate total demand across all zones per hour
    total_hourly = df.groupby("hour")["demand"].sum().reset_index()

    # Calculate 7-day rolling average (168 hours) for clear trend line
    total_hourly["rolling_7d"] = total_hourly["demand"].rolling(168, min_periods=24).mean()

    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
    ax.plot(
        total_hourly["hour"],
        total_hourly["demand"],
        color="#4a90e2",
        alpha=0.35,
        linewidth=0.8,
        label="Hourly Total Demand",
    )
    ax.plot(
        total_hourly["hour"],
        total_hourly["rolling_7d"],
        color="#003366",
        linewidth=2.2,
        label="7-Day Rolling Trend (168-hr Moving Avg)",
    )

    ax.set_title("Total Ride-Share Demand Over Time (Full Year Timeline)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date (2014)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total System Demand (Rides / Hour)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95, fontsize=10)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"    Saved: {output_path}")


def plot_feature_importance(
    model_file: Path = XGBOOST_MODEL_FILE,
    output_path: Path = FEATURE_IMPORTANCE_PLOT,
) -> None:
    """4. Feature importance plot for trained XGBoost model."""
    print("[4] Generating Feature Importance plot...")
    model = joblib.load(model_file)

    # Extract feature importances
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(fi_df)))

    bars = ax.barh(fi_df["Feature"], fi_df["Importance"], color=colors, edgecolor="#222222", linewidth=0.5)

    # Annotate value on each bar
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width:.3f}",
            (width, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color="#333333",
            fontweight="bold",
        )

    ax.set_xlim(0, fi_df["Importance"].max() * 1.15)
    ax.set_title("XGBoost Model Feature Importance (Gini Gain)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Importance Weight", fontsize=11, fontweight="bold")
    ax.set_ylabel("Model Feature", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"    Saved: {output_path}")


def plot_hourly_demand_pattern(
    hourly_file: Path = HOURLY_DEMAND_FILE,
    output_path: Path = HOURLY_DEMAND_PATTERN_PLOT,
) -> None:
    """5. Average demand by hour (hour_of_day vs average demand)."""
    print("[5] Generating Average Demand by Hour plot...")
    df = pd.read_csv(hourly_file)

    # Compute overall hourly average and split by weekday vs weekend
    hourly_overall = df.groupby("hour_of_day")["demand"].mean().reset_index()
    hourly_weekday = df[df["is_weekend"] == 0].groupby("hour_of_day")["demand"].mean().reset_index()
    hourly_weekend = df[df["is_weekend"] == 1].groupby("hour_of_day")["demand"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    ax.plot(
        hourly_overall["hour_of_day"],
        hourly_overall["demand"],
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        label="Overall Average Demand",
    )
    ax.plot(
        hourly_weekday["hour_of_day"],
        hourly_weekday["demand"],
        linestyle="--",
        linewidth=1.8,
        color="#2ca02c",
        label="Weekday (Mon - Fri)",
    )
    ax.plot(
        hourly_weekend["hour_of_day"],
        hourly_weekend["demand"],
        linestyle="--",
        linewidth=1.8,
        color="#ff7f0e",
        label="Weekend (Sat - Sun)",
    )

    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, fontsize=8)
    ax.set_title("Diurnal Demand Pattern (Hour of Day vs. Average Demand)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Hour of Day", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Rides per Zone / Hour", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95, fontsize=10)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"    Saved: {output_path}")


def plot_zone_demand(
    hourly_file: Path = HOURLY_DEMAND_FILE,
    output_path: Path = ZONE_DEMAND_PLOT,
) -> None:
    """6. Average demand by zone (zone vs average demand bar plot)."""
    print("[6] Generating Average Demand by Zone plot...")
    df = pd.read_csv(hourly_file)

    zone_avg = df.groupby("zone")["demand"].mean().reset_index()
    zone_avg = zone_avg.sort_values(by="demand", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    palette = sns.color_palette("mako", len(zone_avg))

    bars = ax.bar(
        [f"Z{z}" for z in zone_avg["zone"]],
        zone_avg["demand"],
        color=palette,
        edgecolor="#333333",
        linewidth=0.6,
    )

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#222222",
        )

    ax.set_title("Average Hourly Demand by Geographical Zone (Sorted)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Geographical Zone ID", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Rides per Hour", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"    Saved: {output_path}")


def run_all_visualizations() -> list[Path]:
    """Generates all 6 Module 8 visualizations."""
    print("=" * 70)
    print(" MODULE 8: VISUALIZATION AND RESULTS")
    print("=" * 70)

    plot_geographical_zones()
    plot_actual_vs_predicted()
    plot_demand_over_time()
    plot_feature_importance()
    plot_hourly_demand_pattern()
    plot_zone_demand()

    generated_plots = [
        GEOGRAPHICAL_ZONES_PLOT,
        ACTUAL_VS_PREDICTED_PLOT,
        DEMAND_OVER_TIME_PLOT,
        FEATURE_IMPORTANCE_PLOT,
        HOURLY_DEMAND_PATTERN_PLOT,
        ZONE_DEMAND_PLOT,
    ]

    print("=" * 70)
    print(" [OK] MODULE 8 VISUALIZATION COMPLETED SUCCESSFULLY")
    print(" Generated PNG files:")
    for p in generated_plots:
        status = "CONFIRMED" if p.exists() else "MISSING"
        size_kb = p.stat().st_size / 1024 if p.exists() else 0
        print(f"   - {p.name:<28}: {size_kb:>7.1f} KB [{status}]")
    print("=" * 70 + "\n")

    return generated_plots


if __name__ == "__main__":
    run_all_visualizations()
