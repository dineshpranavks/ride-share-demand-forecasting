"""Module 3: Geographical Demand Zone Generation Using K-Means.

This module converts raw latitude and longitude pickup locations into discrete
geographical demand zones (clusters) to enable spatial demand forecasting.

Key steps:
1. Load cleaned taxi ride dataset (data/cleaned_taxi.csv).
2. Extract spatial clustering features (latitude, longitude).
3. Train a MiniBatchKMeans clustering model (n_clusters=20, random_state=42).
4. Assign a cluster ID ('zone') to each ride.
5. Save clustered dataset to data/taxi_with_zones.csv.
6. Persist trained clustering model to models/kmeans_zones.pkl.
7. Print zone distribution statistics (clusters, records, counts per zone).
8. Generate and save a high-resolution scatter plot visualization with cluster centers
   to outputs/demand_zones.png.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import MiniBatchKMeans, KMeans

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    CLEANED_DATA_FILE,
    TAXI_WITH_ZONES_FILE,
    KMEANS_MODEL_FILE,
    DEMAND_ZONES_PLOT,
    N_CLUSTERS,
    RANDOM_STATE,
    MODELS_DIR,
    OUTPUTS_DIR,
)


def load_cleaned_data(file_path: Path = CLEANED_DATA_FILE) -> pd.DataFrame:
    """Loads the preprocessed taxi dataset."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at: {file_path}. "
            f"Please run Module 2 (src/preprocess.py) first."
        )
    print(f"[1] Loading cleaned taxi data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"    Loaded {len(df):,} records.")
    return df


def train_clustering_model(
    coordinates: np.ndarray,
    n_clusters: int = N_CLUSTERS,
    random_state: int = RANDOM_STATE,
    use_minibatch: bool = True,
) -> MiniBatchKMeans:
    """Trains a K-Means or MiniBatchKMeans model on geographical coordinates.

    MiniBatchKMeans is utilized by default as it is computationally efficient
    for ~100k geospatial records while achieving convergence quality comparable
    to standard K-Means.
    """
    algorithm_name = "MiniBatchKMeans" if use_minibatch else "KMeans"
    print(f"[2] Training {algorithm_name} model:")
    print(f"    - Number of clusters (n_clusters): {n_clusters}")
    print(f"    - Random state: {random_state}")

    if use_minibatch:
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=2048,
            n_init=3,
        )
    else:
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )

    model.fit(coordinates)
    print("    Model training completed successfully.")
    return model


def plot_demand_zones(
    df: pd.DataFrame,
    cluster_centers: np.ndarray,
    output_path: Path = DEMAND_ZONES_PLOT,
    n_clusters: int = N_CLUSTERS,
) -> None:
    """Generates and saves a scatter plot of geographical demand zones.

    - Longitude on X-axis
    - Latitude on Y-axis
    - Zone as color
    - Cluster centers marked prominently
    """
    print(f"[6] Generating geographical demand zones visualization...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

    # Use a distinct discrete colormap suitable for 20 clusters
    cmap = plt.get_cmap("tab20", n_clusters)

    # Scatter plot of ride pickup coordinates colored by zone
    scatter = ax.scatter(
        df["longitude"],
        df["latitude"],
        c=df["zone"],
        cmap=cmap,
        s=4,
        alpha=0.45,
        edgecolors="none",
        label="Ride Pickups",
    )

    # Display cluster centers (centroids)
    # Note: centers shape is (n_clusters, 2) where col 0 = latitude, col 1 = longitude
    center_lats = cluster_centers[:, 0]
    center_lons = cluster_centers[:, 1]

    ax.scatter(
        center_lons,
        center_lats,
        c="red",
        marker="X",
        s=140,
        linewidths=1.5,
        edgecolors="black",
        zorder=5,
        label="Cluster Centers (Centroids)",
    )

    # Add numeric labels to cluster centers for clarity
    for zone_id, (lat, lon) in enumerate(zip(center_lats, center_lons)):
        ax.annotate(
            f"Z{zone_id}",
            (lon, lat),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
            fontweight="bold",
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray"),
            zorder=6,
        )

    # Colorbar configuration
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(n_clusters), pad=0.02)
    cbar.set_label("Zone ID", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    # Styling and annotations
    ax.set_title(
        f"NYC Ride-Share Demand Zones (K-Means Clustering, K={n_clusters})",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Longitude", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"    Visualization successfully saved to: {output_path}")


def print_clustering_summary(
    df: pd.DataFrame,
    n_clusters: int,
    cluster_centers: np.ndarray,
) -> None:
    """Prints clear and explainable summary metrics for the clustering process."""
    total_records = len(df)
    zone_counts = df["zone"].value_counts().sort_index()

    print("\n" + "=" * 70)
    print(" MODULE 3: K-MEANS DEMAND ZONES SUMMARY")
    print("=" * 70)
    print(f" Total Records Processed : {total_records:,}")
    print(f" Number of Clusters/Zones: {n_clusters}")
    print("-" * 70)
    print(f" {'Zone ID':<8} {'Center Latitude':<18} {'Center Longitude':<18} {'Record Count':<14} {'Demand %':<10}")
    print("-" * 70)

    for zone_id in range(n_clusters):
        count = zone_counts.get(zone_id, 0)
        pct = (count / total_records) * 100 if total_records > 0 else 0
        c_lat = cluster_centers[zone_id, 0]
        c_lon = cluster_centers[zone_id, 1]
        print(
            f" Zone {zone_id:<3} "
            f"{c_lat:<18.6f} "
            f"{c_lon:<18.6f} "
            f"{count:<14,} "
            f"{pct:<9.2f}%"
        )

    print("=" * 70)
    print(f" Highest Demand Zone: Zone {zone_counts.idxmax()} ({zone_counts.max():,} rides)")
    print(f" Lowest Demand Zone : Zone {zone_counts.idxmin()} ({zone_counts.min():,} rides)")
    print("=" * 70 + "\n")


def generate_demand_zones(
    input_file: Path = CLEANED_DATA_FILE,
    output_file: Path = TAXI_WITH_ZONES_FILE,
    model_file: Path = KMEANS_MODEL_FILE,
    plot_file: Path = DEMAND_ZONES_PLOT,
    n_clusters: int = N_CLUSTERS,
    random_state: int = RANDOM_STATE,
    use_minibatch: bool = True,
) -> pd.DataFrame:
    """Executes the full Module 3 pipeline for geospatial clustering."""
    print("=" * 70)
    print(" MODULE 3: GEOGRAPHICAL DEMAND ZONE GENERATION (K-MEANS)")
    print("=" * 70)

    # 1. Load cleaned taxi data
    df = load_cleaned_data(input_file)

    # 2. Extract latitude and longitude features
    required_cols = ["latitude", "longitude"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required spatial features in dataset: {missing}")

    coordinates = df[required_cols].values

    # 3. Train clustering model (MiniBatchKMeans / KMeans)
    model = train_clustering_model(
        coordinates=coordinates,
        n_clusters=n_clusters,
        random_state=random_state,
        use_minibatch=use_minibatch,
    )

    # 4. Assign each ride a zone ID
    df["zone"] = model.predict(coordinates).astype(int)
    print(f"[3] Assigned each ride a zone ID (column 'zone', 0 to {n_clusters - 1}).")

    # 5. Save the result as data/taxi_with_zones.csv
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"[4] Saved clustered dataset to: {output_file}")

    # 6. Save the trained clustering model as models/kmeans_zones.pkl
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file)
    print(f"[5] Saved trained clustering model to: {model_file}")

    # 7. Print summary metrics (clusters, records, records per zone)
    print_clustering_summary(df, n_clusters, model.cluster_centers_)

    # 8. Create and save visualization with cluster centers
    plot_demand_zones(
        df=df,
        cluster_centers=model.cluster_centers_,
        output_path=plot_file,
        n_clusters=n_clusters,
    )

    print("================================================================================")
    print(" [OK] MODULE 3 EXECUTION COMPLETED SUCCESSFULLY")
    print(f"      - Clustered Dataset : {output_file}")
    print(f"      - Trained Model     : {model_file}")
    print(f"      - Visualization     : {plot_file}")
    print("================================================================================\n")

    return df


if __name__ == "__main__":
    generate_demand_zones()
