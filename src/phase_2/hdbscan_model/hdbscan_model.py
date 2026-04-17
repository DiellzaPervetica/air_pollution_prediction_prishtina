from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

try:
    import hdbscan
except Exception as e:
    raise ImportError(
        "Missing package 'hdbscan'. Install with:\n"
        "pip install hdbscan umap-learn pandas numpy scikit-learn scipy joblib plotly kaleido"
    ) from e

try:
    import umap
except Exception as e:
    raise ImportError(
        "Missing package 'umap-learn'. Install with:\n"
        "pip install umap-learn"
    ) from e


# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_PATH = BASE_DIR / "data" / "4E_selected_dataset.csv"

MODEL_DIR = BASE_DIR / "models" / "hdbscan_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = BASE_DIR / "pictures" / "hdbscan_model"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CLUSTERED = BASE_DIR / "data" / "hdbscan_clustered_dataset.csv"
OUTPUT_METRICS = BASE_DIR / "data" / "hdbscan_metrics.csv"
OUTPUT_CLUSTER_SUMMARY = BASE_DIR / "data" / "hdbscan_cluster_summary.csv"
OUTPUT_FEATURE_SUMMARY = BASE_DIR / "data" / "hdbscan_feature_summary.csv"
OUTPUT_RUN_INFO = BASE_DIR / "data" / "hdbscan_run_info.json"

MODEL_PATH = MODEL_DIR / "hdbscan_model.pkl"
SCALER_PATH = MODEL_DIR / "hdbscan_scaler.pkl"
UMAP_PATH = MODEL_DIR / "hdbscan_umap.pkl"

TIME_CANDIDATES = ["datetime", "date"]
TARGET_FOR_INTERPRETATION = "pm25"   # vetëm për interpretim cluster-ash, jo target supervised

# HDBSCAN params
MIN_CLUSTER_SIZE = 80
MIN_SAMPLES = 20
CLUSTER_SELECTION_METHOD = "eom"     # "eom" ose "leaf"
METRIC = "euclidean"

# UMAP params
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.05
UMAP_N_COMPONENTS = 2

RANDOM_STATE = 42
MIN_REQUIRED_ROWS = 300


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# =============================================================================
# HELPERS
# =============================================================================
def detect_time_column(df: pd.DataFrame) -> str | None:
    for col in TIME_CANDIDATES:
        if col in df.columns:
            return col
    return None


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    for c in bool_cols:
        df[c] = df[c].astype(int)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # hiq kolona tipike që nuk duhen si feature në clustering
    drop_like = {
        "unnamed: 0",
        "cluster_label",
        "cluster_probability",
        "outlier_score",
        "umap_1",
        "umap_2",
    }
    num_cols = [c for c in num_cols if c not in drop_like and not c.endswith("_was_missing")]
    return num_cols


def safe_internal_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> dict:
    # Hiq noise (-1) për metrikat klasike
    mask = labels != -1
    X_core = X_scaled[mask]
    y_core = labels[mask]

    unique_clusters = np.unique(y_core)

    result = {
        "silhouette_score": np.nan,
        "davies_bouldin_score": np.nan,
        "calinski_harabasz_score": np.nan,
    }

    if len(X_core) >= 3 and len(unique_clusters) >= 2:
        result["silhouette_score"] = float(silhouette_score(X_core, y_core))
        result["davies_bouldin_score"] = float(davies_bouldin_score(X_core, y_core))
        result["calinski_harabasz_score"] = float(calinski_harabasz_score(X_core, y_core))

    return result


def top_feature_differences(df_clustered: pd.DataFrame, feature_cols: list[str], top_n: int = 15) -> pd.DataFrame:
    rows = []
    global_means = df_clustered[feature_cols].mean()

    cluster_ids = sorted([c for c in df_clustered["cluster_label"].unique() if c != -1])

    for cluster_id in cluster_ids:
        sub = df_clustered[df_clustered["cluster_label"] == cluster_id]
        cluster_means = sub[feature_cols].mean()
        delta = (cluster_means - global_means).abs().sort_values(ascending=False)

        for feat in delta.head(top_n).index:
            rows.append({
                "cluster_label": cluster_id,
                "feature": feat,
                "global_mean": float(global_means[feat]),
                "cluster_mean": float(cluster_means[feat]),
                "absolute_difference": float(abs(cluster_means[feat] - global_means[feat])),
            })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    set_seed(RANDOM_STATE)

    print("=" * 90)
    print("HDBSCAN :: UNSUPERVISED CLUSTERING PIPELINE")
    print("=" * 90)
    print(f"Input dataset: {INPUT_PATH}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    time_col = detect_time_column(df)
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
        if df.duplicated(subset=[time_col]).sum() > 0:
            df = df.drop_duplicates(subset=[time_col], keep="first").reset_index(drop=True)

    feature_cols = numeric_feature_columns(df)

    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found for HDBSCAN.")

    # numeric cleanup
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    print("\n" + "=" * 80)
    print("DATA QUALITY CHECK")
    print("=" * 80)
    print(f"Rows before cleaning: {len(df)}")
    print(f"Detected time column: {time_col}")
    print(f"Numeric feature columns ({len(feature_cols)}):")
    print(feature_cols)

    missing_before = df[feature_cols].isna().sum().sort_values(ascending=False)
    print("\nMissing values before cleaning:")
    print(missing_before[missing_before > 0].to_string() if (missing_before > 0).any() else "No missing values.")

    before_drop = len(df)
    df = df.dropna(subset=feature_cols).copy()
    after_drop = len(df)

    print(f"\nDropped incomplete rows: {before_drop - after_drop}")
    print(f"Rows remaining: {after_drop}")

    if len(df) < MIN_REQUIRED_ROWS:
        raise ValueError(f"Too few rows after cleaning ({len(df)}). Need at least {MIN_REQUIRED_ROWS}.")

    X = df[feature_cols].to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError("Feature matrix still contains NaN/Inf after cleaning.")

    print("\n" + "=" * 80)
    print("SCALING")
    print("=" * 80)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)

    print(f"Scaled matrix shape: {X_scaled.shape}")
    print(f"Saved scaler to: {SCALER_PATH}")

    print("\n" + "=" * 80)
    print("HDBSCAN TRAINING")
    print("=" * 80)
    print(f"min_cluster_size: {MIN_CLUSTER_SIZE}")
    print(f"min_samples     : {MIN_SAMPLES}")
    print(f"selection method: {CLUSTER_SELECTION_METHOD}")
    print(f"metric          : {METRIC}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        metric=METRIC,
        prediction_data=True,
        gen_min_span_tree=True,
    )

    labels = clusterer.fit_predict(X_scaled)
    probabilities = clusterer.probabilities_
    outlier_scores = clusterer.outlier_scores_

    joblib.dump(clusterer, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")

    print("\n" + "=" * 80)
    print("UMAP EMBEDDING")
    print("=" * 80)

    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=UMAP_N_COMPONENTS,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )
    embedding = reducer.fit_transform(X_scaled)
    joblib.dump(reducer, UMAP_PATH)

    print(f"Saved UMAP reducer to: {UMAP_PATH}")

    df_clustered = df.copy()
    df_clustered["cluster_label"] = labels
    df_clustered["cluster_probability"] = probabilities
    df_clustered["outlier_score"] = outlier_scores
    df_clustered["umap_1"] = embedding[:, 0]
    df_clustered["umap_2"] = embedding[:, 1]

    # metrics
    n_noise = int((labels == -1).sum())
    n_non_noise = int((labels != -1).sum())
    unique_non_noise = sorted([int(x) for x in np.unique(labels) if x != -1])
    n_clusters = len(unique_non_noise)
    noise_ratio = float(n_noise / len(labels))

    persistence = getattr(clusterer, "cluster_persistence_", None)
    avg_persistence = float(np.mean(persistence)) if persistence is not None and len(persistence) > 0 else np.nan

    internal = safe_internal_metrics(X_scaled, labels)

    metrics_df = pd.DataFrame([{
        "rows_used": int(len(df_clustered)),
        "n_features": int(len(feature_cols)),
        "n_clusters_excluding_noise": int(n_clusters),
        "n_noise_points": int(n_noise),
        "n_non_noise_points": int(n_non_noise),
        "noise_ratio": noise_ratio,
        "avg_cluster_persistence": avg_persistence,
        "silhouette_score": internal["silhouette_score"],
        "davies_bouldin_score": internal["davies_bouldin_score"],
        "calinski_harabasz_score": internal["calinski_harabasz_score"],
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "min_samples": MIN_SAMPLES,
        "cluster_selection_method": CLUSTER_SELECTION_METHOD,
        "metric": METRIC,
    }])
    metrics_df.to_csv(OUTPUT_METRICS, index=False)

    print("\n" + "=" * 80)
    print("CLUSTERING METRICS")
    print("=" * 80)
    print(metrics_df.to_string(index=False))

    # cluster summary
    group_cols = ["cluster_label"]
    agg_dict = {
        "cluster_probability": ["mean", "min", "max"],
        "outlier_score": ["mean", "max"],
    }

    if TARGET_FOR_INTERPRETATION in df_clustered.columns:
        agg_dict[TARGET_FOR_INTERPRETATION] = ["mean", "median", "min", "max"]

    cluster_summary = (
        df_clustered.groupby(group_cols)
        .agg(
            count=("cluster_label", "size"),
            cluster_probability_mean=("cluster_probability", "mean"),
            cluster_probability_min=("cluster_probability", "min"),
            cluster_probability_max=("cluster_probability", "max"),
            outlier_score_mean=("outlier_score", "mean"),
            outlier_score_max=("outlier_score", "max"),
            **(
                {
                    f"{TARGET_FOR_INTERPRETATION}_mean": (TARGET_FOR_INTERPRETATION, "mean"),
                    f"{TARGET_FOR_INTERPRETATION}_median": (TARGET_FOR_INTERPRETATION, "median"),
                    f"{TARGET_FOR_INTERPRETATION}_min": (TARGET_FOR_INTERPRETATION, "min"),
                    f"{TARGET_FOR_INTERPRETATION}_max": (TARGET_FOR_INTERPRETATION, "max"),
                } if TARGET_FOR_INTERPRETATION in df_clustered.columns else {}
            )
        )
        .reset_index()
        .sort_values(["cluster_label"])
    )
    cluster_summary.to_csv(OUTPUT_CLUSTER_SUMMARY, index=False)

    # feature interpretation
    feature_summary = top_feature_differences(df_clustered, feature_cols, top_n=15)
    feature_summary.to_csv(OUTPUT_FEATURE_SUMMARY, index=False)

    # interactive plot
    print("\n" + "=" * 80)
    print("INTERACTIVE VISUALIZATION")
    print("=" * 80)

    hover_cols = {}
    for col in feature_cols[:10]:
        hover_cols[col] = True
    if TARGET_FOR_INTERPRETATION in df_clustered.columns:
        hover_cols[TARGET_FOR_INTERPRETATION] = True

    df_clustered["cluster_label_str"] = df_clustered["cluster_label"].astype(str)
    df_clustered["is_noise"] = np.where(df_clustered["cluster_label"] == -1, "Noise", "Clustered")

    title = "HDBSCAN Clusters on 4E Selected Dataset (UMAP 2D projection)"
    fig = px.scatter(
        df_clustered,
        x="umap_1",
        y="umap_2",
        color="cluster_label_str",
        symbol="is_noise",
        hover_data=hover_cols,
        title=title,
        opacity=0.85,
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(template="plotly_white", height=720)

    html_path = PLOTS_DIR / "hdbscan_umap_interactive.html"
    png_path = PLOTS_DIR / "hdbscan_umap_interactive.png"

    fig.write_html(str(html_path), include_plotlyjs="cdn")
    try:
        fig.write_image(str(png_path), scale=2)
    except Exception:
        pass

    print(f"Saved interactive plot to: {html_path}")

    # save clustered dataset
    df_clustered.to_csv(OUTPUT_CLUSTERED, index=False)

    run_info = {
        "input_path": str(INPUT_PATH),
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "umap_path": str(UMAP_PATH),
        "outputs": {
            "clustered_dataset": str(OUTPUT_CLUSTERED),
            "metrics": str(OUTPUT_METRICS),
            "cluster_summary": str(OUTPUT_CLUSTER_SUMMARY),
            "feature_summary": str(OUTPUT_FEATURE_SUMMARY),
            "plot_html": str(html_path),
        },
        "config": {
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "min_samples": MIN_SAMPLES,
            "cluster_selection_method": CLUSTER_SELECTION_METHOD,
            "metric": METRIC,
            "umap_n_neighbors": UMAP_N_NEIGHBORS,
            "umap_min_dist": UMAP_MIN_DIST,
            "umap_n_components": UMAP_N_COMPONENTS,
        },
        "feature_columns": feature_cols,
    }

    with open(OUTPUT_RUN_INFO, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Clustered dataset : {OUTPUT_CLUSTERED}")
    print(f"Metrics           : {OUTPUT_METRICS}")
    print(f"Cluster summary   : {OUTPUT_CLUSTER_SUMMARY}")
    print(f"Feature summary   : {OUTPUT_FEATURE_SUMMARY}")
    print(f"Interactive plot  : {html_path}")


if __name__ == "__main__":
    main()