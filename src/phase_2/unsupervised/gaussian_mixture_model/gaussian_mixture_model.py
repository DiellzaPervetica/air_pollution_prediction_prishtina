from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import plotly.express as px
except Exception:
    px = None


matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "phase_2" / "unsupervised" / "gaussian_mixture"
MODELS_DIR = PROJECT_ROOT / "models" / "gaussian_mixture_model"
PLOTS_DIR = PROJECT_ROOT / "pictures" / "phase_2" / "unsupervised" / "gaussian_mixture"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "pm25"
TIME_CANDIDATES = ["datetime", "date"]
RANDOM_STATE = 42
MIN_REQUIRED_ROWS = 300
PCA_VARIANCE_THRESHOLD = 0.95
N_COMPONENT_CANDIDATES = [2, 3, 4, 5, 6]
COVARIANCE_TYPES = ["full", "diag", "tied"]
N_INIT = 10
MIN_CLUSTER_RATIO = 0.05

FEATURE_PRIORITY = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "pollution_stagnation_index",
    "wind_x_vector",
    "wind_y_vector",
    "total_generation_mw",
    "temperature_2m (\u00b0C)",
    "rain (mm)",
    "relative_humidity_2m (%)",
    "wind_speed_10m (km/h)",
]

INPUT_CANDIDATES = [
    PROJECT_ROOT / "data" / "4E_selected_dataset.csv",
    PROJECT_ROOT / "data" / "phase_1" / "4E_selected_dataset.csv",
]
PM25_SCALER_CANDIDATES = [
    PROJECT_ROOT / "models" / "scaler.pkl",
]

OUTPUT_CLUSTERED = DATA_DIR / "gmm_clustered_dataset.csv"
OUTPUT_METRICS = DATA_DIR / "gmm_metrics.csv"
OUTPUT_CLUSTER_SUMMARY = DATA_DIR / "gmm_cluster_summary.csv"
OUTPUT_FEATURE_SUMMARY = DATA_DIR / "gmm_feature_summary.csv"
OUTPUT_MODEL_SELECTION = DATA_DIR / "gmm_model_selection.csv"
OUTPUT_RUN_INFO = DATA_DIR / "gmm_run_info.json"

MODEL_PATH = MODELS_DIR / "gmm_model.pkl"
SCALER_PATH = MODELS_DIR / "gmm_scaler.pkl"
PCA_PATH = MODELS_DIR / "gmm_pca.pkl"
FEATURES_PATH = MODELS_DIR / "gmm_feature_columns.pkl"


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_existing_path(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{label} not found. Checked:\n{searched}")


def load_pm25_scaler() -> object | None:
    try:
        scaler_path = resolve_existing_path(PM25_SCALER_CANDIDATES, "PM2.5 scaler")
        return joblib.load(scaler_path)
    except Exception:
        return None


def inverse_scale_pm25(values: pd.Series | np.ndarray, scaler: object | None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if scaler is None:
        return values

    feature_names = list(getattr(scaler, "feature_names_in_", []))
    if TARGET not in feature_names:
        return values

    idx = feature_names.index(TARGET)
    log_values = values * scaler.scale_[idx] + scaler.mean_[idx]
    return np.expm1(log_values)


def detect_time_column(df: pd.DataFrame) -> str | None:
    for column in TIME_CANDIDATES:
        if column in df.columns:
            return column
    return None


def prepare_dataframe() -> tuple[pd.DataFrame, list[str], Path]:
    input_path = resolve_existing_path(INPUT_CANDIDATES, "Phase 1 selected dataset")
    df = pd.read_csv(input_path)

    time_col = detect_time_column(df)
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
        if df.duplicated(subset=[time_col]).sum() > 0:
            df = df.drop_duplicates(subset=[time_col], keep="first").reset_index(drop=True)
        df["timestamp"] = df[time_col]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [column for column in FEATURE_PRIORITY if column in df.columns]

    if not feature_cols:
        feature_cols = [
            column
            for column in numeric_cols
            if column != TARGET and column != "wind_direction_10m (\u00b0)" and not column.endswith("_was_missing")
        ]

    if len(feature_cols) == 0:
        raise ValueError("No numeric features available for Gaussian Mixture clustering.")

    cols_to_keep = feature_cols.copy()
    if TARGET in df.columns:
        cols_to_keep.append(TARGET)
    if "timestamp" in df.columns:
        cols_to_keep.append("timestamp")

    df = df[cols_to_keep].copy()
    for column in feature_cols + ([TARGET] if TARGET in df.columns else []):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    if len(df) < MIN_REQUIRED_ROWS:
        raise ValueError(f"Too few rows after cleaning ({len(df)}). Need at least {MIN_REQUIRED_ROWS}.")

    return df, feature_cols, input_path


def safe_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {
            "silhouette_score": np.nan,
            "davies_bouldin_score": np.nan,
            "calinski_harabasz_score": np.nan,
        }

    return {
        "silhouette_score": float(silhouette_score(X, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(X, labels)),
    }


def top_feature_differences(df_clustered: pd.DataFrame, feature_cols: list[str], top_n: int = 12) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    global_means = df_clustered[feature_cols].mean()

    for cluster_label in sorted(df_clustered["cluster_label"].unique()):
        cluster_df = df_clustered[df_clustered["cluster_label"] == cluster_label]
        cluster_means = cluster_df[feature_cols].mean()
        deltas = (cluster_means - global_means).abs().sort_values(ascending=False)

        for feature in deltas.head(top_n).index:
            rows.append(
                {
                    "cluster_label": int(cluster_label),
                    "feature": feature,
                    "global_mean": float(global_means[feature]),
                    "cluster_mean": float(cluster_means[feature]),
                    "absolute_difference": float(abs(cluster_means[feature] - global_means[feature])),
                }
            )

    return pd.DataFrame(rows)


def select_best_model(X_pca: np.ndarray) -> tuple[pd.DataFrame, dict[str, object]]:
    selection_rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None

    for n_components in N_COMPONENT_CANDIDATES:
        for covariance_type in COVARIANCE_TYPES:
            model = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                n_init=N_INIT,
                random_state=RANDOM_STATE,
                reg_covar=1e-6,
            )
            model.fit(X_pca)

            labels = model.predict(X_pca)
            probabilities = model.predict_proba(X_pca)
            counts = np.bincount(labels, minlength=n_components)
            cluster_ratios = counts / counts.sum()

            row = {
                "n_components": int(n_components),
                "covariance_type": covariance_type,
                "bic": float(model.bic(X_pca)),
                "aic": float(model.aic(X_pca)),
                "min_cluster_ratio": float(cluster_ratios.min()),
                "max_cluster_ratio": float(cluster_ratios.max()),
                "avg_cluster_confidence": float(probabilities.max(axis=1).mean()),
            }
            row.update(safe_metrics(X_pca, labels))
            selection_rows.append(row)

            if row["min_cluster_ratio"] < MIN_CLUSTER_RATIO:
                continue

            if best_row is None:
                best_row = row
                continue

            if row["bic"] < best_row["bic"] - 1e-9:
                best_row = row
                continue

            if np.isclose(row["bic"], best_row["bic"]) and row["silhouette_score"] > best_row["silhouette_score"]:
                best_row = row

    selection_df = pd.DataFrame(selection_rows).sort_values(
        by=["bic", "silhouette_score", "aic"],
        ascending=[True, False, True],
    )

    if best_row is None:
        best_row = selection_df.iloc[0].to_dict()

    return selection_df, best_row


def save_model_selection_plot(selection_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for covariance_type in COVARIANCE_TYPES:
        subset = selection_df[selection_df["covariance_type"] == covariance_type]
        axes[0].plot(subset["n_components"], subset["bic"], marker="o", label=covariance_type)
        axes[1].plot(subset["n_components"], subset["silhouette_score"], marker="o", label=covariance_type)

    axes[0].set_title("BIC across candidate GMM models")
    axes[0].set_xlabel("Number of components")
    axes[0].set_ylabel("BIC")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].set_title("Silhouette across candidate GMM models")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Silhouette")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    for axis in axes:
        axis.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def save_cluster_profile_heatmap(df_clustered: pd.DataFrame, feature_cols: list[str], output_path: Path) -> None:
    cluster_profiles = df_clustered.groupby("cluster_label")[feature_cols].mean().sort_index()
    values = cluster_profiles.to_numpy()

    fig, ax = plt.subplots(figsize=(max(12, len(feature_cols) * 0.7), max(4, len(cluster_profiles) * 0.8)))
    im = ax.imshow(values, aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(cluster_profiles.index)))
    ax.set_yticklabels([f"Cluster {idx}" for idx in cluster_profiles.index])
    ax.set_title("Gaussian Mixture cluster profiles (feature means)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def build_interactive_plot(df_clustered: pd.DataFrame, html_path: Path, png_path: Path | None = None) -> None:
    if px is None:
        return

    hover_data = {
        "cluster_confidence": True,
        "log_likelihood": True,
    }

    for column in df_clustered.columns:
        if column in {"pca_1", "pca_2", "cluster_label", "cluster_label_str", "cluster_confidence", "log_likelihood"}:
            continue
        if column.startswith("cluster_prob_"):
            continue
        if len(hover_data) >= 12:
            break
        hover_data[column] = True

    fig = px.scatter(
        df_clustered,
        x="pca_1",
        y="pca_2",
        color="cluster_label_str",
        hover_data=hover_data,
        title="Gaussian Mixture clusters on PCA projection",
        opacity=0.82,
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(template="plotly_white", height=720)

    fig.write_html(str(html_path), include_plotlyjs="cdn")
    if png_path is not None:
        try:
            fig.write_image(str(png_path), scale=2)
        except Exception:
            pass


def main() -> None:
    set_seed(RANDOM_STATE)

    print("=" * 90)
    print("GAUSSIAN MIXTURE :: UNSUPERVISED CLUSTERING PIPELINE")
    print("=" * 90)

    pm25_scaler = load_pm25_scaler()
    df, feature_cols, input_path = prepare_dataframe()

    print(f"Input dataset: {input_path}")
    print(f"Rows used    : {len(df)}")
    print(f"Features     : {len(feature_cols)}")
    print("Columns      :", feature_cols)

    print("\n" + "=" * 80)
    print("SCALING + PCA")
    print("=" * 80)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].to_numpy(dtype=float))

    pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, svd_solver="full")
    X_pca = pca.fit_transform(X_scaled)

    if X_pca.shape[1] < 2:
        raise ValueError("PCA reduced the data to fewer than 2 components. Cannot visualize clusters.")

    print(f"Scaled matrix shape : {X_scaled.shape}")
    print(f"PCA matrix shape    : {X_pca.shape}")
    print(f"Explained variance  : {pca.explained_variance_ratio_.sum():.4f}")

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(pca, PCA_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)

    print("\n" + "=" * 80)
    print("MODEL SELECTION")
    print("=" * 80)

    selection_df, best_row = select_best_model(X_pca)
    selection_df.to_csv(OUTPUT_MODEL_SELECTION, index=False)
    print(selection_df.to_string(index=False))

    best_n_components = int(best_row["n_components"])
    best_covariance_type = str(best_row["covariance_type"])

    final_model = GaussianMixture(
        n_components=best_n_components,
        covariance_type=best_covariance_type,
        n_init=N_INIT,
        random_state=RANDOM_STATE,
        reg_covar=1e-6,
    )
    final_model.fit(X_pca)

    labels = final_model.predict(X_pca)
    probabilities = final_model.predict_proba(X_pca)
    cluster_confidence = probabilities.max(axis=1)
    log_likelihood = final_model.score_samples(X_pca)

    joblib.dump(final_model, MODEL_PATH)

    metrics = safe_metrics(X_pca, labels)
    counts = np.bincount(labels, minlength=best_n_components)
    cluster_ratios = counts / counts.sum()

    metrics_df = pd.DataFrame(
        [
            {
                "rows_used": len(df),
                "n_features": len(feature_cols),
                "pca_components": int(X_pca.shape[1]),
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                "n_clusters": int(best_n_components),
                "covariance_type": best_covariance_type,
                "bic": float(final_model.bic(X_pca)),
                "aic": float(final_model.aic(X_pca)),
                "silhouette_score": metrics["silhouette_score"],
                "davies_bouldin_score": metrics["davies_bouldin_score"],
                "calinski_harabasz_score": metrics["calinski_harabasz_score"],
                "min_cluster_ratio": float(cluster_ratios.min()),
                "max_cluster_ratio": float(cluster_ratios.max()),
                "avg_cluster_confidence": float(cluster_confidence.mean()),
            }
        ]
    )
    metrics_df.to_csv(OUTPUT_METRICS, index=False)

    print("\n" + "=" * 80)
    print("FINAL CLUSTERING METRICS")
    print("=" * 80)
    print(metrics_df.to_string(index=False))

    df_clustered = df.copy()
    df_clustered["cluster_label"] = labels
    df_clustered["cluster_label_str"] = df_clustered["cluster_label"].astype(str)
    df_clustered["cluster_confidence"] = cluster_confidence
    df_clustered["log_likelihood"] = log_likelihood
    df_clustered["pca_1"] = X_pca[:, 0]
    df_clustered["pca_2"] = X_pca[:, 1]

    for idx in range(best_n_components):
        df_clustered[f"cluster_prob_{idx}"] = probabilities[:, idx]

    if TARGET in df_clustered.columns:
        df_clustered["pm25_real"] = inverse_scale_pm25(df_clustered[TARGET].to_numpy(), pm25_scaler)

    df_clustered.to_csv(OUTPUT_CLUSTERED, index=False)

    cluster_summary = (
        df_clustered.groupby("cluster_label")
        .agg(
            count=("cluster_label", "size"),
            cluster_ratio=("cluster_label", lambda values: len(values) / len(df_clustered)),
            cluster_confidence_mean=("cluster_confidence", "mean"),
            cluster_confidence_min=("cluster_confidence", "min"),
            cluster_confidence_max=("cluster_confidence", "max"),
            log_likelihood_mean=("log_likelihood", "mean"),
            log_likelihood_min=("log_likelihood", "min"),
            log_likelihood_max=("log_likelihood", "max"),
            **(
                {
                    "pm25_real_mean": ("pm25_real", "mean"),
                    "pm25_real_median": ("pm25_real", "median"),
                    "pm25_real_min": ("pm25_real", "min"),
                    "pm25_real_max": ("pm25_real", "max"),
                }
                if "pm25_real" in df_clustered.columns
                else {}
            ),
        )
        .reset_index()
        .sort_values("cluster_label")
    )
    cluster_summary.to_csv(OUTPUT_CLUSTER_SUMMARY, index=False)

    feature_summary = top_feature_differences(df_clustered, feature_cols, top_n=12)
    feature_summary.to_csv(OUTPUT_FEATURE_SUMMARY, index=False)

    model_selection_plot = PLOTS_DIR / "gmm_model_selection.png"
    save_model_selection_plot(selection_df, model_selection_plot)

    heatmap_path = PLOTS_DIR / "gmm_cluster_profile_heatmap.png"
    save_cluster_profile_heatmap(df_clustered, feature_cols, heatmap_path)

    html_path = PLOTS_DIR / "gmm_pca_interactive.html"
    png_path = PLOTS_DIR / "gmm_pca_interactive.png"
    build_interactive_plot(df_clustered, html_path, png_path)

    run_info = {
        "input_path": str(input_path),
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "pca_path": str(PCA_PATH),
        "feature_columns_path": str(FEATURES_PATH),
        "outputs": {
            "clustered_dataset": str(OUTPUT_CLUSTERED),
            "metrics": str(OUTPUT_METRICS),
            "cluster_summary": str(OUTPUT_CLUSTER_SUMMARY),
            "feature_summary": str(OUTPUT_FEATURE_SUMMARY),
            "model_selection": str(OUTPUT_MODEL_SELECTION),
            "interactive_plot": str(html_path),
            "cluster_profile_heatmap": str(heatmap_path),
            "model_selection_plot": str(model_selection_plot),
        },
        "config": {
            "pca_variance_threshold": PCA_VARIANCE_THRESHOLD,
            "n_component_candidates": N_COMPONENT_CANDIDATES,
            "covariance_types": COVARIANCE_TYPES,
            "n_init": N_INIT,
            "min_cluster_ratio": MIN_CLUSTER_RATIO,
        },
        "selected_model": {
            "n_components": best_n_components,
            "covariance_type": best_covariance_type,
        },
        "feature_columns": feature_cols,
    }

    with open(OUTPUT_RUN_INFO, "w", encoding="utf-8") as run_info_file:
        json.dump(run_info, run_info_file, indent=2, default=str)

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
