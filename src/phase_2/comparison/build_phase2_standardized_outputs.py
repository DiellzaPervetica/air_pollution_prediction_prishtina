from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
PICTURES_DIR = PROJECT_ROOT / "pictures"

PHASE2_DATA_DIR = DATA_DIR / "phase_2"
PHASE2_PLOTS_DIR = PICTURES_DIR / "phase_2"

SUP_DATA_DIR = PHASE2_DATA_DIR / "supervised"
UNSUP_DATA_DIR = PHASE2_DATA_DIR / "unsupervised"
COMP_DATA_DIR = PHASE2_DATA_DIR / "comparison"

SUP_PLOT_DIR = PHASE2_PLOTS_DIR / "supervised"
UNSUP_PLOT_DIR = PHASE2_PLOTS_DIR / "unsupervised"
COMP_PLOT_DIR = PHASE2_PLOTS_DIR / "comparison"

for directory in [
    PHASE2_DATA_DIR,
    PHASE2_PLOTS_DIR,
    SUP_DATA_DIR,
    UNSUP_DATA_DIR,
    COMP_DATA_DIR,
    SUP_PLOT_DIR,
    UNSUP_PLOT_DIR,
    COMP_PLOT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
TARGET = "pm25"


def load_scaler():
    if SCALER_PATH.exists():
        return joblib.load(SCALER_PATH)
    return None


def inverse_scale_feature(values: pd.Series | np.ndarray, scaler, feature_name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if scaler is None:
        return values

    feature_names = list(getattr(scaler, "feature_names_in_", []))
    if feature_name not in feature_names:
        return values

    idx = feature_names.index(feature_name)
    real_values = values * scaler.scale_[idx] + scaler.mean_[idx]
    if feature_name == TARGET:
        return np.expm1(real_values)
    return real_values


def safe_copy(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    if src.resolve() == dst.resolve():
        return str(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def reset_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def copy_artifacts(paths: list[Path], dst_dir: Path) -> list[str]:
    copied: list[str] = []
    for path in paths:
        copied_path = safe_copy(path, dst_dir / path.name)
        if copied_path is not None:
            copied.append(copied_path)
    return copied


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def parse_lightgbm_metrics(text_path: Path) -> dict[str, float]:
    text = text_path.read_text(encoding="utf-8", errors="ignore")

    def extract(pattern: str) -> float:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return float("nan")
        return float(match.group(1))

    return {
        "R2": extract(r"R\S*:\s*([0-9.]+)"),
        "RMSE": extract(r"RMSE:\s*([0-9.]+)"),
        "MAE": extract(r"MAE:\s*([0-9.]+)"),
    }


def save_table_image(df: pd.DataFrame, output_path: Path, title: str) -> None:
    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")

    fig, ax = plt.subplots(figsize=(max(8, len(display_df.columns) * 1.3), max(2.5, len(display_df) * 0.7 + 1.2)))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title(title, fontsize=13, pad=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_metric_bar_chart(df: pd.DataFrame, columns: list[str], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(columns), figsize=(6 * len(columns), 5))
    if len(columns) == 1:
        axes = [axes]

    for ax, column in zip(axes, columns):
        plot_df = df[["model", column]].copy()
        sns.barplot(data=plot_df, x="model", y=column, ax=ax, color="#4C78A8")
        ax.set_title(column)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_feature_panel(feature_panels: list[tuple[str, pd.DataFrame, str]], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(feature_panels), figsize=(6 * len(feature_panels), 6))
    if len(feature_panels) == 1:
        axes = [axes]

    for ax, (label, df, value_col) in zip(axes, feature_panels):
        plot_df = df.head(8).iloc[::-1]
        sns.barplot(data=plot_df, x=value_col, y="feature", ax=ax, color="#72B7B2")
        ax.set_title(label)
        ax.set_xlabel(value_col)
        ax.set_ylabel("")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_actual_vs_predicted_plot(df: pd.DataFrame, output_path: Path, title: str) -> None:
    plot_df = df.copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"], errors="coerce")
    plot_df = plot_df.dropna(subset=["timestamp"]).tail(min(500, len(plot_df)))

    plt.figure(figsize=(14, 5))
    plt.plot(plot_df["timestamp"], plot_df["actual_pm25"], label="Actual", linewidth=1.7)
    plt.plot(plot_df["timestamp"], plot_df["pred_pm25"], label="Predicted", linewidth=1.7, linestyle="--")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("PM2.5")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_cluster_profile_plot(df: pd.DataFrame, label_col: str, value_col: str, output_path: Path, title: str) -> None:
    plot_df = df.copy()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x=label_col, y=value_col, color="#54A24B")
    plt.title(title)
    plt.xlabel(label_col)
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def aggregate_feature_summary(df: pd.DataFrame, feature_col: str = "feature", value_col: str = "absolute_difference") -> pd.DataFrame:
    grouped = df.groupby(feature_col, as_index=False)[value_col].max()
    return grouped.sort_values(value_col, ascending=False).reset_index(drop=True)


def process_lightgbm_improved() -> tuple[dict[str, object], pd.DataFrame]:
    source_dir = PROJECT_ROOT / "src" / "phase_2" / "supervised" / "lightgbm_model" / "improved_model"
    model_data_dir = SUP_DATA_DIR / "lightgbm_improved"
    model_plot_dir = SUP_PLOT_DIR / "lightgbm_improved"
    model_data_dir.mkdir(parents=True, exist_ok=True)
    model_plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = parse_lightgbm_metrics(source_dir / "metrics_summary.txt")
    feature_df = pd.read_csv(source_dir / "feature_importance.csv")
    feature_std = feature_df.rename(
        columns={"Feature": "feature", "Importance_Percentage": "importance_pct"}
    )

    metrics_row = {
        "model": "LightGBM Improved",
        "family": "supervised",
        "evaluation_strategy": "TimeSeriesSplit CV mean",
        "target": TARGET,
        "unit": "real_pm25",
        "n_features": int(len(feature_std)),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE_pct": np.nan,
        "SMAPE_pct": np.nan,
        "R2": metrics["R2"],
        "source_metrics": str(source_dir / "metrics_summary.txt"),
    }
    copied_data = copy_artifacts(
        [
            source_dir / "feature_importance.csv",
            source_dir / "metrics_summary.txt",
        ],
        model_data_dir,
    )
    copied_plots = copy_artifacts(
        [
            source_dir / "actual_vs_predicted.png",
            source_dir / "feature_importance.png",
            source_dir / "learning_curve.png",
        ],
        model_plot_dir,
    )

    manifest = {
        "name": "LightGBM Improved",
        "source_dir": str(source_dir),
        "phase2_data_dir": str(model_data_dir),
        "phase2_plot_dir": str(model_plot_dir),
        "copied_data_files": copied_data,
        "copied_plot_files": copied_plots,
    }
    with open(model_data_dir / "artifact_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return metrics_row, feature_std[["feature", "importance_pct"]]


def process_catboost(scaler) -> tuple[dict[str, object], pd.DataFrame]:
    model_data_dir = SUP_DATA_DIR / "catboost"
    model_plot_dir = SUP_PLOT_DIR / "catboost"
    model_data_dir.mkdir(parents=True, exist_ok=True)
    model_plot_dir.mkdir(parents=True, exist_ok=True)

    forecast = pd.read_csv(model_data_dir / "catboost_forecasts.csv")
    feature_df = pd.read_csv(model_data_dir / "catboost_feature_importance.csv")
    raw_metrics = pd.read_csv(model_data_dir / "catboost_metrics.csv").iloc[0]
    forecast_std = pd.DataFrame({"timestamp": pd.to_datetime(forecast["timestamp"], errors="coerce")})
    forecast_std["actual_scaled"] = pd.to_numeric(
        forecast["actual_scaled"] if "actual_scaled" in forecast.columns else forecast["pm25"],
        errors="coerce",
    )
    forecast_std["pred_scaled"] = pd.to_numeric(forecast["pred"], errors="coerce")
    forecast_std["residual_scaled"] = pd.to_numeric(
        forecast["residual"] if "residual" in forecast.columns else forecast_std["actual_scaled"] - forecast_std["pred_scaled"],
        errors="coerce",
    )
    forecast_std["actual_pm25"] = pd.to_numeric(
        forecast["actual_pm25"] if "actual_pm25" in forecast.columns else inverse_scale_feature(forecast_std["actual_scaled"], scaler, TARGET),
        errors="coerce",
    )
    forecast_std["pred_pm25"] = pd.to_numeric(
        forecast["pred_pm25"] if "pred_pm25" in forecast.columns else inverse_scale_feature(forecast_std["pred_scaled"], scaler, TARGET),
        errors="coerce",
    )
    forecast_std["residual_pm25"] = pd.to_numeric(
        forecast["residual_pm25"] if "residual_pm25" in forecast.columns else forecast_std["actual_pm25"] - forecast_std["pred_pm25"],
        errors="coerce",
    )

    metrics_row = {
        "model": "CatBoost",
        "family": "supervised",
        "evaluation_strategy": "Chronological holdout test",
        "target": TARGET,
        "unit": "real_pm25",
        "n_features": int(raw_metrics["n_features"]),
        "MAE": mae(forecast_std["actual_pm25"], forecast_std["pred_pm25"]),
        "RMSE": rmse(forecast_std["actual_pm25"], forecast_std["pred_pm25"]),
        "MAPE_pct": mape(forecast_std["actual_pm25"], forecast_std["pred_pm25"]),
        "SMAPE_pct": smape(forecast_std["actual_pm25"], forecast_std["pred_pm25"]),
        "R2": float(r2_score(forecast_std["actual_pm25"], forecast_std["pred_pm25"])),
        "source_metrics": str(model_data_dir / "catboost_metrics.csv"),
    }

    feature_std = feature_df.rename(columns={"feature": "feature", "importance": "importance_pct"})
    copied_data = copy_artifacts(
        [
            model_data_dir / "catboost_forecasts.csv",
            model_data_dir / "catboost_metrics.csv",
            model_data_dir / "catboost_feature_importance.csv",
            model_data_dir / "catboost_residuals.csv",
            model_data_dir / "catboost_split_summary.csv",
            model_data_dir / "catboost_run_info.json",
        ],
        model_data_dir,
    )
    copied_plots = copy_artifacts(
        [
            model_plot_dir / "catboost_actual_vs_predicted.png",
            model_plot_dir / "catboost_residual_diagnostics.png",
            model_plot_dir / "catboost_forecast_interactive.html",
            model_plot_dir / "catboost_forecast_interactive.png",
        ],
        model_plot_dir,
    )

    manifest = {
        "name": "CatBoost",
        "source_data": [
            str(model_data_dir / "catboost_forecasts.csv"),
            str(model_data_dir / "catboost_metrics.csv"),
            str(model_data_dir / "catboost_feature_importance.csv"),
        ],
        "phase2_data_dir": str(model_data_dir),
        "phase2_plot_dir": str(model_plot_dir),
        "copied_data_files": copied_data,
        "copied_plot_files": copied_plots,
    }
    with open(model_data_dir / "artifact_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return metrics_row, feature_std[["feature", "importance_pct"]]


def process_sarimax() -> tuple[dict[str, object], pd.DataFrame]:
    model_data_dir = SUP_DATA_DIR / "sarimax"
    model_plot_dir = SUP_PLOT_DIR / "sarimax"
    model_data_dir.mkdir(parents=True, exist_ok=True)
    model_plot_dir.mkdir(parents=True, exist_ok=True)

    raw_metrics = pd.read_csv(model_data_dir / "sarimax_metrics.csv").iloc[0]
    coeff_df = pd.read_csv(model_data_dir / "sarimax_coefficients.csv")

    metrics_row = {
        "model": "SARIMAX",
        "family": "supervised",
        "evaluation_strategy": "Chronological holdout test",
        "target": TARGET,
        "unit": "real_pm25",
        "n_features": int(raw_metrics["n_features"]),
        "MAE": float(raw_metrics["test_MAE"]),
        "RMSE": float(raw_metrics["test_RMSE"]),
        "MAPE_pct": float(raw_metrics["test_MAPE_pct"]),
        "SMAPE_pct": float(raw_metrics["test_SMAPE_pct"]),
        "R2": float(raw_metrics["test_R2"]),
        "source_metrics": str(model_data_dir / "sarimax_metrics.csv"),
    }

    coeff_std = coeff_df.copy()
    coeff_std["feature"] = coeff_std["parameter"]
    coeff_std["effect_strength"] = coeff_std["abs_coefficient"]
    copied_data = copy_artifacts(
        [
            model_data_dir / "sarimax_candidate_results.csv",
            model_data_dir / "sarimax_coefficients.csv",
            model_data_dir / "sarimax_forecasts.csv",
            model_data_dir / "sarimax_metrics.csv",
            model_data_dir / "sarimax_residuals.csv",
            model_data_dir / "sarimax_run_info.json",
            model_data_dir / "sarimax_split_summary.csv",
        ],
        model_data_dir,
    )
    copied_plots = copy_artifacts(
        [
            model_plot_dir / "sarimax_actual_vs_predicted.png",
            model_plot_dir / "sarimax_residual_diagnostics.png",
            model_plot_dir / "sarimax_forecast_interactive.html",
        ],
        model_plot_dir,
    )

    manifest = {
        "name": "SARIMAX",
        "source_data": [
            str(model_data_dir / "sarimax_forecasts.csv"),
            str(model_data_dir / "sarimax_metrics.csv"),
            str(model_data_dir / "sarimax_coefficients.csv"),
        ],
        "phase2_data_dir": str(model_data_dir),
        "phase2_plot_dir": str(model_plot_dir),
        "copied_data_files": copied_data,
        "copied_plot_files": copied_plots,
    }
    with open(model_data_dir / "artifact_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return metrics_row, coeff_std[["feature", "effect_strength"]]


def process_hdbscan(scaler) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    model_data_dir = UNSUP_DATA_DIR / "hdbscan"
    model_plot_dir = UNSUP_PLOT_DIR / "hdbscan"
    model_data_dir.mkdir(parents=True, exist_ok=True)
    model_plot_dir.mkdir(parents=True, exist_ok=True)

    clustered = pd.read_csv(model_data_dir / "hdbscan_clustered_dataset.csv")
    raw_metrics = pd.read_csv(model_data_dir / "hdbscan_metrics.csv").iloc[0]
    feature_summary_raw = pd.read_csv(model_data_dir / "hdbscan_feature_summary.csv")
    cluster_summary = pd.read_csv(model_data_dir / "hdbscan_cluster_summary.csv")

    if "pm25_real" not in clustered.columns and TARGET in clustered.columns:
        clustered["pm25_real"] = inverse_scale_feature(clustered[TARGET], scaler, TARGET)

    if "pm25_real_mean" not in cluster_summary.columns and "pm25_real" in clustered.columns:
        cluster_summary = (
            clustered.groupby("cluster_label")
            .agg(
                count=("cluster_label", "size"),
                cluster_ratio=("cluster_label", lambda values: len(values) / len(clustered)),
                cluster_probability_mean=("cluster_probability", "mean"),
                outlier_score_mean=("outlier_score", "mean"),
                pm25_real_mean=("pm25_real", "mean"),
                pm25_real_median=("pm25_real", "median"),
            )
            .reset_index()
            .sort_values("cluster_label")
        )

    feature_summary = aggregate_feature_summary(feature_summary_raw)

    avg_membership = float(clustered.loc[clustered["cluster_label"] != -1, "cluster_probability"].mean())
    metrics_row = {
        "model": "HDBSCAN",
        "family": "unsupervised",
        "task_type": "clustering",
        "rows_used": int(raw_metrics["rows_used"]),
        "n_features": int(raw_metrics["n_features"]),
        "primary_groups": int(raw_metrics["n_clusters_excluding_noise"]),
        "special_point_ratio": float(raw_metrics["noise_ratio"]),
        "special_point_label": "noise_ratio",
        "avg_confidence_or_severity": avg_membership,
        "silhouette_score": float(raw_metrics["silhouette_score"]),
        "davies_bouldin_score": float(raw_metrics["davies_bouldin_score"]),
        "calinski_harabasz_score": float(raw_metrics["calinski_harabasz_score"]),
        "source_metrics": str(model_data_dir / "hdbscan_metrics.csv"),
    }
    copied_data = copy_artifacts(
        [
            model_data_dir / "hdbscan_clustered_dataset.csv",
            model_data_dir / "hdbscan_cluster_summary.csv",
            model_data_dir / "hdbscan_feature_summary.csv",
            model_data_dir / "hdbscan_metrics.csv",
            model_data_dir / "hdbscan_model_selection.csv",
            model_data_dir / "hdbscan_run_info.json",
        ],
        model_data_dir,
    )
    copied_plots = copy_artifacts(
        [
            model_plot_dir / "hdbscan_umap_interactive.html",
            model_plot_dir / "hdbscan_umap_interactive.png",
            model_plot_dir / "hdbscan_cluster_sizes.png",
            model_plot_dir / "hdbscan_pm25_by_cluster.png",
        ],
        model_plot_dir,
    )

    manifest = {
        "name": "HDBSCAN",
        "source_data": [
            str(model_data_dir / "hdbscan_clustered_dataset.csv"),
            str(model_data_dir / "hdbscan_metrics.csv"),
            str(model_data_dir / "hdbscan_feature_summary.csv"),
        ],
        "phase2_data_dir": str(model_data_dir),
        "phase2_plot_dir": str(model_plot_dir),
        "copied_data_files": copied_data,
        "copied_plot_files": copied_plots,
    }
    with open(model_data_dir / "artifact_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return metrics_row, feature_summary, cluster_summary


def process_gmm() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    model_data_dir = UNSUP_DATA_DIR / "gaussian_mixture"
    model_plot_dir = UNSUP_PLOT_DIR / "gaussian_mixture"
    model_data_dir.mkdir(parents=True, exist_ok=True)
    model_plot_dir.mkdir(parents=True, exist_ok=True)

    raw_metrics = pd.read_csv(model_data_dir / "gmm_metrics.csv").iloc[0]
    feature_summary_raw = pd.read_csv(model_data_dir / "gmm_feature_summary.csv")
    cluster_summary = pd.read_csv(model_data_dir / "gmm_cluster_summary.csv")

    feature_summary = aggregate_feature_summary(feature_summary_raw)

    metrics_row = {
        "model": "Gaussian Mixture",
        "family": "unsupervised",
        "task_type": "clustering",
        "rows_used": int(raw_metrics["rows_used"]),
        "n_features": int(raw_metrics["n_features"]),
        "primary_groups": int(raw_metrics["n_clusters"]),
        "special_point_ratio": 0.0,
        "special_point_label": "noise_ratio",
        "avg_confidence_or_severity": float(raw_metrics["avg_cluster_confidence"]),
        "silhouette_score": float(raw_metrics["silhouette_score"]),
        "davies_bouldin_score": float(raw_metrics["davies_bouldin_score"]),
        "calinski_harabasz_score": float(raw_metrics["calinski_harabasz_score"]),
        "source_metrics": str(model_data_dir / "gmm_metrics.csv"),
    }
    copied_data = copy_artifacts(
        [
            model_data_dir / "gmm_cluster_summary.csv",
            model_data_dir / "gmm_clustered_dataset.csv",
            model_data_dir / "gmm_feature_summary.csv",
            model_data_dir / "gmm_metrics.csv",
            model_data_dir / "gmm_model_selection.csv",
            model_data_dir / "gmm_run_info.json",
        ],
        model_data_dir,
    )
    copied_plots = copy_artifacts(
        [
            model_plot_dir / "gmm_model_selection.png",
            model_plot_dir / "gmm_cluster_profile_heatmap.png",
            model_plot_dir / "gmm_pca_interactive.html",
        ],
        model_plot_dir,
    )

    manifest = {
        "name": "Gaussian Mixture",
        "source_data": [
            str(model_data_dir / "gmm_cluster_summary.csv"),
            str(model_data_dir / "gmm_metrics.csv"),
            str(model_data_dir / "gmm_feature_summary.csv"),
        ],
        "phase2_data_dir": str(model_data_dir),
        "phase2_plot_dir": str(model_plot_dir),
        "copied_data_files": copied_data,
        "copied_plot_files": copied_plots,
    }
    with open(model_data_dir / "artifact_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return metrics_row, feature_summary, cluster_summary


def process_isolation_forest() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    model_data_dir = UNSUP_DATA_DIR / "isolation_forest"
    model_plot_dir = UNSUP_PLOT_DIR / "isolation_forest"
    model_data_dir.mkdir(parents=True, exist_ok=True)
    model_plot_dir.mkdir(parents=True, exist_ok=True)

    raw_metrics = pd.read_csv(model_data_dir / "isolation_forest_metrics.csv").iloc[0]
    feature_summary = pd.read_csv(model_data_dir / "isolation_forest_feature_summary.csv")
    scored = pd.read_csv(model_data_dir / "isolation_forest_scored_dataset.csv")

    metrics_row = {
        "model": "Isolation Forest",
        "family": "unsupervised",
        "task_type": "anomaly_detection",
        "rows_used": int(raw_metrics["rows_used"]),
        "n_features": int(raw_metrics["n_features"]),
        "primary_groups": 2,
        "special_point_ratio": float(raw_metrics["anomaly_ratio"]),
        "special_point_label": "anomaly_ratio",
        "avg_confidence_or_severity": float(raw_metrics["severity_mean_anomaly"]),
        "silhouette_score": np.nan,
        "davies_bouldin_score": np.nan,
        "calinski_harabasz_score": np.nan,
        "source_metrics": str(model_data_dir / "isolation_forest_metrics.csv"),
    }

    if "pm25_real" in scored.columns:
        pm25_profile = (
            scored.assign(group=np.where(scored["anomaly"] == -1, "Anomaly", "Normal"))
            .groupby("group", as_index=False)["pm25_real"]
            .mean()
            .rename(columns={"pm25_real": "pm25_real_mean"})
        )
    else:
        pm25_profile = pd.DataFrame({"group": ["Normal", "Anomaly"], "pm25_real_mean": [np.nan, np.nan]})
    copied_data = copy_artifacts(
        [
            model_data_dir / "isolation_forest_scored_dataset.csv",
            model_data_dir / "isolation_forest_metrics.csv",
            model_data_dir / "isolation_forest_feature_summary.csv",
            model_data_dir / "isolation_forest_top_anomalies.csv",
            model_data_dir / "isolation_forest_run_info.json",
        ],
        model_data_dir,
    )
    copied_plots = copy_artifacts(
        [
            model_plot_dir / "isolation_forest_pm25.png",
            model_plot_dir / "isolation_forest_energy.png",
            model_plot_dir / "isolation_forest_pm25_zoom.png",
            model_plot_dir / "isolation_forest_scatter.png",
            model_plot_dir / "isolation_forest_feature_shift.png",
            model_plot_dir / "isolation_forest_score_distribution.png",
        ],
        model_plot_dir,
    )

    manifest = {
        "name": "Isolation Forest",
        "source_data": [
            str(model_data_dir / "isolation_forest_scored_dataset.csv"),
            str(model_data_dir / "isolation_forest_metrics.csv"),
            str(model_data_dir / "isolation_forest_feature_summary.csv"),
        ],
        "phase2_data_dir": str(model_data_dir),
        "phase2_plot_dir": str(model_plot_dir),
        "copied_data_files": copied_data,
        "copied_plot_files": copied_plots,
    }
    with open(model_data_dir / "artifact_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return metrics_row, feature_summary, pm25_profile


def build_supervised_outputs(scaler) -> dict[str, object]:
    lightgbm_row, lightgbm_features = process_lightgbm_improved()
    catboost_row, catboost_features = process_catboost(scaler)
    sarimax_row, sarimax_features = process_sarimax()

    comparison_df = pd.DataFrame([lightgbm_row, catboost_row, sarimax_row])
    comparison_df.to_csv(COMP_DATA_DIR / "supervised_model_comparison.csv", index=False)

    save_metric_bar_chart(
        comparison_df,
        ["MAE", "RMSE"],
        COMP_PLOT_DIR / "supervised_error_metrics.png",
        "Supervised model comparison: MAE and RMSE",
    )
    save_metric_bar_chart(
        comparison_df,
        ["R2"],
        COMP_PLOT_DIR / "supervised_r2_comparison.png",
        "Supervised model comparison: R2",
    )
    save_feature_panel(
        [
            ("LightGBM Improved", lightgbm_features.rename(columns={"importance_pct": "importance_pct"}), "importance_pct"),
            ("CatBoost", catboost_features.rename(columns={"importance_pct": "importance_pct"}), "importance_pct"),
            ("SARIMAX", sarimax_features.rename(columns={"effect_strength": "effect_strength"}), "effect_strength"),
        ],
        COMP_PLOT_DIR / "supervised_feature_panels.png",
        "Supervised model feature summaries",
    )
    save_table_image(
        comparison_df[["model", "evaluation_strategy", "MAE", "RMSE", "R2", "MAPE_pct", "SMAPE_pct"]],
        COMP_PLOT_DIR / "supervised_comparison_table.png",
        "Supervised model comparison table",
    )

    return {
        "comparison_csv": str(COMP_DATA_DIR / "supervised_model_comparison.csv"),
        "plots": [
            str(COMP_PLOT_DIR / "supervised_error_metrics.png"),
            str(COMP_PLOT_DIR / "supervised_r2_comparison.png"),
            str(COMP_PLOT_DIR / "supervised_feature_panels.png"),
            str(COMP_PLOT_DIR / "supervised_comparison_table.png"),
        ],
    }


def build_unsupervised_outputs(scaler) -> dict[str, object]:
    hdbscan_row, hdbscan_features, hdbscan_cluster_summary = process_hdbscan(scaler)
    gmm_row, gmm_features, gmm_cluster_summary = process_gmm()
    if_row, if_features, if_pm25_profile = process_isolation_forest()

    comparison_df = pd.DataFrame([hdbscan_row, gmm_row, if_row])
    comparison_df.to_csv(COMP_DATA_DIR / "unsupervised_model_comparison.csv", index=False)

    save_metric_bar_chart(
        comparison_df,
        ["special_point_ratio", "primary_groups"],
        COMP_PLOT_DIR / "unsupervised_special_ratio_and_groups.png",
        "Unsupervised comparison: special-point ratio and primary groups",
    )

    clustering_only = comparison_df[comparison_df["task_type"] == "clustering"].copy()
    save_metric_bar_chart(
        clustering_only,
        ["silhouette_score", "davies_bouldin_score", "calinski_harabasz_score"],
        COMP_PLOT_DIR / "unsupervised_clustering_quality.png",
        "Clustering quality comparison",
    )
    save_feature_panel(
        [
            ("HDBSCAN", hdbscan_features, "absolute_difference"),
            ("Gaussian Mixture", gmm_features, "absolute_difference"),
            ("Isolation Forest", if_features, "absolute_difference"),
        ],
        COMP_PLOT_DIR / "unsupervised_feature_panels.png",
        "Unsupervised feature profile summaries",
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.barplot(data=hdbscan_cluster_summary[hdbscan_cluster_summary["cluster_label"] != -1], x="cluster_label", y="pm25_real_mean", ax=axes[0], color="#4C78A8")
    axes[0].set_title("HDBSCAN PM2.5 by cluster")
    axes[0].set_xlabel("cluster_label")
    axes[0].set_ylabel("pm25_real_mean")

    sns.barplot(data=gmm_cluster_summary, x="cluster_label", y="pm25_real_mean", ax=axes[1], color="#72B7B2")
    axes[1].set_title("GMM PM2.5 by cluster")
    axes[1].set_xlabel("cluster_label")
    axes[1].set_ylabel("pm25_real_mean")

    sns.barplot(data=if_pm25_profile, x="group", y="pm25_real_mean", ax=axes[2], color="#E45756")
    axes[2].set_title("Isolation Forest PM2.5 profile")
    axes[2].set_xlabel("group")
    axes[2].set_ylabel("pm25_real_mean")

    plt.tight_layout()
    plt.savefig(COMP_PLOT_DIR / "unsupervised_pm25_profiles.png", dpi=300, bbox_inches="tight")
    plt.close()

    save_table_image(
        comparison_df[
            [
                "model",
                "task_type",
                "special_point_ratio",
                "primary_groups",
                "avg_confidence_or_severity",
                "silhouette_score",
                "davies_bouldin_score",
            ]
        ],
        COMP_PLOT_DIR / "unsupervised_comparison_table.png",
        "Unsupervised model comparison table",
    )

    return {
        "comparison_csv": str(COMP_DATA_DIR / "unsupervised_model_comparison.csv"),
        "plots": [
            str(COMP_PLOT_DIR / "unsupervised_special_ratio_and_groups.png"),
            str(COMP_PLOT_DIR / "unsupervised_clustering_quality.png"),
            str(COMP_PLOT_DIR / "unsupervised_feature_panels.png"),
            str(COMP_PLOT_DIR / "unsupervised_pm25_profiles.png"),
            str(COMP_PLOT_DIR / "unsupervised_comparison_table.png"),
        ],
    }


def main() -> None:
    scaler = load_scaler()
    reset_directory(COMP_DATA_DIR)
    reset_directory(COMP_PLOT_DIR)

    supervised_manifest = build_supervised_outputs(scaler)
    unsupervised_manifest = build_unsupervised_outputs(scaler)

    manifest = {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(PHASE2_DATA_DIR),
        "plots_dir": str(PHASE2_PLOTS_DIR),
        "supervised": supervised_manifest,
        "unsupervised": unsupervised_manifest,
    }

    with open(PHASE2_DATA_DIR / "phase2_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("Saved phase-2 outputs:")
    print(f"- {COMP_DATA_DIR / 'supervised_model_comparison.csv'}")
    print(f"- {COMP_DATA_DIR / 'unsupervised_model_comparison.csv'}")
    print(f"- {PHASE2_DATA_DIR / 'phase2_manifest.json'}")
    print(f"- {COMP_PLOT_DIR}")


if __name__ == "__main__":
    main()
