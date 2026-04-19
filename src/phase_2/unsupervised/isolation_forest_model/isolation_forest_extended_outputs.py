from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "phase_2" / "unsupervised" / "isolation_forest"
MODEL_DIR = PROJECT_ROOT / "models" / "isolation_forest_model"
PLOTS_DIR = PROJECT_ROOT / "pictures" / "phase_2" / "unsupervised" / "isolation_forest"
LEGACY_RESULTS_DIR = CURRENT_DIR / "isolation_forest_results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LEGACY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [
    PROJECT_ROOT / "data" / "4E_selected_dataset.csv",
    PROJECT_ROOT / "data" / "phase_1" / "4E_selected_dataset.csv",
]
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

TIME_CANDIDATES = ["datetime", "date"]
TARGET = "pm25"
ENERGY_FEATURE = "total_generation_mw"

FEATURE_PRIORITY = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "pollution_stagnation_index",
    "wind_x_vector",
    "wind_y_vector",
    "total_generation_mw",
    "temperature_2m (°C)",
    "rain (mm)",
    "relative_humidity_2m (%)",
    "wind_speed_10m (km/h)",
    "pm25",
]

N_ESTIMATORS = 100
CONTAMINATION = 0.05
RANDOM_STATE = 42
N_JOBS = 1

OUTPUT_SCORED = DATA_DIR / "isolation_forest_scored_dataset.csv"
OUTPUT_METRICS = DATA_DIR / "isolation_forest_metrics.csv"
OUTPUT_FEATURE_SUMMARY = DATA_DIR / "isolation_forest_feature_summary.csv"
OUTPUT_TOP_ANOMALIES = DATA_DIR / "isolation_forest_top_anomalies.csv"
OUTPUT_RUN_INFO = DATA_DIR / "isolation_forest_run_info.json"

MODEL_PATH = MODEL_DIR / "isolation_forest_model.pkl"
FEATURES_PATH = MODEL_DIR / "isolation_forest_feature_columns.pkl"


plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")


def resolve_existing_path(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{label} not found. Checked:\n{searched}")


def detect_time_column(df: pd.DataFrame) -> str:
    for column in TIME_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(f"No time column found. Tried: {TIME_CANDIDATES}")


def load_optional_scaler():
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


def prepare_dataframe() -> tuple[pd.DataFrame, list[str], Path]:
    input_path = resolve_existing_path(INPUT_CANDIDATES, "Phase 1 selected dataset")
    df = pd.read_csv(input_path)

    time_col = detect_time_column(df)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    df["timestamp"] = df[time_col]

    if df.duplicated(subset=["timestamp"]).sum() > 0:
        df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [column for column in FEATURE_PRIORITY if column in df.columns]

    if not feature_cols:
        feature_cols = [
            column
            for column in numeric_cols
            if not column.endswith("_was_missing") and column not in {"unnamed: 0"}
        ]

    for column in feature_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    return df, feature_cols, input_path


def build_feature_summary(df_scored: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    anomalies = df_scored[df_scored["anomaly"] == -1]
    normal = df_scored[df_scored["anomaly"] == 1]

    if anomalies.empty or normal.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "normal_mean",
                "anomaly_mean",
                "absolute_difference",
            ]
        )

    rows = []
    for feature in feature_cols:
        normal_mean = float(normal[feature].mean())
        anomaly_mean = float(anomalies[feature].mean())
        rows.append(
            {
                "feature": feature,
                "normal_mean": normal_mean,
                "anomaly_mean": anomaly_mean,
                "absolute_difference": abs(anomaly_mean - normal_mean),
            }
        )

    return pd.DataFrame(rows).sort_values("absolute_difference", ascending=False).reset_index(drop=True)


def save_pm25_plot(df_scored: pd.DataFrame, output_path: Path) -> None:
    if "pm25_real" not in df_scored.columns:
        return

    anomalies = df_scored[df_scored["anomaly"] == -1]
    plt.figure(figsize=(15, 6))
    plt.plot(
        df_scored["timestamp"],
        df_scored["pm25_real"],
        color="royalblue",
        alpha=0.7,
        linewidth=1.4,
        label="PM2.5",
    )
    plt.scatter(
        anomalies["timestamp"],
        anomalies["pm25_real"],
        color="crimson",
        s=26,
        zorder=5,
        label="Anomalies",
    )
    plt.title("Isolation Forest: PM2.5 anomalies (real units)")
    plt.xlabel("Time")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_energy_plot(df_scored: pd.DataFrame, output_path: Path) -> None:
    column = "total_generation_mw_real" if "total_generation_mw_real" in df_scored.columns else ENERGY_FEATURE
    if column not in df_scored.columns:
        return

    anomalies = df_scored[df_scored["anomaly"] == -1]
    plt.figure(figsize=(15, 6))
    plt.plot(
        df_scored["timestamp"],
        df_scored[column],
        color="forestgreen",
        alpha=0.7,
        linewidth=1.4,
        label="Generation",
    )
    plt.scatter(
        anomalies["timestamp"],
        anomalies[column],
        color="darkred",
        s=26,
        zorder=5,
        label="Anomalies",
    )
    plt.title("Isolation Forest: energy anomalies")
    plt.xlabel("Time")
    plt.ylabel("Total generation (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_zoom_plot(df_scored: pd.DataFrame, output_path: Path) -> None:
    if "pm25_real" not in df_scored.columns:
        return

    df_zoom = df_scored.tail(min(1000, len(df_scored))).copy()
    anomalies_zoom = df_zoom[df_zoom["anomaly"] == -1]

    plt.figure(figsize=(15, 6))
    plt.plot(
        df_zoom["timestamp"],
        df_zoom["pm25_real"],
        color="navy",
        alpha=0.75,
        linewidth=1.8,
        label="PM2.5",
    )
    plt.scatter(
        anomalies_zoom["timestamp"],
        anomalies_zoom["pm25_real"],
        color="red",
        s=55,
        zorder=5,
        label="Anomalies",
    )
    plt.title("Isolation Forest: recent PM2.5 anomaly zoom")
    plt.xlabel("Time")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_scatter_plot(df_scored: pd.DataFrame, output_path: Path) -> None:
    x_col = "total_generation_mw_real" if "total_generation_mw_real" in df_scored.columns else ENERGY_FEATURE
    if x_col not in df_scored.columns or "pm25_real" not in df_scored.columns:
        return

    anomalies = df_scored[df_scored["anomaly"] == -1]
    normal = df_scored[df_scored["anomaly"] == 1]

    plt.figure(figsize=(10, 8))
    plt.scatter(
        normal[x_col],
        normal["pm25_real"],
        color="lightgray",
        alpha=0.35,
        s=14,
        label="Normal hours",
    )
    plt.scatter(
        anomalies[x_col],
        anomalies["pm25_real"],
        color="darkred",
        alpha=0.85,
        s=36,
        edgecolor="black",
        linewidth=0.4,
        label="Anomaly hours",
    )
    plt.title("Isolation Forest: PM2.5 vs total generation")
    plt.xlabel("Total generation (MW)")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_feature_shift_plot(feature_summary: pd.DataFrame, output_path: Path) -> None:
    if feature_summary.empty:
        return

    top_df = feature_summary.head(12).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top_df["feature"], top_df["absolute_difference"], color="teal")
    plt.title("Isolation Forest: top anomaly feature shifts")
    plt.xlabel("Absolute mean difference (scaled space)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_score_distribution(df_scored: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df_scored,
        x="anomaly_score",
        hue="anomaly_label",
        bins=60,
        stat="density",
        common_norm=False,
        element="step",
    )
    plt.title("Isolation Forest: anomaly score distribution")
    plt.xlabel("Decision function score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    print("=" * 90)
    print("ISOLATION FOREST :: EXTENDED OUTPUT PIPELINE")
    print("=" * 90)

    df, feature_cols, input_path = prepare_dataframe()
    scaler = load_optional_scaler()

    print(f"Input dataset : {input_path}")
    print(f"Rows used     : {len(df)}")
    print(f"Features      : {len(feature_cols)}")
    print(feature_cols)

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    X = df[feature_cols].to_numpy(dtype=float)
    df_scored = df.copy()
    df_scored["anomaly"] = model.fit_predict(X)
    df_scored["anomaly_score"] = model.decision_function(X)
    df_scored["anomaly_severity"] = -df_scored["anomaly_score"]
    df_scored["anomaly_label"] = np.where(df_scored["anomaly"] == -1, "Anomaly", "Normal")

    if TARGET in df_scored.columns:
        df_scored["pm25_real"] = inverse_scale_feature(df_scored[TARGET], scaler, TARGET)
    if ENERGY_FEATURE in df_scored.columns:
        df_scored["total_generation_mw_real"] = inverse_scale_feature(
            df_scored[ENERGY_FEATURE], scaler, ENERGY_FEATURE
        )

    anomalies = df_scored[df_scored["anomaly"] == -1].copy()
    anomalies = anomalies.sort_values("anomaly_score").reset_index(drop=True)
    anomalies["anomaly_rank"] = np.arange(1, len(anomalies) + 1)

    feature_summary = build_feature_summary(df_scored, feature_cols)

    metrics = {
        "rows_used": int(len(df_scored)),
        "n_features": int(len(feature_cols)),
        "n_anomalies": int(len(anomalies)),
        "anomaly_ratio": float(len(anomalies) / len(df_scored)),
        "contamination": CONTAMINATION,
        "n_estimators": N_ESTIMATORS,
        "score_mean_all": float(df_scored["anomaly_score"].mean()),
        "score_mean_normal": float(df_scored.loc[df_scored["anomaly"] == 1, "anomaly_score"].mean()),
        "score_mean_anomaly": float(anomalies["anomaly_score"].mean()),
        "severity_mean_anomaly": float(anomalies["anomaly_severity"].mean()),
    }

    if "pm25_real" in df_scored.columns:
        metrics["pm25_real_mean_normal"] = float(df_scored.loc[df_scored["anomaly"] == 1, "pm25_real"].mean())
        metrics["pm25_real_mean_anomaly"] = float(anomalies["pm25_real"].mean())
        metrics["pm25_real_p95_anomaly"] = float(anomalies["pm25_real"].quantile(0.95))

    if "total_generation_mw_real" in df_scored.columns:
        metrics["total_generation_real_mean_normal"] = float(
            df_scored.loc[df_scored["anomaly"] == 1, "total_generation_mw_real"].mean()
        )
        metrics["total_generation_real_mean_anomaly"] = float(anomalies["total_generation_mw_real"].mean())

    metrics_df = pd.DataFrame([metrics])

    cols_for_top = ["timestamp"]
    if "pm25_real" in anomalies.columns:
        cols_for_top.append("pm25_real")
    if TARGET in anomalies.columns:
        cols_for_top.append(TARGET)
    cols_for_top.extend(["anomaly_score", "anomaly_severity", "anomaly_rank"])
    if "total_generation_mw_real" in anomalies.columns:
        cols_for_top.append("total_generation_mw_real")
    elif ENERGY_FEATURE in anomalies.columns:
        cols_for_top.append(ENERGY_FEATURE)

    top_anomalies = anomalies[cols_for_top].head(100).copy()

    metrics_df.to_csv(OUTPUT_METRICS, index=False)
    feature_summary.to_csv(OUTPUT_FEATURE_SUMMARY, index=False)
    df_scored.to_csv(OUTPUT_SCORED, index=False)
    top_anomalies.to_csv(OUTPUT_TOP_ANOMALIES, index=False)
    top_anomalies.to_csv(LEGACY_RESULTS_DIR / "top_anomalies_list.csv", index=False)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)

    pm25_plot = PLOTS_DIR / "isolation_forest_pm25.png"
    energy_plot = PLOTS_DIR / "isolation_forest_energy.png"
    zoom_plot = PLOTS_DIR / "isolation_forest_pm25_zoom.png"
    scatter_plot = PLOTS_DIR / "isolation_forest_scatter.png"
    feature_shift_plot = PLOTS_DIR / "isolation_forest_feature_shift.png"
    score_distribution_plot = PLOTS_DIR / "isolation_forest_score_distribution.png"

    save_pm25_plot(df_scored, pm25_plot)
    save_energy_plot(df_scored, energy_plot)
    save_zoom_plot(df_scored, zoom_plot)
    save_scatter_plot(df_scored, scatter_plot)
    save_feature_shift_plot(feature_summary, feature_shift_plot)
    save_score_distribution(df_scored, score_distribution_plot)

    run_info = {
        "input_path": str(input_path),
        "model_path": str(MODEL_PATH),
        "feature_columns_path": str(FEATURES_PATH),
        "outputs": {
            "scored_dataset": str(OUTPUT_SCORED),
            "metrics": str(OUTPUT_METRICS),
            "feature_summary": str(OUTPUT_FEATURE_SUMMARY),
            "top_anomalies": str(OUTPUT_TOP_ANOMALIES),
            "legacy_top_anomalies": str(LEGACY_RESULTS_DIR / "top_anomalies_list.csv"),
            "pm25_plot": str(pm25_plot),
            "energy_plot": str(energy_plot),
            "zoom_plot": str(zoom_plot),
            "scatter_plot": str(scatter_plot),
            "feature_shift_plot": str(feature_shift_plot),
            "score_distribution_plot": str(score_distribution_plot),
        },
        "config": {
            "n_estimators": N_ESTIMATORS,
            "contamination": CONTAMINATION,
            "random_state": RANDOM_STATE,
            "n_jobs": N_JOBS,
        },
        "feature_columns": feature_cols,
    }

    with open(OUTPUT_RUN_INFO, "w", encoding="utf-8") as file:
        json.dump(run_info, file, indent=2, default=str)

    print("\nSaved outputs:")
    print(f"- {OUTPUT_SCORED}")
    print(f"- {OUTPUT_METRICS}")
    print(f"- {OUTPUT_FEATURE_SUMMARY}")
    print(f"- {OUTPUT_TOP_ANOMALIES}")
    print(f"- {OUTPUT_RUN_INFO}")
    print(f"- {pm25_plot}")
    print(f"- {feature_shift_plot}")


if __name__ == "__main__":
    main()
