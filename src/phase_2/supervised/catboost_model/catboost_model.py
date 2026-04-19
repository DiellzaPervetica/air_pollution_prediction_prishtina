from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json
import random

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

INPUT_CANDIDATES = [
    BASE_DIR / "data" / "4E_selected_dataset.csv",
    BASE_DIR / "data" / "phase_1" / "4E_selected_dataset.csv",
]
SCALER_CANDIDATES = [
    BASE_DIR / "models" / "scaler.pkl",
]

PHASE2_DATA_DIR = BASE_DIR / "data" / "phase_2" / "supervised" / "catboost"
PHASE2_PLOTS_DIR = BASE_DIR / "pictures" / "phase_2" / "supervised" / "catboost"

MODEL_DIR = BASE_DIR / "models" / "catboost_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = PHASE2_PLOTS_DIR
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PHASE2_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FORECASTS = PHASE2_DATA_DIR / "catboost_forecasts.csv"
OUTPUT_METRICS = PHASE2_DATA_DIR / "catboost_metrics.csv"
OUTPUT_FEATURES = PHASE2_DATA_DIR / "catboost_feature_importance.csv"
OUTPUT_SPLIT_SUMMARY = PHASE2_DATA_DIR / "catboost_split_summary.csv"
OUTPUT_RUN_INFO = PHASE2_DATA_DIR / "catboost_run_info.json"
OUTPUT_RESIDUALS = PHASE2_DATA_DIR / "catboost_residuals.csv"

MODEL_PATH = MODEL_DIR / "catboost_pm25_model.cbm"

TARGET = "pm25"
TIME_CANDIDATES = ["datetime", "date"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

ENABLE_PM25_LAGS = True
PM25_LAGS = [1, 24]

# CatBoost params
ITERATIONS = 600
LEARNING_RATE = 0.03
DEPTH = 6
LOSS_FUNCTION = "RMSE"
EVAL_METRIC = "RMSE"
EARLY_STOPPING_ROUNDS = 50


# =============================================================================
# HELPERS
# =============================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def detect_time_column(df: pd.DataFrame) -> str:
    for col in TIME_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"No time column found. Tried: {TIME_CANDIDATES}")


def resolve_existing_path(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{label} not found. Checked:\n{searched}")


def load_optional_scaler():
    for candidate in SCALER_CANDIDATES:
        if candidate.exists():
            return joblib.load(candidate)
    return None


def inverse_scale_feature(values, scaler, feature_name: str = TARGET) -> np.ndarray:
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


def add_optional_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    if not ENABLE_PM25_LAGS or TARGET not in df.columns:
        return df

    df = df.copy()
    for lag in PM25_LAGS:
        lag_col = f"{TARGET}_lag_{lag}"
        if lag_col not in df.columns:
            df[lag_col] = df[TARGET].shift(lag)
    return df


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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


def all_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        raise ValueError("No finite values available for metric computation.")

    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_pct": mape(y_true, y_pred),
        "SMAPE_pct": smape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "n_eval_points": int(len(y_true)),
    }


def build_interactive_plot(df_all, test_df, preds, html_path, png_path=None, train_end=None, val_end=None):
    plot_df = test_df.copy()
    plot_df["pred"] = preds
    plot_df["residual"] = plot_df[TARGET] - plot_df["pred"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_all["timestamp"],
        y=df_all[TARGET],
        mode="lines",
        name="Observed",
        line=dict(width=2),
        hovertemplate="Time=%{x}<br>Observed=%{y:.3f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["pred"],
        mode="lines+markers",
        name="Predicted",
        line=dict(width=2, dash="dash"),
        marker=dict(size=5),
        customdata=plot_df[[TARGET, "residual"]].values,
        hovertemplate="Time=%{x}<br>Predicted=%{y:.3f}<br>Actual=%{customdata[0]:.3f}<br>Residual=%{customdata[1]:.3f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df["residual"],
        mode="markers",
        name="Residuals",
        yaxis="y2",
        marker=dict(size=6, symbol="diamond"),
        hovertemplate="Time=%{x}<br>Residual=%{y:.3f}<extra></extra>",
    ))

    if train_end is not None:
        fig.add_vline(x=train_end, line_dash="dot")
        fig.add_vline(x=val_end, line_dash="dot")

    fig.update_layout(
        title="CatBoost Forecast vs Observed PM2.5 (chronological split)",
        xaxis=dict(
            title="Time",
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=[
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=30, label="30d", step="day", stepmode="backward"),
                    dict(count=90, label="90d", step="day", stepmode="backward"),
                    dict(step="all"),
                ]
            ),
        ),
        yaxis=dict(title="PM2.5"),
        yaxis2=dict(title="Residual", overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
        legend=dict(orientation="h"),
        template="plotly_white",
        height=720,
    )

    fig.write_html(str(html_path), include_plotlyjs="cdn")
    if png_path is not None:
        try:
            fig.write_image(str(png_path), scale=2)
        except Exception:
            pass


def save_static_forecast_plot(plot_df: pd.DataFrame, output_path: Path) -> None:
    subset = plot_df.tail(min(500, len(plot_df))).copy()
    plt.figure(figsize=(14, 5))
    plt.plot(subset["timestamp"], subset["actual_pm25"], label="Actual", linewidth=1.7)
    plt.plot(subset["timestamp"], subset["pred_pm25"], label="Predicted", linewidth=1.7, linestyle="--")
    plt.title("CatBoost actual vs predicted PM2.5")
    plt.xlabel("Time")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_residual_plot(plot_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(plot_df["residual_pm25"], bins=40, color="steelblue", alpha=0.85)
    plt.axvline(plot_df["residual_pm25"].mean(), color="darkred", linestyle="--", linewidth=2)
    plt.title("CatBoost residual distribution")
    plt.xlabel("Residual (µg/m³)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(RANDOM_STATE)
    scaler = load_optional_scaler()
    input_path = resolve_existing_path(INPUT_CANDIDATES, "Phase 1 selected dataset")

    print("=" * 90)
    print("CATBOOST :: SUPERVISED FORECASTING PIPELINE")
    print("=" * 90)
    print(f"Input dataset: {input_path}")

    df = pd.read_csv(input_path)

    time_col = detect_time_column(df)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    # remove duplicate timestamps if any
    if df.duplicated(subset=[time_col]).sum() > 0:
        df = df.drop_duplicates(subset=[time_col], keep="first").reset_index(drop=True)

    # keep timestamp separately
    df["timestamp"] = df[time_col]
    df = add_optional_lag_features(df)

    # bool -> int
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    for c in bool_cols:
        df[c] = df[c].astype(int)

    # numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET not in numeric_cols:
        raise ValueError(f"Target '{TARGET}' not found among numeric columns.")

    feature_cols = [c for c in numeric_cols if c != TARGET and not c.endswith("_was_missing")]

    # numeric cleanup
    for c in [TARGET] + feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[[TARGET] + feature_cols] = df[[TARGET] + feature_cols].replace([np.inf, -np.inf], np.nan)

    print("\n" + "=" * 80)
    print("DATA QUALITY CHECK")
    print("=" * 80)
    print(f"Rows before cleaning: {len(df)}")
    print(f"Target: {TARGET}")
    print(f"Feature count: {len(feature_cols)}")
    print("Features:", feature_cols)

    missing_before = df[[TARGET] + feature_cols].isna().sum().sort_values(ascending=False)
    print("\nMissing values before cleaning:")
    print(missing_before[missing_before > 0].to_string() if (missing_before > 0).any() else "No missing values.")

    before_drop = len(df)
    df = df.dropna(subset=[TARGET] + feature_cols).copy()
    after_drop = len(df)

    print(f"\nDropped incomplete rows: {before_drop - after_drop}")
    print(f"Rows remaining: {after_drop}")

    if after_drop < 300:
        raise ValueError("Too few rows remaining after cleaning.")

    # chronological split
    n = len(df)
    train_end_idx = int(n * TRAIN_RATIO)
    val_end_idx = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end_idx].copy()
    val_df = df.iloc[train_end_idx:val_end_idx].copy()
    test_df = df.iloc[val_end_idx:].copy()

    print("\n" + "=" * 80)
    print("CHRONOLOGICAL SPLIT SUMMARY")
    print("=" * 80)
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows  : {len(val_df)}")
    print(f"Test rows : {len(test_df)}")
    print(f"Train range: {train_df['timestamp'].min()} -> {train_df['timestamp'].max()}")
    print(f"Val range  : {val_df['timestamp'].min()} -> {val_df['timestamp'].max()}")
    print(f"Test range : {test_df['timestamp'].min()} -> {test_df['timestamp'].max()}")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]

    X_val = val_df[feature_cols]
    y_val = val_df[TARGET]

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET]

    model = CatBoostRegressor(
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        depth=DEPTH,
        loss_function=LOSS_FUNCTION,
        eval_metric=EVAL_METRIC,
        random_seed=RANDOM_STATE,
        verbose=100
    )

    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS
    )

    model.save_model(str(MODEL_PATH))
    joblib.dump(feature_cols, MODEL_DIR / "catboost_feature_columns.pkl")

    print(f"Saved model to: {MODEL_PATH}")

    print("\n" + "=" * 80)
    print("PREDICTION + METRICS")
    print("=" * 80)

    y_val_pred = model.predict(X_val)
    y_pred = model.predict(X_test)

    validation_metrics_scaled = all_metrics(y_val.values, y_val_pred)
    test_metrics_scaled = all_metrics(y_test.values, y_pred)

    y_val_real = inverse_scale_feature(y_val.values, scaler, TARGET)
    y_val_pred_real = inverse_scale_feature(y_val_pred, scaler, TARGET)
    y_test_real = inverse_scale_feature(y_test.values, scaler, TARGET)
    y_pred_real = inverse_scale_feature(y_pred, scaler, TARGET)

    validation_metrics_real = all_metrics(y_val_real, y_val_pred_real)
    test_metrics_real = all_metrics(y_test_real, y_pred_real)

    metrics_df = pd.DataFrame([{
        "target": TARGET,
        "n_features": len(feature_cols),
        "iterations": ITERATIONS,
        "learning_rate": LEARNING_RATE,
        "depth": DEPTH,
        "lags_enabled": ENABLE_PM25_LAGS,
        "lag_columns": ",".join([f"{TARGET}_lag_{lag}" for lag in PM25_LAGS]) if ENABLE_PM25_LAGS else "",
        "validation_MAE_scaled": validation_metrics_scaled["MAE"],
        "validation_RMSE_scaled": validation_metrics_scaled["RMSE"],
        "validation_R2_scaled": validation_metrics_scaled["R2"],
        "test_MAE_scaled": test_metrics_scaled["MAE"],
        "test_RMSE_scaled": test_metrics_scaled["RMSE"],
        "test_R2_scaled": test_metrics_scaled["R2"],
        "validation_MAE": validation_metrics_real["MAE"],
        "validation_RMSE": validation_metrics_real["RMSE"],
        "validation_MAPE_pct": validation_metrics_real["MAPE_pct"],
        "validation_SMAPE_pct": validation_metrics_real["SMAPE_pct"],
        "validation_R2": validation_metrics_real["R2"],
        "MAE": test_metrics_real["MAE"],
        "RMSE": test_metrics_real["RMSE"],
        "MAPE_pct": test_metrics_real["MAPE_pct"],
        "SMAPE_pct": test_metrics_real["SMAPE_pct"],
        "R2": test_metrics_real["R2"],
        "n_eval_points": test_metrics_real["n_eval_points"],
        "metric_space": "real_pm25",
    }])
    metrics_df.to_csv(OUTPUT_METRICS, index=False)

    print(metrics_df.to_string(index=False))

    forecast_df = test_df[["timestamp", TARGET]].copy()
    forecast_df["actual_scaled"] = forecast_df[TARGET]
    forecast_df["pred"] = y_pred
    forecast_df["residual"] = forecast_df[TARGET] - forecast_df["pred"]
    forecast_df["actual_pm25"] = y_test_real
    forecast_df["pred_pm25"] = y_pred_real
    forecast_df["residual_pm25"] = y_test_real - y_pred_real
    forecast_df.to_csv(OUTPUT_FORECASTS, index=False)

    residual_df = forecast_df[["timestamp", "actual_pm25", "pred_pm25", "residual_pm25"]].copy()
    residual_df.to_csv(OUTPUT_RESIDUALS, index=False)

    importances = model.get_feature_importance()
    feat_imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    feat_imp_df.to_csv(OUTPUT_FEATURES, index=False)

    split_summary = pd.DataFrame([{
        "rows_total": len(df),
        "rows_train": len(train_df),
        "rows_val": len(val_df),
        "rows_test": len(test_df),
        "target": TARGET,
        "n_features": len(feature_cols),
        "train_end": train_df["timestamp"].max(),
        "validation_end": val_df["timestamp"].max(),
        "test_start": test_df["timestamp"].min(),
    }])
    split_summary.to_csv(OUTPUT_SPLIT_SUMMARY, index=False)

    html_path = PLOTS_DIR / "catboost_forecast_interactive.html"
    png_path = PLOTS_DIR / "catboost_forecast_interactive.png"
    static_plot_path = PLOTS_DIR / "catboost_actual_vs_predicted.png"
    residual_plot_path = PLOTS_DIR / "catboost_residual_diagnostics.png"

    df_all_real = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        TARGET: inverse_scale_feature(df[TARGET].values, scaler, TARGET),
    })
    test_df_real = pd.DataFrame({
        "timestamp": forecast_df["timestamp"].values,
        TARGET: forecast_df["actual_pm25"].values,
    })

    build_interactive_plot(
        df_all=df_all_real,
        test_df=test_df_real,
        preds=y_pred_real,
        html_path=html_path,
        png_path=png_path,
        train_end=train_df["timestamp"].max(),
        val_end=val_df["timestamp"].max()
    )
    save_static_forecast_plot(residual_df, static_plot_path)
    save_residual_plot(residual_df, residual_plot_path)

    run_info = {
        "input_path": str(input_path),
        "model_path": str(MODEL_PATH),
        "outputs": {
            "forecasts": str(OUTPUT_FORECASTS),
            "metrics": str(OUTPUT_METRICS),
            "feature_importance": str(OUTPUT_FEATURES),
            "split_summary": str(OUTPUT_SPLIT_SUMMARY),
            "residuals": str(OUTPUT_RESIDUALS),
            "interactive_plot": str(html_path),
            "static_plot": str(static_plot_path),
            "residual_plot": str(residual_plot_path),
        },
        "config": {
            "target": TARGET,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "enable_pm25_lags": ENABLE_PM25_LAGS,
            "pm25_lags": PM25_LAGS,
            "iterations": ITERATIONS,
            "learning_rate": LEARNING_RATE,
            "depth": DEPTH,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        },
        "feature_columns": feature_cols,
    }

    with open(OUTPUT_RUN_INFO, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Forecasts        : {OUTPUT_FORECASTS}")
    print(f"Metrics          : {OUTPUT_METRICS}")
    print(f"Feature importance: {OUTPUT_FEATURES}")
    print(f"Interactive plot : {html_path}")


if __name__ == "__main__":
    main()
