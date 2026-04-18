from __future__ import annotations

import json
import random
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import plotly.graph_objects as go
except Exception:
    go = None

try:
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception as exc:
    raise ImportError(
        "Missing package 'statsmodels'. Run the script with the project's virtual "
        "environment, for example: .\\.venv\\Scripts\\python.exe "
        "src\\phase_2\\sarimax_model\\sarimax_model.py"
    ) from exc


matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "sarimax_model"
PLOTS_DIR = PROJECT_ROOT / "pictures" / "sarimax_model"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "pm25"
TIME_CANDIDATES = ["datetime", "date"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42
FORECAST_HORIZON = 24
MAXITER = 70

EXOG_FEATURE_PRIORITY = [
    "hour_sin",
    "hour_cos",
    "pollution_stagnation_index",
    "wind_x_vector",
    "wind_y_vector",
    "total_generation_mw",
    "temperature_2m (\u00b0C)",
    "rain (mm)",
    "relative_humidity_2m (%)",
]

MODEL_CANDIDATES = [
    {"order": (1, 0, 1), "seasonal_order": (1, 0, 1, FORECAST_HORIZON), "trend": "c"},
    {"order": (2, 0, 1), "seasonal_order": (1, 0, 1, FORECAST_HORIZON), "trend": "c"},
    {"order": (1, 0, 2), "seasonal_order": (1, 0, 1, FORECAST_HORIZON), "trend": "c"},
    {"order": (1, 0, 1), "seasonal_order": (1, 1, 1, FORECAST_HORIZON), "trend": "c"},
]

INPUT_CANDIDATES = [
    DATA_DIR / "4E_selected_dataset.csv",
    DATA_DIR / "phase_1" / "4E_selected_dataset.csv",
]
SCALER_CANDIDATES = [
    PROJECT_ROOT / "models" / "scaler.pkl",
]

OUTPUT_FORECASTS = DATA_DIR / "sarimax_forecasts.csv"
OUTPUT_METRICS = DATA_DIR / "sarimax_metrics.csv"
OUTPUT_COEFFICIENTS = DATA_DIR / "sarimax_coefficients.csv"
OUTPUT_CANDIDATES = DATA_DIR / "sarimax_candidate_results.csv"
OUTPUT_SPLIT_SUMMARY = DATA_DIR / "sarimax_split_summary.csv"
OUTPUT_RUN_INFO = DATA_DIR / "sarimax_run_info.json"
OUTPUT_RESIDUALS = DATA_DIR / "sarimax_residuals.csv"

MODEL_PATH = MODELS_DIR / "sarimax_pm25_model.pkl"
SUMMARY_PATH = MODELS_DIR / "sarimax_summary.txt"
FEATURES_PATH = MODELS_DIR / "sarimax_feature_columns.pkl"


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_pct": mape(y_true, y_pred),
        "SMAPE_pct": smape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        "n_eval_points": int(len(y_true)),
    }


def load_scaler() -> object:
    scaler_path = resolve_existing_path(SCALER_CANDIDATES, "Scaler")
    return joblib.load(scaler_path)


def inverse_scale_pm25(values: pd.Series | np.ndarray, scaler: object) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    feature_names = list(getattr(scaler, "feature_names_in_", []))

    if TARGET not in feature_names:
        return values

    idx = feature_names.index(TARGET)
    log_values = values * scaler.scale_[idx] + scaler.mean_[idx]
    return np.expm1(log_values)


def prepare_dataframe() -> tuple[pd.DataFrame, list[str], Path]:
    input_path = resolve_existing_path(INPUT_CANDIDATES, "Phase 1 selected dataset")
    df = pd.read_csv(input_path)

    time_col = detect_time_column(df)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    if df.duplicated(subset=[time_col]).sum() > 0:
        df = df.drop_duplicates(subset=[time_col], keep="first").reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET not in numeric_cols:
        raise ValueError(f"Target '{TARGET}' not found among numeric columns.")

    available_priority = [col for col in EXOG_FEATURE_PRIORITY if col in df.columns]
    if not available_priority:
        available_priority = [
            col for col in numeric_cols
            if col != TARGET and col != "wind_direction_10m (\u00b0)" and not col.endswith("_was_missing")
        ]

    required_cols = [time_col, TARGET] + available_priority
    df = df[required_cols].copy()

    for column in [TARGET] + available_priority:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[[TARGET] + available_priority] = df[[TARGET] + available_priority].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET] + available_priority).reset_index(drop=True)
    df["timestamp"] = df[time_col]

    if len(df) < 500:
        raise ValueError("Too few rows remain after cleaning for SARIMAX training.")

    return df, available_priority, input_path


def split_chronologically(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_rows = len(df)
    train_end = int(n_rows * TRAIN_RATIO)
    val_end = int(n_rows * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def fit_sarimax(
    y_train: pd.Series,
    x_train: pd.DataFrame,
    candidate: dict[str, object],
):
    model = SARIMAX(
        endog=y_train,
        exog=x_train,
        order=candidate["order"],
        seasonal_order=candidate["seasonal_order"],
        trend=candidate["trend"],
        concentrate_scale=True,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=MAXITER)


def extend_one_step_predictions(
    result,
    y_future: pd.Series,
    exog_future: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    extended_result = result.extend(endog=y_future, exog=exog_future)
    prediction = extended_result.get_prediction(start=0, end=len(y_future) - 1)
    predictions = np.asarray(prediction.predicted_mean, dtype=float)
    conf_int = prediction.conf_int(alpha=0.05)

    if isinstance(conf_int, pd.DataFrame):
        lower = np.asarray(conf_int.iloc[:, 0], dtype=float)
        upper = np.asarray(conf_int.iloc[:, 1], dtype=float)
    else:
        lower = np.asarray(conf_int[:, 0], dtype=float)
        upper = np.asarray(conf_int[:, 1], dtype=float)

    return predictions, lower, upper


def evaluate_candidate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    candidate: dict[str, object],
    scaler: object,
) -> tuple[dict[str, object], object | None]:
    started = time.perf_counter()
    record: dict[str, object] = {
        "order": str(candidate["order"]),
        "seasonal_order": str(candidate["seasonal_order"]),
        "trend": candidate["trend"],
        "status": "failed",
        "validation_MAE": np.nan,
        "validation_RMSE": np.nan,
        "validation_MAPE_pct": np.nan,
        "validation_SMAPE_pct": np.nan,
        "validation_R2": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "fit_seconds": np.nan,
        "error": "",
    }

    try:
        result = fit_sarimax(train_df[TARGET], train_df[feature_cols], candidate)
        val_forecast_scaled, _, _ = extend_one_step_predictions(
            result,
            val_df[TARGET],
            val_df[feature_cols],
        )

        y_val_real = inverse_scale_pm25(val_df[TARGET].to_numpy(), scaler)
        y_pred_real = inverse_scale_pm25(val_forecast_scaled, scaler)
        metrics = all_metrics(y_val_real, y_pred_real)

        record.update(
            {
                "status": "ok",
                "validation_MAE": metrics["MAE"],
                "validation_RMSE": metrics["RMSE"],
                "validation_MAPE_pct": metrics["MAPE_pct"],
                "validation_SMAPE_pct": metrics["SMAPE_pct"],
                "validation_R2": metrics["R2"],
                "aic": float(result.aic),
                "bic": float(result.bic),
            }
        )

        elapsed = time.perf_counter() - started
        record["fit_seconds"] = round(elapsed, 3)
        return record, result
    except Exception as exc:
        elapsed = time.perf_counter() - started
        record["fit_seconds"] = round(elapsed, 3)
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record, None


def build_interactive_plot(forecast_df: pd.DataFrame, html_path: Path, png_path: Path | None = None) -> None:
    if go is None:
        return

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["actual_pm25"],
            mode="lines",
            name="Observed",
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["pred_pm25"],
            mode="lines+markers",
            name="Predicted",
            line=dict(width=2, dash="dash"),
            marker=dict(size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["pred_lower_pm25"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["pred_upper_pm25"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(99, 110, 250, 0.16)",
            line=dict(width=0),
            name="95% interval",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="SARIMAX Forecast vs Observed PM2.5 (chronological test window)",
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
        yaxis=dict(title="PM2.5 (ug/m3)"),
        template="plotly_white",
        legend=dict(orientation="h"),
        height=720,
        hovermode="x unified",
    )

    fig.write_html(str(html_path), include_plotlyjs="cdn")
    if png_path is not None:
        try:
            fig.write_image(str(png_path), scale=2)
        except Exception:
            pass


def save_residual_diagnostics(residual_df: pd.DataFrame, output_path: Path) -> float:
    residuals = residual_df["residual_pm25"].to_numpy(dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    ljung_box = acorr_ljungbox(residuals, lags=[min(24, max(2, len(residuals) // 5))], return_df=True)
    ljung_box_pvalue = float(ljung_box["lb_pvalue"].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(residual_df["timestamp"], residual_df["residual_pm25"], color="tab:blue", linewidth=1.2)
    axes[0, 0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Residuals over time")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Residual")

    axes[0, 1].hist(residuals, bins=35, color="tab:green", alpha=0.8, edgecolor="white")
    axes[0, 1].set_title("Residual distribution")
    axes[0, 1].set_xlabel("Residual")
    axes[0, 1].set_ylabel("Count")

    plot_acf(residuals, ax=axes[1, 0], lags=min(48, len(residuals) - 1), zero=False)
    axes[1, 0].set_title("Residual ACF")

    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q plot")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    return ljung_box_pvalue


def main() -> None:
    set_seed(RANDOM_STATE)

    print("=" * 90)
    print("SARIMAX :: SUPERVISED FORECASTING PIPELINE")
    print("=" * 90)

    scaler = load_scaler()
    df, feature_cols, input_path = prepare_dataframe()
    train_df, val_df, test_df = split_chronologically(df)

    print(f"Input dataset : {input_path}")
    print(f"Rows used     : {len(df)}")
    print(f"Target        : {TARGET}")
    print(f"Exog features : {len(feature_cols)}")
    print("Features      :", feature_cols)

    print("\n" + "=" * 80)
    print("CHRONOLOGICAL SPLIT SUMMARY")
    print("=" * 80)
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows  : {len(val_df)}")
    print(f"Test rows : {len(test_df)}")
    print(f"Train range: {train_df['timestamp'].min()} -> {train_df['timestamp'].max()}")
    print(f"Val range  : {val_df['timestamp'].min()} -> {val_df['timestamp'].max()}")
    print(f"Test range : {test_df['timestamp'].min()} -> {test_df['timestamp'].max()}")

    print("\n" + "=" * 80)
    print("MODEL SELECTION")
    print("=" * 80)

    candidate_rows: list[dict[str, object]] = []
    best_record: dict[str, object] | None = None

    for candidate in MODEL_CANDIDATES:
        record, _ = evaluate_candidate(train_df, val_df, feature_cols, candidate, scaler)
        candidate_rows.append(record)

        print(
            f"{record['order']} x {record['seasonal_order']} :: {record['status']} "
            f"| val_RMSE={record['validation_RMSE']}"
        )

        if record["status"] != "ok":
            continue

        if best_record is None or float(record["validation_RMSE"]) < float(best_record["validation_RMSE"]):
            best_record = record

    candidate_df = pd.DataFrame(candidate_rows).sort_values(
        by=["status", "validation_RMSE", "aic"],
        ascending=[False, True, True],
        na_position="last",
    )
    candidate_df.to_csv(OUTPUT_CANDIDATES, index=False)

    if best_record is None:
        raise RuntimeError("All SARIMAX candidate models failed. Check sarimax_candidate_results.csv.")

    best_order = tuple(int(value.strip()) for value in str(best_record["order"]).strip("()").split(","))
    best_seasonal_order = tuple(
        int(value.strip()) for value in str(best_record["seasonal_order"]).strip("()").split(",")
    )
    best_trend = str(best_record["trend"])

    final_candidate = {
        "order": best_order,
        "seasonal_order": best_seasonal_order,
        "trend": best_trend,
    }

    print("\nBest validation model:")
    print(
        f"order={final_candidate['order']} seasonal_order={final_candidate['seasonal_order']} "
        f"trend={final_candidate['trend']} val_RMSE={best_record['validation_RMSE']:.4f}"
    )

    combined_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    final_result = fit_sarimax(combined_df[TARGET], combined_df[feature_cols], final_candidate)

    print("\n" + "=" * 80)
    print("TEST FORECAST + METRICS")
    print("=" * 80)

    test_pred_scaled, test_lower_scaled, test_upper_scaled = extend_one_step_predictions(
        final_result,
        test_df[TARGET],
        test_df[feature_cols],
    )

    y_test_real = inverse_scale_pm25(test_df[TARGET].to_numpy(), scaler)
    y_pred_real = inverse_scale_pm25(test_pred_scaled, scaler)
    y_lower_real = inverse_scale_pm25(test_lower_scaled, scaler)
    y_upper_real = inverse_scale_pm25(test_upper_scaled, scaler)

    validation_pred_scaled, _, _ = extend_one_step_predictions(
        fit_sarimax(train_df[TARGET], train_df[feature_cols], final_candidate),
        val_df[TARGET],
        val_df[feature_cols],
    )
    y_val_real = inverse_scale_pm25(val_df[TARGET].to_numpy(), scaler)
    y_val_pred_real = inverse_scale_pm25(validation_pred_scaled, scaler)

    validation_metrics = all_metrics(y_val_real, y_val_pred_real)
    test_metrics = all_metrics(y_test_real, y_pred_real)

    residual_df = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"].to_numpy(),
            "actual_pm25": y_test_real,
            "pred_pm25": y_pred_real,
            "residual_pm25": y_test_real - y_pred_real,
        }
    )
    residual_df.to_csv(OUTPUT_RESIDUALS, index=False)

    diagnostics_path = PLOTS_DIR / "sarimax_residual_diagnostics.png"
    ljung_box_pvalue = save_residual_diagnostics(residual_df, diagnostics_path)

    metrics_df = pd.DataFrame(
        [
            {
                "target": TARGET,
                "n_rows_total": len(df),
                "n_rows_train": len(train_df),
                "n_rows_val": len(val_df),
                "n_rows_test": len(test_df),
                "n_features": len(feature_cols),
                "order": str(final_candidate["order"]),
                "seasonal_order": str(final_candidate["seasonal_order"]),
                "trend": final_candidate["trend"],
                "forecast_mode": "state_space_one_step_ahead",
                "validation_MAE": validation_metrics["MAE"],
                "validation_RMSE": validation_metrics["RMSE"],
                "validation_MAPE_pct": validation_metrics["MAPE_pct"],
                "validation_SMAPE_pct": validation_metrics["SMAPE_pct"],
                "validation_R2": validation_metrics["R2"],
                "test_MAE": test_metrics["MAE"],
                "test_RMSE": test_metrics["RMSE"],
                "test_MAPE_pct": test_metrics["MAPE_pct"],
                "test_SMAPE_pct": test_metrics["SMAPE_pct"],
                "test_R2": test_metrics["R2"],
                "aic": float(final_result.aic),
                "bic": float(final_result.bic),
                "ljung_box_pvalue": ljung_box_pvalue,
            }
        ]
    )
    metrics_df.to_csv(OUTPUT_METRICS, index=False)
    print(metrics_df.to_string(index=False))

    forecast_df = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"].to_numpy(),
            "actual_scaled": test_df[TARGET].to_numpy(),
            "pred_scaled": test_pred_scaled,
            "pred_lower_scaled": test_lower_scaled,
            "pred_upper_scaled": test_upper_scaled,
            "actual_pm25": y_test_real,
            "pred_pm25": y_pred_real,
            "pred_lower_pm25": y_lower_real,
            "pred_upper_pm25": y_upper_real,
            "residual_pm25": y_test_real - y_pred_real,
        }
    )
    forecast_df.to_csv(OUTPUT_FORECASTS, index=False)

    params = final_result.params
    pvalues = final_result.pvalues.reindex(params.index)
    coefficients_df = pd.DataFrame(
        {
            "parameter": params.index,
            "coefficient": params.values,
            "abs_coefficient": np.abs(params.values),
            "pvalue": pvalues.values,
        }
    ).sort_values("abs_coefficient", ascending=False)
    coefficients_df.to_csv(OUTPUT_COEFFICIENTS, index=False)

    split_summary = pd.DataFrame(
        [
            {
                "rows_total": len(df),
                "rows_train": len(train_df),
                "rows_val": len(val_df),
                "rows_test": len(test_df),
                "train_start": train_df["timestamp"].min(),
                "train_end": train_df["timestamp"].max(),
                "val_start": val_df["timestamp"].min(),
                "val_end": val_df["timestamp"].max(),
                "test_start": test_df["timestamp"].min(),
                "test_end": test_df["timestamp"].max(),
            }
        ]
    )
    split_summary.to_csv(OUTPUT_SPLIT_SUMMARY, index=False)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as summary_file:
        summary_file.write(final_result.summary().as_text())

    joblib.dump(final_result, MODEL_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)

    html_path = PLOTS_DIR / "sarimax_forecast_interactive.html"
    png_path = PLOTS_DIR / "sarimax_forecast_interactive.png"
    build_interactive_plot(forecast_df, html_path, png_path)

    static_plot_path = PLOTS_DIR / "sarimax_actual_vs_predicted.png"
    plt.figure(figsize=(15, 6))
    plt.plot(forecast_df["timestamp"], forecast_df["actual_pm25"], label="Actual", linewidth=1.8)
    plt.plot(forecast_df["timestamp"], forecast_df["pred_pm25"], label="Predicted", linewidth=1.6, linestyle="--")
    plt.fill_between(
        forecast_df["timestamp"],
        forecast_df["pred_lower_pm25"],
        forecast_df["pred_upper_pm25"],
        alpha=0.18,
        label="95% interval",
    )
    plt.title("SARIMAX forecast vs observed PM2.5")
    plt.xlabel("Time")
    plt.ylabel("PM2.5 (ug/m3)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(static_plot_path, dpi=300)
    plt.close()

    run_info = {
        "input_path": str(input_path),
        "model_path": str(MODEL_PATH),
        "summary_path": str(SUMMARY_PATH),
        "feature_columns_path": str(FEATURES_PATH),
        "outputs": {
            "forecasts": str(OUTPUT_FORECASTS),
            "metrics": str(OUTPUT_METRICS),
            "coefficients": str(OUTPUT_COEFFICIENTS),
            "candidate_results": str(OUTPUT_CANDIDATES),
            "split_summary": str(OUTPUT_SPLIT_SUMMARY),
            "residuals": str(OUTPUT_RESIDUALS),
            "interactive_plot": str(html_path),
            "static_plot": str(static_plot_path),
            "residual_diagnostics": str(diagnostics_path),
        },
        "config": {
            "target": TARGET,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "candidate_models": MODEL_CANDIDATES,
            "maxiter": MAXITER,
            "forecast_horizon": FORECAST_HORIZON,
        },
        "selected_model": {
            "order": final_candidate["order"],
            "seasonal_order": final_candidate["seasonal_order"],
            "trend": final_candidate["trend"],
        },
        "feature_columns": feature_cols,
    }

    with open(OUTPUT_RUN_INFO, "w", encoding="utf-8") as run_info_file:
        json.dump(run_info, run_info_file, indent=2, default=str)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Forecasts         : {OUTPUT_FORECASTS}")
    print(f"Metrics           : {OUTPUT_METRICS}")
    print(f"Coefficients      : {OUTPUT_COEFFICIENTS}")
    print(f"Candidate results : {OUTPUT_CANDIDATES}")
    print(f"Interactive plot  : {html_path}")


if __name__ == "__main__":
    main()
