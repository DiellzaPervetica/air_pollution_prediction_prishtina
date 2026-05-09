from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# SAFE CATBOOST IMPORT
# ---------------------------------------------------------
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
    CATBOOST_IMPORT_ERROR = ""
except Exception as e:
    CatBoostRegressor = None
    CATBOOST_AVAILABLE = False
    CATBOOST_IMPORT_ERROR = str(e)

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Prishtina PM2.5 Forecast Studio",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_CANDIDATES = [
    BASE_DIR / "data" / "phase_1" / "4E_selected_dataset.csv",
    BASE_DIR / "data" / "4E_selected_dataset.csv",
]

SCALER_CANDIDATES = [
    BASE_DIR / "models" / "scaler.pkl",
]

CATBOOST_MODEL_CANDIDATES = [
    BASE_DIR / "models" / "phase_3" / "catboost_tuned" / "catboost_phase3_tuned_model.cbm",
    BASE_DIR / "models" / "catboost_model" / "catboost_pm25_model.cbm",
]

CATBOOST_FEATURE_CANDIDATES = [
    BASE_DIR / "models" / "phase_3" / "catboost_tuned" / "catboost_phase3_feature_columns.pkl",
    BASE_DIR / "models" / "catboost_model" / "catboost_feature_columns.pkl",
]

CATBOOST_RUN_INFO_CANDIDATES = [
    BASE_DIR / "data" / "phase_3" / "supervised" / "catboost_tuned" / "catboost_tuned_run_info.json",
    BASE_DIR / "data" / "phase_2" / "supervised" / "catboost" / "catboost_run_info.json",
]

CATBOOST_FORECAST_CANDIDATES = [
    BASE_DIR / "data" / "phase_2" / "supervised" / "catboost" / "catboost_forecasts.csv",
]

CATBOOST_METRICS_CANDIDATES = [
    BASE_DIR / "data" / "phase_2" / "supervised" / "catboost" / "catboost_metrics.csv",
]

CATBOOST_IMPORTANCE_CANDIDATES = [
    BASE_DIR / "data" / "phase_2" / "supervised" / "catboost" / "catboost_feature_importance.csv",
]

SUPERVISED_COMPARISON_CANDIDATES = [
    BASE_DIR / "data" / "phase_2" / "comparison" / "supervised_model_comparison.csv",
]

UNSUPERVISED_COMPARISON_CANDIDATES = [
    BASE_DIR / "data" / "phase_2" / "comparison" / "unsupervised_model_comparison.csv",
]

PHASE3_METRICS_CANDIDATES = [
    BASE_DIR / "data" / "phase_3" / "supervised" / "catboost_tuned" / "catboost_tuned_metrics.csv",
]

PHASE3_IMPROVEMENT_CANDIDATES = [
    BASE_DIR / "data" / "phase_3" / "comparison" / "catboost_phase2_vs_phase3_improvement.csv",
]

PHASE3_SHAP_CANDIDATES = [
    BASE_DIR / "data" / "phase_3" / "supervised" / "catboost_tuned" / "catboost_tuned_shap_global_importance.csv",
]

PHASE3_SEASONAL_CANDIDATES = [
    BASE_DIR / "data" / "phase_3" / "supervised" / "catboost_tuned" / "catboost_tuned_seasonal_stability.csv",
]

PHASE3_DAILY_SNAPSHOT_CANDIDATES = [
    BASE_DIR / "data" / "phase_3" / "forecasting" / "next_day_pm25_daily_summary_snapshot.csv",
]

PHASE3_HOURLY_SNAPSHOT_CANDIDATES = [
    BASE_DIR / "data" / "phase_3" / "forecasting" / "next_day_pm25_hourly_forecast_snapshot.csv",
]

TARGET = "pm25"
TIME_CANDIDATES = ["timestamp", "datetime", "date"]

DIRECT_CONTROL_COLS = [
    "total_generation_mw",
    "temperature_2m (°C)",
    "rain (mm)",
    "relative_humidity_2m (%)",
    "wind_direction_10m (°)",
    "wind_speed_10m (km/h)",
]

DEFAULT_CATBOOST_FEATURES = [
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
    "wind_direction_10m (°)",
    "wind_speed_10m (km/h)",
    "pm25_lag_1",
    "pm25_lag_24",
]

DISPLAY_NAMES = {
    "pm25": "PM2.5",
    "total_generation_mw": "Total generation",
    "temperature_2m (°C)": "Temperature",
    "rain (mm)": "Rain",
    "relative_humidity_2m (%)": "Humidity",
    "wind_direction_10m (°)": "Wind direction",
    "wind_speed_10m (km/h)": "Wind speed",
    "pollution_stagnation_index": "Stagnation index",
    "wind_x_vector": "Wind X",
    "wind_y_vector": "Wind Y",
}

PRESETS = {
    "Balanced": {
        "generation_pct": 0,
        "temperature_delta": 0.0,
        "rain_pct": 0,
        "humidity_delta": 0.0,
        "wind_speed_pct": 0,
        "wind_direction_shift": 0,
    },
    "Cold stagnant evening": {
        "generation_pct": 18,
        "temperature_delta": -4.0,
        "rain_pct": -100,
        "humidity_delta": 8.0,
        "wind_speed_pct": -35,
        "wind_direction_shift": 0,
    },
    "Windy clean day": {
        "generation_pct": -10,
        "temperature_delta": 3.0,
        "rain_pct": 0,
        "humidity_delta": -6.0,
        "wind_speed_pct": 45,
        "wind_direction_shift": 20,
    },
    "Rain washout": {
        "generation_pct": 0,
        "temperature_delta": 0.0,
        "rain_pct": 120,
        "humidity_delta": 6.0,
        "wind_speed_pct": 15,
        "wind_direction_shift": 0,
    },
    "High generation night": {
        "generation_pct": 25,
        "temperature_delta": -1.0,
        "rain_pct": -100,
        "humidity_delta": 5.0,
        "wind_speed_pct": -15,
        "wind_direction_shift": 0,
    },
}

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def detect_time_col(df: pd.DataFrame) -> str:
    for col in TIME_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Nuk u gjet kolona kohore. U provuan: {TIME_CANDIDATES}")


@st.cache_data(show_spinner=False)
def load_optional_csv(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_phase1_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = detect_time_col(df)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).copy()
    df = df.rename(columns={time_col: "timestamp"}).sort_values("timestamp").reset_index(drop=True)

    numeric_cols = [c for c in df.columns if c != "timestamp"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET not in df.columns:
        raise ValueError(f"Mungon kolona target '{TARGET}' në dataset.")

    return df


@st.cache_resource(show_spinner=False)
def load_pickle(path: str | None):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_catboost_bundle(
    model_path: str | None,
    feature_path: str | None,
    run_info_path: str | None,
    scaler_path: str | None,
):
    scaler = load_pickle(scaler_path)

    feature_cols = None
    if feature_path is not None and Path(feature_path).exists():
        with open(feature_path, "rb") as f:
            feature_cols = pickle.load(f)

    if feature_cols is None and run_info_path is not None and Path(run_info_path).exists():
        with open(run_info_path, "r", encoding="utf-8") as f:
            run_info = json.load(f)
        feature_cols = run_info.get("feature_columns")

    if not feature_cols:
        feature_cols = DEFAULT_CATBOOST_FEATURES

    # if catboost not available or model missing, return cleanly
    if (not CATBOOST_AVAILABLE) or model_path is None or (not Path(model_path).exists()):
        return None, feature_cols, scaler

    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return model, feature_cols, scaler


def scaler_feature_index(scaler) -> dict[str, int]:
    names = getattr(scaler, "feature_names_in_", None)
    if scaler is None or names is None:
        return {}
    return {name: idx for idx, name in enumerate(names)}


def inverse_scale_value(value: float, feature_name: str, scaler) -> float:
    val = float(value)

    idx_map = scaler_feature_index(scaler)
    if scaler is not None and feature_name in idx_map:
        idx = idx_map[feature_name]
        val = val * float(scaler.scale_[idx]) + float(scaler.mean_[idx])

    if feature_name == TARGET:
        try:
            val = np.expm1(val)
        except Exception:
            pass
        val = max(val, 0.0)

    return float(val)


def scale_value(value: float, feature_name: str, scaler) -> float:
    val = float(value)

    if feature_name == TARGET:
        val = np.log1p(max(val, 0.0))

    idx_map = scaler_feature_index(scaler)
    if scaler is not None and feature_name in idx_map:
        idx = idx_map[feature_name]
        denom = float(scaler.scale_[idx]) if float(scaler.scale_[idx]) != 0 else 1.0
        val = (val - float(scaler.mean_[idx])) / denom

    return float(val)


@st.cache_data(show_spinner=False)
def build_display_frame(df_model_space: pd.DataFrame, scaler_path: str | None) -> pd.DataFrame:
    scaler = load_pickle(scaler_path)
    out = df_model_space.copy()

    cols_to_inverse = [c for c in DIRECT_CONTROL_COLS + [TARGET] if c in out.columns]
    for col in cols_to_inverse:
        out[col] = out[col].apply(lambda x: inverse_scale_value(x, col, scaler))

    out["hour"] = out["timestamp"].dt.hour
    out["month"] = out["timestamp"].dt.month
    out["day_name"] = out["timestamp"].dt.day_name()

    return out


def get_context_subset(df_display: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    subset = df_display[(df_display["month"] == ts.month) & (df_display["hour"] == ts.hour)].copy()
    if len(subset) >= 20:
        return subset

    subset = df_display[df_display["hour"] == ts.hour].copy()
    if len(subset) >= 20:
        return subset

    return df_display.copy()


def profile_controls(df_display: pd.DataFrame, ts: pd.Timestamp) -> dict[str, float]:
    subset = get_context_subset(df_display, ts)
    values = {}
    for col in DIRECT_CONTROL_COLS:
        if col not in subset.columns:
            values[col] = 0.0
            continue
        series = pd.to_numeric(subset[col], errors="coerce").dropna()
        values[col] = float(series.median()) if not series.empty else 0.0
    return values


def nearest_historical_controls(df_display: pd.DataFrame, ts: pd.Timestamp) -> dict[str, float]:
    exact = df_display.loc[df_display["timestamp"] == ts]
    if not exact.empty:
        row = exact.iloc[0]
        return {col: float(row[col]) for col in DIRECT_CONTROL_COLS if col in exact.columns}
    return profile_controls(df_display, ts)


def apply_scenario_adjustments(base: dict[str, float], settings: dict[str, float]) -> dict[str, float]:
    adjusted = dict(base)

    adjusted["total_generation_mw"] = max(
        0.0, base["total_generation_mw"] * (1.0 + settings["generation_pct"] / 100.0)
    )
    adjusted["temperature_2m (°C)"] = base["temperature_2m (°C)"] + settings["temperature_delta"]
    adjusted["rain (mm)"] = max(
        0.0, base["rain (mm)"] * (1.0 + settings["rain_pct"] / 100.0)
    )
    adjusted["relative_humidity_2m (%)"] = float(
        np.clip(base["relative_humidity_2m (%)"] + settings["humidity_delta"], 0.0, 100.0)
    )
    adjusted["wind_speed_10m (km/h)"] = max(
        0.0, base["wind_speed_10m (km/h)"] * (1.0 + settings["wind_speed_pct"] / 100.0)
    )
    adjusted["wind_direction_10m (°)"] = (
        base["wind_direction_10m (°)"] + settings["wind_direction_shift"]
    ) % 360.0

    return adjusted


def build_catboost_row(
    ts: pd.Timestamp,
    controls_display: dict[str, float],
    lag1_model_space: float,
    lag24_model_space: float,
    feature_cols: list[str],
    scaler,
) -> pd.DataFrame:
    gen = float(controls_display["total_generation_mw"])
    temp = float(controls_display["temperature_2m (°C)"])
    rain = max(0.0, float(controls_display["rain (mm)"]))
    humidity = float(np.clip(controls_display["relative_humidity_2m (%)"], 0.0, 100.0))
    wind_dir = float(controls_display["wind_direction_10m (°)"]) % 360.0
    wind_speed = max(0.0, float(controls_display["wind_speed_10m (km/h)"]))

    hour = int(ts.hour)
    month = int(ts.month)

    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)

    stagnation = gen / (wind_speed + 0.1)
    wd_rad = np.deg2rad(wind_dir)
    wind_x = wind_speed * np.cos(wd_rad)
    wind_y = wind_speed * np.sin(wd_rad)

    raw_values = {
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "pollution_stagnation_index": stagnation,
        "wind_x_vector": wind_x,
        "wind_y_vector": wind_y,
        "total_generation_mw": gen,
        "temperature_2m (°C)": temp,
        "rain (mm)": rain,
        "relative_humidity_2m (%)": humidity,
        "wind_direction_10m (°)": wind_dir,
        "wind_speed_10m (km/h)": wind_speed,
    }

    row = {}
    for col in feature_cols:
        if col == "pm25_lag_1":
            row[col] = float(lag1_model_space)
        elif col == "pm25_lag_24":
            row[col] = float(lag24_model_space)
        else:
            row[col] = scale_value(raw_values.get(col, 0.0), col, scaler)

    return pd.DataFrame([row], columns=feature_cols)


def predict_pm25(model, row: pd.DataFrame, scaler) -> tuple[float, float]:
    pred_model = float(model.predict(row)[0])
    pred_real = inverse_scale_value(pred_model, TARGET, scaler)
    return pred_model, pred_real


def get_lag(history_model: pd.Series, ts: pd.Timestamp, hours: int) -> float:
    lag_ts = ts - pd.Timedelta(hours=hours)

    if lag_ts in history_model.index:
        return float(history_model.loc[lag_ts])

    older = history_model.loc[history_model.index <= lag_ts]
    if not older.empty:
        return float(older.iloc[-1])

    return float(history_model.iloc[0])


def run_historical_counterfactual(
    ts: pd.Timestamp,
    df_model_space: pd.DataFrame,
    df_display: pd.DataFrame,
    model,
    feature_cols: list[str],
    scaler,
    settings: dict[str, float],
) -> dict[str, object]:
    history_model = pd.Series(
        df_model_space[TARGET].values,
        index=df_model_space["timestamp"]
    ).sort_index()

    base_controls = nearest_historical_controls(df_display, ts)
    scenario_controls = apply_scenario_adjustments(base_controls, settings)

    lag1 = get_lag(history_model, ts, 1)
    lag24 = get_lag(history_model, ts, 24)

    base_row = build_catboost_row(ts, base_controls, lag1, lag24, feature_cols, scaler)
    scenario_row = build_catboost_row(ts, scenario_controls, lag1, lag24, feature_cols, scaler)

    _, pred_base_real = predict_pm25(model, base_row, scaler)
    _, pred_scenario_real = predict_pm25(model, scenario_row, scaler)

    actual_model = float(history_model.loc[history_model.index <= ts].iloc[-1])
    actual_real = inverse_scale_value(actual_model, TARGET, scaler)

    return {
        "base_controls": base_controls,
        "scenario_controls": scenario_controls,
        "pred_base_real": pred_base_real,
        "pred_scenario_real": pred_scenario_real,
        "actual_real": actual_real,
    }


def run_recursive_future_forecast(
    start_ts: pd.Timestamp,
    horizon: int,
    df_model_space: pd.DataFrame,
    df_display: pd.DataFrame,
    model,
    feature_cols: list[str],
    scaler,
    settings: dict[str, float],
) -> pd.DataFrame:
    history_model = pd.Series(
        df_model_space[TARGET].values,
        index=df_model_space["timestamp"]
    ).sort_index()

    rows = []

    for step in range(horizon):
        ts = start_ts + pd.Timedelta(hours=step)

        base_controls = profile_controls(df_display, ts)
        scenario_controls = apply_scenario_adjustments(base_controls, settings)

        lag1 = get_lag(history_model, ts, 1)
        lag24 = get_lag(history_model, ts, 24)

        row = build_catboost_row(ts, scenario_controls, lag1, lag24, feature_cols, scaler)
        pred_model, pred_real = predict_pm25(model, row, scaler)

        history_model.loc[ts] = pred_model

        rows.append(
            {
                "timestamp": ts,
                "PM2.5 forecast": pred_real,
                "Baseline generation": base_controls["total_generation_mw"],
                "Scenario generation": scenario_controls["total_generation_mw"],
                "Baseline temperature": base_controls["temperature_2m (°C)"],
                "Scenario temperature": scenario_controls["temperature_2m (°C)"],
                "Baseline wind": base_controls["wind_speed_10m (km/h)"],
                "Scenario wind": scenario_controls["wind_speed_10m (km/h)"],
                "Baseline rain": base_controls["rain (mm)"],
                "Scenario rain": scenario_controls["rain (mm)"],
            }
        )

    return pd.DataFrame(rows)


def prepare_backtest_frame(df: pd.DataFrame | None, scaler) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    time_col = next((c for c in TIME_CANDIDATES if c in df.columns), None)
    if time_col is None:
        return None

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).rename(columns={time_col: "timestamp"})

    actual_candidates = ["actual_pm25", "actual_real", "actual", TARGET, "actual_scaled"]
    pred_candidates = ["pred_pm25", "pred_real", "pred", "forecast", "prediction"]

    actual_col = next((c for c in actual_candidates if c in out.columns), None)
    pred_col = next((c for c in pred_candidates if c in out.columns), None)

    if actual_col is None or pred_col is None:
        return None

    out["actual_plot"] = pd.to_numeric(out[actual_col], errors="coerce")
    out["pred_plot"] = pd.to_numeric(out[pred_col], errors="coerce")

    if actual_col == "actual_scaled":
        out["actual_plot"] = out["actual_plot"].apply(
            lambda x: inverse_scale_value(x, TARGET, scaler)
        )

    if pred_col == "pred":
        out["pred_plot"] = out["pred_plot"].apply(
            lambda x: inverse_scale_value(x, TARGET, scaler)
        )

    out = out.dropna(subset=["timestamp", "actual_plot", "pred_plot"]).sort_values("timestamp")
    return out[["timestamp", "actual_plot", "pred_plot"]]


def build_feature_importance_chart(df: pd.DataFrame | None) -> go.Figure | None:
    if df is None or df.empty:
        return None

    work = df.copy()
    feature_col = "feature" if "feature" in work.columns else work.columns[0]
    value_col = "importance" if "importance" in work.columns else work.columns[1]

    work = work[[feature_col, value_col]].copy()
    work.columns = ["feature", "importance"]
    work["label"] = work["feature"].map(lambda x: DISPLAY_NAMES.get(x, x))
    work = work.sort_values("importance", ascending=False).head(12)

    fig = px.bar(
        work,
        x="importance",
        y="label",
        orientation="h",
        title="CatBoost feature importance",
    )
    fig.update_layout(height=420, yaxis_title="")
    return fig


def build_metric_figure(actual: float, base: float, scenario: float) -> go.Figure:
    max_val = max(actual, base, scenario, 1.0) * 1.25

    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=scenario,
            delta={"reference": base},
            title={"text": "PM2.5 scenario forecast"},
            gauge={
                "axis": {"range": [0, max_val]},
                "threshold": {"line": {"color": "black", "width": 4}, "value": actual},
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)

# ---------------------------------------------------------
# LOAD FILES
# ---------------------------------------------------------
data_path = first_existing(DATA_CANDIDATES)
scaler_path = first_existing(SCALER_CANDIDATES)
catboost_model_path = first_existing(CATBOOST_MODEL_CANDIDATES)
catboost_feature_path = first_existing(CATBOOST_FEATURE_CANDIDATES)
catboost_run_info_path = first_existing(CATBOOST_RUN_INFO_CANDIDATES)
catboost_forecast_path = first_existing(CATBOOST_FORECAST_CANDIDATES)
catboost_metrics_path = first_existing(CATBOOST_METRICS_CANDIDATES)
catboost_importance_path = first_existing(CATBOOST_IMPORTANCE_CANDIDATES)
supervised_comparison_path = first_existing(SUPERVISED_COMPARISON_CANDIDATES)
unsupervised_comparison_path = first_existing(UNSUPERVISED_COMPARISON_CANDIDATES)
phase3_metrics_path = first_existing(PHASE3_METRICS_CANDIDATES)
phase3_improvement_path = first_existing(PHASE3_IMPROVEMENT_CANDIDATES)
phase3_shap_path = first_existing(PHASE3_SHAP_CANDIDATES)
phase3_seasonal_path = first_existing(PHASE3_SEASONAL_CANDIDATES)
phase3_daily_snapshot_path = first_existing(PHASE3_DAILY_SNAPSHOT_CANDIDATES)
phase3_hourly_snapshot_path = first_existing(PHASE3_HOURLY_SNAPSHOT_CANDIDATES)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("Prishtina PM2.5 Forecast Studio")
st.caption(
    "Ky version është i lidhur me artefaktet e fazës 2 dhe fazës 3: CatBoost i tunuar, "
    "krahasime, SHAP, stabilitet sezonal dhe snapshot offline për parashikim të ditës."
)

# ---------------------------------------------------------
# BASIC FILE CHECK
# ---------------------------------------------------------
if data_path is None:
    st.error("Nuk u gjet dataset-i `4E_selected_dataset.csv`.")
    st.stop()

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
try:
    df_model = load_phase1_frame(str(data_path))
    model, feature_cols, scaler = load_catboost_bundle(
        str(catboost_model_path) if catboost_model_path else None,
        str(catboost_feature_path) if catboost_feature_path else None,
        str(catboost_run_info_path) if catboost_run_info_path else None,
        str(scaler_path) if scaler_path else None,
    )
    df_display = build_display_frame(df_model, str(scaler_path) if scaler_path else None)
except Exception as exc:
    st.exception(exc)
    st.stop()

MODEL_READY = model is not None

supervised_df = load_optional_csv(str(supervised_comparison_path) if supervised_comparison_path else None)
unsupervised_df = load_optional_csv(str(unsupervised_comparison_path) if unsupervised_comparison_path else None)
catboost_forecast_df = load_optional_csv(str(catboost_forecast_path) if catboost_forecast_path else None)
catboost_metrics_df = load_optional_csv(str(catboost_metrics_path) if catboost_metrics_path else None)
catboost_importance_df = load_optional_csv(str(catboost_importance_path) if catboost_importance_path else None)
phase3_metrics_df = load_optional_csv(str(phase3_metrics_path) if phase3_metrics_path else None)
phase3_improvement_df = load_optional_csv(str(phase3_improvement_path) if phase3_improvement_path else None)
phase3_shap_df = load_optional_csv(str(phase3_shap_path) if phase3_shap_path else None)
phase3_seasonal_df = load_optional_csv(str(phase3_seasonal_path) if phase3_seasonal_path else None)
phase3_daily_snapshot_df = load_optional_csv(str(phase3_daily_snapshot_path) if phase3_daily_snapshot_path else None)
phase3_hourly_snapshot_df = load_optional_csv(str(phase3_hourly_snapshot_path) if phase3_hourly_snapshot_path else None)
backtest_df = prepare_backtest_frame(catboost_forecast_df, scaler)

last_hist_ts = pd.Timestamp(df_model["timestamp"].max())
first_hist_ts = pd.Timestamp(df_model["timestamp"].min())

# ---------------------------------------------------------
# WARNINGS
# ---------------------------------------------------------
if not CATBOOST_AVAILABLE:
    st.warning(
        "Paketa `catboost` nuk është e instaluar. "
        "App-i do hapet, por tab-et e forecast-it do të jenë vetëm informative derisa ta instalosh."
    )
    st.code("python -m pip install catboost", language="powershell")

elif catboost_model_path is None:
    st.warning(
        "Paketa `catboost` ekziston, por nuk u gjet file-i i modelit "
        "`models/catboost_model/catboost_pm25_model.cbm`."
    )

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Scenario setup")

preset_name = st.sidebar.selectbox("Preset", list(PRESETS.keys()))
preset_values = PRESETS[preset_name]

generation_pct = st.sidebar.slider("Generation shift (%)", -40, 50, int(preset_values["generation_pct"]))
temperature_delta = st.sidebar.slider("Temperature shift", -10.0, 10.0, float(preset_values["temperature_delta"]), 0.5)
rain_pct = st.sidebar.slider("Rain shift (%)", -100, 300, int(preset_values["rain_pct"]))
humidity_delta = st.sidebar.slider("Humidity shift", -30.0, 30.0, float(preset_values["humidity_delta"]), 1.0)
wind_speed_pct = st.sidebar.slider("Wind speed shift (%)", -60, 80, int(preset_values["wind_speed_pct"]))
wind_direction_shift = st.sidebar.slider("Wind direction shift (°)", -180, 180, int(preset_values["wind_direction_shift"]))

settings = {
    "generation_pct": generation_pct,
    "temperature_delta": temperature_delta,
    "rain_pct": rain_pct,
    "humidity_delta": humidity_delta,
    "wind_speed_pct": wind_speed_pct,
    "wind_direction_shift": wind_direction_shift,
}

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Historical scenario replay", "Future forecast", "Model center"]
)

# ---------------------------------------------------------
# TAB 1 - OVERVIEW
# ---------------------------------------------------------
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("History start", first_hist_ts.strftime("%Y-%m-%d %H:%M"))
    c2.metric("History end", last_hist_ts.strftime("%Y-%m-%d %H:%M"))
    c3.metric("Rows", f"{len(df_model):,}")

    left, right = st.columns([1.2, 1])

    with left:
        pm25_daily = (
            df_display.set_index("timestamp")[TARGET]
            .resample("D")
            .mean()
            .reset_index()
        )
        fig_daily = px.line(
            pm25_daily,
            x="timestamp",
            y=TARGET,
            title="Daily mean PM2.5 from processed dataset"
        )
        fig_daily.update_layout(height=380, yaxis_title="PM2.5")
        st.plotly_chart(fig_daily, use_container_width=True)

    with right:
        st.markdown("### Status")
        if MODEL_READY:
            st.success("CatBoost modeli është gati për forecast.")
        else:
            st.info("App-i është hapur, por CatBoost forecast nuk është aktiv ende.")

        st.write(
            """
            Ky app:
            - lexon dataset-in final të fazës së parë,
            - përdor CatBoost për replay historik dhe forecast të ardhshëm,
            - dhe shfaq tabelat krahasuese të modeleve të fazës së dytë.
            """
        )

        if supervised_df is not None and not supervised_df.empty and "R2" in supervised_df.columns:
            best_row = supervised_df.sort_values("R2", ascending=False).iloc[0]
            st.success(
                f"Best supervised model nga CSV: {best_row['model']} | R² = {safe_float(best_row['R2']):.4f}"
            )

    st.markdown("### Latest observed PM2.5 values")
    latest_real = df_display[["timestamp", TARGET]].tail(24).copy()
    latest_real.columns = ["timestamp", "PM2.5"]
    st.dataframe(latest_real, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 2 - HISTORICAL
# ---------------------------------------------------------
with tab2:
    st.markdown("### Historical counterfactual")

    if not MODEL_READY:
        st.warning(
            "Ky seksion kërkon CatBoost model aktiv. "
            "Instalo `catboost` dhe sigurohu që ekziston file-i i modelit."
        )
    else:
        cols = st.columns([1, 1, 1])
        hist_date = cols[0].date_input(
            "Date",
            value=last_hist_ts.date(),
            min_value=first_hist_ts.date(),
            max_value=last_hist_ts.date(),
            key="hist_date",
        )
        hist_hour = cols[1].slider("Hour", 0, 23, int(last_hist_ts.hour), key="hist_hour")
        run_hist = cols[2].button("Run historical scenario", use_container_width=True)

        selected_ts = pd.Timestamp(hist_date) + pd.Timedelta(hours=int(hist_hour))

        if run_hist or "hist_result" not in st.session_state:
            st.session_state["hist_result"] = run_historical_counterfactual(
                ts=selected_ts,
                df_model_space=df_model,
                df_display=df_display,
                model=model,
                feature_cols=feature_cols,
                scaler=scaler,
                settings=settings,
            )

        hist_result = st.session_state["hist_result"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Actual PM2.5", f"{hist_result['actual_real']:.2f}")
        m2.metric("Model baseline", f"{hist_result['pred_base_real']:.2f}")
        m3.metric(
            "Scenario PM2.5",
            f"{hist_result['pred_scenario_real']:.2f}",
            f"{hist_result['pred_scenario_real'] - hist_result['pred_base_real']:+.2f}",
        )

        left, right = st.columns([1, 1])

        with left:
            st.plotly_chart(
                build_metric_figure(
                    actual=hist_result["actual_real"],
                    base=hist_result["pred_base_real"],
                    scenario=hist_result["pred_scenario_real"],
                ),
                use_container_width=True,
            )

        with right:
            compare_df = pd.DataFrame(
                {
                    "Version": ["Actual", "Baseline", "Scenario"],
                    "PM2.5": [
                        hist_result["actual_real"],
                        hist_result["pred_base_real"],
                        hist_result["pred_scenario_real"],
                    ],
                }
            )
            fig_compare = px.bar(compare_df, x="Version", y="PM2.5", title="Actual vs baseline vs scenario")
            fig_compare.update_layout(height=320)
            st.plotly_chart(fig_compare, use_container_width=True)

        controls_df = pd.DataFrame(
            {
                "Variable": [DISPLAY_NAMES.get(c, c) for c in DIRECT_CONTROL_COLS],
                "Baseline": [hist_result["base_controls"][c] for c in DIRECT_CONTROL_COLS],
                "Scenario": [hist_result["scenario_controls"][c] for c in DIRECT_CONTROL_COLS],
            }
        )
        st.dataframe(controls_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 3 - FUTURE
# ---------------------------------------------------------
with tab3:
    st.markdown("### Recursive future forecast")

    if phase3_daily_snapshot_df is not None and not phase3_daily_snapshot_df.empty:
        st.markdown("#### Stored next-day snapshot")
        snapshot = phase3_daily_snapshot_df.iloc[0]
        st.info(
            "Ky seksion lexon rezultatet e ruajtura nga faza 3. "
            "Nuk ka nevojë për shkarkim live nga KOSTT gjatë demos."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Forecast date", str(snapshot.get("forecast_date", "")))
        c2.metric("Mean PM2.5", f"{safe_float(snapshot.get('pm25_mean_forecast')):.2f}")
        c3.metric("Peak PM2.5", f"{safe_float(snapshot.get('pm25_max_forecast')):.2f}")
        c4.metric("Risk", str(snapshot.get("risk_category", "")))

        if phase3_hourly_snapshot_df is not None and not phase3_hourly_snapshot_df.empty:
            hourly_snapshot = phase3_hourly_snapshot_df.copy()
            hourly_snapshot["timestamp"] = pd.to_datetime(hourly_snapshot["timestamp"], errors="coerce")
            hourly_snapshot = hourly_snapshot.dropna(subset=["timestamp"])
            fig_snapshot = px.line(
                hourly_snapshot,
                x="timestamp",
                y="pm25_forecast",
                markers=True,
                title="Stored 24h PM2.5 forecast from KOSTT + weather snapshot",
            )
            fig_snapshot.update_layout(height=340, yaxis_title="PM2.5")
            st.plotly_chart(fig_snapshot, use_container_width=True)

    if not MODEL_READY:
        st.warning(
            "Ky seksion kërkon CatBoost model aktiv. "
            "Pasi ta instalosh `catboost`, forecast-i do punojë pa ndryshim tjetër në kod."
        )
    else:
        cols = st.columns([1, 1, 1])

        default_future = (last_hist_ts + pd.Timedelta(hours=1)).date()
        future_date = cols[0].date_input(
            "Future start date",
            value=default_future,
            min_value=default_future,
            key="future_date",
        )
        future_hour = cols[1].slider("Future start hour", 0, 23, int((last_hist_ts + pd.Timedelta(hours=1)).hour))
        horizon = cols[2].selectbox("Horizon (hours)", [24, 48, 72, 96, 168], index=1)

        start_future_ts = pd.Timestamp(future_date) + pd.Timedelta(hours=int(future_hour))

        if start_future_ts <= last_hist_ts:
            st.warning("Zgjidh një kohë që është pas fundit të historikut.")
        else:
            forecast_df = run_recursive_future_forecast(
                start_ts=start_future_ts,
                horizon=int(horizon),
                df_model_space=df_model,
                df_display=df_display,
                model=model,
                feature_cols=feature_cols,
                scaler=scaler,
                settings=settings,
            )

            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Start", start_future_ts.strftime("%Y-%m-%d %H:%M"))
            a2.metric("End", forecast_df["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M"))
            a3.metric("Peak PM2.5", f"{forecast_df['PM2.5 forecast'].max():.2f}")
            a4.metric("Mean PM2.5", f"{forecast_df['PM2.5 forecast'].mean():.2f}")

            fig_forecast = px.line(
                forecast_df,
                x="timestamp",
                y="PM2.5 forecast",
                markers=True,
                title="Future PM2.5 forecast path",
            )
            fig_forecast.update_layout(height=380, yaxis_title="PM2.5")
            st.plotly_chart(fig_forecast, use_container_width=True)

            fig_gen = go.Figure()
            fig_gen.add_trace(
                go.Scatter(
                    x=forecast_df["timestamp"],
                    y=forecast_df["Baseline generation"],
                    mode="lines",
                    name="Baseline generation",
                )
            )
            fig_gen.add_trace(
                go.Scatter(
                    x=forecast_df["timestamp"],
                    y=forecast_df["Scenario generation"],
                    mode="lines",
                    name="Scenario generation",
                )
            )
            fig_gen.update_layout(
                title="Baseline vs scenario generation",
                height=320,
                yaxis_title="Generation"
            )
            st.plotly_chart(fig_gen, use_container_width=True)

            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 4 - MODEL CENTER
# ---------------------------------------------------------
with tab4:
    st.markdown("### Saved model outputs and comparisons")

    if phase3_metrics_df is not None and not phase3_metrics_df.empty:
        st.markdown("#### Phase 3 tuned CatBoost")
        show_cols = [
            "selected_candidate",
            "MAE",
            "RMSE",
            "R2",
            "MAE_improvement",
            "RMSE_improvement",
            "R2_improvement",
            "RMSE_improvement_pct",
        ]
        show_cols = [col for col in show_cols if col in phase3_metrics_df.columns]
        st.dataframe(phase3_metrics_df[show_cols], use_container_width=True, hide_index=True)

    if phase3_improvement_df is not None and not phase3_improvement_df.empty:
        st.markdown("#### CatBoost phase 2 vs phase 3")
        st.dataframe(phase3_improvement_df, use_container_width=True, hide_index=True)

        plot_df = phase3_improvement_df[phase3_improvement_df["metric"].isin(["MAE", "RMSE", "R2"])].copy()
        if not plot_df.empty:
            plot_long = plot_df.melt(
                id_vars=["metric"],
                value_vars=["phase2_catboost", "phase3_tuned_catboost"],
                var_name="Version",
                value_name="Value",
            )
            fig_phase3 = px.bar(
                plot_long,
                x="metric",
                y="Value",
                color="Version",
                barmode="group",
                title="CatBoost improvement after tuning",
            )
            fig_phase3.update_layout(height=350)
            st.plotly_chart(fig_phase3, use_container_width=True)

    if phase3_shap_df is not None and not phase3_shap_df.empty:
        st.markdown("#### Explainable AI: SHAP")
        shap_plot = phase3_shap_df.head(10).copy()
        fig_shap = px.bar(
            shap_plot.iloc[::-1],
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Top SHAP drivers of PM2.5 prediction",
        )
        fig_shap.update_layout(height=420, yaxis_title="")
        st.plotly_chart(fig_shap, use_container_width=True)

    if phase3_seasonal_df is not None and not phase3_seasonal_df.empty:
        st.markdown("#### Seasonal stability")
        st.dataframe(phase3_seasonal_df, use_container_width=True, hide_index=True)
        if "season" in phase3_seasonal_df.columns and "RMSE" in phase3_seasonal_df.columns:
            fig_season = px.bar(
                phase3_seasonal_df,
                x="season",
                y="RMSE",
                title="Out-of-fold RMSE by season",
                text_auto=".2f",
            )
            fig_season.update_layout(height=350)
            st.plotly_chart(fig_season, use_container_width=True)

    if supervised_df is not None and not supervised_df.empty:
        st.markdown("#### Supervised comparison")
        st.dataframe(supervised_df, use_container_width=True, hide_index=True)

        if "model" in supervised_df.columns and "R2" in supervised_df.columns:
            fig_sup = px.bar(
                supervised_df,
                x="model",
                y="R2",
                title="Supervised model comparison by R²",
                text_auto=".4f",
            )
            fig_sup.update_layout(height=350)
            st.plotly_chart(fig_sup, use_container_width=True)

    if unsupervised_df is not None and not unsupervised_df.empty:
        st.markdown("#### Unsupervised comparison")
        st.dataframe(unsupervised_df, use_container_width=True, hide_index=True)

    if backtest_df is not None and not backtest_df.empty:
        st.markdown("#### CatBoost backtest")
        fig_backtest = go.Figure()
        fig_backtest.add_trace(
            go.Scatter(
                x=backtest_df["timestamp"],
                y=backtest_df["actual_plot"],
                mode="lines",
                name="Actual",
            )
        )
        fig_backtest.add_trace(
            go.Scatter(
                x=backtest_df["timestamp"],
                y=backtest_df["pred_plot"],
                mode="lines",
                name="Predicted",
            )
        )
        fig_backtest.update_layout(height=380, title="CatBoost observed vs predicted")
        st.plotly_chart(fig_backtest, use_container_width=True)

    if catboost_importance_df is not None and not catboost_importance_df.empty:
        fig_imp = build_feature_importance_chart(catboost_importance_df)
        if fig_imp is not None:
            st.plotly_chart(fig_imp, use_container_width=True)

    if catboost_metrics_df is not None and not catboost_metrics_df.empty:
        st.markdown("#### Saved CatBoost metrics")
        st.dataframe(catboost_metrics_df, use_container_width=True, hide_index=True)
