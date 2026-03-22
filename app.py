from pathlib import Path
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Prishtina Live Pollution Simulator",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "2D_validated_final_dataset.csv"

POLLUTANTS = ["co", "no2", "o3", "pm10", "pm25", "so2"]

ENERGY_COLS = ["A3_MW", "A4_MW", "A5_MW", "B1_MW", "B2_MW"]

WEATHER_COLS = [
    "temperature_2m (°C)",
    "rain (mm)",
    "snowfall (cm)",
    "relative_humidity_2m (%)",
    "wind_direction_10m (°)",
    "wind_speed_10m (km/h)",
]

RAW_CONTROL_COLS = ENERGY_COLS + WEATHER_COLS

MODEL_FEATURES = [
    *ENERGY_COLS,
    "total_generation_mw",
    "temperature_2m (°C)",
    "rain (mm)",
    "snowfall (cm)",
    "relative_humidity_2m (%)",
    "wind_direction_10m (°)",
    "wind_speed_10m (km/h)",
    "day_of_week",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "temp_wind_interact",
    "generation_humidity_interact",
    "pollution_stagnation_index",
    "wind_x_vector",
    "wind_y_vector",
]

DISPLAY_NAMES = {
    "A3_MW": "A3 (MW)",
    "A4_MW": "A4 (MW)",
    "A5_MW": "A5 (MW)",
    "B1_MW": "B1 (MW)",
    "B2_MW": "B2 (MW)",
    "temperature_2m (°C)": "Temperature (°C)",
    "rain (mm)": "Rain (mm)",
    "snowfall (cm)": "Snowfall (cm)",
    "relative_humidity_2m (%)": "Relative Humidity (%)",
    "wind_direction_10m (°)": "Wind Direction (°)",
    "wind_speed_10m (km/h)": "Wind Speed (km/h)",
    "pm25": "PM2.5",
    "pm10": "PM10",
    "co": "CO",
    "no2": "NO2",
    "o3": "O3",
    "so2": "SO2",
}

SESSION_KEYS = {
    "A3_MW": "a3_mw",
    "A4_MW": "a4_mw",
    "A5_MW": "a5_mw",
    "B1_MW": "b1_mw",
    "B2_MW": "b2_mw",
    "temperature_2m (°C)": "temperature",
    "rain (mm)": "rain",
    "snowfall (cm)": "snowfall",
    "relative_humidity_2m (%)": "humidity",
    "wind_direction_10m (°)": "wind_direction",
    "wind_speed_10m (km/h)": "wind_speed",
}

def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    df = safe_numeric(df, RAW_CONTROL_COLS + POLLUTANTS)

    df["total_generation_mw"] = df[ENERGY_COLS].sum(axis=1)

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["temp_wind_interact"] = df["temperature_2m (°C)"] * df["wind_speed_10m (km/h)"]
    df["generation_humidity_interact"] = (
        df["total_generation_mw"] * df["relative_humidity_2m (%)"]
    )

    df["pollution_stagnation_index"] = (
        df["total_generation_mw"] / (df["wind_speed_10m (km/h)"] + 0.1)
    )

    wd_rad = np.deg2rad(df["wind_direction_10m (°)"])
    wv = df["wind_speed_10m (km/h)"]
    df["wind_x_vector"] = wv * np.cos(wd_rad)
    df["wind_y_vector"] = wv * np.sin(wd_rad)

    return df


@st.cache_data(show_spinner=False)
def load_training_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["datetime", *RAW_CONTROL_COLS, *POLLUTANTS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Mungojnë kolonat e nevojshme në dataset: " + ", ".join(missing)
        )

    df = add_engineered_features(df)
    df = df.dropna(subset=MODEL_FEATURES + POLLUTANTS).reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=True)
def train_model(path: str):
    df = load_training_frame(path)

    # Time-aware split: first 80% train, last 20% test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[MODEL_FEATURES]
    y_train = train_df[POLLUTANTS]

    X_test = test_df[MODEL_FEATURES]
    y_test = test_df[POLLUTANTS]

    model = ExtraTreesRegressor(
        n_estimators=350,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    pred_test = pd.DataFrame(
        model.predict(X_test),
        columns=POLLUTANTS,
        index=y_test.index,
    )

    metrics = []
    for pollutant in POLLUTANTS:
        y_true = y_test[pollutant].values
        y_pred = pred_test[pollutant].values

        metrics.append(
            {
                "Pollutant": DISPLAY_NAMES.get(pollutant, pollutant),
                "R2": round(r2_score(y_true, y_pred), 4),
                "MAE": round(mean_absolute_error(y_true, y_pred), 4),
                "Actual Mean": round(np.mean(y_true), 4),
                "Predicted Mean": round(np.mean(y_pred), 4),
            }
        )

    metrics_df = pd.DataFrame(metrics)
    return model, metrics_df


def build_feature_row(
    selected_date,
    selected_hour: int,
    raw_inputs: dict,
) -> pd.DataFrame:
    ts = pd.Timestamp(selected_date) + pd.Timedelta(hours=int(selected_hour))

    total_generation_mw = sum(raw_inputs[col] for col in ENERGY_COLS)

    hour_sin = np.sin(2 * np.pi * selected_hour / 24)
    hour_cos = np.cos(2 * np.pi * selected_hour / 24)
    month_sin = np.sin(2 * np.pi * ts.month / 12)
    month_cos = np.cos(2 * np.pi * ts.month / 12)

    temp = raw_inputs["temperature_2m (°C)"]
    rain = raw_inputs["rain (mm)"]
    snowfall = raw_inputs["snowfall (cm)"]
    humidity = raw_inputs["relative_humidity_2m (%)"]
    wind_dir = raw_inputs["wind_direction_10m (°)"]
    wind_speed = raw_inputs["wind_speed_10m (km/h)"]

    temp_wind_interact = temp * wind_speed
    generation_humidity_interact = total_generation_mw * humidity
    pollution_stagnation_index = total_generation_mw / (wind_speed + 0.1)

    wd_rad = np.deg2rad(wind_dir)
    wind_x_vector = wind_speed * np.cos(wd_rad)
    wind_y_vector = wind_speed * np.sin(wd_rad)

    row = {
        "A3_MW": raw_inputs["A3_MW"],
        "A4_MW": raw_inputs["A4_MW"],
        "A5_MW": raw_inputs["A5_MW"],
        "B1_MW": raw_inputs["B1_MW"],
        "B2_MW": raw_inputs["B2_MW"],
        "total_generation_mw": total_generation_mw,
        "temperature_2m (°C)": temp,
        "rain (mm)": rain,
        "snowfall (cm)": snowfall,
        "relative_humidity_2m (%)": humidity,
        "wind_direction_10m (°)": wind_dir,
        "wind_speed_10m (km/h)": wind_speed,
        "day_of_week": ts.dayofweek,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "temp_wind_interact": temp_wind_interact,
        "generation_humidity_interact": generation_humidity_interact,
        "pollution_stagnation_index": pollution_stagnation_index,
        "wind_x_vector": wind_x_vector,
        "wind_y_vector": wind_y_vector,
    }

    return pd.DataFrame([row], columns=MODEL_FEATURES)


def predict_scenario(model, feature_df: pd.DataFrame) -> pd.Series:
    pred = model.predict(feature_df)[0]
    pred = np.clip(pred, a_min=0, a_max=None)
    return pd.Series(pred, index=POLLUTANTS)


def get_ranges(df: pd.DataFrame) -> dict:
    ranges = {}

    for col in RAW_CONTROL_COLS:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        if col in ENERGY_COLS:
            lo = 0.0
            hi = max(float(s.quantile(0.995)), float(s.max()))
            default = float(s.median())
            step = 1.0

        elif col == "temperature_2m (°C)":
            lo = float(np.floor(s.quantile(0.01) - 3))
            hi = float(np.ceil(s.quantile(0.99) + 3))
            default = float(s.median())
            step = 0.5

        elif col == "rain (mm)":
            lo = 0.0
            hi = float(np.ceil(max(s.quantile(0.995), 5)))
            default = float(s.median())
            step = 0.1

        elif col == "snowfall (cm)":
            lo = 0.0
            hi = float(np.ceil(max(s.quantile(0.995), 2)))
            default = float(s.median())
            step = 0.1

        elif col == "relative_humidity_2m (%)":
            lo = 0.0
            hi = 100.0
            default = float(s.median())
            step = 1.0

        elif col == "wind_direction_10m (°)":
            lo = 0
            hi = 359
            default = int(round(s.median()))
            step = 1

        elif col == "wind_speed_10m (km/h)":
            lo = 0.0
            hi = float(np.ceil(max(s.quantile(0.995), 20)))
            default = float(s.median())
            step = 0.5

        else:
            lo = float(s.min())
            hi = float(s.max())
            default = float(s.median())
            step = 0.1

        ranges[col] = {
            "min": lo,
            "max": hi,
            "default": default,
            "step": step,
        }

    return ranges


def initialize_session_state(ranges: dict, df: pd.DataFrame):
    latest_date = df["datetime"].max().date()

    if "scenario_date" not in st.session_state:
        st.session_state.scenario_date = latest_date

    if "scenario_hour" not in st.session_state:
        st.session_state.scenario_hour = 12

    if "selected_target" not in st.session_state:
        st.session_state.selected_target = "pm25"

    if "sensitivity_feature" not in st.session_state:
        st.session_state.sensitivity_feature = "wind_speed_10m (km/h)"

    for col, key in SESSION_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = ranges[col]["default"]


def apply_preset(preset_name: str, df: pd.DataFrame, ranges: dict):
    q = df[RAW_CONTROL_COLS].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()

    def qv(col, quant):
        return float(q[float(quant)][col])

    if preset_name == "Winter stagnation":
        values = {
            "A3_MW": qv("A3_MW", 0.75),
            "A4_MW": qv("A4_MW", 0.75),
            "A5_MW": qv("A5_MW", 0.60),
            "B1_MW": qv("B1_MW", 0.75),
            "B2_MW": qv("B2_MW", 0.75),
            "temperature_2m (°C)": qv("temperature_2m (°C)", 0.10),
            "rain (mm)": 0.0,
            "snowfall (cm)": qv("snowfall (cm)", 0.50),
            "relative_humidity_2m (%)": qv("relative_humidity_2m (%)", 0.90),
            "wind_direction_10m (°)": qv("wind_direction_10m (°)", 0.50),
            "wind_speed_10m (km/h)": max(qv("wind_speed_10m (km/h)", 0.10), 0.5),
        }
        st.session_state.scenario_hour = 21

    elif preset_name == "Windy clean day":
        values = {
            "A3_MW": qv("A3_MW", 0.40),
            "A4_MW": qv("A4_MW", 0.40),
            "A5_MW": qv("A5_MW", 0.30),
            "B1_MW": qv("B1_MW", 0.40),
            "B2_MW": qv("B2_MW", 0.40),
            "temperature_2m (°C)": qv("temperature_2m (°C)", 0.75),
            "rain (mm)": qv("rain (mm)", 0.10),
            "snowfall (cm)": 0.0,
            "relative_humidity_2m (%)": qv("relative_humidity_2m (%)", 0.40),
            "wind_direction_10m (°)": qv("wind_direction_10m (°)", 0.50),
            "wind_speed_10m (km/h)": qv("wind_speed_10m (km/h)", 0.90),
        }
        st.session_state.scenario_hour = 14

    elif preset_name == "High generation night":
        values = {
            "A3_MW": qv("A3_MW", 0.90),
            "A4_MW": qv("A4_MW", 0.90),
            "A5_MW": qv("A5_MW", 0.75),
            "B1_MW": qv("B1_MW", 0.90),
            "B2_MW": qv("B2_MW", 0.90),
            "temperature_2m (°C)": qv("temperature_2m (°C)", 0.25),
            "rain (mm)": 0.0,
            "snowfall (cm)": 0.0,
            "relative_humidity_2m (%)": qv("relative_humidity_2m (%)", 0.75),
            "wind_direction_10m (°)": qv("wind_direction_10m (°)", 0.50),
            "wind_speed_10m (km/h)": qv("wind_speed_10m (km/h)", 0.20),
        }
        st.session_state.scenario_hour = 23

    elif preset_name == "Rain washout":
        values = {
            "A3_MW": qv("A3_MW", 0.60),
            "A4_MW": qv("A4_MW", 0.60),
            "A5_MW": qv("A5_MW", 0.50),
            "B1_MW": qv("B1_MW", 0.60),
            "B2_MW": qv("B2_MW", 0.60),
            "temperature_2m (°C)": qv("temperature_2m (°C)", 0.50),
            "rain (mm)": max(qv("rain (mm)", 0.90), 1.0),
            "snowfall (cm)": 0.0,
            "relative_humidity_2m (%)": qv("relative_humidity_2m (%)", 0.90),
            "wind_direction_10m (°)": qv("wind_direction_10m (°)", 0.50),
            "wind_speed_10m (km/h)": qv("wind_speed_10m (km/h)", 0.60),
        }
        st.session_state.scenario_hour = 10

    else:
        return

    for col, value in values.items():
        key = SESSION_KEYS[col]
        lo = ranges[col]["min"]
        hi = ranges[col]["max"]
        st.session_state[key] = max(lo, min(hi, value))


def get_current_raw_inputs() -> dict:
    return {
        col: float(st.session_state[SESSION_KEYS[col]])
        for col in RAW_CONTROL_COLS
    }


def get_baseline_inputs(df: pd.DataFrame, selected_date, selected_hour: int) -> dict:
    month = pd.Timestamp(selected_date).month

    subset = df[
        (df["datetime"].dt.month == month) &
        (df["datetime"].dt.hour == selected_hour)
    ].copy()

    if len(subset) < 80:
        subset = df[df["datetime"].dt.hour == selected_hour].copy()

    if len(subset) < 80:
        subset = df.copy()

    baseline = {col: float(subset[col].median()) for col in RAW_CONTROL_COLS}
    return baseline


def make_gauge(value: float, reference: float, pollutant: str, history: pd.Series) -> go.Figure:
    q25 = float(history.quantile(0.25))
    q50 = float(history.quantile(0.50))
    q75 = float(history.quantile(0.75))
    q99 = float(history.quantile(0.99))
    max_range = max(q99, value * 1.15, reference * 1.15, 1.0)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=float(value),
            delta={"reference": float(reference), "relative": False},
            title={"text": DISPLAY_NAMES.get(pollutant, pollutant)},
            gauge={
                "axis": {"range": [0, max_range]},
                "bar": {"thickness": 0.35},
                "steps": [
                    {"range": [0, q25], "color": "#d8f3dc"},
                    {"range": [q25, q50], "color": "#ffe8a1"},
                    {"range": [q50, q75], "color": "#ffd6a5"},
                    {"range": [q75, max_range], "color": "#ffadad"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.85,
                    "value": float(value),
                },
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def make_pollutant_bar(pred_current: pd.Series, pred_baseline: pd.Series) -> go.Figure:
    df_plot = pd.DataFrame({
        "Pollutant": [DISPLAY_NAMES.get(p, p) for p in POLLUTANTS] * 2,
        "Value": list(pred_baseline.values) + list(pred_current.values),
        "Scenario": ["Baseline"] * len(POLLUTANTS) + ["Current scenario"] * len(POLLUTANTS),
    })

    fig = px.bar(
        df_plot,
        x="Pollutant",
        y="Value",
        color="Scenario",
        barmode="group",
        title="Baseline vs Current Scenario",
        text_auto=".2f",
    )
    fig.update_layout(height=420)
    return fig


def make_distribution_chart(history: pd.Series, current_value: float, baseline_value: float, pollutant: str) -> go.Figure:
    fig = px.histogram(
        x=history,
        nbins=50,
        title=f"Historical Distribution of {DISPLAY_NAMES.get(pollutant, pollutant)}",
    )
    fig.add_vline(
        x=float(baseline_value),
        line_dash="dash",
        annotation_text="Baseline",
        annotation_position="top",
    )
    fig.add_vline(
        x=float(current_value),
        line_dash="solid",
        annotation_text="Scenario",
        annotation_position="top",
    )
    fig.update_layout(
        xaxis_title=DISPLAY_NAMES.get(pollutant, pollutant),
        yaxis_title="Frequency",
        height=380,
    )
    return fig


def make_daily_history_chart(df: pd.DataFrame, pollutant: str, current_value: float) -> go.Figure:
    daily = (
        df.set_index("datetime")[pollutant]
        .resample("D")
        .mean()
        .reset_index()
    )

    fig = px.line(
        daily,
        x="datetime",
        y=pollutant,
        title=f"Daily Average Historical {DISPLAY_NAMES.get(pollutant, pollutant)}",
    )
    fig.add_hline(
        y=float(current_value),
        line_dash="dash",
        annotation_text="Scenario prediction",
        annotation_position="top left",
    )
    fig.update_layout(height=380, xaxis_title="Date", yaxis_title=DISPLAY_NAMES.get(pollutant, pollutant))
    return fig


def make_sensitivity_curve(
    model,
    selected_date,
    selected_hour: int,
    current_raw: dict,
    feature_name: str,
    feature_min: float,
    feature_max: float,
    pollutant: str,
    n_points: int = 40,
) -> go.Figure:
    sweep_values = np.linspace(feature_min, feature_max, n_points)
    rows = []

    for value in sweep_values:
        raw_copy = current_raw.copy()
        raw_copy[feature_name] = float(value)

        # Keep wind direction within valid range
        if feature_name == "wind_direction_10m (°)":
            raw_copy[feature_name] = float(value % 360)

        feature_row = build_feature_row(selected_date, selected_hour, raw_copy)
        pred = predict_scenario(model, feature_row)

        rows.append({
            feature_name: float(value),
            pollutant: float(pred[pollutant]),
        })

    sweep_df = pd.DataFrame(rows)

    fig = px.line(
        sweep_df,
        x=feature_name,
        y=pollutant,
        title=f"Sensitivity of {DISPLAY_NAMES.get(pollutant, pollutant)} to {DISPLAY_NAMES.get(feature_name, feature_name)}",
    )
    fig.add_vline(
        x=float(current_raw[feature_name]),
        line_dash="dash",
        annotation_text="Current value",
        annotation_position="top",
    )
    fig.update_layout(
        height=420,
        xaxis_title=DISPLAY_NAMES.get(feature_name, feature_name),
        yaxis_title=DISPLAY_NAMES.get(pollutant, pollutant),
    )
    return fig


def render_metric_cards(pred_current: pd.Series, pred_baseline: pd.Series):
    cols = st.columns(len(POLLUTANTS))
    for i, pollutant in enumerate(POLLUTANTS):
        delta = float(pred_current[pollutant] - pred_baseline[pollutant])
        cols[i].metric(
            DISPLAY_NAMES.get(pollutant, pollutant),
            f"{pred_current[pollutant]:.2f}",
            f"{delta:+.2f}",
        )


st.title("Prishtina Live Pollution Simulator")
st.caption(
    "Interactive scenario explorer for electricity generation, weather conditions and live pollution response."
)

if not DATA_PATH.exists():
    st.error(
        f"Nuk u gjet dataset-i te kjo rrugë: {DATA_PATH}\n\n"
        "Vendose `app.py` në root të projektit dhe sigurohu që ekziston `data/2D_validated_final_dataset.csv`."
    )
    st.stop()

try:
    training_df = load_training_frame(str(DATA_PATH))
    model, metrics_df = train_model(str(DATA_PATH))
except Exception as e:
    st.exception(e)
    st.stop()

ranges = get_ranges(training_df)
initialize_session_state(ranges, training_df)

st.sidebar.header("Scenario Controls")

preset_name = st.sidebar.selectbox(
    "Preset scenario",
    ["Custom", "Winter stagnation", "Windy clean day", "High generation night", "Rain washout"],
)

if st.sidebar.button("Load preset"):
    if preset_name != "Custom":
        apply_preset(preset_name, training_df, ranges)
        st.rerun()

st.sidebar.markdown("---")

date_min = training_df["datetime"].min().date()
date_max = training_df["datetime"].max().date()

st.sidebar.date_input(
    "Scenario date",
    min_value=date_min,
    max_value=date_max,
    key="scenario_date",
)

st.sidebar.slider(
    "Scenario hour",
    min_value=0,
    max_value=23,
    key="scenario_hour",
)

st.sidebar.markdown("### Power Generation")

for col in ENERGY_COLS:
    cfg = ranges[col]
    st.sidebar.slider(
        DISPLAY_NAMES[col],
        min_value=float(cfg["min"]),
        max_value=float(cfg["max"]),
        step=float(cfg["step"]),
        key=SESSION_KEYS[col],
    )

st.sidebar.markdown("### Weather Conditions")

for col in WEATHER_COLS:
    cfg = ranges[col]

    if col == "wind_direction_10m (°)":
        st.sidebar.slider(
            DISPLAY_NAMES[col],
            min_value=int(cfg["min"]),
            max_value=int(cfg["max"]),
            step=int(cfg["step"]),
            key=SESSION_KEYS[col],
        )
    else:
        st.sidebar.slider(
            DISPLAY_NAMES[col],
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            step=float(cfg["step"]),
            key=SESSION_KEYS[col],
        )

st.sidebar.markdown("---")

st.sidebar.selectbox(
    "Main displayed pollutant",
    options=POLLUTANTS,
    format_func=lambda x: DISPLAY_NAMES.get(x, x),
    key="selected_target",
)

selected_date = st.session_state.scenario_date
selected_hour = int(st.session_state.scenario_hour)
selected_target = st.session_state.selected_target

current_raw = get_current_raw_inputs()
baseline_raw = get_baseline_inputs(training_df, selected_date, selected_hour)

current_features = build_feature_row(selected_date, selected_hour, current_raw)
baseline_features = build_feature_row(selected_date, selected_hour, baseline_raw)

pred_current = predict_scenario(model, current_features)
pred_baseline = predict_scenario(model, baseline_features)

render_metric_cards(pred_current, pred_baseline)

with st.expander("Model quality on holdout period"):
    st.dataframe(metrics_df, use_container_width=True)

tab1, tab2, tab3 = st.tabs(
    ["Live Simulator", "Sensitivity Explorer", "Historical Context"]
)

with tab1:
    left, right = st.columns([1, 1])

    with left:
        st.plotly_chart(
            make_gauge(
                value=float(pred_current[selected_target]),
                reference=float(pred_baseline[selected_target]),
                pollutant=selected_target,
                history=training_df[selected_target],
            ),
            use_container_width=True,
        )

    with right:
        st.plotly_chart(
            make_pollutant_bar(pred_current, pred_baseline),
            use_container_width=True,
        )

    st.markdown("### Scenario summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.write(
        f"""
**Date/Hour**
- Date: `{selected_date}`
- Hour: `{selected_hour:02d}:00`
- Day of week: `{pd.Timestamp(selected_date).day_name()}`
        """
    )

    summary_col2.write(
        f"""
**Generation**
- Total generation: `{sum(current_raw[c] for c in ENERGY_COLS):.2f} MW`
- A units total: `{current_raw['A3_MW'] + current_raw['A4_MW'] + current_raw['A5_MW']:.2f} MW`
- B units total: `{current_raw['B1_MW'] + current_raw['B2_MW']:.2f} MW`
        """
    )

    summary_col3.write(
        f"""
**Weather**
- Temp: `{current_raw['temperature_2m (°C)']:.2f} °C`
- Humidity: `{current_raw['relative_humidity_2m (%)']:.2f} %`
- Wind: `{current_raw['wind_speed_10m (km/h)']:.2f} km/h`
- Rain: `{current_raw['rain (mm)']:.2f} mm`
        """
    )

    st.markdown("### Animated transition")
    st.caption("Animates the model response from the historical baseline to your custom scenario.")

    animate = st.button("Play transition animation")

    gauge_placeholder = st.empty()
    bar_placeholder = st.empty()
    line_placeholder = st.empty()

    if animate:
        frames = 30
        progress = st.progress(0)
        transition_rows = []

        for i in range(frames):
            alpha = i / (frames - 1)

            raw_interp = {}
            for col in RAW_CONTROL_COLS:
                start_val = float(baseline_raw[col])
                end_val = float(current_raw[col])
                val = start_val + alpha * (end_val - start_val)

                if col == "wind_direction_10m (°)":
                    val = val % 360

                raw_interp[col] = float(val)

            interp_features = build_feature_row(selected_date, selected_hour, raw_interp)
            interp_pred = predict_scenario(model, interp_features)

            transition_rows.append(
                {
                    "Frame": i + 1,
                    selected_target: float(interp_pred[selected_target]),
                }
            )

            gauge_placeholder.plotly_chart(
                make_gauge(
                    value=float(interp_pred[selected_target]),
                    reference=float(pred_baseline[selected_target]),
                    pollutant=selected_target,
                    history=training_df[selected_target],
                ),
                use_container_width=True,
            )

            bar_placeholder.plotly_chart(
                make_pollutant_bar(interp_pred, pred_baseline),
                use_container_width=True,
            )

            transition_df = pd.DataFrame(transition_rows)
            line_fig = px.line(
                transition_df,
                x="Frame",
                y=selected_target,
                markers=True,
                title=f"Animated path of {DISPLAY_NAMES[selected_target]}",
            )
            line_fig.update_layout(height=300)
            line_placeholder.plotly_chart(line_fig, use_container_width=True)

            progress.progress((i + 1) / frames)
            time.sleep(0.05)

with tab2:
    st.markdown("### Sensitivity explorer")
    st.caption("Change one variable across its full range while keeping all others fixed.")

    sens_col1, sens_col2 = st.columns([1, 1])

    with sens_col1:
        feature_name = st.selectbox(
            "Variable to sweep",
            options=RAW_CONTROL_COLS,
            format_func=lambda x: DISPLAY_NAMES.get(x, x),
            index=RAW_CONTROL_COLS.index(st.session_state.sensitivity_feature)
            if st.session_state.sensitivity_feature in RAW_CONTROL_COLS else 0,
        )

    with sens_col2:
        n_points = st.slider("Number of points", 20, 80, 40)

    st.session_state.sensitivity_feature = feature_name

    fig_sensitivity = make_sensitivity_curve(
        model=model,
        selected_date=selected_date,
        selected_hour=selected_hour,
        current_raw=current_raw,
        feature_name=feature_name,
        feature_min=float(ranges[feature_name]["min"]),
        feature_max=float(ranges[feature_name]["max"]),
        pollutant=selected_target,
        n_points=n_points,
    )
    st.plotly_chart(fig_sensitivity, use_container_width=True)

    st.markdown("### Current scenario inputs")
    st.dataframe(
        pd.DataFrame(
            {
                "Feature": [DISPLAY_NAMES.get(c, c) for c in RAW_CONTROL_COLS],
                "Value": [current_raw[c] for c in RAW_CONTROL_COLS],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    hist_col1, hist_col2 = st.columns(2)

    with hist_col1:
        st.plotly_chart(
            make_distribution_chart(
                history=training_df[selected_target],
                current_value=float(pred_current[selected_target]),
                baseline_value=float(pred_baseline[selected_target]),
                pollutant=selected_target,
            ),
            use_container_width=True,
        )

    with hist_col2:
        st.plotly_chart(
            make_daily_history_chart(
                df=training_df,
                pollutant=selected_target,
                current_value=float(pred_current[selected_target]),
            ),
            use_container_width=True,
        )

    st.markdown("### Latest real observations")
    latest_real = training_df[["datetime", *POLLUTANTS]].tail(10).copy()
    latest_real.columns = [
        "datetime",
        "CO",
        "NO2",
        "O3",
        "PM10",
        "PM2.5",
        "SO2",
    ]
    st.dataframe(latest_real, use_container_width=True)