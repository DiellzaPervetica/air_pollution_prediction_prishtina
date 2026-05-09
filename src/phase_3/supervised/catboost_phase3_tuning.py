from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = PROJECT_ROOT / "data" / "phase_1" / "4E_selected_dataset.csv"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

PHASE2_METRICS_PATH = PROJECT_ROOT / "data" / "phase_2" / "supervised" / "catboost" / "catboost_metrics.csv"

DATA_DIR = PROJECT_ROOT / "data" / "phase_3" / "supervised" / "catboost_tuned"
PLOTS_DIR = PROJECT_ROOT / "pictures" / "phase_3" / "supervised" / "catboost_tuned"
MODEL_DIR = PROJECT_ROOT / "models" / "phase_3" / "catboost_tuned"

for directory in [DATA_DIR, PLOTS_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

TARGET = "pm25"
TIME_COLS = ["datetime", "date", "timestamp"]
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

PHASE2_REFERENCE = {
    "candidate": "phase2_reference",
    "iterations": 600,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_strength": 1,
    "bagging_temperature": 1.0,
    "early_stopping_rounds": 50,
}

TUNING_CANDIDATES = [
    PHASE2_REFERENCE,
    {
        "candidate": "regularized_depth5",
        "iterations": 900,
        "learning_rate": 0.025,
        "depth": 5,
        "l2_leaf_reg": 5,
        "random_strength": 1,
        "bagging_temperature": 0.5,
        "early_stopping_rounds": 80,
    },
    {
        "candidate": "slow_regularized_depth5",
        "iterations": 1200,
        "learning_rate": 0.02,
        "depth": 5,
        "l2_leaf_reg": 7,
        "random_strength": 1.5,
        "bagging_temperature": 0.7,
        "early_stopping_rounds": 100,
    },
    {
        "candidate": "compact_depth4",
        "iterations": 900,
        "learning_rate": 0.03,
        "depth": 4,
        "l2_leaf_reg": 5,
        "random_strength": 1,
        "bagging_temperature": 0.5,
        "early_stopping_rounds": 80,
    },
    {
        "candidate": "strong_regularized_depth6",
        "iterations": 1200,
        "learning_rate": 0.02,
        "depth": 6,
        "l2_leaf_reg": 10,
        "random_strength": 2,
        "bagging_temperature": 0.8,
        "early_stopping_rounds": 100,
    },
    {
        "candidate": "compact_strong_regularization",
        "iterations": 1200,
        "learning_rate": 0.02,
        "depth": 4,
        "l2_leaf_reg": 10,
        "random_strength": 2,
        "bagging_temperature": 0.8,
        "early_stopping_rounds": 100,
    },
    {
        "candidate": "best_validation_candidate",
        "iterations": 1500,
        "learning_rate": 0.015,
        "depth": 5,
        "l2_leaf_reg": 12,
        "random_strength": 2,
        "bagging_temperature": 0.8,
        "early_stopping_rounds": 120,
    },
]


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def detect_time_column(df: pd.DataFrame) -> str:
    for column in TIME_COLS:
        if column in df.columns:
            return column
    raise ValueError(f"No time column found. Tried: {TIME_COLS}")


def inverse_scale_pm25(values: np.ndarray | pd.Series, scaler: object) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    feature_names = list(getattr(scaler, "feature_names_in_", []))
    if TARGET not in feature_names:
        return values

    idx = feature_names.index(TARGET)
    log_values = values * scaler.scale_[idx] + scaler.mean_[idx]
    return np.expm1(log_values)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    smape_denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, 1e-8)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_pct": float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0),
        "SMAPE_pct": float(np.mean(np.abs(y_true - y_pred) / smape_denom) * 100.0),
        "R2": float(r2_score(y_true, y_pred)),
        "n_eval_points": int(len(y_true)),
    }


def prepare_data() -> tuple[pd.DataFrame, list[str], object]:
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(INPUT_PATH)
    time_col = detect_time_column(df)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).drop_duplicates(time_col).reset_index(drop=True)
    df["timestamp"] = df[time_col]

    for lag in [1, 24]:
        lag_col = f"{TARGET}_lag_{lag}"
        if lag_col not in df.columns:
            df[lag_col] = df[TARGET].shift(lag)

    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != TARGET and not col.endswith("_was_missing")]
    df[[TARGET] + feature_cols] = df[[TARGET] + feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET] + feature_cols).reset_index(drop=True)

    return df, feature_cols, scaler


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_rows = len(df)
    train_end = int(n_rows * TRAIN_RATIO)
    val_end = int(n_rows * (TRAIN_RATIO + VAL_RATIO))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def build_model(params: dict[str, object]) -> CatBoostRegressor:
    model_params = {
        "iterations": params["iterations"],
        "learning_rate": params["learning_rate"],
        "depth": params["depth"],
        "l2_leaf_reg": params["l2_leaf_reg"],
        "random_strength": params["random_strength"],
        "bagging_temperature": params["bagging_temperature"],
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "allow_writing_files": False,
    }
    return CatBoostRegressor(**model_params)


def evaluate_model(
    model: CatBoostRegressor,
    frame: pd.DataFrame,
    feature_cols: list[str],
    scaler: object,
    label: str,
) -> dict[str, float | str]:
    pred_scaled = model.predict(frame[feature_cols])
    y_real = inverse_scale_pm25(frame[TARGET].values, scaler)
    pred_real = inverse_scale_pm25(pred_scaled, scaler)
    metrics = calculate_metrics(y_real, pred_real)
    return {f"{label}_{key}": value for key, value in metrics.items()}


def save_tuning_plot(results_df: pd.DataFrame) -> None:
    plot_df = results_df.sort_values("validation_RMSE")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=plot_df, x="candidate", y="validation_RMSE", ax=axes[0], color="#4C78A8")
    axes[0].set_title("Validation RMSE by candidate")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=25)

    sns.barplot(data=plot_df, x="candidate", y="test_RMSE", ax=axes[1], color="#72B7B2")
    axes[1].set_title("Holdout test RMSE by candidate")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=25)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "catboost_tuning_candidates.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_actual_vs_predicted(
    test_df: pd.DataFrame,
    pred_real: np.ndarray,
    scaler: object,
) -> pd.DataFrame:
    actual_real = inverse_scale_pm25(test_df[TARGET].values, scaler)
    out = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
        "actual_pm25": actual_real,
        "pred_pm25": pred_real,
        "residual_pm25": actual_real - pred_real,
    })
    out.to_csv(DATA_DIR / "catboost_tuned_forecasts.csv", index=False)

    plot_df = out.tail(min(500, len(out))).copy()
    plt.figure(figsize=(14, 5))
    plt.plot(plot_df["timestamp"], plot_df["actual_pm25"], label="Actual", linewidth=1.7)
    plt.plot(plot_df["timestamp"], plot_df["pred_pm25"], label="Tuned CatBoost", linestyle="--", linewidth=1.7)
    plt.title("Phase 3 tuned CatBoost actual vs predicted PM2.5")
    plt.xlabel("Time")
    plt.ylabel("PM2.5")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "catboost_tuned_actual_vs_predicted.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(out["residual_pm25"], bins=40, color="#4C78A8", alpha=0.85)
    plt.axvline(out["residual_pm25"].mean(), color="#D62728", linestyle="--", linewidth=2)
    plt.title("Tuned CatBoost residual distribution")
    plt.xlabel("Residual PM2.5")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "catboost_tuned_residual_diagnostics.png", dpi=300)
    plt.close()

    return out


def save_shap_outputs(model: CatBoostRegressor, test_df: pd.DataFrame, feature_cols: list[str]) -> None:
    sample = test_df[feature_cols].sample(n=min(1200, len(test_df)), random_state=RANDOM_STATE)
    shap_values_full = model.get_feature_importance(data=Pool(sample, feature_names=feature_cols), type="ShapValues")
    shap_values = shap_values_full[:, :-1]

    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df["direction_simple"] = np.where(
        shap_df["mean_shap"] >= 0,
        "higher values tend to increase prediction",
        "higher values tend to decrease prediction",
    )
    shap_df.to_csv(DATA_DIR / "catboost_tuned_shap_global_importance.csv", index=False)

    top_df = shap_df.head(12).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top_df["feature"], top_df["mean_abs_shap"], color="#59A14F")
    plt.title("SHAP global importance for tuned CatBoost")
    plt.xlabel("Mean absolute SHAP value")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "catboost_tuned_shap_global_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    top_features = shap_df.head(8)["feature"].tolist()
    long_rows = []
    for feature in top_features:
        idx = feature_cols.index(feature)
        feature_values = sample[feature].to_numpy()
        shap_feature_values = shap_values[:, idx]
        long_rows.append(pd.DataFrame({
            "feature": feature,
            "feature_value": feature_values,
            "shap_value": shap_feature_values,
        }))
    long_df = pd.concat(long_rows, ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.stripplot(
        data=long_df,
        x="shap_value",
        y="feature",
        hue="feature",
        palette="tab10",
        size=3,
        alpha=0.35,
        legend=False,
    )
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("SHAP direction of top PM2.5 drivers")
    plt.xlabel("SHAP value in model space")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "catboost_tuned_shap_direction.png", dpi=300, bbox_inches="tight")
    plt.close()


def season_name(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Autumn"


def save_time_stability_outputs(df: pd.DataFrame, feature_cols: list[str], params: dict[str, object], scaler: object) -> None:
    tscv = TimeSeriesSplit(n_splits=5)
    rows = []
    prediction_frames = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df), start=1):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        model = build_model(params)
        model.fit(
            train_df[feature_cols],
            train_df[TARGET],
            eval_set=(val_df[feature_cols], val_df[TARGET]),
            use_best_model=True,
            early_stopping_rounds=int(params["early_stopping_rounds"]),
        )
        pred_scaled = model.predict(val_df[feature_cols])
        actual_real = inverse_scale_pm25(val_df[TARGET].values, scaler)
        pred_real = inverse_scale_pm25(pred_scaled, scaler)
        metrics = calculate_metrics(actual_real, pred_real)
        rows.append({
            "fold": fold,
            "start": val_df["timestamp"].min(),
            "end": val_df["timestamp"].max(),
            **metrics,
        })
        fold_pred = pd.DataFrame({
            "fold": fold,
            "timestamp": val_df["timestamp"].values,
            "actual_pm25": actual_real,
            "pred_pm25": pred_real,
        })
        prediction_frames.append(fold_pred)

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(DATA_DIR / "catboost_tuned_timeseries_fold_metrics.csv", index=False)

    oof_df = pd.concat(prediction_frames, ignore_index=True)
    oof_df["timestamp"] = pd.to_datetime(oof_df["timestamp"], errors="coerce")
    oof_df["month"] = oof_df["timestamp"].dt.month
    oof_df["season"] = oof_df["month"].map(season_name)
    oof_df.to_csv(DATA_DIR / "catboost_tuned_oof_predictions.csv", index=False)

    season_rows = []
    for season, group in oof_df.groupby("season"):
        season_rows.append({
            "season": season,
            **calculate_metrics(group["actual_pm25"].values, group["pred_pm25"].values),
        })
    season_df = pd.DataFrame(season_rows).sort_values("season")
    season_df.to_csv(DATA_DIR / "catboost_tuned_seasonal_stability.csv", index=False)

    month_rows = []
    for month, group in oof_df.groupby("month"):
        month_rows.append({
            "month": int(month),
            **calculate_metrics(group["actual_pm25"].values, group["pred_pm25"].values),
        })
    month_df = pd.DataFrame(month_rows).sort_values("month")
    month_df.to_csv(DATA_DIR / "catboost_tuned_monthly_stability.csv", index=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=season_df, x="season", y="RMSE", color="#F28E2B")
    plt.title("Tuned CatBoost seasonal stability")
    plt.xlabel("")
    plt.ylabel("OOF RMSE")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "catboost_tuned_seasonal_stability.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=month_df, x="month", y="RMSE", marker="o", color="#4C78A8")
    plt.title("Tuned CatBoost monthly stability")
    plt.xlabel("Month")
    plt.ylabel("OOF RMSE")
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "catboost_tuned_monthly_stability.png", dpi=300)
    plt.close()


def load_phase2_metrics() -> dict[str, float]:
    if not PHASE2_METRICS_PATH.exists():
        return {}
    df = pd.read_csv(PHASE2_METRICS_PATH)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {
        "phase2_MAE": float(row.get("MAE", np.nan)),
        "phase2_RMSE": float(row.get("RMSE", np.nan)),
        "phase2_R2": float(row.get("R2", np.nan)),
        "phase2_MAPE_pct": float(row.get("MAPE_pct", np.nan)),
        "phase2_SMAPE_pct": float(row.get("SMAPE_pct", np.nan)),
    }


def main() -> None:
    set_seed()
    df, feature_cols, scaler = prepare_data()
    train_df, val_df, test_df = split_data(df)

    results = []
    trained_models: dict[str, CatBoostRegressor] = {}

    print("=" * 88)
    print("PHASE 3 :: CATBOOST TUNING, EXPLAINABILITY AND STABILITY")
    print("=" * 88)
    print(f"Rows: {len(df)} | Features: {len(feature_cols)}")
    print(f"Train: {train_df['timestamp'].min()} -> {train_df['timestamp'].max()}")
    print(f"Val  : {val_df['timestamp'].min()} -> {val_df['timestamp'].max()}")
    print(f"Test : {test_df['timestamp'].min()} -> {test_df['timestamp'].max()}")

    for params in TUNING_CANDIDATES:
        candidate = str(params["candidate"])
        print(f"\nTraining candidate: {candidate}")
        model = build_model(params)
        model.fit(
            train_df[feature_cols],
            train_df[TARGET],
            eval_set=(val_df[feature_cols], val_df[TARGET]),
            use_best_model=True,
            early_stopping_rounds=int(params["early_stopping_rounds"]),
        )
        trained_models[candidate] = model
        row = {
            **params,
            "best_iteration": int(model.get_best_iteration() or params["iterations"]),
            **evaluate_model(model, val_df, feature_cols, scaler, "validation"),
            **evaluate_model(model, test_df, feature_cols, scaler, "test"),
        }
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(DATA_DIR / "catboost_tuning_candidates.csv", index=False)
    save_tuning_plot(results_df)

    best_row = results_df.sort_values(["validation_RMSE", "validation_MAE"]).iloc[0]
    best_candidate = str(best_row["candidate"])
    best_model = trained_models[best_candidate]
    best_params = next(candidate for candidate in TUNING_CANDIDATES if candidate["candidate"] == best_candidate)

    best_model.save_model(str(MODEL_DIR / "catboost_phase3_tuned_model.cbm"))
    joblib.dump(feature_cols, MODEL_DIR / "catboost_phase3_feature_columns.pkl")

    best_pred_scaled = best_model.predict(test_df[feature_cols])
    best_pred_real = inverse_scale_pm25(best_pred_scaled, scaler)
    save_actual_vs_predicted(test_df, best_pred_real, scaler)

    final_metrics = {
        "model": "CatBoost Tuned",
        "phase": "phase_3",
        "selection_strategy": "lowest validation RMSE among conservative candidates",
        "target": TARGET,
        "unit": "real_pm25",
        "n_features": len(feature_cols),
        "selected_candidate": best_candidate,
        "best_iteration": int(best_row["best_iteration"]),
        "validation_MAE": float(best_row["validation_MAE"]),
        "validation_RMSE": float(best_row["validation_RMSE"]),
        "validation_MAPE_pct": float(best_row["validation_MAPE_pct"]),
        "validation_SMAPE_pct": float(best_row["validation_SMAPE_pct"]),
        "validation_R2": float(best_row["validation_R2"]),
        "MAE": float(best_row["test_MAE"]),
        "RMSE": float(best_row["test_RMSE"]),
        "MAPE_pct": float(best_row["test_MAPE_pct"]),
        "SMAPE_pct": float(best_row["test_SMAPE_pct"]),
        "R2": float(best_row["test_R2"]),
        "n_eval_points": int(best_row["test_n_eval_points"]),
    }

    phase2 = load_phase2_metrics()
    if phase2:
        final_metrics.update({
            "phase2_MAE": phase2["phase2_MAE"],
            "phase2_RMSE": phase2["phase2_RMSE"],
            "phase2_R2": phase2["phase2_R2"],
            "MAE_improvement": phase2["phase2_MAE"] - final_metrics["MAE"],
            "RMSE_improvement": phase2["phase2_RMSE"] - final_metrics["RMSE"],
            "R2_improvement": final_metrics["R2"] - phase2["phase2_R2"],
            "RMSE_improvement_pct": ((phase2["phase2_RMSE"] - final_metrics["RMSE"]) / phase2["phase2_RMSE"]) * 100.0,
        })

    pd.DataFrame([final_metrics]).to_csv(DATA_DIR / "catboost_tuned_metrics.csv", index=False)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": best_model.get_feature_importance(),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importance_df.to_csv(DATA_DIR / "catboost_tuned_feature_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    plot_df = importance_df.head(12).iloc[::-1]
    plt.barh(plot_df["feature"], plot_df["importance"], color="#4C78A8")
    plt.title("Tuned CatBoost feature importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "catboost_tuned_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    save_shap_outputs(best_model, test_df, feature_cols)
    save_time_stability_outputs(df, feature_cols, best_params, scaler)

    run_info = {
        "input_path": str(INPUT_PATH),
        "model_path": str(MODEL_DIR / "catboost_phase3_tuned_model.cbm"),
        "feature_columns_path": str(MODEL_DIR / "catboost_phase3_feature_columns.pkl"),
        "selected_candidate": best_candidate,
        "selection_strategy": "lowest validation RMSE",
        "outputs": {
            "metrics": str(DATA_DIR / "catboost_tuned_metrics.csv"),
            "tuning_candidates": str(DATA_DIR / "catboost_tuning_candidates.csv"),
            "forecasts": str(DATA_DIR / "catboost_tuned_forecasts.csv"),
            "shap": str(DATA_DIR / "catboost_tuned_shap_global_importance.csv"),
            "seasonal_stability": str(DATA_DIR / "catboost_tuned_seasonal_stability.csv"),
        },
        "feature_columns": feature_cols,
    }
    with open(DATA_DIR / "catboost_tuned_run_info.json", "w", encoding="utf-8") as file:
        json.dump(run_info, file, indent=2, default=str)

    print("\nSelected candidate:", best_candidate)
    print(pd.DataFrame([final_metrics]).to_string(index=False))
    print("\nSaved phase 3 CatBoost outputs to:", DATA_DIR)


if __name__ == "__main__":
    main()
