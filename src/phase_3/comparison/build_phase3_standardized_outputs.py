from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PHASE2_SUPERVISED = PROJECT_ROOT / "data" / "phase_2" / "comparison" / "supervised_model_comparison.csv"
PHASE3_METRICS = PROJECT_ROOT / "data" / "phase_3" / "supervised" / "catboost_tuned" / "catboost_tuned_metrics.csv"
PHASE3_TUNING = PROJECT_ROOT / "data" / "phase_3" / "supervised" / "catboost_tuned" / "catboost_tuning_candidates.csv"
PHASE3_DAILY_SNAPSHOT = PROJECT_ROOT / "data" / "phase_3" / "forecasting" / "next_day_pm25_daily_summary_snapshot.csv"

DATA_DIR = PROJECT_ROOT / "data" / "phase_3" / "comparison"
PLOTS_DIR = PROJECT_ROOT / "pictures" / "phase_3" / "comparison"

for directory in [DATA_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def save_table_image(df: pd.DataFrame, output_path: Path, title: str) -> None:
    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda value: "N/A" if pd.isna(value) else f"{value:.4f}")
        else:
            display_df[column] = display_df[column].map(lambda value: "N/A" if pd.isna(value) else str(value))

    fig_width = max(10, len(display_df.columns) * 1.65)
    fig_height = max(2.8, len(display_df) * 0.7 + 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.55)
    ax.set_title(title, fontsize=13, pad=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_csv(path)


def build_phase2_supervised_snapshot() -> pd.DataFrame:
    df = load_required(PHASE2_SUPERVISED, "Phase 2 supervised comparison")
    keep_cols = ["model", "evaluation_strategy", "MAE", "RMSE", "R2", "MAPE_pct", "SMAPE_pct"]
    keep_cols = [col for col in keep_cols if col in df.columns]
    out = df[keep_cols].copy()
    out.to_csv(DATA_DIR / "phase2_supervised_reference.csv", index=False)
    save_table_image(out, PLOTS_DIR / "phase2_supervised_reference_table.png", "Phase 2 supervised model comparison")

    metrics = [metric for metric in ["MAE", "RMSE", "R2"] if metric in out.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.8))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sns.barplot(data=out, x="model", y=metric, ax=ax, color="#4C78A8")
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Phase 2 supervised models: where phase 3 starts", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "phase2_supervised_metrics_reference.png", dpi=300, bbox_inches="tight")
    plt.close()
    return out


def build_catboost_improvement_summary(phase2_df: pd.DataFrame) -> pd.DataFrame:
    phase3 = load_required(PHASE3_METRICS, "Phase 3 CatBoost metrics").iloc[0].to_dict()
    catboost_phase2 = phase2_df[phase2_df["model"].astype(str).str.contains("CatBoost", case=False, na=False)]
    if catboost_phase2.empty:
        raise ValueError("CatBoost row was not found in phase 2 supervised comparison.")
    phase2 = catboost_phase2.iloc[0].to_dict()

    rows = []
    for metric in ["MAE", "RMSE", "R2", "MAPE_pct", "SMAPE_pct"]:
        if metric not in phase2 or metric not in phase3:
            continue
        old = float(phase2[metric])
        new = float(phase3[metric])
        if metric == "R2":
            direction = "higher is better"
            absolute_improvement = new - old
            relative_improvement_pct = ((new - old) / abs(old)) * 100.0 if old != 0 else np.nan
        else:
            direction = "lower is better"
            absolute_improvement = old - new
            relative_improvement_pct = ((old - new) / old) * 100.0 if old != 0 else np.nan
        rows.append({
            "metric": metric,
            "phase2_catboost": old,
            "phase3_tuned_catboost": new,
            "absolute_improvement": absolute_improvement,
            "relative_improvement_pct": relative_improvement_pct,
            "direction": direction,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(DATA_DIR / "catboost_phase2_vs_phase3_improvement.csv", index=False)
    save_table_image(summary, PLOTS_DIR / "catboost_phase2_vs_phase3_improvement_table.png", "CatBoost phase 2 vs phase 3")

    plot_df = summary[summary["metric"].isin(["MAE", "RMSE", "R2"])].copy()
    plot_long = plot_df.melt(
        id_vars=["metric"],
        value_vars=["phase2_catboost", "phase3_tuned_catboost"],
        var_name="version",
        value_name="value",
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_long, x="metric", y="value", hue="version", palette=["#4C78A8", "#F28E2B"])
    plt.title("CatBoost improvement after phase 3 tuning")
    plt.xlabel("")
    plt.ylabel("Metric value")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "catboost_phase2_vs_phase3_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    return summary


def build_tuning_reference() -> None:
    if not PHASE3_TUNING.exists():
        return
    tuning = pd.read_csv(PHASE3_TUNING)
    keep_cols = [
        "candidate",
        "depth",
        "learning_rate",
        "l2_leaf_reg",
        "best_iteration",
        "validation_RMSE",
        "test_RMSE",
        "test_R2",
    ]
    keep_cols = [col for col in keep_cols if col in tuning.columns]
    out = tuning[keep_cols].copy().sort_values("validation_RMSE")
    out.to_csv(DATA_DIR / "catboost_phase3_tuning_reference.csv", index=False)
    save_table_image(out, PLOTS_DIR / "catboost_phase3_tuning_reference_table.png", "Phase 3 CatBoost tuning candidates")


def build_snapshot_reference() -> None:
    if not PHASE3_DAILY_SNAPSHOT.exists():
        return
    snapshot = pd.read_csv(PHASE3_DAILY_SNAPSHOT)
    snapshot.to_csv(DATA_DIR / "next_day_forecast_snapshot_reference.csv", index=False)
    keep_cols = [
        "forecast_date",
        "generation_forecast_mwh",
        "pm25_mean_forecast",
        "pm25_max_forecast",
        "risk_category",
        "snapshot_note",
    ]
    keep_cols = [col for col in keep_cols if col in snapshot.columns]
    save_table_image(snapshot[keep_cols], PLOTS_DIR / "next_day_forecast_snapshot_table.png", "Stored next-day PM2.5 snapshot")


def main() -> None:
    print("=" * 88)
    print("PHASE 3 :: STANDARDIZED COMPARISON OUTPUTS")
    print("=" * 88)
    phase2_df = build_phase2_supervised_snapshot()
    improvement = build_catboost_improvement_summary(phase2_df)
    build_tuning_reference()
    build_snapshot_reference()

    output_index = {
        "data_outputs": {
            "phase2_supervised_reference": str(DATA_DIR / "phase2_supervised_reference.csv"),
            "catboost_improvement": str(DATA_DIR / "catboost_phase2_vs_phase3_improvement.csv"),
            "tuning_reference": str(DATA_DIR / "catboost_phase3_tuning_reference.csv"),
            "forecast_snapshot_reference": str(DATA_DIR / "next_day_forecast_snapshot_reference.csv"),
        },
        "picture_outputs": {
            "phase2_supervised_metrics": str(PLOTS_DIR / "phase2_supervised_metrics_reference.png"),
            "catboost_improvement_metrics": str(PLOTS_DIR / "catboost_phase2_vs_phase3_metrics.png"),
            "catboost_improvement_table": str(PLOTS_DIR / "catboost_phase2_vs_phase3_improvement_table.png"),
            "forecast_snapshot_table": str(PLOTS_DIR / "next_day_forecast_snapshot_table.png"),
        },
    }
    with open(DATA_DIR / "phase3_comparison_run_info.json", "w", encoding="utf-8") as file:
        json.dump(output_index, file, indent=2)

    print(improvement.to_string(index=False))
    print("\nSaved phase 3 comparison outputs to:", DATA_DIR)


if __name__ == "__main__":
    main()
