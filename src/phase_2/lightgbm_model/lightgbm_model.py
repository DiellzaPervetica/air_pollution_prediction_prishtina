from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pickle
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
INPUT_PATH = BASE_DIR / "data" / "4E_selected_dataset.csv"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

INCLUDE_ADDITIONAL_FEATURES = True
SCENARIO_NAME = "improved_model" if INCLUDE_ADDITIONAL_FEATURES else "baseline_model"
OUTPUT_DIR = CURRENT_DIR / SCENARIO_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)

def inverse_scale_feature(values, scaler, feature_name="pm25"):
    feature_names = list(scaler.feature_names_in_)
    idx = feature_names.index(feature_name)
    values = np.asarray(values, dtype=float)
    real_values = values * scaler.scale_[idx] + scaler.mean_[idx]
    if feature_name == "pm25":
        return np.expm1(real_values)
    return real_values

def load_and_preprocess_data(use_lags=True):
    df = pd.read_csv(INPUT_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    target = 'pm25'
    
    if use_lags:
        for lag in [1, 24]:
            df[f'pm25_lag_{lag}'] = df[target].shift(lag)
    
    df = df.dropna().reset_index(drop=True)
    return df, target

def plot_feature_importance(importance_df, output_dir):
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance_Percentage', y='Feature', data=importance_df.head(15), palette='viridis')
    plt.title(f'Feature importance (%) ({SCENARIO_NAME.replace("_", " ").title()})', fontsize=14)
    plt.xlabel('Relative importance (%)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    for index, value in enumerate(importance_df['Importance_Percentage'].head(15)):
        plt.text(value, index, f' {value:.2f}', va='center', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=300)
    plt.close()

def plot_actual_vs_predicted(dates, y_true, y_pred, output_dir):
    plt.figure(figsize=(15, 6))
    plot_limit = -500 
    plt.plot(dates[plot_limit:], y_true[plot_limit:], label='Actual values', color='blue', alpha=0.7, linewidth=1.5)
    plt.plot(dates[plot_limit:], y_pred[plot_limit:], label='Predicted values', color='red', alpha=0.7, linewidth=1.5, linestyle='--')
    plt.title(f'Actual vs Predicted values of PM2.5 ({SCENARIO_NAME.replace("_", " ").title()})', fontsize=14)
    plt.xlabel('Date and time', fontsize=12)
    plt.ylabel('PM2.5 (µg/m³)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=300)
    plt.close()

def plot_learning_curve(evals_result, output_dir):
    if not evals_result:
        return
        
    plt.figure(figsize=(10, 6))
    for data_name in evals_result.keys():
        metric_name = list(evals_result[data_name].keys())[0]
        plt.plot(evals_result[data_name][metric_name], label=data_name)
        
    plt.title('Learning Curve', fontsize=14)
    plt.xlabel('Iterations (Trees)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curve.png", dpi=300)
    plt.close()

def train_and_evaluate():
    df, target = load_and_preprocess_data(use_lags=INCLUDE_ADDITIONAL_FEATURES)
    scaler = load_scaler()

    drop_cols = ['datetime', 'date', target]
    features = [c for c in df.columns if c not in drop_cols]
    X = df[features]
    y = df[target]

    tscv = TimeSeriesSplit(n_splits=5)
    all_mae_real, all_rmse_real, all_r2_real = [], [], []

    params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'importance_type': 'gain',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    last_fold_dates, last_fold_y_true, last_fold_y_pred = None, None, None
    last_model = None
    last_X_test = None
    evals_result = {} 

    print(f"\nDuke trajnuar modelin LightGBM: {SCENARIO_NAME.replace('_', ' ').upper()}")
    print("-" * 50)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = lgb.LGBMRegressor(**params)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.record_evaluation(evals_result)
        ]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_names=['Train', 'Validation'],
            callbacks=callbacks
        )

        preds = model.predict(X_test)
        y_test_real = inverse_scale_feature(y_test.to_numpy(), scaler, feature_name=target)
        preds_real = inverse_scale_feature(preds, scaler, feature_name=target)
        
        all_mae_real.append(mean_absolute_error(y_test_real, preds_real))
        all_rmse_real.append(np.sqrt(mean_squared_error(y_test_real, preds_real)))
        all_r2_real.append(r2_score(y_test_real, preds_real))

        if fold == tscv.n_splits - 1:
            last_fold_dates = df.iloc[test_idx]['datetime'].values
            last_fold_y_true = y_test_real
            last_fold_y_pred = preds_real
            last_model = model
            last_X_test = X_test

    mean_r2 = np.mean(all_r2_real)
    mean_rmse = np.mean(all_rmse_real)
    mean_mae = np.mean(all_mae_real)

    print(f"Rezultatet e evaluimit të modelit: R²: {mean_r2:.4f} | RMSE: {mean_rmse:.4f} µg/m³ | MAE: {mean_mae:.4f} µg/m³\n")

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': last_model.feature_importances_
    })
    total_importance = importance_df['Importance'].sum()
    importance_df['Importance_Percentage'] = (importance_df['Importance'] / total_importance) * 100
    
    importance_df = importance_df[['Feature', 'Importance_Percentage']].sort_values('Importance_Percentage', ascending=False)
    
    report_text = (
        f"Skenari: {SCENARIO_NAME.replace('_', ' ').title()}\n"
        f"=====================================\n"
        f"R²: {mean_r2:.4f}\n"
        f"RMSE:     {mean_rmse:.4f} µg/m³\n"
        f"MAE:      {mean_mae:.4f} µg/m³\n\n"
        f"Top 5 veçoritë më të rëndësishme:\n"
    )
    for i, row in importance_df.head(5).iterrows():
        report_text += f"- {row['Feature']}: {row['Importance_Percentage']:.2f}%\n"

    with open(OUTPUT_DIR / "metrics_summary.txt", "w", encoding='utf-8') as f:
        f.write(report_text)
    
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    print("Duke gjeneruar vizualizimet...")
    plot_feature_importance(importance_df, OUTPUT_DIR)
    plot_actual_vs_predicted(last_fold_dates, last_fold_y_true, last_fold_y_pred, OUTPUT_DIR)
    plot_learning_curve(evals_result, OUTPUT_DIR)
    
    sample_size = min(2000, len(last_X_test))
    
    joblib.dump(last_model, OUTPUT_DIR / f"{SCENARIO_NAME}.joblib")
    print(f"[SUKSES] Të gjitha artifaktet u ruajtën në dosjen: '{SCENARIO_NAME}'")

if __name__ == "__main__":
    train_and_evaluate()