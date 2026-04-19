from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[3]
INPUT_PATH = REPO_ROOT / "data" / "phase_1" / "4E_selected_dataset.csv"

OUTPUT_DIR = CURRENT_DIR / "isolation_forest_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_and_preprocess_data():
    print(f"1. Duke lexuar të dhënat nga: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    df.set_index('datetime', inplace=True)
    
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
        
    df.dropna(inplace=True)
    return df

def run_isolation_forest():
    df = load_and_preprocess_data()

    print("2. Duke përgatitur features...")
    features = df.columns.tolist()
    
    print("3. Duke trajnuar modelin Isolation Forest...")
    iso_forest = IsolationForest(n_estimators=100, 
                                 contamination=0.05, 
                                 random_state=42,
                                 n_jobs=-1)
    
    df['anomaly'] = iso_forest.fit_predict(df[features])
    df['anomaly_score'] = iso_forest.decision_function(df[features])
    
    anomalies = df[df['anomaly'] == -1]
    
    print(f"Trajnimi përfundoi! U gjetën {len(anomalies)} anomali nga {len(df)} rreshta në total.")
    print("-" * 50)

    print("4. Duke gjeneruar vizualizimet...")
    
    target_col = 'pm25' if 'pm25' in df.columns else 'pm2.5'
    energy_col = 'total_generation_mw'
    
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df[target_col], color='lightblue', alpha=0.7, label='Nivelet normale (PM2.5)', linewidth=1.5)
    plt.scatter(anomalies.index, anomalies[target_col], color='red', s=25, label='Anomali', zorder=5)
    plt.title('Isolation Forest: Zbulimi i anomalive në ndotjen e ajrit (PM2.5)', fontsize=14, fontweight='bold')
    plt.xlabel('Data dhe Koha', fontsize=12)
    plt.ylabel('PM2.5', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "isolation_forest_pm25.png", dpi=300)
    plt.close()

    if energy_col in df.columns:
        plt.figure(figsize=(15, 6))
        plt.plot(df.index, df[energy_col], color='lightgreen', alpha=0.7, label='Prodhimi Normal', linewidth=1.5)
        plt.scatter(anomalies.index, anomalies[energy_col], color='darkred', s=25, label='Anomali në Prodhim', zorder=5)
        plt.title('Isolation Forest: Zbulimi i anomalive në prodhimin e energjisë', fontsize=14, fontweight='bold')
        plt.xlabel('Data dhe koha', fontsize=12)
        plt.ylabel('Prodhimi total (MW)', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "isolation_forest_energy.png", dpi=300)
        plt.close()

    plt.figure(figsize=(15, 6))
    zoom_start = '2025-11-15'
    zoom_end = '2025-12-05'
    
    if zoom_start in df.index and zoom_end in df.index:
        df_zoom = df.loc[zoom_start:zoom_end]
        anomalies_zoom = anomalies.loc[zoom_start:zoom_end]
    else:
        df_zoom = df.tail(1000)
        anomalies_zoom = anomalies[anomalies.index >= df_zoom.index[0]]

    plt.plot(df_zoom.index, df_zoom[target_col], color='royalblue', alpha=0.7, label='PM2.5 (Ecuria Normale)', linewidth=2)
    plt.scatter(anomalies_zoom.index, anomalies_zoom[target_col], color='red', s=60, label='Goditjet Anormale', zorder=5)
    plt.title('Zoom-in: Goditjet e ndotjes (Nëntor - Dhjetor 2025)', fontsize=14, fontweight='bold')
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('PM2.5 (E skaluar)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "isolation_forest_pm25_zoom.png", dpi=300)
    plt.close()

    if energy_col in df.columns:
        plt.figure(figsize=(10, 8))
        
        plt.scatter(df[energy_col], df[target_col], color='lightgray', alpha=0.4, label='Orët Normale', s=15)
        plt.scatter(anomalies[energy_col], anomalies[target_col], color='darkred', alpha=0.8, label='Orët Anormale', s=40, edgecolor='black', linewidth=0.5)
        
        plt.title('Korrelacioni i anomalive: ndotja vs prodhimi i energjisë', fontsize=14, fontweight='bold')
        plt.xlabel('Prodhimi total i energjisë (MW)', fontsize=12)
        plt.ylabel('Ndotja e ajrit - PM2.5', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "isolation_forest_scatter.png", dpi=300)
        plt.close()

    print("Duke ruajtur listën e anomalive ekstreme...")
    top_anomalies = anomalies.sort_values('anomaly_score').head(100)
    
    cols_to_save = [target_col, 'anomaly_score']
    if energy_col in df.columns: cols_to_save.append(energy_col)
    
    top_anomalies[cols_to_save].to_csv(OUTPUT_DIR / "top_anomalies_list.csv")
    
    print(f"\n[SUKSES] Të gjitha artifaktet u ruajtën në dosjen: '{OUTPUT_DIR.name}'")

if __name__ == "__main__":
    run_isolation_forest()
