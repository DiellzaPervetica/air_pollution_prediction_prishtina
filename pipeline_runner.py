import subprocess
import sys
from pathlib import Path

def run_script(script_path):

    print(f"\n[EKZEKUTIMI] Duke nisur: {script_path}...")
    try:
        result = subprocess.run([sys.executable, str(script_path)], check=True, capture_output=False)
        print(f"[SUKSES] Perfundoi me sukses: {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[GABIM] Skripta {script_path} deshtoi!")
        return False

def main():
    BASE_DIR = Path(__file__).resolve().parent

    pipeline = [
        BASE_DIR / "1A_merge_data.py",

        BASE_DIR / "1B_distinct_values" / "1B1_pollution_distinct.py",
        BASE_DIR / "1B_distinct_values" / "1B2_weather_distinct.py",
        BASE_DIR / "1B_distinct_values" / "1B3_energy_distinct.py",

        BASE_DIR / "1C_datetime_and_duplicates" / "1C1_datetime_cleaning.py",
        BASE_DIR / "1C_datetime_and_duplicates" / "1C2_extract_duplicates.py",
        BASE_DIR / "1C_datetime_and_duplicates" / "1C3_remove_duplicates.py",

        BASE_DIR / "1D_data_quality_cleaning" / "1D1_clean_pollution.py",
        BASE_DIR / "1D_data_quality_cleaning" / "1D2_clean_wind.py",
        BASE_DIR / "1D_data_quality_cleaning" / "1D3_clean_energy_weather.py",
        BASE_DIR / "1D_data_quality_cleaning" / "1D4_precision_control.py",
        BASE_DIR / "1D_data_quality_cleaning" / "1D5_logical_validation.py",

        BASE_DIR / "1E_missing_values_handling" / "1E1_missing_values_report.py",
        BASE_DIR / "1E_missing_values_handling" / "1E2_impute_pm_particles.py",
        BASE_DIR / "1E_missing_values_handling" / "1E3_impute_gases.py",
        BASE_DIR / "1E_missing_values_handling" / "1E4_impute_finalize.py",

        BASE_DIR / "1F_Advanced_Consistency_Checks" / "1F1_pm_ratio_consistency.py",
        BASE_DIR / "1F_Advanced_Consistency_Checks" / "1F2_timeline_gap_validation.py",
        BASE_DIR / "1F_Advanced_Consistency_Checks" / "1F3_final_integrity_audit.py"
    ]

    print("="*50)
    print("NISJA E PIPELINE Te PASTRIMIT Te Te DHeNAVE")
    print("="*50)

    for script in pipeline:
        if not script.exists():
            print(f"[KUJDES] File nuk u gjet: {script}. Duke e anashkaluar...")
            continue
        
        success = run_script(script)
        
        if not success:
            print("\n[STOP] Pipeline u nderpre per shkak te nje gabimi.")
            break
    else:
        print("\n" + "="*50)
        print("PIPELINE PERFUNDOI ME SUKSES Te PLOTE!")
        print("="*50)

if __name__ == "__main__":
    main()