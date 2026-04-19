# Prishtina Air Pollution, Weather and Energy Production Pipeline (2023–2026)

<table>
  <tr>
    <td width="150" align="center" valign="center">
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/University_of_Prishtina_logo.svg/1200px-University_of_Prishtina_logo.svg.png" width="120" alt="University Logo" />
    </td>
    <td valign="top">
      <p><strong>Universiteti i Prishtinës</strong></p>
      <p>Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike</p>
      <p>Inxhinieri Kompjuterike dhe Softuerike – Programi Master</p>
      <p><strong>Projekti nga lënda:</strong> Përgatitja dhe vizualizimi i të dhënave</p>
      <p><strong>Profesor:</strong> Dr. Sc. Lule Ahmedi</p>
      <p><strong>Asistent:</strong> Dr. Sc. Mërgim H. Hoti</p>
      <p><strong>Studentët:</strong></p>
      <ul>
        <li>Diellza Përvetica</li>
        <li>Fatjeta Gashi</li>
        <li>Festina Klinaku</li>
      </ul>
    </td>
  </tr>
</table>

---

## Përmbajtja

1. [Përmbledhje e projektit](#përmbledhje-e-projektit)
2. [Qëllimi i punimit](#qëllimi-i-punimit)
3. [Dashboard Streamlit](#dashboard-streamlit)
4. [Struktura e repository-t](#struktura-e-repository-t)
5. [01 Përgatitja e modelit](#01-përgatitja-e-modelit)
   - [Burimet e të dhënave](#burimet-e-të-dhënave)
   - [Përshkrimi i dataset-eve hyrëse](#përshkrimi-i-dataset-eve-hyrëse)
   - [Topologjia e pipeline-it](#topologjia-e-pipeline-it)
   - [Përshkrimi i detajuar i çdo skripte](#përshkrimi-i-detajuar-i-çdo-skripte)
     - [Data collection](#data-collection)
     - [Integration](#integration)
     - [Distinct values](#distinct-values)
     - [Data cleaning](#data-cleaning)
     - [Feature engineering](#feature-engineering)
     - [Preprocessing](#preprocessing)
   - [Artefaktet dhe output-et e krijuara](#artefaktet-dhe-output-et-e-krijuara)
   - [Vizualizimet e gjeneruara](#vizualizimet-e-gjeneruara)
   - [Teknikat e zbatuara dhe lidhja me lëndën](#teknikat-e-zbatuara-dhe-lidhja-me-lëndën)
   - [Ekzekutimi i projektit](#ekzekutimi-i-projektit)
   - [Rezultati final i pipeline-it](#rezultati-final-i-pipeline-it)
   - [Zgjerime në vazhdim](#zgjerime-në-vazhdim)
6. [02 Modelimi dhe analiza](#02-modelimi-dhe-analiza)
   - [Qasja e përgjithshme](#qasja-e-përgjithshme)
   - [CatBoost për parashikimin e PM2.5](#catboost-për-parashikimin-e-pm25)
   - [LightGBM për parashikimin e PM2.5](#lightgbm-për-parashikimin-e-pm25)
   - [SARIMAX për parashikimin e PM2.5](#sarimax-për-parashikimin-e-pm25)
   - [HDBSCAN për analizë unsupervised](#hdbscan-për-analizë-unsupervised)
   - [Gaussian Mixture për analizë unsupervised](#gaussian-mixture-për-analizë-unsupervised)
   - [Isolation Forest për analizë unsupervised](#isolation-forest-për-analizë-unsupervised)
   - [Ekzekutimi i fazës së dytë](#ekzekutimi-i-fazës-së-dytë)
   - [Krahasimi i harmonizuar i modeleve](#krahasimi-i-harmonizuar-i-modeleve)
   - [Rezultatet kryesore të fazës së dytë](#rezultatet-kryesore-të-fazës-së-dytë)
   - [Metrikat dhe interpretimi i rezultateve](#metrikat-dhe-interpretimi-i-rezultateve)
   - [Artefaktet e krijuara nga modelet](#artefaktet-e-krijuara-nga-modelet)
   - [Vizualizimet interaktive](#vizualizimet-interaktive)
   - [Rezultati i zgjeruar i pipeline-it](#rezultati-i-zgjeruar-i-pipeline-it)
7. [Zgjerime në vazhdim](#zgjerime-në-vazhdim)
8. [Anëtarët e grupit](#anëtarët-e-grupit)
9. [Acknowledgments](#acknowledgments)

---

## Përmbledhje e projektit

Ky projekt implementon një pipeline të plotë, modular dhe të riprodhueshëm për ndërtimin e një dataset-i analitik dhe model-ready për analizën dhe parashikimin e ndotjes së ajrit në Prishtinë, me fokus të veçantë te `PM2.5`.

Pipeline-i ndërtohet mbi integrimin e tre burimeve të ndryshme të të dhënave, të mbledhura për periudhën 2023–2026:

1. të dhënat për prodhimin e energjisë elektrike nga termocentralet e Kosovës,
2. të dhënat meteorologjike për Prishtinën,
3. të dhënat për ndotjen e ajrit në Prishtinë.

Më pas, këto burime:

- harmonizohen në nivel kohor orë-pas-ore,
- pastrohen,
- validohen,
- plotësohen për vlerat mungesë,
- pasurohen me karakteristika të reja,
- trajtohen për outlier-a dhe skewness,
- standardizohen,
- dhe në fund reduktohen në një subset tiparesh më të qëndrueshëm për modelim.

Ky projekt demonstron të gjithë ciklin e përgatitjes së të dhënave: nga kolektimi, integrimi dhe kontrolli i cilësisë, deri te feature engineering, transformimi statistikor dhe feature selection.

Në fazën e dytë, dataset-i final `data/phase_1/4E_selected_dataset.csv` është përdorur për modelim, krahasim dhe interpretim të avancuar. Konkretisht, janë implementuar tre modele supervised (`CatBoostRegressor`, `LightGBM`, `SARIMAX`) për parashikimin e `PM2.5`, si dhe tre modele unsupervised (`HDBSCAN`, `Gaussian Mixture`, `Isolation Forest`) për identifikimin e regjimeve mjedisore, cluster-ëve, noise points dhe anomalive. Për më tepër, rezultatet e të gjitha modeleve janë harmonizuar në `data/phase_2/` dhe `pictures/phase_2/`, në mënyrë që krahasimi dhe dokumentimi të jenë sa më të qarta dhe të riprodhueshme.

---

## Qëllimi i punimit

Qëllimi kryesor i këtij projekti është të ndërtojë një dataset të pastër dhe analitikisht të qëndrueshëm për të studiuar marrëdhëniet ndërmjet:

- prodhimit të energjisë elektrike,
- kushteve meteorologjike,
- dhe ndotësve atmosferikë në Prishtinë,

me fokus të veçantë në përdorimin e këtyre të dhënave për parashikimin e `PM2.5`.

Nga pikëpamja akademike, projekti është ndërtuar në dy shtresa të lidhura ngushtë:

- **faza e parë**, ku ndërtohet dataset-i final i pastër dhe model-ready;
- **faza e dytë**, ku testohet vlera reale e këtij dataset-i për regresion, clustering dhe anomaly detection.

Një komponent plotësues i rëndësishëm është edhe `app.py`, i cili shërben si dashboard interaktiv dhe e bën më të lehtë demonstrimin vizual të ndikimit të energjisë dhe motit në ndotjen e ajrit.

Objektivat kryesore janë:

- të integrohen burime heterogjene të të dhënave në një bosht të përbashkët kohor;
- të kontrollohet cilësia e të dhënave dhe të korrigjohen vlera të pasakta;
- të trajtohen mungesat pa humbur informacion të vlefshëm;
- të krijohen tipare të reja kohore, meteorologjike dhe ndërvepruese;
- të zbutet ndikimi i outlier-ave dhe shpërndarjeve shumë të shtrembëruara;
- të standardizohet dataset-i për përdorim në modele statistikore dhe machine learning;
- të eliminohet multikolineariteti i tepërt përmes VIF-based feature selection;
- të përdoret dataset-i final i përzgjedhur për ndërtimin dhe validimin e modeleve supervised për parashikimin e `PM2.5`;
- të analizohet struktura e brendshme e të dhënave përmes metodave unsupervised clustering dhe anomaly detection;
- të ndërtohet një dokumentim i plotë, i detajuar dhe profesional për të gjithë ciklin e projektit.

---

## Dashboard Streamlit

Ky projekt përfshin edhe një dashboard interaktiv të ndërtuar me Streamlit në `app.py`, i cili shërben si shtresë vizuale dhe demonstrative mbi të gjithë pipeline-in. Në versionin aktual, dashboard-i nuk është më vetëm një simulator i thjeshtë, por një mjedis i zgjeruar ku bashkohen eksplorimi i të dhënave historike, simulimi i faktorëve kryesorë dhe prezantimi i rezultateve të modeleve të fazës së dytë.

Nga pikëpamja funksionale, dashboard-i lejon:

- manipulimin interaktiv të parametrave energjetikë dhe meteorologjikë;
- shikimin e ndikimit të tyre në ndotësit kryesorë atmosferikë;
- paraqitjen më të qartë të rolit të `PM2.5` si target kryesor i projektit;
- dhe interpretimin më intuitiv të rezultateve të modeleve supervised dhe unsupervised.

Në këtë version të zgjeruar, dashboard-i pasqyron edhe gjashtë modelet kryesore të përdorura në fazën e dytë:

- `CatBoost`, `LightGBM` dhe `SARIMAX` për forecast të `PM2.5`;
- `HDBSCAN`, `Gaussian Mixture` dhe `Isolation Forest` për clustering, profile mjedisore dhe anomaly detection.

Kjo do të thotë se `app.py` funksionon si nyje lidhëse mes:

- të dhënave të pastruara dhe të përpunuara;
- modeleve të trajnuara;
- figurave dhe rezultateve të ruajtura në repo;
- dhe prezantimit praktik të projektit në formë të kuptueshme për përdoruesin ose profesorin.

Prandaj, nga ana e dokumentimit, është më e arsyeshme që ky komponent të paraqitet menjëherë pas qëllimit të punimit, sepse e tregon që projekti nuk përfundon vetëm me skripta analitikë, por shtrihet edhe në një shtresë prezantimi interaktiv.

<img width="1920" height="901" alt="image" src="https://github.com/user-attachments/assets/66cd2439-bb97-440b-baf5-5ec97bbc7aee" />
<img width="954" height="451" alt="image" src="https://github.com/user-attachments/assets/108326b0-a581-4491-972a-4fb8c3f6aedc" />
<img width="958" height="448" alt="{39B7F118-D825-4E3C-94A7-73B1600AE34F}" src="https://github.com/user-attachments/assets/50fb329f-fafe-4573-81de-e3a17e06ae05" />
<img width="959" height="474" alt="{C69C8447-CABA-4644-B13C-6B22E50C8374}" src="https://github.com/user-attachments/assets/62bc85f7-544e-49f2-85ac-c864c3143746" />
<img width="958" height="472" alt="{83EC6A35-6FE5-4103-9C78-1DFBB82FD6EF}" src="https://github.com/user-attachments/assets/8237cfd0-2672-4f91-a1f1-a7b556b66237" />
<img width="958" height="451" alt="{4B83DF69-60B7-4F85-B0DB-46FC1EDD743D}" src="https://github.com/user-attachments/assets/238d6ae6-5986-4e17-9b7f-b3abdbcaf4f8" />
<img width="957" height="478" alt="{B50EBF5D-65EE-4730-BE13-D8C2F21334DA}" src="https://github.com/user-attachments/assets/f21634cc-099a-4ef2-8341-891684b80d7d" />

---

## Struktura e repository-t

Struktura e mëposhtme paraqet gjendjen aktuale të repository-t dhe fokusohet te file-t kryesorë të projektit: skriptat e pipeline-it, modelet, dataset-et dhe output-et më të rëndësishme për dokumentim. Për qartësi nuk paraqiten folderët teknikë si `.venv`, `.idea`, `.vscode` dhe `__pycache__`.

```text
AIR_POLLUTION_PREDICTION_PRISHTINA/
│
├── app.py
├── README.md
├── test.py
│
├── src/
│   ├── phase_1/
│   │   ├── data_collection/
│   │   │   ├── get_kosova_air_quality_data.ps1
│   │   │   └── get_prishtina_air_quality_data.ipynb
│   │   ├── integration/
│   │   │   └── 1A_merge_data.py
│   │   ├── distinct_values/
│   │   │   └── 1B_distinct_values.py
│   │   ├── data_cleaning/
│   │   │   ├── 2A_datetime_and_duplicates.py
│   │   │   ├── 2B_data_quality_cleaning.py
│   │   │   ├── 2C_missing_values_handling.py
│   │   │   └── 2D_validate_final_dataset.py
│   │   ├── feature_engineering/
│   │   │   ├── 3A_target_analysis.py
│   │   │   └── 3B_feature_engineering.py
│   │   └── preprocessing/
│   │       ├── 4A_outlier_treatment.py
│   │       ├── 4B_skewness_treatment.py
│   │       ├── 4C_visualization_before_after.py
│   │       ├── 4D_feature_scaling.py
│   │       └── 4E_feature_selection.py
│   │
│   └── phase_2/
│       ├── supervised/
│       │   ├── catboost_model/
│       │   │   └── catboost_model.py
│       │   ├── lightgbm_model/
│       │   │   ├── lightgbm_model.py
│       │   │   ├── baseline_model/
│       │   │   └── improved_model/
│       │   └── sarimax_model/
│       │       └── sarimax_model.py
│       ├── unsupervised/
│       │   ├── gaussian_mixture_model/
│       │   │   └── gaussian_mixture_model.py
│       │   ├── hdbscan_model/
│       │   │   └── hdbscan_model.py
│       │   └── isolation_forest_model/
│       │       ├── isolation_forest_model.py
│       │       └── isolation_forest_extended_outputs.py
│       └── comparison/
│           └── build_phase2_standardized_outputs.py
│
├── data/
│   ├── raw/
│   │   ├── prishtina_air_quality_2023_2025.csv
│   │   ├── prishtina_energy_production_2023_2026.csv
│   │   ├── prishtina_weather_2023_2026.csv
│   │   ├── thermocentral_A_emissions.csv
│   │   └── thermocentral_B_emissions.csv
│   ├── phase_1/
│   │   ├── 1A_merged_data_hourly_2023_2025.csv
│   │   ├── 1B_distinct_values/
│   │   ├── 2A_cleaned_no_duplicates.csv
│   │   ├── 2B_quality_checked.csv
│   │   ├── 2C_missing_values_handled.csv
│   │   ├── 2D_validated_final_dataset.csv
│   │   ├── 3B_engineered_dataset.csv
│   │   ├── 4A_outliers_handled.csv
│   │   ├── 4B_skewness_handled.csv
│   │   ├── 4D_scaled_dataset.csv
│   │   └── 4E_selected_dataset.csv
│   └── phase_2/
│       ├── supervised/
│       │   ├── catboost/
│       │   │   ├── catboost_feature_importance.csv
│       │   │   ├── catboost_forecasts.csv
│       │   │   ├── catboost_metrics.csv
│       │   │   ├── catboost_run_info.json
│       │   │   └── catboost_split_summary.csv
│       │   ├── lightgbm_improved/
│       │   │   ├── feature_importance.csv
│       │   │   └── metrics_summary.txt
│       │   └── sarimax/
│       │       ├── sarimax_candidate_results.csv
│       │       ├── sarimax_coefficients.csv
│       │       ├── sarimax_forecasts.csv
│       │       ├── sarimax_metrics.csv
│       │       ├── sarimax_residuals.csv
│       │       ├── sarimax_run_info.json
│       │       └── sarimax_split_summary.csv
│       ├── unsupervised/
│       │   ├── gaussian_mixture/
│       │   │   ├── gmm_clustered_dataset.csv
│       │   │   ├── gmm_cluster_summary.csv
│       │   │   ├── gmm_feature_summary.csv
│       │   │   ├── gmm_metrics.csv
│       │   │   ├── gmm_model_selection.csv
│       │   │   └── gmm_run_info.json
│       │   ├── hdbscan/
│       │   │   ├── hdbscan_clustered_dataset.csv
│       │   │   ├── hdbscan_cluster_summary.csv
│       │   │   ├── hdbscan_feature_summary.csv
│       │   │   ├── hdbscan_metrics.csv
│       │   │   └── hdbscan_run_info.json
│       │   └── isolation_forest/
│       │       ├── isolation_forest_feature_summary.csv
│       │       ├── isolation_forest_metrics.csv
│       │       ├── isolation_forest_run_info.json
│       │       ├── isolation_forest_scored_dataset.csv
│       │       └── isolation_forest_top_anomalies.csv
│       ├── comparison/
│       │   ├── supervised_model_comparison.csv
│       │   └── unsupervised_model_comparison.csv
│       └── phase2_manifest.json
│
├── models/
│   ├── scaler.pkl
│   ├── catboost_model/
│   │   ├── catboost_feature_columns.pkl
│   │   └── catboost_pm25_model.cbm
│   ├── gaussian_mixture_model/
│   │   ├── gmm_feature_columns.pkl
│   │   ├── gmm_model.pkl
│   │   ├── gmm_pca.pkl
│   │   └── gmm_scaler.pkl
│   ├── hdbscan_model/
│   │   ├── hdbscan_model.pkl
│   │   ├── hdbscan_scaler.pkl
│   │   └── hdbscan_umap.pkl
│   ├── isolation_forest_model/
│   │   ├── isolation_forest_feature_columns.pkl
│   │   └── isolation_forest_model.pkl
│   └── sarimax_model/
│       ├── sarimax_feature_columns.pkl
│       ├── sarimax_pm25_model.pkl
│       └── sarimax_summary.txt
│
└── pictures/
    ├── img.png
    ├── phase_1/
    │   ├── pollutant_correlation_heatmap.png
    │   ├── pollutant_vs_predictors_heatmap.png
    │   └── 4C_visualization_before_after/
    └── phase_2/
        ├── supervised/
        │   ├── catboost/
        │   │   ├── catboost_actual_vs_predicted.png
        │   │   ├── catboost_feature_importance.png
        │   │   ├── catboost_forecast_interactive.html
        │   │   ├── catboost_metrics_table.png
        │   │   └── catboost_residual_diagnostics.png
        │   ├── lightgbm_improved/
        │   │   ├── lightgbm_actual_vs_predicted.png
        │   │   ├── lightgbm_feature_importance.png
        │   │   ├── lightgbm_learning_curve.png
        │   │   └── lightgbm_metrics_table.png
        │   └── sarimax/
        │       ├── sarimax_actual_vs_predicted.png
        │       ├── sarimax_coefficients.png
        │       ├── sarimax_forecast_interactive.html
        │       ├── sarimax_metrics_table.png
        │       └── sarimax_residual_diagnostics.png
        ├── unsupervised/
        │   ├── gaussian_mixture/
        │   │   ├── gmm_cluster_profile_heatmap.png
        │   │   ├── gmm_metrics_table.png
        │   │   ├── gmm_model_selection.png
        │   │   ├── gmm_pca_interactive.html
        │   │   ├── gmm_pm25_by_cluster.png
        │   │   └── gmm_scatter.png
        │   ├── hdbscan/
        │   │   ├── hdbscan_feature_shift_panel.png
        │   │   ├── hdbscan_metrics_table.png
        │   │   ├── hdbscan_pm25_by_cluster.png
        │   │   ├── hdbscan_scatter.png
        │   │   └── hdbscan_umap_interactive.html
        │   └── isolation_forest/
        │       ├── isolation_forest_energy.png
        │       ├── isolation_forest_metrics_table.png
        │       ├── isolation_forest_pm25.png
        │       ├── isolation_forest_pm25_zoom.png
        │       ├── isolation_forest_scatter.png
        │       └── isolation_forest_score_distribution.png
        └── comparison/
            ├── supervised_comparison_table.png
            ├── supervised_error_metrics.png
            ├── supervised_feature_panels.png
            ├── supervised_r2_comparison.png
            ├── unsupervised_clustering_quality.png
            ├── unsupervised_comparison_table.png
            ├── unsupervised_feature_panels.png
            ├── unsupervised_pm25_profiles.png
            └── unsupervised_special_ratio_and_groups.png
```

Kjo strukturë është e favorshme për dokumentim profesional sepse:

- ndan qartë fazën e përgatitjes së të dhënave nga faza e modelimit;
- e bën të dallueshme rrjedhën `raw -> phase_1 -> phase_2`;
- mban artefaktet e secilit model në një vend të qëndrueshëm;
- ndan qartë `src`, `data`, `models` dhe `pictures`, pra kodin, të dhënat, modelet e ruajtura dhe vizualizimet;
- lejon krahasim më të lehtë mes modeleve supervised dhe unsupervised;
- dhe e bën repo-n më të pastër për dorëzim, prezantim dhe mirëmbajtje.

---

## 01 Përgatitja e modelit

### Burimet e të dhënave

Ky projekt bazohet në tre burime kryesore të të dhënave:

#### 1. Prodhimi i energjisë elektrike nga termocentralet e Kosovës

Dataset-i përmban prodhimin orar të njësive energjetike:

- `A3_MW`
- `A4_MW`
- `A5_MW`
- `B1_MW`
- `B2_MW`

Nga këto është ndërtuar edhe:

- `total_generation_mw`

Të dhënat janë marrë nga KOSTT dhe janë harmonizuar në nivel orar.

#### 2. Të dhënat meteorologjike për Prishtinën

Dataset-i meteorologjik përmban atribute si:

- temperatura,
- reshjet,
- bora,
- lagështia relative,
- drejtimi i erës,
- shpejtësia e erës.

Këto të dhëna janë përdorur për të modeluar kushtet atmosferike që ndikojnë në përhapjen ose stagnimin e ndotjes. Të dhënat janë marrë nga OpenMeteo.

#### 3. Të dhënat e ndotjes së ajrit në Prishtinë

Dataset-i i cilësisë së ajrit përmban matje të ndotësve:

- `co`
- `no2`
- `o3`
- `pm10`
- `pm25`
- `so2`

Këto të dhëna janë mbledhur dhe konsoliduar për Prishtinën përmes burimeve të tipit OpenAQ / arkivave përkatëse / notebook-ut të kolektimit të përdorur në projekt.

#### Shtrirja kohore

Burimet hyrëse mbulojnë periudhën 2023–2026. Megjithatë, dataset-i i integruar final ruan vetëm intervalin ku të tre burimet kanë mbulim të përbashkët orar, prandaj output-i i parë i integruar ruhet si:

- `1A_merged_data_hourly_2023_2025.csv`

Kjo e bën integrimin kohor të saktë dhe shmang boshllëqet e krijuara nga mungesa e përbashkët midis burimeve.

#### Dataset-i i integruar

Pas bashkimit (`merge`) të tre burimeve me `inner join`, dataset-i final përmban vetëm intervalin e përbashkët kohor:

- Numri i rreshtave: **9,370**
- Numri i kolonave: **22**
- Numri total i vlerave: **206,140**
- Intervali kohor: **2023-08-01 → 2025-11-27**

- Reduktimi i numrit të rreshtave është rezultat i sinkronizimit strikt kohor ndërmjet burimeve, ku ruhen vetëm momentet për të cilat ekzistojnë të dhëna në të tre dataset-et.

---

### Përshkrimi i dataset-eve hyrëse

Pipeline-i përdor tre skedarë bruto të ruajtur në `data/raw/`:

- `prishtina_air_quality_2023_2025.csv`
- `prishtina_weather_2023_2026.csv`
- `prishtina_energy_production_2023_2026.csv`

#### Dataset-i i ndotjes së ajrit

Përmban kolonën `datetime` dhe ndotësit kryesorë atmosferikë:

- `co`
- `no2`
- `o3`
- `pm10`
- `pm25`
- `so2`

Karakteristikat e dataset-it:

- Numri i rreshtave: **10,147**
- Numri i kolonave: **7**
- Numri total i vlerave: **71,029**
- Intervali kohor: **2023-03-14 → 2025-11-27**

#### Dataset-i meteorologjik

Përmban kolonën kohore dhe atributet:

- `temperature_2m (°C)`
- `rain (mm)`
- `snowfall (cm)`
- `relative_humidity_2m (%)`
- `wind_direction_10m (°)`
- `wind_speed_10m (km/h)`

Karakteristikat e dataset-it:

- Numri i rreshtave: **27,813**
- Numri i kolonave: **7**
- Numri total i vlerave: **194,691**
- Intervali kohor: **2023-01-01 → 2026-03-05**

#### Dataset-i i energjisë

Përmban:

- kolonën e datës,
- kolonën e orës,
- prodhimin për secilën njësi termocentrali,
- dhe totalin e gjenerimit të energjisë.

Gjatë leximit, ky dataset kërkon pastrim shtesë të header-it, sepse struktura e tij fillestare nuk është menjëherë tabulare në formën standarde CSV.

Karakteristikat e dataset-it:

- Numri i rreshtave: **22,581**
- Numri i kolonave: **7**
- Numri total i vlerave: **158,067**
- Intervali kohor: **2023-08-01 → 2026-03-03**

---

### Topologjia e pipeline-it

Pipeline-i është ndërtuar si një sekuencë hapash modularë, ku secili skript:

- lexon një output të fazës paraprake,
- kryen një transformim të caktuar,
- dhe shkruan një output të ri të versionuar.

Rrjedha logjike është kjo:

1. **Mbledhja e të dhënave**  
   Shkarkimi / përgatitja e burimeve bruto.

2. **Integrimi i të dhënave**  
   Bashkimi i ndotjes, motit dhe energjisë në një dataset të përbashkët orar.

3. **Distinct value profiling**  
   Nxjerrja e vlerave unike për atribute kyçe numerike.

4. **Data cleaning dhe quality checks**  
   Heqja e duplikateve, korrigjimi i vlerave jo-logjike, plotësimi i mungesave, validimi kronologjik dhe fizik.

5. **Target analysis dhe exploratory correlation analysis**  
   Analiza statistikore fillestare e ndotësve dhe lidhjeve me tiparet shpjeguese.

6. **Feature engineering**  
   Krijimi i tipareve kohore, lag-ve, rolling windows, ndërveprimeve dhe vektorëve të erës.

7. **Outlier handling**  
   Kufizimi i vlerave ekstreme me quantile capping.

8. **Skewness handling**  
   Transformime `log1p` dhe `Yeo-Johnson` për kolonat e shtrembëruara.

9. **Before/after visualization**  
   Krahasime histogramash para dhe pas transformimeve.

10. **Scaling**  
    Standardizimi i të gjitha kolonave numerike.

11. **Feature selection**  
    Heqja e tipareve problematike dhe reduktimi i multikolinearitetit me VIF.

---

### Përshkrimi i detajuar i çdo skripte

### Data collection

#### `get_kosova_air_quality_data.ps1`

Ky skript PowerShell përdoret për shkarkimin e të dhënave arkivore nga OpenAQ për disa `location IDs` të lidhura me Prishtinën ose pikat përkatëse të matjes.

##### Çfarë bën skripta

- krijon folder-in bazë të ruajtjes në disk,
- iteron mbi një listë `location IDs`,
- për secilin lokacion përdor komandën `aws s3 cp` për të shkarkuar skedarët `.csv.gz` nga arkiva publike e OpenAQ,
- ruan të dhënat në nënfolderë të ndarë sipas `location ID`.

##### Qëllimi

Ky hap siguron mbledhjen e të dhënave bruto të ndotjes / matjeve për përpunim të mëtejshëm.

##### Lokacionet e përdorura

Në versionin aktual përdoren:

- `2536`
- `7674`
- `7931`
- `7933`
- `9337`

##### Output

Skedarët bruto ruhen lokalisht në strukturë të ndarë sipas lokacionit.

---

#### `get_prishtina_air_quality_data.ipynb`

Ky notebook shërben si mjedis interaktiv për mbledhje, eksplorim, filtrime dhe/ose konsolidim të të dhënave të cilësisë së ajrit për Prishtinën.

Meqë logjika e plotë e notebook-ut nuk është përfshirë këtu në README, roli i tij në projekt është:

- të ndihmojë në eksplorimin fillestar të të dhënave,
- të përgatisë ose eksportojë skedarët bruto/finalë të përdorur më pas në pipeline,
- të shërbejë si hap ndërmjetës midis burimeve online dhe CSV-ve në `data/raw/`.

---

### Integration

#### `1A_merge_data.py`

Ky është hapi themelor i integrimit të të tre burimeve.

##### Input

- `data/raw/prishtina_air_quality_2023_2025.csv`
- `data/raw/prishtina_weather_2023_2026.csv`
- `data/raw/prishtina_energy_production_2023_2026.csv`

##### Hapat kryesorë

1. Lexon dataset-in e ndotjes së ajrit.
2. Lexon dataset-in meteorologjik, duke anashkaluar rreshtat hyrës jo-standardë.
3. Lexon dataset-in e energjisë pa header standard dhe e zbulon automatikisht rreshtin e header-it.

<img width="366" height="54" alt="{BE5A24DC-1B28-4178-AC88-BC896FC2D274}" src="https://github.com/user-attachments/assets/1e1f6cd1-363b-4a64-9c9e-fcf356cfb1f6" />
   
5. Normalizon emrat e kolonave të energjisë:
   - `Ora Hour` → `hour`
   - `DATA Date` → `date`
   - `A3 (MW)` → `A3_MW`
   - `A4 (MW)` → `A4_MW`
   - `A5 (MW)` → `A5_MW`
   - `B1 (MW)` → `B1_MW`
   - `B2 (MW)` → `B2_MW`

<img width="369" height="137" alt="{AAE917B3-90A4-4AEF-972B-944317A01B36}" src="https://github.com/user-attachments/assets/7b1a9d31-108a-4b06-ae95-51c1ed11c883" />

6. Konverton kolonat kohore në `datetime`.
7. Harmonizon timezone-in e ndotjes dhe motit në `Europe/Belgrade`, pastaj i kthen në naive timestamps.

<img width="575" height="221" alt="{F4E9923A-69C5-4D33-80AC-C79D01092939}" src="https://github.com/user-attachments/assets/52283448-fda9-4aa2-947c-2261663d4255" />

10. Pastron duplikatet sipas `datetime`.
11. Për dataset-in e energjisë:

- konverton `date`,
- konverton `hour`,
- krijon `datetime`,
- llogarit `total_generation_mw`.

<img width="592" height="81" alt="image" src="https://github.com/user-attachments/assets/8e114cc1-a1fa-4fbd-9461-357d0e7721be" />

11. Zgjedh vetëm kolonat relevante nga secili burim.

<img width="381" height="57" alt="{65723B3D-B2FE-4A3A-84F0-49593283C896}" src="https://github.com/user-attachments/assets/98af75ad-9e60-4a80-8558-0e910875bb02" />

13. Kryen dy merge-e me `how="inner"`:
    - ndotja + moti,
    - pastaj rezultati + energjia.
14. Krijon kolonat: - `date` - `hour` - `interval_start`
    <img width="431" height="94" alt="{AA095FE6-7145-4932-98A4-BCCD0F0B1ACA}" src="https://github.com/user-attachments/assets/9cac6b45-b4fd-47a2-b479-650faa2d1d9f" />

##### Output

- `data/phase_1/1A_merged_data_hourly_2023_2025.csv`

<img width="542" height="133" alt="image" src="https://github.com/user-attachments/assets/4718329f-b2cc-4645-948e-5eace36d9ec4" />

<img width="473" height="373" alt="{E2F813E0-8D5D-442E-B540-48CA917DFA39}" src="https://github.com/user-attachments/assets/a7af8314-e52b-465e-8099-6a97b644b2bf" />

##### Roli në pipeline

Ky skript krijon dataset-in e parë të integruar orar, që shërben si bazë për të gjitha hapat pasues.

---

### Distinct values

#### `1B_distinct_values.py`

Ky skript bën profilizimin e vlerave unike për një grup kolonash kryesore.

##### Input

- `data/phase_1/1A_merged_data_hourly_2023_2025.csv`

##### Kolonat e përfshira

- ndotësit: `co`, `no2`, `o3`, `pm10`, `pm25`, `so2`
- atributet meteorologjike:
  - temperatura
  - reshjet
  - bora
  - lagështia relative
  - drejtimi i erës
  - shpejtësia e erës
- kolonat e energjisë:
  - `A3_MW`
  - `A4_MW`
  - `A5_MW`
  - `B1_MW`
  - `B2_MW`
  - `total_generation_mw`

##### Çfarë bën

- lexon dataset-in e integruar,

<img width="428" height="126" alt="{85DD1928-3765-4E4A-B0D3-D437772217AC}" src="https://github.com/user-attachments/assets/012286f2-7b62-4f35-90db-f70fb9c366c6" />

- për secilën kolonë nxjerr vlerat unike jo-null,
- i rendit,
- dhe i ruan si CSV të ndarë në folderin `data/phase_1/1B_distinct_values/`.

<img width="523" height="140" alt="{1410133E-14B9-47EE-8AA0-816CBF5B5718}" src="https://github.com/user-attachments/assets/a5667111-5910-4add-9ea8-036b7ce44bf7" />

##### Output

Folderi `1B_distinct_values/` përmban një skedar të veçantë për secilin atribut, p.sh.:

- `distinct_co.csv`
- `distinct_no2.csv`
- `distinct_o3.csv`
- `distinct_pm10.csv`
- `distinct_pm25.csv`
- `distinct_so2.csv`
- `distinct_a3_mw.csv`
- `distinct_a4_mw.csv`
- `distinct_a5_mw.csv`
- `distinct_b1_mw.csv`
- `distinct_b2_mw.csv`
- `distinct_total_generation_mw.csv`
- si dhe skedarët për atributet meteorologjike të pastruara sipas emërtimit.

Pamje nga skedaret unik:

<img width="216" height="289" alt="{DBB27AF7-8935-4189-90AB-624587087BFA}" src="https://github.com/user-attachments/assets/32f47cab-4898-4f79-9eab-887c81351c11" />

##### Roli ne pipeline

Ky hap mbështet eksplorimin fillestar të shpërndarjeve dhe kontrollin e domenit të vlerave.

---

### Data cleaning

#### `2A_datetime_and_duplicates.py`

Ky skript kryen pastrimin fillestar të dimensionit kohor dhe duplikateve.

##### Input

- `data/phase_1/1A_merged_data_hourly_2023_2025.csv`

##### Çarë bën

- konverton `datetime` në format korrekt,
- heq rreshtat ku `datetime` është invalid,
- rendit dataset-in sipas kohës,

<img width="495" height="33" alt="{7CEB88CC-989B-436C-8FA7-0419144A38ED}" src="https://github.com/user-attachments/assets/cec00aab-5620-4c60-9428-4f12eb715584" />

- numëron duplikatet,
- heq duplikatet e plota.

<img width="308" height="93" alt="{1DBC8645-552B-4E2F-A0CB-606E6BD3F65A}" src="https://github.com/user-attachments/assets/50220b1f-63cd-4f8e-a1ef-962ad42637eb" />

##### Output

- `data/phase_1/2A_cleaned_no_duplicates.csv`

##### Roli ne pipeline

Siguron që dataset-i i integruar të ketë rend kronologjik korrekt dhe të mos ketë rreshta të përsëritur.

---

#### `2B_data_quality_cleaning.py`

Ky skript zbaton rregulla të cilësisë së të dhënave.

##### Input

- `data/phase_1/2A_cleaned_no_duplicates.csv`

##### Cfarë bën

1. Për ndotësit:
   - zëvendëson vlerat negative me `NaN`, sepse fizikisht nuk kanë kuptim.

<img width="512" height="93" alt="{0C81D64B-A74D-4E95-A0A6-3984C71E3294}" src="https://github.com/user-attachments/assets/938f0433-10ef-49db-9912-c8ed171f60ae" />

2. Për drejtimin e erës:
   - normalizon këndet me operatorin `% 360`.

<img width="489" height="67" alt="{F02A6AFC-BA27-434F-95B7-86B1A65A9967}" src="https://github.com/user-attachments/assets/ebf9098e-5ae0-4999-ab6c-5c0b10ae8838" />

3. Për reshjet dhe borën:
   - kufizon vlerat minimale në `0`.

<img width="460" height="79" alt="{914CA309-CC22-417F-A765-2859E4665F16}" src="https://github.com/user-attachments/assets/560223dd-1a43-48b4-97a8-32d7618d3001" />

4. Për kolonat e energjisë:
   - kufizon vlerat negative në `0`.

<img width="509" height="91" alt="{5F4D0161-B98D-4182-AC4B-3BFE405120E0}" src="https://github.com/user-attachments/assets/966bc5f6-dbbd-4496-bdaf-e05d89be397a" />

5. Për lagështinë relative:
   - kufizon vlerat në intervalin `[0, 100]`.

<img width="606" height="76" alt="{E9F73771-109C-4B82-9280-FBEE45ED2B89}" src="https://github.com/user-attachments/assets/70822281-b25c-4c46-89d9-1e8164a29079" />

6. Për `total_generation_mw`:
   - e rillogarit nga `A3_MW + A4_MW + A5_MW + B1_MW + B2_MW`
   - dhe korrigjon mospërputhjet me totalin ekzistues.

<img width="549" height="112" alt="{0A3BF60A-A524-4BEA-8DF1-47B6D5D51A61}" src="https://github.com/user-attachments/assets/d4b50c73-9640-48be-8fb8-f56c2cb3412b" />

7. Rrumbullakon kolonat numerike në 3 shifra dhjetore.

<img width="424" height="79" alt="{3661720A-987A-41F9-9DD0-CF8A14E2B71F}" src="https://github.com/user-attachments/assets/bb783372-bf4d-4ca5-8513-540fa23d363c" />

##### Output

- `data/phase_1/2B_quality_checked.csv`

##### Roli në pipeline

Ky hap vendos validim fizik dhe konsistencë numerike mbi të dhënat.

---

#### `2C_missing_values_handling.py`

Ky skript trajton vlerat mungesë.

##### Input

- `data/phase_1/2B_quality_checked.csv`

##### Strategjia e trajtimit

- `pm10` dhe `pm25`: plotësohen me `backfill`
- `co`, `no2`, `o3`, `so2`: plotësohen me `forward fill`
- në fund aplikohet kombinimi `ffill().bfill()` për gjithë dataset-in

##### Çfarë bën

- llogarit mungesat për kolonë dhe përqindjen e tyre,

<img width="486" height="55" alt="{01A3889B-10DA-4CF6-B7CB-E035F8E86192}" src="https://github.com/user-attachments/assets/928a6ed8-7b39-4275-a0f9-2e7ab8a9ee39" />

- raporton sa vlera janë plotësuar për secilin ndotës,

<img width="436" height="38" alt="{CE119CE1-8023-47EF-9506-C10DD4FDF390}" src="https://github.com/user-attachments/assets/4b7ee0b4-ddf2-4a0f-b0f4-05e0661173b7" />

- plotëson vlerat mungesë sipas logjikës së përcaktuar,

<img width="303" height="129" alt="{3540B4D7-B0C9-4BCB-AEF1-1243168EF91D}" src="https://github.com/user-attachments/assets/7ed3034b-734f-434e-bea2-6701b38ef879" />

<img width="303" height="129" alt="image" src="https://github.com/user-attachments/assets/41ba9138-c6ac-4886-8df8-5fabeab93f7c" />

- verifikon sa `NULL` mbeten në fund.

<img width="303" height="40" alt="{73B7F4D2-33C2-4F4A-A6D8-15DA761D7F8F}" src="https://github.com/user-attachments/assets/b9de8d37-492e-48c5-b67d-1d8a163e05f0" />

##### Output

- `data/phase_1/2C_missing_values_handled.csv`

##### Roli në pipeline

Ky hap shmang humbjen e rreshtave dhe prodhon një dataset të plotë për analizat pasuese.

---

#### `2D_validate_final_dataset.py`

Ky skript bën validimin final të dataset-it pas trajtimit të mungesave.

##### Input

- `data/phase_1/2C_missing_values_handled.csv`

##### Çfarë bën

1. Kontrollon raportin fizik ndërmjet:
   - `pm25`
   - `pm10`

   dhe korrigjon rastet kur `pm25 > pm10` duke vendosur `pm25 = pm10`.

<img width="" height="110" alt="image" src="https://github.com/user-attachments/assets/4f5c7fa0-b2b9-4571-916d-129fafd8d098" />

3. Kontrollon gaps kohore:
   - konverton `datetime`,
   - llogarit diferencën ndërmjet rreshtave,
   - numëron boshllëqet më të mëdha se 1 orë.

<img width="366" height="181" alt="image" src="https://github.com/user-attachments/assets/06b9f87f-0840-4ed7-a164-e96a28f134a7" />

3. Kontrollon nëse kanë mbetur `NULL`.

<img width="366" height="141" alt="{F77E8282-105B-45A2-ADFE-DB03A3297653}" src="https://github.com/user-attachments/assets/8b5ce324-d735-43f6-814e-692895bf63d5" />

##### Output

- `data/phase_1/2D_validated_final_dataset.csv`

<img width="925" height="379" alt="{6F19998C-61F9-47E1-9FA8-C1FC8054795B}" src="https://github.com/user-attachments/assets/8bfc891e-b0cc-4c97-b800-451f3fde22b4" />

##### Roli në pipeline

Ky është dataset-i final i pastruar dhe validuar, mbi të cilin kryhen analiza dhe inxhinierim tiparesh.

---

### Feature engineering

#### `3A_target_analysis.py`

Ky skript kryen analizën fillestare të target-it dhe marrëdhënieve të tij me tiparet shpjeguese.

##### Input

- `data/phase_1/2D_validated_final_dataset.csv`

##### Çfarë bën

1. Gjeneron statistika përmbledhëse për ndotësit:
   - `co`
   - `no2`
   - `o3`
   - `pm10`
   - `pm25`
   - `so2`

<img width="" height="60" alt="image" src="https://github.com/user-attachments/assets/1487f6c8-0454-49a3-8a8d-17ede5f5cd2c" />

2. Formon një subset me:
   - ndotësit,
   - kolonat e energjisë,
   - kolonat meteorologjike.

3. Llogarit matricën e korrelacionit.

  <img width="508" height="111" alt="{38275DD5-5A2E-4CFF-91C0-5C666AFF3DFE}" src="https://github.com/user-attachments/assets/86d203a6-4fcc-454a-8efe-d5aeaa473b77" />

5. Krijon dy heatmap-a:
   - korrelacioni i ndotësve me energjinë dhe motin,
   - korrelacioni mes vetë ndotësve.

##### Output

- `pictures/phase_1/pollutant_vs_predictors_heatmap.png`
- `pictures/phase_1/pollutant_correlation_heatmap.png`

##### Roli në pipeline

Ky hap ndihmon në identifikimin e lidhjeve lineare fillestare dhe në justifikimin e tipareve të përdorura më pas në feature engineering.

---

#### `3B_feature_engineering.py`

Ky skript ndërton dataset-in e pasuruar me tipare të reja.

##### Input

- `data/phase_1/2D_validated_final_dataset.csv`

##### Target

- `pm25`

##### Çfarë bën

###### 1. Përgatitje kohore

- konverton `datetime`,
- rendit dataset-in kronologjikisht,
- nxjerr:
  - `hour`
  - `day_of_week`
  - `month`

<img width="507" height="93" alt="{9D5E10B1-7451-40A8-BA92-01DE19B074E0}" src="https://github.com/user-attachments/assets/382465fc-9cae-4af7-925a-1ff1dc0ae6a1" />

###### 2. Encodim ciklik

Krijon:

- `hour_sin`
- `hour_cos`
- `month_sin`
- `month_cos`

<img width="388" height="68" alt="{027EC64A-7F94-4673-8856-A2FA95B7FD55}" src="https://github.com/user-attachments/assets/e6ef76dd-d39e-4241-aadf-313f80d33cb4" />

Qëllimi është të përfaqësojë natyrën ciklike të orës dhe muajit.

###### 3. Lag features

Për kolonat:

- `total_generation_mw`
- `wind_speed_10m (km/h)`
- `temperature_2m (°C)`

krijohen lag-e:

- `lag_1h`
- `lag_3h`
- `lag_6h`

<img width="611" height="128" alt="{0129CD0F-43C3-46C7-857F-CC79A2E4E235}" src="https://github.com/user-attachments/assets/7f5885ea-3b9b-41b0-a1f2-24b55c428940" />

###### 4. Rolling features

Krijohen:

- `total_gen_rolling_sum_12h`
- `total_gen_rolling_sum_24h`

<img width="604" height="34" alt="{84F7C7EB-928A-4284-92DF-77244C86351B}" src="https://github.com/user-attachments/assets/8cd91bcd-5ad6-4c38-b1e7-a03d92793557" />

###### 5. Interaction features

Krijohen:

- `temp_wind_interact`
- `generation_humidity_interact`

<img width="595" height="41" alt="{CB1FE4F3-C35F-4AB7-8E84-B5113E104D46}" src="https://github.com/user-attachments/assets/161bea77-ad84-4b1c-a731-d92b75a50321" />

###### 6. Stagnation proxy

Krijohet:

- `pollution_stagnation_index = total_generation_mw / (wind_speed + 0.1)`

Ky indikator përpiqet të përfaqësojë situatat kur ka prodhim të lartë dhe erë të ulët, pra kushte më të favorshme për grumbullim ndotjesh.

<img width="593" height="31" alt="{7736A968-7617-4B05-AB85-04C693E45840}" src="https://github.com/user-attachments/assets/0b615ce4-1d7f-4258-bfb9-1d04e49561ac" />

###### 7. Wind vector decomposition

Nga shpejtësia dhe drejtimi i erës krijohen:

- `wind_x_vector`
- `wind_y_vector`

<img width="322" height="69" alt="{9EE0FD41-466C-406C-A328-75084CFF86E6}" src="https://github.com/user-attachments/assets/69a803b2-5af1-4a22-beb9-808ec06a6aeb" />

###### 8. Heqja e rreshtave me `NaN`

Pas krijimit të lag-eve dhe rolling windows hiqen rreshtat fillestarë që mbeten pa vlera të plota.

<img width="252" height="34" alt="{9214D524-77A2-425F-88FE-1406798AAE8D}" src="https://github.com/user-attachments/assets/367b4a97-7563-4864-ab33-e71c1d7bd6ea" />

##### Output

- `data/phase_1/3B_engineered_dataset.csv`

##### Roli në pipeline

Ky është dataset-i i parë i pasuruar me tipare që modelojnë dinamikat kohore, ndikimet meteorologjike dhe ndërveprimet me prodhimin e energjisë.

---

### Preprocessing

#### `4A_outlier_treatment.py`

Ky skript trajton outlier-at me quantile capping.

##### Input

- `data/phase_1/3B_engineered_dataset.csv`

##### Strategjia

Për secilën kolonë numerike kandidate:

- kufiri i poshtëm = quantile `0.1%`
- kufiri i sipërm = quantile `99%`

Vlerat jashtë këtij intervali nuk fshihen, por priten në kufijtë përkatës.

##### Kolonat e përjashtuara

- `datetime`
- `date`
- disa tipare ciklike dhe vektorë strukturorë si:
  - `hour_sin`
  - `hour_cos`
  - `month_sin`
  - `month_cos`
  - `wind_x_vector`
  - `wind_y_vector`

##### Çfarë bën

- identifikon kolonat numerike kandidate,

<img width="" height="100" alt="image" src="https://github.com/user-attachments/assets/1277d0ad-20f8-4edd-aa83-7c3518c640b5" />

- llogarit kufijtë e poshtëm dhe të sipërm,

<img width="333" height="264" alt="{7637B515-ECC8-42F7-93E4-F51A36886583}" src="https://github.com/user-attachments/assets/5dc44eae-bd83-4230-8c0c-96b366d56d3f" />

- numëron sa vlera u cap-en në secilin krah,

<img width="300" height="64" alt="{868F8310-591A-4475-A72B-6B3610709F19}" src="https://github.com/user-attachments/assets/cc9dd492-043c-4cf0-8eea-6e39a24f3fd0" />

- krijon një raport për tiparet me më shumë vlera të kufizuara.

<img width="304" height="92" alt="{5C06F89A-7C0E-4C98-B3D6-7EB360549105}" src="https://github.com/user-attachments/assets/f774921a-69c9-493e-bb9d-55b0fe23b267" />

##### Output

- `data/phase_1/4A_outliers_handled.csv`

##### Roli në pipeline

Ky hap redukton ndikimin e vlerave ekstreme pa humbur rreshta.

---

#### `4B_skewness_treatment.py`

Ky skript trajton shtrembërimin e shpërndarjes së kolonave numerike.

##### Input

- `data/phase_1/4A_outliers_handled.csv`

##### Strategjia

Për secilën kolonë numerike:

- llogaritet skewness,
- nëse `|skew| > 1.0`, zbatohet transformim.

##### Llojet e transformimit

- nëse kolona ka vetëm vlera jo-negative:
  - përdoret `log1p`
- ndryshe:
  - përdoret `PowerTransformer(method="yeo-johnson")`

##### Çfarë bën

- krahason skewness para dhe pas transformimit,

<img width="293" height="140" alt="{ACF2D39A-7132-44DE-94FD-02FBADE7EFE2}" src="https://github.com/user-attachments/assets/52ef2624-8f11-497a-a0c8-219132acfe5e" />

- ruan metodën e përdorur për secilën kolonë,

<img width="600" height="283" alt="{54D65469-B661-45E2-8812-109DCE98FE9B}" src="https://github.com/user-attachments/assets/3de4131c-acf9-4a06-80d8-beb30e751223" />

- raporton mean absolute skewness dhe median absolute skewness para/pas.

<img width="" height="86" alt="{27116D5F-1372-4BBB-8835-D8036D487641}" src="https://github.com/user-attachments/assets/33186496-a0f3-45a0-a447-d9ac72219563" />

##### Output

- `data/phase_1/4B_skewness_handled.csv`

<img width="520" height="328" alt="image" src="https://github.com/user-attachments/assets/cf173685-fe02-43a0-b660-4421f45afdb7" />

##### Roli në pipeline

Ky hap i bën shpërndarjet më të përshtatshme për standardizim, analiza lineare dhe modele machine learning.

---

#### `4C_visualization_before_after.py`

Ky skript gjeneron histogramat krahasuese para dhe pas trajtimit të outlier-ave dhe skewness.

##### Input

- `data/phase_1/3B_engineered_dataset.csv`
- `data/phase_1/4A_outliers_handled.csv`
- `data/phase_1/4B_skewness_handled.csv`

##### Tiparet e vizualizuara

- `pm25`
- `total_generation_mw`
- `pollution_stagnation_index`
- `rain (mm)`
- `temp_wind_interact`

##### Çfarë bën

Për secilin atribut:

- vizaton tre histogramë në të njëjtën figurë:
  - para trajtimit,
  - pas trajtimit të outlier-ave,
  - pas trajtimit të skewness.

##### Output

Folderi:

- `pictures/phase_1/4C_visualization_before_after/`

me figurat:

##### PM2.5 Distribution Comparison

![PM2.5](pictures/phase_1/4C_visualization_before_after/pm25_distribution_comparison.png)

##### Total Generation MW Distribution Comparison

![Total Generation](pictures/phase_1/4C_visualization_before_after/total_generation_mw_distribution_comparison.png)

##### Pollution Stagnation Index Distribution Comparison

![Stagnation](pictures/phase_1/4C_visualization_before_after/pollution_stagnation_index_distribution_comparison.png)

##### Rain (mm) Distribution Comparison

![Rain](pictures/phase_1/4C_visualization_before_after/rain_mm_distribution_comparison.png)

##### Temperature-Wind Interaction Distribution Comparison

![Temp Wind](pictures/phase_1/4C_visualization_before_after/temp_wind_interact_distribution_comparison.png)

##### Roli ne pipeline

Ky hap dokumenton vizualisht efektin e transformimeve statistikore.

---

#### `4D_feature_scaling.py`

Ky skript standardizon të gjitha kolonat numerike.

##### Input

- `data/phase_1/4B_skewness_handled.csv`

##### Çfarë bën

- ndan kolonat jo-numerike:
  - `datetime`
  - `date`
    <img width="245" height="40" alt="image" src="https://github.com/user-attachments/assets/3b8d43de-8615-4725-8c5f-c773c74ec3f4" />

- standardizon të gjitha kolonat e tjera me `StandardScaler`,

<img width="245" height="107" alt="{51FDEC3D-668F-4A86-8459-33B135196EF7}" src="https://github.com/user-attachments/assets/3f8a062d-ef76-49dc-b919-029f3b898aca" />

- rikombinon kolonat kohore me kolonat e shkallëzuara,

<img width="345" height="40" alt="image" src="https://github.com/user-attachments/assets/e5a9f4c5-a3b1-4bf1-a60e-6a9f03a08a4e" />

- ruan scaler-in e trajnuar.

<img width="239" height="73" alt="{90AB8B67-BC65-4DC9-BC92-374F23CB0AF9}" src="https://github.com/user-attachments/assets/1a0cae99-1434-486b-b814-794fc2c30c57" />

##### Output

- `data/phase_1/4D_scaled_dataset.csv`
- `models/scaler.pkl`

##### Roli në pipeline

Ky hap siguron që tiparet numerike të jenë në të njëjtën shkallë dhe gati për feature selection ose modelim.

---

#### `4E_feature_selection.py`

Ky skript kryen reduktimin final të tipareve.

##### Input

- `data/phase_1/4D_scaled_dataset.csv`

##### Target

- `pm25`

##### Strategjia e seleksionimit

###### 1. Heqje manuale e kolonave jo të dëshiruara

Hiqen:

- ndotësit e tjerë si variabla hyrëse:
  - `co`
  - `no2`
  - `o3`
  - `pm10`
  - `so2`
- kolona strukturore:
  - `A3_MW`
  - `A4_MW`
  - `A5_MW`
  - `B1_MW`
  - `B2_MW`
  - `hour`
  - `month`
  - `day_of_week`
- të gjitha kolonat me `lag` në emër
- çdo kolonë tjetër që përmban `pm25` përveç target-it

<img width="600" height="170" alt="{EFCA4BD3-8CC8-415E-9549-24C0D552CEE1}" src="https://github.com/user-attachments/assets/4c08beec-c313-4a65-a8df-404cf0206fad" />

###### 2. Heqje e kolonave konstante ose pothuajse konstante

- kolona me vetëm 1 vlerë unike
- kolona me devijim standard pothuajse zero

<img width="433" height="108" alt="{4008EF22-1994-45BF-999C-9B987BC2C534}" src="https://github.com/user-attachments/assets/ee46b9fb-c185-4479-baff-13c2531c6685" />

###### 3. VIF-based elimination

Për kolonat e mbetura:

- llogaritet `Variance Inflation Factor (VIF)`
- hiqet iterativisht kolona me VIF më të lartë derisa:
  - VIF maksimal të jetë më i vogël ose i barabartë me `7.0`

<div>
<img width="" height="50" alt="image" src="https://github.com/user-attachments/assets/21f5dded-12fd-4750-b587-5a92fffa0e48" />
</div>

<div>
<img width="" height="200" alt="image" src="https://github.com/user-attachments/assets/4829919c-e535-4a39-a6cd-b2f6862d06c1" />
</div>

###### 4. Raportim

Në fund raportohet:

- madhësia e dataset-it fillestar,
- madhësia e dataset-it final,
- numri i tipareve finale,
- tiparet e mbajtura, të renditura sipas korrelacionit absolut me `pm25`.

<img width="506" height="151" alt="{7C2D8392-1353-40D2-937B-7035E866EA08}" src="https://github.com/user-attachments/assets/3e56b2a1-22dc-4f9f-843c-b03bd3c7eaee" />

##### Output

- `data/phase_1/4E_selected_dataset.csv`

<img width="1091" height="703" alt="image" src="https://github.com/user-attachments/assets/66b3f216-ac68-45de-915e-7c55089049b7" />

##### Roli në pipeline

Ky është dataset-i final i reduktuar, i përgatitur për modelim statistikor ose machine learning me target `pm25`.

---

### Artefaktet dhe output-et e krijuara

#### Dataset-et e ruajtura ne `data/`

- `1A_merged_data_hourly_2023_2025.csv`  
  Dataset-i i parë i integruar orar.

- `2A_cleaned_no_duplicates.csv`  
  Versioni pa duplikate dhe me `datetime` të validuar.

- `2B_quality_checked.csv`  
  Versioni pas rregullave të cilësisë.

- `2C_missing_values_handled.csv`  
  Versioni pas imputimit dhe plotësimit të mungesave.

- `2D_validated_final_dataset.csv`  
  Dataset-i final i pastruar dhe validuar.

- `3B_engineered_dataset.csv`  
  Dataset-i me tipare të reja.

- `4A_outliers_handled.csv`  
  Dataset-i pas outlier capping.

- `4B_skewness_handled.csv`  
  Dataset-i pas transformimeve kundër skewness.

- `4D_scaled_dataset.csv`  
  Dataset-i i standardizuar.

- `4E_selected_dataset.csv`  
  Dataset-i final i reduktuar për modelim.

#### Artefakte shtesë

- `models/scaler.pkl`  
  Objekti i `StandardScaler` për ripërdorim në inferencë ose pipeline të mëtejshme.

- `data/phase_1/1B_distinct_values/`  
  Folder me vlera unike për atributet kryesore.

---

### Vizualizimet e gjeneruara

#### 1. Heatmap-at nga analiza fillestare

##### `pictures/phase_1/pollutant_vs_predictors_heatmap.png`

Paraqet korrelacionin ndërmjet ndotësve dhe tipareve të energjisë + motit.

##### `pictures/phase_1/pollutant_correlation_heatmap.png`

Paraqet korrelacionin ndërmjet vetë ndotësve atmosferikë.

#### 2. Histogramat krahasuese para/pas

Folderi `pictures/phase_1/4C_visualization_before_after/` përmban figura që krahasojnë shpërndarjen:

- para trajtimit,
- pas trajtimit të outlier-ave,
- pas trajtimit të skewness.

##### Figurat aktuale

- `pm25_distribution_comparison.png`
- `pollution_stagnation_index_distribution_comparison.png`
- `rain_mm_distribution_comparison.png`
- `temp_wind_interact_distribution_comparison.png`
- `total_generation_mw_distribution_comparison.png`

#### Figurat e projektit

##### Pollutant vs Predictors Heatmap

![Pollutant vs Predictors](pictures/phase_1/pollutant_vs_predictors_heatmap.png)

##### Pollutant Correlation Heatmap

![Pollutant Correlation](pictures/phase_1/pollutant_correlation_heatmap.png)

##### PM2.5 Distribution Comparison

![PM2.5](pictures/phase_1/4C_visualization_before_after/pm25_distribution_comparison.png)

##### Total Generation MW Distribution Comparison

![Total Generation](pictures/phase_1/4C_visualization_before_after/total_generation_mw_distribution_comparison.png)

##### Pollution Stagnation Index Distribution Comparison

![Stagnation](pictures/phase_1/4C_visualization_before_after/pollution_stagnation_index_distribution_comparison.png)

##### Rain (mm) Distribution Comparison

![Rain](pictures/phase_1/4C_visualization_before_after/rain_mm_distribution_comparison.png)

##### Temperature-Wind Interaction Distribution Comparison

![Temp Wind](pictures/phase_1/4C_visualization_before_after/temp_wind_interact_distribution_comparison.png)

### Teknikat e zbatuara dhe lidhja me lëndën

Ky projekt përmbush në mënyrë të drejtpërdrejtë temat kryesore të lëndës “Machine Learning”.

#### 1. Data collection

- Shkarkim dhe konsolidim i të dhënave nga burime të ndryshme.
- Përdorim i PowerShell, notebook-ut dhe CSV-ve bruto.

#### 2. Data integration

- Bashkim i tre burimeve heterogjene mbi bosht kohor të përbashkët.
- Harmonizim i formateve të kohës dhe timezone.

#### 3. Data cleaning

- Heqja e duplikateve.
- Korrigjimi i vlerave jo-logjike.
- Kufizim i vlerave fizike jashtë intervaleve të pranueshme.

#### 4. Missing value handling

- Forward fill
- Backfill
- Plotësim i të dhënave pa heqje agresive të rreshtave

#### 5. Validation

- Kontrolli fizik `PM2.5 <= PM10`
- Kontrolli i gaps kohore
- Kontrolli final i `NULL`

#### 6. Exploratory data analysis

- Statistika përmbledhëse
- Matrica korrelacioni
- Heatmap-a për target-in dhe predictor-at

#### 7. Feature engineering

- Encodim ciklik i kohës
- Lag features
- Rolling features
- Interaction terms
- Wind decomposition
- Domain-inspired stagnation index

#### 8. Outlier handling

- Quantile capping me kufijtë `0.5%` dhe `99.5%`
- Qasje robuste pa fshirje të rreshtave

#### 9. Skewness handling

- `log1p`
- `Yeo-Johnson`
- Krahasim para/pas me statistika dhe vizualizime

#### 10. Scaling

- Standardizim i kolonave numerike me `StandardScaler`

#### 11. Feature selection

- Heqje manuale e kolonave jorelevante ose problematike
- Heqje e kolonave konstante
- Reduktim i multikolinearitetit përmes `VIF`

---

### Ekzekutimi i projektit

#### Parakushtet

- Python 3.10+ ose më i ri
- `pip`
- mjedis virtual i rekomanduar
- për skriptin PowerShell: qasje në `aws cli` nëse përdoret shkarkimi nga OpenAQ archive

#### Instalimi i librarive

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

#### Ekzekutimi i pipeline-it

Skriptat ekzekutohen sipas rendit logjik:

```bash
python src/phase_1/integration/1A_merge_data.py
python src/phase_1/distinct_values/1B_distinct_values.py

python src/phase_1/data_cleaning/2A_datetime_and_duplicates.py
python src/phase_1/data_cleaning/2B_data_quality_cleaning.py
python src/phase_1/data_cleaning/2C_missing_values_handling.py
python src/phase_1/data_cleaning/2D_validate_final_dataset.py

python src/phase_1/feature_engineering/3A_target_analysis.py
python src/phase_1/feature_engineering/3B_feature_engineering.py

python src/phase_1/preprocessing/4A_outlier_treatment.py
python src/phase_1/preprocessing/4B_skewness_treatment.py
python src/phase_1/preprocessing/4C_visualization_before_after.py
python src/phase_1/preprocessing/4D_feature_scaling.py
python src/phase_1/preprocessing/4E_feature_selection.py
```

#### Renditja e varësive

Çdo skript varet nga output-i i mëparshëm. Prandaj rekomandohet ekzekutimi në rend strikt.

---

### Rezultati final i pipeline-it

Produkti final i këtij projekti është:

- një dataset i integruar, i pastër dhe i validuar,
- një dataset i pasuruar me tipare domethënëse kohore dhe meteorologjike,
- një version i trajtuar për outlier-a dhe skewness,
- një version i standardizuar,
- dhe në fund një subset final tiparesh me multikolinearitet të reduktuar.

Dataset-i final:

- `data/phase_1/4E_selected_dataset.csv`

është forma më e përshtatshme për:

- modelim prediktiv të `PM2.5`,
- regresion,
- krahasim modelesh machine learning,
- analiza statistikore të marrëdhënieve mes energjisë, motit dhe ndotjes.

---

## 02 Modelimi dhe analiza

Pas përfundimit të pipeline-it të përgatitjes së të dhënave, dataset-i final `data/phase_1/4E_selected_dataset.csv` është përdorur si hyrje për një fazë të dytë të projektit, e fokusuar në modelim dhe analizë të avancuar. Kjo fazë e zgjeron projektin nga një pipeline i pastrimit dhe përgatitjes së të dhënave në një workflow të plotë të machine learning dhe data analysis.

Në këtë fazë janë zhvilluar disa qasje komplementare:

- qasje **supervised**, për parashikimin e `PM2.5` me `CatBoostRegressor`, `LightGBM` dhe `SARIMAX`;
- qasje **unsupervised**, për analizimin e strukturës së brendshme të të dhënave me `HDBSCAN`, `Gaussian Mixture` dhe `Isolation Forest`.

Qëllimi i kësaj pjese nuk është vetëm ndërtimi i modeleve, por edhe demonstrimi që dataset-i final i krijuar nga pipeline-i është realisht i përdorshëm për:

- parashikim,
- validim korrekt kohor,
- interpretim të tipareve,
- dhe eksplorim të cluster-ëve dhe outlier-ave në të dhënat mjedisore dhe energjetike.

---

### Qasja e përgjithshme

Faza e modelimit është ndërtuar mbi parimet e mëposhtme:

1. **Përdorim i dataset-it final të selektuar**
   - Input kryesor për modelet është:
     - `data/phase_1/4E_selected_dataset.csv`

2. **Ruajtje e rendit kronologjik**
   - Për modelin supervised, ndarja e të dhënave është bërë sipas kohës dhe jo rastësisht, për të shmangur leakage dhe për të simuluar më mirë një skenar real parashikimi.

3. **Përdorim i tipareve numerike të përzgjedhura**
   - Dataset-i final tashmë përmban një përzgjedhje tiparesh të reduktuara përmes preprocessing dhe VIF-based feature selection, prandaj është përdorur drejtpërdrejt si bazë për modelim.

4. **Ruajtje e artefakteve**
   - Çdo model ruan output-et e veta në `data/phase_2/`, `models/` dhe `pictures/phase_2/`, në mënyrë që rezultatet të jenë të gjurmueshme dhe të riprodhueshme.

5. **Harmonizim për krahasim**
   - Përtej output-eve native të modeleve, është ndërtuar edhe një shtresë standardizimi me `src/phase_2/comparison/build_phase2_standardized_outputs.py`, e cila mbledh metrikat, figurat dhe tabelat krahasuese në një strukturë të përbashkët për dokumentim.

---

### CatBoost për parashikimin e PM2.5

Për modelimin supervised është përdorur `CatBoostRegressor`, një algoritëm gradient boosting shumë i përshtatshëm për të dhëna tabulare, marrëdhënie jo-lineare dhe ndërveprime komplekse ndërmjet tipareve meteorologjike, energjetike dhe kohore.

Ky model është zgjedhur sepse:

- punon shumë mirë me të dhëna tabulare të përpunuara paraprakisht,
- është më i lehtë për t’u trajnuar sesa modelet deep learning të tipit time-series,
- është i qëndrueshëm ndaj noise-it dhe feature interactions,
- dhe jep lehtësisht interpretim përmes `feature importance`.

#### Input

Modeli lexon dataset-in final:

- `data/phase_1/4E_selected_dataset.csv`

dhe identifikon kolonën kohore (`datetime` ose `date`) për të ruajtur renditjen kronologjike të vëzhgimeve.

#### Target

Target-i i përzgjedhur për modelin supervised është:

- `pm25`

#### Feature-at hyrëse

Pas leximit të dataset-it:

- kolonat boolean, nëse ekzistojnë, kthehen në `int`,
- mbahen kolonat numerike,
- target-i hiqet nga lista e feature-ave,
- përjashtohen kolonat teknike me prapashtesë `"_was_missing"` nëse ekzistojnë.

Në ekzekutimin aktual, modeli ka përdorur këto feature-a:

- `hour_sin`
- `hour_cos`
- `month_sin`
- `month_cos`
- `pollution_stagnation_index`
- `wind_x_vector`
- `wind_y_vector`
- `total_generation_mw`
- `temperature_2m (°C)`
- `rain (mm)`
- `relative_humidity_2m (%)`
- `wind_direction_10m (°)`
- `wind_speed_10m (km/h)`

#### Fragment kyç i kodit: konfigurimi i hyrjes

```python
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent

INPUT_CANDIDATES = [
    BASE_DIR / "data" / "4E_selected_dataset.csv",
    BASE_DIR / "data" / "phase_1" / "4E_selected_dataset.csv",
]

MODEL_DIR = BASE_DIR / "models" / "catboost_model"
PLOTS_DIR = BASE_DIR / "pictures" / "phase_2" / "supervised" / "catboost"
PHASE2_DATA_DIR = BASE_DIR / "data" / "phase_2" / "supervised" / "catboost"

OUTPUT_FORECASTS = PHASE2_DATA_DIR / "catboost_forecasts.csv"
OUTPUT_METRICS = PHASE2_DATA_DIR / "catboost_metrics.csv"
OUTPUT_FEATURES = PHASE2_DATA_DIR / "catboost_feature_importance.csv"
OUTPUT_SPLIT_SUMMARY = PHASE2_DATA_DIR / "catboost_split_summary.csv"

TARGET = "pm25"
TIME_CANDIDATES = ["datetime", "date"]
```

#### Data quality check në këtë fazë

Para trajnimit, skripta bën kontrollin bazë të cilësisë për këtë fazë të modelimit:

- kontrollon ekzistencën e target-it,
- kontrollon mungesat në target dhe feature-a,
- zëvendëson `inf` dhe `-inf` me `NaN`,
- dhe heq rreshtat jo të plotë vetëm nëse janë të nevojshëm.

Në ekzekutimin e raportuar:

- numri i rreshtave hyrës ka qenë **9347**
- numri i feature-ave ka qenë **13**
- mungesa në kolonat e modelit kanë qenë **0**
- rreshta të hequr pas cleaning: **0**

#### Fragment kyç i kodit: kontrollet para modelit

```python
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c != TARGET and not c.endswith("_was_missing")]

for c in [TARGET] + feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df[[TARGET] + feature_cols] = df[[TARGET] + feature_cols].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[TARGET] + feature_cols).copy()
```

#### Validimi korrekt pa leakage

Për këtë model nuk është përdorur `random train_test_split`, por një ndarje kronologjike në tri pjesë:

- `train`
- `validation`
- `test`

Kjo qasje është shumë e rëndësishme për problemin tonë, sepse të dhënat janë kohore dhe modeli duhet të testojë aftësinë për të parashikuar të ardhmen nga e kaluara, jo nga vlera të përziera rastësisht.

Në ekzekutimin aktual, ndarja ka qenë:

- `Train rows: 6542`
- `Val rows: 1402`
- `Test rows: 1403`

me intervale:

- `Train range: 2023-08-18 09:00:00 -> 2025-07-17 21:00:00`
- `Val range: 2025-07-17 22:00:00 -> 2025-09-18 12:00:00`
- `Test range: 2025-09-18 13:00:00 -> 2025-11-27 19:00:00`

#### Fragment kyç i kodit: ndarja kronologjike

```python
n = len(df)
train_end_idx = int(n * TRAIN_RATIO)
val_end_idx = int(n * (TRAIN_RATIO + VAL_RATIO))

train_df = df.iloc[:train_end_idx].copy()
val_df = df.iloc[train_end_idx:val_end_idx].copy()
test_df = df.iloc[val_end_idx:].copy()
```

#### Parametrat e modelit

Modeli `CatBoostRegressor` është inicializuar me parametrat:

- `iterations = 600`
- `learning_rate = 0.03`
- `depth = 6`
- `loss_function = "RMSE"`
- `eval_metric = "RMSE"`
- `early_stopping_rounds = 50`

Ky konfigurim është zgjedhur për të krijuar një model mjaftueshëm të fuqishëm për parashikim, por njëkohësisht praktik për trajnim dhe debug në mjedis lokal.

#### Fragment kyç i kodit: inicializimi i modelit

```python
model = CatBoostRegressor(
    iterations=600,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100
)
```

#### Trajnimi

Gjatë trajnimit, skripta:

- përdor `train` për mësim,
- përdor `validation` për kontroll të performancës,
- aktivizon `use_best_model=True`,
- dhe përdor `early_stopping_rounds=50`.

#### Fragment kyç i kodit: trajnimi dhe validimi

```python
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True,
    early_stopping_rounds=50
)
```

Në ekzekutimin aktual, modeli ka arritur:

- `bestTest = 0.7030203514`
- `bestIteration = 599`

dhe është ruajtur në:

- `models/catboost_model/catboost_pm25_model.cbm`

#### Predikimi dhe metrikat

Pas trajnimit, modeli gjeneron parashikime mbi test set-in dhe llogarit metrikat:

- `MAE`
- `RMSE`
- `MAPE`
- `SMAPE`
- `R²`

Në ekzekutimin e raportuar, rezultatet kanë qenë:

- `MAE = 0.800051`
- `RMSE = 1.005971`
- `MAPE_pct = 357.542306`
- `SMAPE_pct = 108.638466`
- `R2 = 0.331006`
- `n_eval_points = 1403`

#### Fragment kyç i kodit: metrikat

```python
metrics = {
    "MAE": mae(y_true, y_pred),
    "RMSE": rmse(y_true, y_pred),
    "MAPE_pct": mape(y_true, y_pred),
    "SMAPE_pct": smape(y_true, y_pred),
    "R2": float(r2_score(y_true, y_pred))
}
```

#### Çfarë printohet gjatë ekzekutimit

Skripta e CatBoost-it printon në console këto seksione:

- `DATA QUALITY CHECK`
- `CHRONOLOGICAL SPLIT SUMMARY`
- `TRAINING`
- `PREDICTION + METRICS`
- `DONE`

Pra, gjatë ekzekutimit përdoruesi mund të shohë në mënyrë të drejtpërdrejtë:

- numrin e rreshtave hyrës,
- numrin e feature-ave,
- mungesat para cleaning,
- ndarjen train/val/test,
- progresin e trajnimit,
- metrikat finale,
- dhe rrugët ku ruhen file-t.

#### Artefaktet e gjeneruara nga CatBoost

Skripta ruan këto output-e:

- `data/phase_2/supervised/catboost/catboost_forecasts.csv`
  Parashikimet në test set bashkë me vlerat reale dhe residuals.

- `data/phase_2/supervised/catboost/catboost_metrics.csv`
  Tabela e metrikave finale.

- `data/phase_2/supervised/catboost/catboost_feature_importance.csv`
  Rëndësia e secilit feature.

- `data/phase_2/supervised/catboost/catboost_split_summary.csv`
  Përmbledhja e ndarjes kronologjike.

- `models/catboost_model/catboost_pm25_model.cbm`
  Modeli i trajnuar.

- `data/phase_2/supervised/catboost/catboost_run_info.json`
  Përmbledhje e konfigurimit dhe output-eve.

#### Vizualizimi interaktiv

Skripta përfshin edhe ndërtimin e një grafiku interaktiv `Observed vs Predicted` me Plotly, ku parashikohet ruajtja e figurave në:

- `pictures/phase_2/supervised/catboost/catboost_forecast_interactive.html`
- `pictures/phase_2/supervised/catboost/catboost_forecast_interactive.png`

Në versionin aktual të repo-s, krahas figurës interaktive janë gjeneruar edhe figurat statike të standardizuara për krahasim:

- `pictures/phase_2/supervised/catboost/catboost_actual_vs_predicted.png`
- `pictures/phase_2/supervised/catboost/catboost_residual_diagnostics.png`
- `pictures/phase_2/supervised/catboost/catboost_feature_importance.png`
- `pictures/phase_2/supervised/catboost/catboost_metrics_table.png`

---

### LightGBM për parashikimin e PM2.5

#### Pse LightGBM?

- **Shpejtësi dhe efikasitet:** Algoritmi përdor _Gradient Boosting_ të bazuar në pemë, duke ofruar performancë të lartë në të dhëna tabelare.
- **Qartësi analitike (Feature Importance):** Ofron transparencë të plotë mbi peshën që ka secili faktor (prodhimi i energjisë elektrike, era, temperatura) në rezultatin final.

#### 1. Skenari izolues (Modeli baseline)

Në këtë fazë, modeli u trajnua duke përdorur ekskluzivisht prodhimin e energjisë dhe kushtet meteorologjike, pa përfshirë informacion mbi ndotjen e orëve të kaluara.

<img width="800" height="120" alt="image" src="https://github.com/user-attachments/assets/8f29a1b5-b0c6-4f6f-9358-e43522753493" /> <br />

- **R² Score:** 0.1944
- **Konstatimi:** Rezultati prej ~19% është një gjetje e rëndësishme. Kjo vërteton shkencërisht se sado të sakta të jenë të dhënat e motit apo raportimet e energjisë, PM2.5 nuk mund të parashikohet saktë vetëm përmes inputeve të çastit, pasi grimcat kanë një natyrë të lartë akumuluese në atmosferë.

#### 2. Modeli dinamik (Skenari me "lags")

Për të rregulluar këtë dhe për t'u dhënë kuptim faktorëve tanë kryesorë, modeli u përmirësua duke përfshirë variablat autogresive **pm25_lag_1** dhe **pm25_lag_24** (kujtesa e ajrit). Kjo rriti saktësinë në mënyrë drastike nga **0.19 në 0.73**.

<img width="800" height="120" alt="image" src="https://github.com/user-attachments/assets/7a87d2be-a7ae-47b8-a219-3c9c764669d7" /> <br />

**Roli i motit dhe energjisë në modelin e ri:**
Përfshirja e _lags_ nuk i zhvlerëson variablat tona, por i vendos në kontekstin e duhur fizik:

- **Gjendja bazë:** Ndotja paraprake (`lag`) përcakton sasinë e smogut që tashmë gjendet në atmosferë.
- **Katalizatorët:** Prodhimi i energjisë dhe moti veprojnë si agjentët kryesorë që rrisin këtë bazë (përmes emetimeve shtesë) ose e ulin atë (përmes shpërndarjes nga era).

#### Rezultatet dhe vizualizimet

**Rëndësia e veçorive:**
Analiza e fitimit (gain) vërteton hipotezën e projektit tonë: ndërkohë që inercia kohore (`lag`) vendos bazën, **temperatura** dhe **prodhimi i energjisë** renditen menjëherë pas saj si faktorët e jashtëm me ndikimin më të lartë në luhatjen e cilësisë së ajrit.

![Feature Importance](pictures/phase_2/supervised/lightgbm_improved/lightgbm_feature_importance.png)

**Parashikimi vs Realiteti:**
Grafiku tregon se modeli i përmirësuar dinamik arrin të ndjekë me saktësi pikat ekstreme (peaks) të ndotjes gjatë kombinimit të prodhimit të lartë me kushtet atmosferike jo të favorshme.

![Actual vs Predicted](pictures/phase_2/supervised/lightgbm_improved/lightgbm_actual_vs_predicted.png)

Në përditësimin më të fundit të këtij modeli janë raportuar edhe metrikat relative të gabimit:

- `MAE = 2.0827`
- `RMSE = 3.2537`
- `R² = 0.7454`
- `MAPE = 20.78%`
- `SMAPE = 19.90%`

#### Konkluzioni

Ky eksperiment vërteton se ndikimi i termocentraleve dhe motit në Prishtinë është shumë domethënës, por efekti i tyre i vërtetë mund të matet dhe parashikohet saktë vetëm kur modeli merr parasysh natyrën akumuluese të smogut në atmosferë.

#### Artifaktet e gjeneruara

- `baseline_model.joblib` / `improved_model.joblib`: Modelet e ruajtura.
- `metrics_summary.txt`: Përmbledhja e metrikave (`MAE`, `RMSE`, `R²`, `MAPE`, `SMAPE`).
- `feature_importance.csv` dhe `feature_importance.png`: Pesha e saktë e ndikimit për çdo variabël.
- `actual_vs_predicted.png`: Grafiku kohor i përputhshmërisë mes parashikimit të modelit dhe ndotjes reale.
- `learning_curve.png`: Kurba e rënies së gabimit gjatë procesit të trajnimit të modelit.
- Për dokumentim të harmonizuar në fazën e dytë përdoren edhe:
  - `data/phase_2/supervised/lightgbm_improved/feature_importance.csv`
  - `data/phase_2/supervised/lightgbm_improved/metrics_summary.txt`
  - `pictures/phase_2/supervised/lightgbm_improved/lightgbm_actual_vs_predicted.png`
  - `pictures/phase_2/supervised/lightgbm_improved/lightgbm_feature_importance.png`
  - `pictures/phase_2/supervised/lightgbm_improved/lightgbm_learning_curve.png`
  - `pictures/phase_2/supervised/lightgbm_improved/lightgbm_metrics_table.png`

---

### SARIMAX për parashikimin e PM2.5

Përveç modeleve tree-based, në këtë projekt është implementuar edhe `SARIMAX` (`Seasonal AutoRegressive Integrated Moving Average with eXogenous variables`), një model statistikor shumë i përshtatshëm për seri kohore me:

- varësi autoregresive,
- sezonalitet të qartë,
- dhe ndikim nga variabla të jashtëm si moti dhe prodhimi i energjisë.

Kjo e bën `SARIMAX` një zgjedhje shumë të fortë akademikisht për temën tonë, sepse jo vetëm parashikon `PM2.5`, por edhe lejon interpretim të drejtpërdrejtë të:

- memorjes kohore të ndotjes,
- sezonalitetit 24-orësh,
- dhe rolit të faktorëve meteorologjikë dhe energjetikë si variabla exogenous.

Implementimi ndodhet në:

- `src/phase_2/supervised/sarimax_model/sarimax_model.py`

#### Pse SARIMAX?

Ky model është zgjedhur sepse:

- është benchmark statistikor i fortë për seri kohore mjedisore;
- kap njëkohësisht komponentin autoregresiv, moving average dhe sezonalitetin ditor;
- lejon shtimin e feature-ave exogenous pa e humbur interpretueshmërinë;
- dhe prodhon koeficientë statistikisht të lexueshëm, gjë shumë e vlefshme për dokumentim akademik.

Në termat e projektit tonë, `SARIMAX` i përgjigjet drejtpërdrejt pyetjes nëse `PM2.5` në Prishtinë mund të shpjegohet si kombinim i:

- gjendjes së vet në të kaluarën,
- ciklit ditor të ndotjes,
- kushteve atmosferike,
- dhe prodhimit të energjisë.

#### Input

Skripta përdor si lokacion standard dataset-in final:

- `data/phase_1/4E_selected_dataset.csv`

Për kompatibilitet me versionet e mëhershme të repo-s, ajo kontrollon edhe fallback-un:

- `data/4E_selected_dataset.csv`

Në ekzekutimin aktual të raportuar në repo, input-i real ka qenë:

- `data/phase_1/4E_selected_dataset.csv`

#### Target

Target-i i modelit supervised është:

- `pm25`

#### Feature-at exogenous të përdorura

Në konfigurimin final janë përdorur 9 feature-a exogenous:

- `hour_sin`
- `hour_cos`
- `pollution_stagnation_index`
- `wind_x_vector`
- `wind_y_vector`
- `total_generation_mw`
- `temperature_2m (°C)`
- `rain (mm)`
- `relative_humidity_2m (%)`

Kjo zgjedhje është shumë e arsyeshme për një model statistikor si `SARIMAX`, sepse mban vetëm tiparet më kuptimplota dhe shmang fryrjen e panevojshme të modelit me shumë variabla të njëkohshme.

#### Fragment kyç i kodit: konfigurimi i modelit

```python
TARGET = "pm25"
FORECAST_HORIZON = 24

EXOG_FEATURE_PRIORITY = [
    "hour_sin",
    "hour_cos",
    "pollution_stagnation_index",
    "wind_x_vector",
    "wind_y_vector",
    "total_generation_mw",
    "temperature_2m (°C)",
    "rain (mm)",
    "relative_humidity_2m (%)",
]

MODEL_CANDIDATES = [
    {"order": (1, 0, 1), "seasonal_order": (1, 0, 1, 24), "trend": "c"},
    {"order": (2, 0, 1), "seasonal_order": (1, 0, 1, 24), "trend": "c"},
    {"order": (1, 0, 2), "seasonal_order": (1, 0, 1, 24), "trend": "c"},
    {"order": (1, 0, 1), "seasonal_order": (1, 1, 1, 24), "trend": "c"},
]
```

#### Përgatitja e të dhënave

Para trajnimit, skripta:

- identifikon kolonën kohore (`datetime` ose `date`);
- i rendit vëzhgimet në mënyrë kronologjike;
- heq duplikatet eventuale sipas timestamp-it;
- konverton target-in dhe feature-at në formë numerike;
- zëvendëson `inf` dhe `-inf` me `NaN`;
- dhe ruan vetëm rreshtat validë për target-in dhe feature-at exogenous.

Pas këtij hapi janë përdorur:

- `9347` rreshta totale
- `9` feature-a exogenous

#### Validimi korrekt pa leakage

Një pikë shumë e rëndësishme metodologjikisht është se `SARIMAX` nuk është trajnuar me ndarje rastësore, por me ndarje kronologjike `train/validation/test`. Kjo është qasja e duhur për seri kohore, sepse modeli duhet të parashikojë të ardhmen nga e kaluara, jo nga të dhëna të përziera.

Ndarja finale ka qenë:

- `Train rows: 6542`
- `Validation rows: 1402`
- `Test rows: 1403`

me intervale:

- `Train range: 2023-08-18 09:00:00 -> 2025-07-17 21:00:00`
- `Validation range: 2025-07-17 22:00:00 -> 2025-09-18 12:00:00`
- `Test range: 2025-09-18 13:00:00 -> 2025-11-27 19:00:00`

#### Fragment kyç i kodit: ndarja kronologjike

```python
n = len(df)
train_end_idx = int(n * TRAIN_RATIO)
val_end_idx = int(n * (TRAIN_RATIO + VAL_RATIO))

train_df = df.iloc[:train_end_idx].copy()
val_df = df.iloc[train_end_idx:val_end_idx].copy()
test_df = df.iloc[val_end_idx:].copy()
```

#### Zgjedhja e modelit final

Përzgjedhja nuk është bërë me vetëm një konfigurim të vetëm, por me krahasim të katër kandidatëve `SARIMAX` mbi validation set. Kjo është shumë e rëndësishme për dokumentim akademik, sepse tregon se modeli final është zgjedhur mbi bazë performance dhe jo vetëm mbi intuitë.

Kandidatët e testuar kanë qenë:

- `(1, 0, 1) x (1, 0, 1, 24)` me `trend = c`
- `(2, 0, 1) x (1, 0, 1, 24)` me `trend = c`
- `(1, 0, 2) x (1, 0, 1, 24)` me `trend = c`
- `(1, 0, 1) x (1, 1, 1, 24)` me `trend = c`

Sipas `validation_RMSE`, modeli më i mirë ka dalë:

- `order = (1, 0, 1)`
- `seasonal_order = (1, 0, 1, 24)`
- `trend = "c"`

me rezultat:

- `Validation RMSE = 2.0431`
- `Validation R² = 0.7140`

Modeli final më pas është ritrajnuar mbi `train + validation`, ndërsa testimi final është bërë vetëm mbi `test`, duke ruajtur një holdout të pastër kohor.

#### Fragment kyç i kodit: zgjedhja dhe ritrajnimi final

```python
for candidate in MODEL_CANDIDATES:
    record, _ = evaluate_candidate(train_df, val_df, feature_cols, candidate, scaler)
    candidate_rows.append(record)

combined_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
final_result = fit_sarimax(combined_df[TARGET], combined_df[feature_cols], final_candidate)
```

#### Mënyra e parashikimit

Parashikimi në test set është kryer me qasje `state_space_one_step_ahead`, pra modeli ecën hap pas hapi në kohë duke përditësuar gjendjen e tij. Kjo është një mënyrë shumë e përshtatshme për një problem real forecast-imi.

Për secilin timestamp në test ruhen:

- vlera reale e `PM2.5`,
- parashikimi i modelit,
- kufiri i poshtëm i intervalit,
- kufiri i sipërm i intervalit,
- dhe residual-i.

#### Rezultatet e raportuara

Në `validation` modeli ka arritur:

- `MAE = 1.3354`
- `RMSE = 2.0431`
- `MAPE = 19.12%`
- `SMAPE = 17.96%`
- `R² = 0.7140`

Në `test` modeli ka arritur:

- `MAE = 3.1125`
- `RMSE = 4.7654`
- `MAPE = 28.01%`
- `SMAPE = 25.47%`
- `R² = 0.7748`

Gjithashtu janë ruajtur edhe metrika shtesë të modelit statistik:

- `AIC = 9795.08`
- `BIC = 9899.73`

Këto rezultate e bëjnë `SARIMAX` një model shumë serioz dhe të balancuar për raportim akademik: ai është më i interpretueshëm sesa boosting methods dhe njëkohësisht jep performancë të mirë në të dhëna reale.

#### Interpretimi i koeficientëve

Nga `data/phase_2/supervised/sarimax/sarimax_coefficients.csv`, koeficientët më domethënës janë:

- `ar.L1 = 0.8815`, që tregon memorje të fortë autoregresive të `PM2.5`;
- `ar.S.L24 = 0.7981`, që konfirmon sezonalitetin ditor 24-orësh;
- `ma.S.L24 = -0.5190`, që tregon korrigjim sezonal në komponentin e gabimit;
- `hour_sin = 0.3071` dhe `hour_cos = 0.1411`, që tregojnë ndikim të qartë të ciklit ditor;
- `temperature_2m (°C) = 0.0985`, që ka dalë pozitiv dhe statistikisht i rëndësishëm;
- `wind_x_vector = -0.0287`, që sugjeron efekt shpërndarës të erës në një nga komponentët e saj.

Në aspekt interpretimi, këto vlera tregojnë se:

- `PM2.5` në Prishtinë ka inercion të fortë kohor;
- ekziston një ritëm i qartë ditor në nivelin e ndotjes;
- dhe moti nuk vepron i izoluar, por si modulator mbi një proces që tashmë ka kujtesë atmosferike.

#### Vizualizimet

Grafiku kryesor i parashikimit:

![SARIMAX Actual vs Predicted](pictures/phase_2/supervised/sarimax/sarimax_actual_vs_predicted.png)

Diagnostika e residualeve:

![SARIMAX Residual Diagnostics](pictures/phase_2/supervised/sarimax/sarimax_residual_diagnostics.png)

Paneli shtesë i koeficientëve më të fortë:

![SARIMAX Coefficients](pictures/phase_2/supervised/sarimax/sarimax_coefficients.png)

Përveç figurave statike, është ruajtur edhe vizualizimi interaktiv:

- `pictures/phase_2/supervised/sarimax/sarimax_forecast_interactive.html`

#### Artefaktet e gjeneruara nga SARIMAX

Skripta ruan këto output-e:

- `data/phase_2/supervised/sarimax/sarimax_forecasts.csv`
  Parashikimet në test set, intervalet e besimit dhe residuals.

- `data/phase_2/supervised/sarimax/sarimax_metrics.csv`
  Metrikat në validation dhe test, si dhe AIC/BIC.

- `data/phase_2/supervised/sarimax/sarimax_coefficients.csv`
  Koeficientët finalë dhe p-value-t për secilin parametër.

- `data/phase_2/supervised/sarimax/sarimax_candidate_results.csv`
  Krahasimi i konfigurimeve kandidate gjatë model selection.

- `data/phase_2/supervised/sarimax/sarimax_split_summary.csv`
  Përmbledhja e ndarjes kronologjike.

- `data/phase_2/supervised/sarimax/sarimax_residuals.csv`
  Residuals në njësinë reale të `PM2.5`.

- `data/phase_2/supervised/sarimax/sarimax_run_info.json`
  Konfigurimi i plotë i ekzekutimit dhe rrugët e output-eve.

- `models/sarimax_model/sarimax_pm25_model.pkl`
  Modeli final i trajnuar.

- `models/sarimax_model/sarimax_summary.txt`
  Përmbledhja statistikore e `statsmodels`.

- `models/sarimax_model/sarimax_feature_columns.pkl`
  Lista e feature-ave exogenous të përdorura.

- `pictures/phase_2/supervised/sarimax/sarimax_actual_vs_predicted.png`
- `pictures/phase_2/supervised/sarimax/sarimax_residual_diagnostics.png`
- `pictures/phase_2/supervised/sarimax/sarimax_forecast_interactive.html`
- `pictures/phase_2/supervised/sarimax/sarimax_coefficients.png`

---

### HDBSCAN për analizë unsupervised

Për analizën unsupervised është përdorur `HDBSCAN`, një algoritëm clustering i bazuar në densitet, i cili nuk kërkon përcaktim paraprak të numrit të cluster-ëve dhe është shumë i përshtatshëm për të dhëna reale me shape të parregullt, densitete të ndryshme dhe presence të outlier-ave.

Kjo pjesë është ndërtuar për të eksploruar strukturën latente të dataset-it final dhe për të identifikuar:

- profile të ngjashme të vëzhgimeve,
- cluster-a me kushte të ngjashme meteorologjike dhe energjetike,
- si dhe pikat që sillen si noise ose anomali.

#### Input

Si edhe te CatBoost, hyrja është:

- `data/phase_1/4E_selected_dataset.csv`

#### Përgatitja e feature-ave

Për HDBSCAN përdoren kolonat numerike të dataset-it final. Në këtë fazë:

- kolonat boolean, nëse ekzistojnë, kthehen në `int`,
- zgjidhen kolonat numerike,
- përjashtohen kolonat teknike ose kolonat që krijohen nga vetë clustering-u,
- përjashtohen kolonat me prapashtesë `"_was_missing"`.

#### Fragment kyç i kodit: përzgjedhja e kolonave numerike

```python
def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    for c in bool_cols:
        df[c] = df[c].astype(int)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    drop_like = {
        "unnamed: 0",
        "cluster_label",
        "cluster_probability",
        "outlier_score",
        "umap_1",
        "umap_2",
    }

    num_cols = [c for c in num_cols if c not in drop_like and not c.endswith("_was_missing")]
    return num_cols
```

#### Standardizimi

Para clustering-ut, tiparet standardizohen me `StandardScaler`, në mënyrë që kolonat me shkallë të ndryshme të mos dominojnë ndërtimin e cluster-ëve.

#### Fragment kyç i kodit: scaling

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)
```

#### Parametrat e HDBSCAN

Modeli është konfiguruar me:

- `min_cluster_size = 80`
- `min_samples = 20`
- `cluster_selection_method = "eom"`
- `metric = "euclidean"`

#### Fragment kyç i kodit: inicializimi i HDBSCAN

```python
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    cluster_selection_method=CLUSTER_SELECTION_METHOD,
    metric=METRIC,
    prediction_data=True,
    gen_min_span_tree=True,
)
```

#### Çfarë prodhon HDBSCAN

Pas trajnimit, modeli gjeneron për çdo vëzhgim:

- `cluster_label`
- `cluster_probability`
- `outlier_score`

Këto kolona shtohen në dataset-in final të cluster-uar.

#### Reduktimi dimensional për vizualizim

Për të vizualizuar cluster-at në 2 dimensione, skripta përdor `UMAP` me konfigurim:

- `n_neighbors = 30`
- `min_dist = 0.05`
- `n_components = 2`

#### Fragment kyç i kodit: UMAP

```python
reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.05,
    n_components=2,
    metric="euclidean",
    random_state=42,
)

embedding = reducer.fit_transform(X_scaled)
```

Pas këtij hapi krijohen kolonat:

- `umap_1`
- `umap_2`

të cilat përdoren për vizualizimin interaktiv të cluster-ëve.

#### Metrikat e clustering-ut

Për vlerësimin e strukturës së cluster-ëve, skripta llogarit:

- `silhouette_score`
- `davies_bouldin_score`
- `calinski_harabasz_score`

duke përjashtuar pikat `noise` (`cluster_label = -1`) aty ku kërkohet.

#### Fragment kyç i kodit: metrikat e brendshme

```python
internal = {
    "silhouette_score": silhouette_score(X_core, y_core),
    "davies_bouldin_score": davies_bouldin_score(X_core, y_core),
    "calinski_harabasz_score": calinski_harabasz_score(X_core, y_core),
}
```

#### Çfarë printohet gjatë ekzekutimit

Skripta e HDBSCAN është ndërtuar që të printojë në console këto seksione:

- `DATA QUALITY CHECK`
- `SCALING`
- `HDBSCAN TRAINING`
- `UMAP EMBEDDING`
- `CLUSTERING METRICS`
- `INTERACTIVE VISUALIZATION`
- `DONE`

Pra, gjatë ekzekutimit përdoruesi mund të shohë:

- sa rreshta ka dataset-i para dhe pas cleaning,
- cilat feature përdoren,
- metrikat e clustering-ut,
- sa cluster-a janë gjetur,
- sa pika janë klasifikuar si noise,
- dhe ku janë ruajtur output-et.

#### Artefaktet e gjeneruara nga HDBSCAN

Skripta ruan këto output-e:

- `data/phase_2/unsupervised/hdbscan/hdbscan_clustered_dataset.csv`
  Dataset-i final me kolonat `cluster_label`, `cluster_probability`, `outlier_score`, `umap_1`, `umap_2`.

- `data/phase_2/unsupervised/hdbscan/hdbscan_metrics.csv`
  Metrikat e clustering-ut dhe përmbledhja e modelit.

- `data/phase_2/unsupervised/hdbscan/hdbscan_cluster_summary.csv`
  Përmbledhje statistikore për çdo cluster.

- `data/phase_2/unsupervised/hdbscan/hdbscan_feature_summary.csv`
  Përmbledhje e tipareve që dallojnë më shumë cluster-at.

- `models/hdbscan_model/hdbscan_model.pkl`
  Modeli i trajnuar.

- `models/hdbscan_model/hdbscan_scaler.pkl`
  Scaler-i i përdorur për standardizim.

- `models/hdbscan_model/hdbscan_umap.pkl`
  Objekti i ruajtur i reduktimit dimensional.

- `data/phase_2/unsupervised/hdbscan/hdbscan_run_info.json`
  Informacion për konfigurimin dhe rrugët e output-eve.

#### Vizualizimi interaktiv

Vizualizimi interaktiv i cluster-ëve gjenerohet në:

- `pictures/phase_2/unsupervised/hdbscan/hdbscan_umap_interactive.html`
- `pictures/phase_2/unsupervised/hdbscan/hdbscan_umap_interactive.png`

Për krahasim të harmonizuar janë gjeneruar edhe:

- `pictures/phase_2/unsupervised/hdbscan/hdbscan_pm25_by_cluster.png`
- `pictures/phase_2/unsupervised/hdbscan/hdbscan_pm25_timeline.png`
- `pictures/phase_2/unsupervised/hdbscan/hdbscan_pm25_zoom.png`
- `pictures/phase_2/unsupervised/hdbscan/hdbscan_scatter.png`
- `pictures/phase_2/unsupervised/hdbscan/hdbscan_confidence_distribution.png`
- `pictures/phase_2/unsupervised/hdbscan/hdbscan_feature_shift_panel.png`
- `pictures/phase_2/unsupervised/hdbscan/hdbscan_metrics_table.png`

Ky vizualizim lejon:

- dallimin e cluster-ëve në plan 2D,
- evidentimin e noise/outlier points,
- dhe inspektimin e feature-ave kryesore për secilin vëzhgim përmes hover.

---

### Gaussian Mixture për analizë unsupervised

Përveç `HDBSCAN`, në këtë projekt është implementuar edhe `Gaussian Mixture Model (GMM)`, një metodë probabilistike clustering që modelon të dhënat si kombinim i disa shpërndarjeve Gaussiane.

Kjo qasje është shumë e vlefshme në kontekstin tonë, sepse kushtet atmosferike dhe regjimet e ndotjes nuk janë gjithmonë të ndara në cluster-a të prerë fort. Shpesh kemi profile që mbivendosen, dhe `GMM` e kap pikërisht këtë me:

- anëtarësim probabilistik në cluster,
- fleksibilitet në forma eliptike të shpërndarjes,
- dhe interpretim të regjimeve mjedisore si profile të buta, jo si ndarje strikte.

Implementimi ndodhet në:

- `src/phase_2/unsupervised/gaussian_mixture_model/gaussian_mixture_model.py`

#### Pse Gaussian Mixture?

`Gaussian Mixture` është zgjedhur sepse:

- jep soft clustering, jo vetëm etiketim të fortë;
- është shumë i përshtatshëm kur profilet mjedisore mbivendosen pjesërisht;
- lejon krahasim formal modelesh me `BIC` dhe `AIC`;
- dhe shërben si kundërpeshë metodologjike ndaj `HDBSCAN`, i cili është density-based.

Nga pikëpamja akademike, kjo e forcon shumë projektin, sepse demonstron dy filozofi të ndryshme unsupervised:

- një qasje me densitet dhe noise handling (`HDBSCAN`);
- dhe një qasje probabilistike me model-based clustering (`Gaussian Mixture`).

#### Input

Ashtu si te `SARIMAX`, skripta përdor si lokacion standard dataset-in final:

- `data/phase_1/4E_selected_dataset.csv`

dhe ruan edhe një fallback për kompatibilitet me strukturat e mëhershme:

- `data/4E_selected_dataset.csv`

Në ekzekutimin aktual është përdorur:

- `data/phase_1/4E_selected_dataset.csv`

#### Feature-at e përdorura për clustering

Modeli është trajnuar mbi 12 feature-a numerike:

- `hour_sin`
- `hour_cos`
- `month_sin`
- `month_cos`
- `pollution_stagnation_index`
- `wind_x_vector`
- `wind_y_vector`
- `total_generation_mw`
- `temperature_2m (°C)`
- `rain (mm)`
- `relative_humidity_2m (%)`
- `wind_speed_10m (km/h)`

Është shumë e rëndësishme të theksohet se:

- `pm25` nuk përdoret si input për të ndërtuar cluster-at;
- `pm25` përdoret vetëm pas trajnimit për interpretim post-hoc, përmes kolonës `pm25_real`.

Kjo e bën analizën më të pastër metodologjikisht: cluster-at nuk “detyrohen” të formohen sipas target-it, por më pas kontrollohet si sillet `PM2.5` brenda secilit regjim të zbuluar.

#### Fragment kyç i kodit: konfigurimi i feature-ave dhe kandidatëve

```python
PCA_VARIANCE_THRESHOLD = 0.95
N_COMPONENT_CANDIDATES = [2, 3, 4, 5, 6]
COVARIANCE_TYPES = ["full", "diag", "tied"]
MIN_CLUSTER_RATIO = 0.05

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
]
```

#### Standardizimi dhe PCA

Para clustering-ut, tiparet standardizohen me `StandardScaler`, që asnjë kolonë me shkallë më të madhe të mos dominojë modelin.

Më pas aplikohet `PCA` me prag:

- `explained variance >= 95%`

Në ekzekutimin aktual kjo ka dhënë:

- `12` feature-a hyrëse
- `9` PCA components
- `97.12%` explained variance

Kjo do të thotë se reduktimi dimensional e ruan pothuajse të gjithë informacionin kryesor, ndërsa e bën clustering-un më stabil dhe më të lehtë për vizualizim.

#### Fragment kyç i kodit: scaling dhe PCA

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols].to_numpy(dtype=float))

pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, svd_solver="full")
X_pca = pca.fit_transform(X_scaled)
```

#### Përzgjedhja e modelit final

Përzgjedhja e modelit është bërë në mënyrë të strukturuar, duke testuar kombinime të:

- `n_components = 2, 3, 4, 5, 6`
- `covariance_type = full, diag, tied`

Për secilin kandidat janë llogaritur:

- `BIC`
- `AIC`
- `silhouette_score`
- `davies_bouldin_score`
- `calinski_harabasz_score`
- `min_cluster_ratio`
- `avg_cluster_confidence`

Një kandidat konsiderohet valid për zgjedhje finale vetëm nëse:

- `min_cluster_ratio >= 0.05`

Modeli final zgjidhet kryesisht sipas `BIC` më të ulët, dhe në rast afërsie përdoret `silhouette_score` si kriter dytësor.

Konfigurimi më i mirë në këtë repo ka dalë:

- `n_components = 6`
- `covariance_type = "full"`

#### Fragment kyç i kodit: logjika e model selection

```python
if row["min_cluster_ratio"] < MIN_CLUSTER_RATIO:
    continue

if best_row is None:
    best_row = row
    continue

if row["bic"] < best_row["bic"] - 1e-9:
    best_row = row
    continue

if np.isclose(row["bic"], best_row["bic"]) and row["silhouette_score"] > best_row["silhouette_score"]:
    best_row = row
```

#### Rezultatet e raportuara

Në konfigurimin final janë marrë këto rezultate:

- `Rows used = 9347`
- `Number of clusters = 6`
- `Covariance type = full`
- `BIC = 154996.08`
- `AIC = 152646.10`
- `Silhouette score = 0.0899`
- `Davies-Bouldin score = 2.0751`
- `Calinski-Harabasz score = 950.04`
- `Min cluster ratio = 0.1128`
- `Max cluster ratio = 0.2679`
- `Average cluster confidence = 0.9688`

Këtu vlen një interpretim i kujdesshëm akademik:

- `silhouette_score` është relativisht i ulët, që tregon se cluster-at nuk janë të ndarë në mënyrë shumë të fortë;
- kjo është normale për të dhëna reale mjedisore, ku regjimet atmosferike shpesh mbivendosen;
- ndërkohë, `avg_cluster_confidence = 0.9688` dhe raportet e balancuara të cluster-ëve tregojnë se modeli po prodhon ndarje të përdorshme dhe të qëndrueshme.

Pra, te `GMM` nuk duhet parë vetëm silhouette, por kombinimi i:

- `BIC/AIC`,
- balancës së cluster-ëve,
- probabiliteteve të anëtarësimit,
- dhe interpretueshmërisë së profileve të gjetura.

#### Interpretimi i cluster-ëve

Nga `data/phase_2/unsupervised/gaussian_mixture/gmm_cluster_summary.csv` dhe `data/phase_2/unsupervised/gaussian_mixture/gmm_feature_summary.csv`, dalin disa profile shumë interesante:

- `Cluster 3` është regjimi me ndotjen mesatare më të lartë:
  - `pm25_real_mean = 17.51`
  - lidhet me temperaturë nën mesatare, lagështi më të lartë, prodhim energjie më të lartë dhe stagnation mbi mesatare.

- `Cluster 0` paraqet kushte më të pastra dhe më të ajrosura:
  - `pm25_real_mean = 7.56`
  - ka komponentë më të fortë të erës dhe stagnation më të ulët.

- `Cluster 5` përfaqëson regjim të lidhur me reshje:
  - `pm25_real_mean = 8.15`
  - `rain (mm)` është feature-i më devijues pozitiv në këtë cluster.

Kjo do të thotë se `Gaussian Mixture` nuk po ndan të dhënat vetëm sipas një variable të vetme, por po zbulon regjime mjedisore me kombinime të ndryshme të:

- stinës,
- orës së ditës,
- reshjeve,
- stagnation-it atmosferik,
- erës,
- dhe intensitetit të prodhimit energjetik.

#### Vizualizimet

Krahasimi i kandidatëve gjatë model selection:

![GMM Model Selection](pictures/phase_2/unsupervised/gaussian_mixture/gmm_model_selection.png)

Heatmap-i i profileve të cluster-ëve:

![GMM Cluster Profile Heatmap](pictures/phase_2/unsupervised/gaussian_mixture/gmm_cluster_profile_heatmap.png)

Profili i `PM2.5` sipas cluster-it:

![GMM PM25 by Cluster](pictures/phase_2/unsupervised/gaussian_mixture/gmm_pm25_by_cluster.png)

Shpërndarja e cluster-ëve në raport me energjinë dhe `PM2.5`:

![GMM Scatter](pictures/phase_2/unsupervised/gaussian_mixture/gmm_scatter.png)

Vizualizimi interaktiv në hapësirën e reduktuar me `PCA` ruhet në:

- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_pca_interactive.html`

#### Artefaktet e gjeneruara nga Gaussian Mixture

Skripta ruan këto output-e:

- `data/phase_2/unsupervised/gaussian_mixture/gmm_clustered_dataset.csv`
  Dataset-i me etiketat e cluster-it, probabilitetet e anëtarësimit dhe koordinatat `PCA`.

- `data/phase_2/unsupervised/gaussian_mixture/gmm_metrics.csv`
  Përmbledhja e metrikave finale të clustering-ut.

- `data/phase_2/unsupervised/gaussian_mixture/gmm_cluster_summary.csv`
  Statistika për secilin cluster, përfshirë `pm25_real_mean`.

- `data/phase_2/unsupervised/gaussian_mixture/gmm_feature_summary.csv`
  Feature-at që dallojnë më së shumti secilin cluster nga mesatarja globale.

- `data/phase_2/unsupervised/gaussian_mixture/gmm_model_selection.csv`
  Tabela e plotë e kandidatëve të testuar.

- `data/phase_2/unsupervised/gaussian_mixture/gmm_run_info.json`
  Informacioni i konfigurimit dhe rrugët e output-eve.

- `models/gaussian_mixture_model/gmm_model.pkl`
  Modeli final i trajnuar.

- `models/gaussian_mixture_model/gmm_scaler.pkl`
  Scaler-i i përdorur për standardizim.

- `models/gaussian_mixture_model/gmm_pca.pkl`
  Objekti `PCA` i ruajtur.

- `models/gaussian_mixture_model/gmm_feature_columns.pkl`
  Lista e feature-ave të përdorura.

- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_model_selection.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_cluster_profile_heatmap.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_pca_interactive.html`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_pm25_by_cluster.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_pm25_timeline.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_pm25_zoom.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_scatter.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_confidence_distribution.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_feature_shift_panel.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_metrics_table.png`

---

### Isolation Forest për analizë unsupervised

#### Pse Isolation Forest?
Ndryshe nga modelet e tjera që përpiqen të mësojnë se çfarë është "normale", ky algoritëm izolon drejtpërdrejt pikat që sillen në mënyrë të pazakontë (anomalitë). Modeli nuk kërkon thjesht vlera ekstreme (p.sh. ajër shumë i ndotur), por gjen situata ku variablat nuk përputhen logjikisht me njëra-tjetrën (p.sh. ndotje e ulët kur prodhimi i energjisë është i lartë).

#### Identifikimi i goditjeve kohore (Zoom-in Analysis)
Për të kuptuar saktësinë e modelit, kemi zmadhuar dritaren kohore në periudhën kritike Nëntor–Dhjetor 2025.

![Isolation Forest Zoom-in PM2.5](pictures/phase_2/unsupervised/isolation_forest/isolation_forest_pm25_zoom.png)

Konstatimi: Siç shihet nga pikat e kuqe, modeli nuk gjeneron alarme të rreme gjatë luhatjeve të zakonshme. Ai shënjestron me saktësi vetëm momentet (peaks) kur niveli i PM2.5 del plotësisht jashtë kontrollit apo ritmit sezonal.

#### Korrelacioni vizual: Ndotja vs. Energjia (Scatter Plot)
Ky grafik ballafaqon prodhimin total të energjisë (MW) me nivelet e PM2.5 për të parë se ku "thyhet" normaliteti.

![Isolation Forest Energy vs PM2.5 Scatter](pictures/phase_2/unsupervised/isolation_forest/isolation_forest_scatter.png)

Interpretimi i shpërndarjes:

Reja e pikave gri (Normaliteti): Përfaqëson 95% të ditëve ku prodhimi dhe ndotja ndjekin një trend të parashikueshëm.

Korniza e kuqe (Anomalitë): Anomalitë grumbullohen në skajet ekstreme. Vlen të theksohet se pikat në pjesën e poshtme djathtas (prodhim i lartë, ndotje e ulët) janë lënë si gri nga modeli kur të dhënat e motit kanë treguar prani të erës apo shiut, duke vërtetuar se modeli kupton kontekstin meteorologjik.

#### Skenarët kryesorë të zbuluar
Nga analiza e 100 anomalive më ekstreme, veçuam dy raste që vërtetojnë lidhjen mes energjisë dhe mjedisit:

"Mrekullia e 27 Nëntorit" (Anomalia e ajrit të pastër):
Modeli kapi një moment ku KEK-u po punonte me kapacitet maksimal dimëror, por ndotja ishte pothuajse zero. Kjo u izolua si anomali pasi theu lidhjen logjike prodhim-ndotje; shpjegimi fizik ishte prania e një stuhie që pastroi ajrin pavarësisht emetimeve.

"Ndotja e Natës" (2 Prill 2024, ora 04:00):
Në një kohë kur trafiku i qytetit është zero, modeli zbuloi një rritje të papritur të PM2.5. Duke parë variablat e energjisë, u vërtetua se fiks në atë orë prodhimi ishte në mbingarkesë, duke bërë lidhjen direkte mes punës së termocentraleve dhe smogut nocturn.

#### Konkluzioni
Ndërkohë që algoritmet supervised si LightGBM parashikojnë trendin e përgjithshëm, Isolation Forest shërben si një "radar" për të identifikuar dështimet e sistemit dhe goditjet mjedisore. Ky model vërteton se ndikimi i energjisë në ajër është i pranishëm, por mund të modifikohet drastikisht nga kushtet atmosferike.

Artifaktet e gjeneruara

![Isolation Forest PM2.5 Trend](pictures/phase_2/unsupervised/isolation_forest/isolation_forest_pm25.png)

![Isolation Forest Energy Trend](pictures/phase_2/unsupervised/isolation_forest/isolation_forest_energy.png)

- `pictures/phase_2/unsupervised/isolation_forest/isolation_forest_pm25_zoom.png`: Pamja e zmadhuar e anomalive.
- `pictures/phase_2/unsupervised/isolation_forest/isolation_forest_scatter.png`: Grafiku i korrelacionit `PM2.5` vs energji.
- `pictures/phase_2/unsupervised/isolation_forest/isolation_forest_pm25.png`: Ecuria kohore e anomalive për `PM2.5`.
- `pictures/phase_2/unsupervised/isolation_forest/isolation_forest_energy.png`: Ecuria kohore e prodhimit energjetik.
- `pictures/phase_2/unsupervised/isolation_forest/isolation_forest_score_distribution.png`: Shpërndarja e score-ve të anomalive.
- `data/phase_2/unsupervised/isolation_forest/isolation_forest_top_anomalies.csv`: Regjistri i rasteve më anormale për analizë të detajuar.

---

### Ekzekutimi i fazës së dytë

Pas sqarimit të secilit model, riprodhueshmëria e pipeline-it të fazës së dytë mund të dokumentohet qartë duke i ekzekutuar skriptat individualisht nga root-i i projektit me interpreter-in e environment-it virtual:

```powershell
.\.venv\Scripts\python.exe src\phase_2\supervised\catboost_model\catboost_model.py
.\.venv\Scripts\python.exe src\phase_2\supervised\lightgbm_model\lightgbm_model.py
.\.venv\Scripts\python.exe src\phase_2\supervised\sarimax_model\sarimax_model.py

.\.venv\Scripts\python.exe src\phase_2\unsupervised\hdbscan_model\hdbscan_model.py
.\.venv\Scripts\python.exe src\phase_2\unsupervised\gaussian_mixture_model\gaussian_mixture_model.py
.\.venv\Scripts\python.exe src\phase_2\unsupervised\isolation_forest_model\isolation_forest_extended_outputs.py

.\.venv\Scripts\python.exe src\phase_2\comparison\build_phase2_standardized_outputs.py
```

Skripta e fundit është veçanërisht e rëndësishme për README dhe raportim, sepse:

- gjeneron `data/phase_2/comparison/supervised_model_comparison.csv`;
- gjeneron `data/phase_2/comparison/unsupervised_model_comparison.csv`;
- krijon figura krahasuese në `pictures/phase_2/comparison/`;
- dhe harmonizon vizualizimet statike për modelet supervised dhe unsupervised.

Kjo renditje e seksionit e bën README-n më të natyrshëm akademikisht: fillimisht prezantohen modelet dhe interpretimi i tyre, ndërsa në fund tregohet qartë si mund të riprodhohen rezultatet.

---

### Krahasimi i harmonizuar i modeleve

Për të pasur krahasim sa më të qartë dhe sa më profesional, output-et e fazës së dytë janë harmonizuar në dy familje:

- **supervised**, ku modelet krahasohen sipas `MAE`, `RMSE`, `R²`, `MAPE` dhe `SMAPE`;
- **unsupervised**, ku modelet krahasohen sipas numrit të grupeve kryesore, raportit të pikave speciale (`noise` ose `anomaly ratio`), si dhe metrikave të brendshme të clustering-ut kur këto janë të aplikueshme.

Ky harmonizim është shumë i rëndësishëm akademikisht, sepse:

- shmang krahasimet e paqarta mes output-eve heterogjene;
- bën të mundur dokumentimin paralel të modeleve të ndryshme;
- dhe lejon interpretim më të drejtë të rolit të secilit algoritëm brenda projektit.

Figura e mëposhtme përmbledh krahasimin standard të modeleve supervised:

![Supervised Comparison Table](pictures/phase_2/comparison/supervised_comparison_table.png)

Kjo figurë tregon në një tabelë të vetme strategjinë e evaluimit dhe metrikat kryesore për `LightGBM`, `CatBoost` dhe `SARIMAX`. Ajo e bën të qartë që jo të gjitha modelet janë testuar me të njëjtën strategji vlerësimi, prandaj interpretimi duhet bërë me kujdes.

Për supervised, është gjeneruar edhe paneli i përbashkët i tipareve më të rëndësishme:

![Supervised Feature Panels](pictures/phase_2/comparison/supervised_feature_panels.png)

Ky panel ndihmon në krahasimin e tre logjikave të ndryshme të interpretimit:

- `feature importance` për modelet boosting;
- koeficientët për `SARIMAX`;
- dhe rolin e tipareve energjetike, meteorologjike dhe kohore në të tre qasjet.

Për unsupervised, tabela e harmonizuar është kjo:

![Unsupervised Comparison Table](pictures/phase_2/comparison/unsupervised_comparison_table.png)

Kjo figurë vendos në të njëjtin kornizë `HDBSCAN`, `Gaussian Mixture` dhe `Isolation Forest`, duke i interpretuar sipas natyrës së tyre: clustering për dy të parat dhe anomaly detection për të fundit.

Për profilin e `PM2.5` në secilin model unsupervised është gjeneruar edhe figura:

![Unsupervised PM25 Profiles](pictures/phase_2/comparison/unsupervised_pm25_profiles.png)

Kjo figurë ndihmon shumë në raport, sepse e bën të dukshme si ndryshon niveli mesatar i `PM2.5` ndërmjet cluster-ëve apo ndërmjet grupeve `Normal/Anomaly`.

---

### Rezultatet kryesore të fazës së dytë

#### Rezultatet supervised

| Modeli | Strategjia e evaluimit | MAE | RMSE | R² | MAPE (%) | SMAPE (%) |
|---|---|---:|---:|---:|---:|---:|
| LightGBM Improved | TimeSeriesSplit CV mean | 2.0827 | 3.2537 | 0.7454 | 20.78 | 19.90 |
| CatBoost | Chronological holdout test | 2.6918 | 4.3210 | 0.8147 | 23.4860 | 21.5382 |
| SARIMAX | Chronological holdout test | 3.1125 | 4.7654 | 0.7748 | 28.0101 | 25.4697 |

Interpretimi i drejtë i kësaj tabele është:

- `CatBoost` ka performancën më të fortë në holdout test sipas `R²`;
- `LightGBM` ka gabimet mesatare më të ulëta në `TimeSeriesSplit CV`, por kjo nuk është plotësisht e krahasueshme një-me-një me holdout test;
- `SARIMAX` mbetet modeli më i interpretueshëm statistikisht dhe njëkohësisht jep performancë solide në test set.

#### Rezultatet unsupervised

| Modeli | Lloji | Rows | Feature-a | Grupet kryesore | Noise / Anomaly Ratio | Avg. Confidence / Severity | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HDBSCAN | Clustering | 9347 | 14 | 2 | 0.7354 | 0.9882 | 0.2658 | 1.2785 | 266.1120 |
| Gaussian Mixture | Clustering | 9347 | 12 | 6 | 0.0000 | 0.9688 | 0.0899 | 2.0751 | 950.0418 |
| Isolation Forest | Anomaly Detection | 9347 | 13 | 2 | 0.0501 | 0.0267 | N/A | N/A | N/A |

Interpretimi akademik i rezultateve unsupervised është:

- `HDBSCAN` ofron ndarje më të fortë mes cluster-ëve bazë, por me shumë pika të klasifikuara si `noise`, gjë që është normale për një metodë density-based konservative;
- `Gaussian Mixture` prodhon një ndarje më të imët në 6 regjime dhe është shumë i vlefshëm për interpretim probabilistik të profileve mjedisore;
- `Isolation Forest` nuk është model clustering, prandaj nuk krahasohet me `silhouette` ose `Davies-Bouldin`, por me cilësinë e zbulimit të anomalive dhe interpretimin e rasteve të veçanta.

---

### Metrikat dhe interpretimi i rezultateve

Në këtë fazë janë përdorur dy nivele interpretimi:

#### 1. Interpretimi supervised

Te modelet supervised (`CatBoost`, `LightGBM`, `SARIMAX`), interpretimi bazohet në:

- metrikat e regresionit,
- krahasimin ndërmjet vlerave reale dhe të parashikuara,
- residuals,
- rëndësinë e feature-ave,
- dhe, në rastin e `SARIMAX`, edhe koeficientët statistikorë, `AIC/BIC` dhe diagnostikën e residualeve.

Kjo ndihmon në kuptimin se:

- sa mirë modeli e parashikon `PM2.5`,
- cilat tipare ndikojnë më shumë në parashikim,
- sa i fortë është komponenti kohor dhe sezonal,
- dhe sa e qëndrueshme është performanca në validation dhe test set.

#### 2. Interpretimi unsupervised

Te modelet unsupervised (`HDBSCAN`, `Gaussian Mixture` dhe `Isolation Forest`), interpretimi bazohet në:

- numrin dhe përmasat e cluster-ëve,
- pikat noise,
- probabilitetet e anëtarësimit në cluster,
- outlier scores,
- raportin e anomalive,
- `BIC/AIC` te modelet probabilistike,
- dhe përmbledhjet statistikore të feature-ave sipas cluster-it.

Kjo ndihmon për të kuptuar:

- nëse të dhënat ndahen në profile natyrore,
- nëse ekzistojnë regjime të ndryshme të ndotjes,
- sa të ndara apo të mbivendosura janë këto regjime,
- dhe cilat kombinime të motit dhe energjisë shfaqin sjellje të ngjashme.

---

### Artefaktet e krijuara nga modelet

Pas fazës së dytë të projektit, përveç output-eve të pipeline-it të përgatitjes së të dhënave, janë krijuar edhe artefakte të reja modelimi.

#### CatBoost

- `data/phase_2/supervised/catboost/catboost_forecasts.csv`
- `data/phase_2/supervised/catboost/catboost_metrics.csv`
- `data/phase_2/supervised/catboost/catboost_feature_importance.csv`
- `data/phase_2/supervised/catboost/catboost_split_summary.csv`
- `data/phase_2/supervised/catboost/catboost_run_info.json`
- `models/catboost_model/catboost_pm25_model.cbm`
- `pictures/phase_2/supervised/catboost/catboost_forecast_interactive.html`
- `pictures/phase_2/supervised/catboost/catboost_forecast_interactive.png`
- `pictures/phase_2/supervised/catboost/catboost_actual_vs_predicted.png`
- `pictures/phase_2/supervised/catboost/catboost_residual_diagnostics.png`
- `pictures/phase_2/supervised/catboost/catboost_feature_importance.png`
- `pictures/phase_2/supervised/catboost/catboost_metrics_table.png`

#### LightGBM

- `src/phase_2/supervised/lightgbm_model/improved_model/improved_model.joblib`
- `src/phase_2/supervised/lightgbm_model/improved_model/metrics_summary.txt`
- `src/phase_2/supervised/lightgbm_model/improved_model/feature_importance.csv`
- `src/phase_2/supervised/lightgbm_model/improved_model/feature_importance.png`
- `src/phase_2/supervised/lightgbm_model/improved_model/actual_vs_predicted.png`
- `src/phase_2/supervised/lightgbm_model/improved_model/learning_curve.png`
- `data/phase_2/supervised/lightgbm_improved/metrics_summary.txt`
- `data/phase_2/supervised/lightgbm_improved/feature_importance.csv`
- `pictures/phase_2/supervised/lightgbm_improved/lightgbm_actual_vs_predicted.png`
- `pictures/phase_2/supervised/lightgbm_improved/lightgbm_feature_importance.png`
- `pictures/phase_2/supervised/lightgbm_improved/lightgbm_learning_curve.png`
- `pictures/phase_2/supervised/lightgbm_improved/lightgbm_metrics_table.png`

#### HDBSCAN

- `data/phase_2/unsupervised/hdbscan/hdbscan_clustered_dataset.csv`
- `data/phase_2/unsupervised/hdbscan/hdbscan_metrics.csv`
- `data/phase_2/unsupervised/hdbscan/hdbscan_cluster_summary.csv`
- `data/phase_2/unsupervised/hdbscan/hdbscan_feature_summary.csv`
- `data/phase_2/unsupervised/hdbscan/hdbscan_run_info.json`
- `models/hdbscan_model/hdbscan_model.pkl`
- `models/hdbscan_model/hdbscan_scaler.pkl`
- `models/hdbscan_model/hdbscan_umap.pkl`
- `pictures/phase_2/unsupervised/hdbscan/hdbscan_umap_interactive.html`
- `pictures/phase_2/unsupervised/hdbscan/hdbscan_umap_interactive.png`
- `pictures/phase_2/unsupervised/hdbscan/hdbscan_pm25_by_cluster.png`
- `pictures/phase_2/unsupervised/hdbscan/hdbscan_pm25_timeline.png`
- `pictures/phase_2/unsupervised/hdbscan/hdbscan_pm25_zoom.png`
- `pictures/phase_2/unsupervised/hdbscan/hdbscan_scatter.png`
- `pictures/phase_2/unsupervised/hdbscan/hdbscan_confidence_distribution.png`
- `pictures/phase_2/unsupervised/hdbscan/hdbscan_feature_shift_panel.png`
- `pictures/phase_2/unsupervised/hdbscan/hdbscan_metrics_table.png`

#### SARIMAX

- `data/phase_2/supervised/sarimax/sarimax_forecasts.csv`
- `data/phase_2/supervised/sarimax/sarimax_metrics.csv`
- `data/phase_2/supervised/sarimax/sarimax_coefficients.csv`
- `data/phase_2/supervised/sarimax/sarimax_candidate_results.csv`
- `data/phase_2/supervised/sarimax/sarimax_split_summary.csv`
- `data/phase_2/supervised/sarimax/sarimax_residuals.csv`
- `data/phase_2/supervised/sarimax/sarimax_run_info.json`
- `models/sarimax_model/sarimax_pm25_model.pkl`
- `models/sarimax_model/sarimax_summary.txt`
- `models/sarimax_model/sarimax_feature_columns.pkl`
- `pictures/phase_2/supervised/sarimax/sarimax_actual_vs_predicted.png`
- `pictures/phase_2/supervised/sarimax/sarimax_residual_diagnostics.png`
- `pictures/phase_2/supervised/sarimax/sarimax_forecast_interactive.html`
- `pictures/phase_2/supervised/sarimax/sarimax_coefficients.png`
- `pictures/phase_2/supervised/sarimax/sarimax_metrics_table.png`

#### Gaussian Mixture

- `data/phase_2/unsupervised/gaussian_mixture/gmm_clustered_dataset.csv`
- `data/phase_2/unsupervised/gaussian_mixture/gmm_metrics.csv`
- `data/phase_2/unsupervised/gaussian_mixture/gmm_cluster_summary.csv`
- `data/phase_2/unsupervised/gaussian_mixture/gmm_feature_summary.csv`
- `data/phase_2/unsupervised/gaussian_mixture/gmm_model_selection.csv`
- `data/phase_2/unsupervised/gaussian_mixture/gmm_run_info.json`
- `models/gaussian_mixture_model/gmm_model.pkl`
- `models/gaussian_mixture_model/gmm_scaler.pkl`
- `models/gaussian_mixture_model/gmm_pca.pkl`
- `models/gaussian_mixture_model/gmm_feature_columns.pkl`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_model_selection.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_cluster_profile_heatmap.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_pca_interactive.html`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_pm25_by_cluster.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_pm25_timeline.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_pm25_zoom.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_scatter.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_confidence_distribution.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_feature_shift_panel.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_metrics_table.png`

#### Isolation Forest

- `data/phase_2/unsupervised/isolation_forest/isolation_forest_scored_dataset.csv`
- `data/phase_2/unsupervised/isolation_forest/isolation_forest_metrics.csv`
- `data/phase_2/unsupervised/isolation_forest/isolation_forest_feature_summary.csv`
- `data/phase_2/unsupervised/isolation_forest/isolation_forest_top_anomalies.csv`
- `data/phase_2/unsupervised/isolation_forest/isolation_forest_run_info.json`
- `models/isolation_forest_model/isolation_forest_model.pkl`
- `models/isolation_forest_model/isolation_forest_feature_columns.pkl`
- `pictures/phase_2/unsupervised/isolation_forest/isolation_forest_pm25.png`
- `pictures/phase_2/unsupervised/isolation_forest/isolation_forest_energy.png`
- `pictures/phase_2/unsupervised/isolation_forest/isolation_forest_pm25_zoom.png`
- `pictures/phase_2/unsupervised/isolation_forest/isolation_forest_scatter.png`
- `pictures/phase_2/unsupervised/isolation_forest/isolation_forest_score_distribution.png`
- `pictures/phase_2/unsupervised/isolation_forest/isolation_forest_feature_shift_panel.png`
- `pictures/phase_2/unsupervised/isolation_forest/isolation_forest_metrics_table.png`

#### Output-et krahasuese të fazës së dytë

- `data/phase_2/comparison/supervised_model_comparison.csv`
- `data/phase_2/comparison/unsupervised_model_comparison.csv`
- `pictures/phase_2/comparison/supervised_comparison_table.png`
- `pictures/phase_2/comparison/supervised_error_metrics.png`
- `pictures/phase_2/comparison/supervised_r2_comparison.png`
- `pictures/phase_2/comparison/supervised_feature_panels.png`
- `pictures/phase_2/comparison/unsupervised_comparison_table.png`
- `pictures/phase_2/comparison/unsupervised_special_ratio_and_groups.png`
- `pictures/phase_2/comparison/unsupervised_clustering_quality.png`
- `pictures/phase_2/comparison/unsupervised_feature_panels.png`
- `pictures/phase_2/comparison/unsupervised_pm25_profiles.png`

---

### Vizualizimet interaktive

Në këtë fazë janë ndërtuar jo vetëm vizualizime interaktive, por edhe figura statike të standardizuara për krahasim akademik mes modeleve.

#### Vizualizimi i CatBoost

Grafiku interaktiv `Observed vs Predicted` është konceptuar për të paraqitur:

- serinë reale të `PM2.5`,
- parashikimet në test set,
- residuals,
- dhe kufijtë kohorë ndërmjet `train`, `validation` dhe `test`.

Ky vizualizim ruhet në:

- `pictures/phase_2/supervised/catboost/catboost_forecast_interactive.html`

#### Vizualizimi i HDBSCAN

Vizualizimi 2D me `UMAP` është konceptuar për të paraqitur:

- shpërndarjen e vëzhgimeve në hapësirë të reduktuar,
- cluster-at e gjetur nga HDBSCAN,
- noise points,
- dhe karakteristikat kryesore në hover.

Ky vizualizim ruhet në:

- `pictures/phase_2/unsupervised/hdbscan/hdbscan_umap_interactive.html`

#### Vizualizimi i SARIMAX

Vizualizimi i `SARIMAX` është ndërtuar për të paraqitur:

- serinë reale të `PM2.5`,
- parashikimin në `test set`,
- intervalet e besimit,
- si dhe diagnostikën e residualeve në formë statike.

Artefaktet kryesore të vizualizimit janë:

- `pictures/phase_2/supervised/sarimax/sarimax_forecast_interactive.html`
- `pictures/phase_2/supervised/sarimax/sarimax_actual_vs_predicted.png`
- `pictures/phase_2/supervised/sarimax/sarimax_residual_diagnostics.png`
- `pictures/phase_2/supervised/sarimax/sarimax_coefficients.png`

#### Vizualizimi i Gaussian Mixture

Vizualizimi i `Gaussian Mixture` është ndërtuar për të paraqitur:

- cluster-at probabilistikë në hapësirën e reduktuar me `PCA`,
- krahasimin e kandidatëve gjatë model selection,
- dhe profilet mesatare të cluster-ëve përmes heatmap-it.

Artefaktet kryesore të vizualizimit janë:

- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_pca_interactive.html`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_model_selection.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_cluster_profile_heatmap.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_pm25_by_cluster.png`
- `pictures/phase_2/unsupervised/gaussian_mixture/gmm_scatter.png`

#### Vizualizimi i Isolation Forest

Vizualizimet e `Isolation Forest` janë ndërtuar për të paraqitur:

- ecurinë kohore të anomalive në `PM2.5`,
- lidhjen mes prodhimit të energjisë dhe ndotjes,
- shpërndarjen e score-ve të anomalisë,
- dhe profilin `Normal vs Anomaly`.

Artefaktet kryesore të vizualizimit janë:

- `pictures/phase_2/unsupervised/isolation_forest/isolation_forest_pm25.png`
- `pictures/phase_2/unsupervised/isolation_forest/isolation_forest_pm25_zoom.png`
- `pictures/phase_2/unsupervised/isolation_forest/isolation_forest_scatter.png`
- `pictures/phase_2/unsupervised/isolation_forest/isolation_forest_score_distribution.png`
- `pictures/phase_2/unsupervised/isolation_forest/isolation_forest_pm25_profile.png`
- `pictures/phase_2/unsupervised/isolation_forest/isolation_forest_metrics_table.png`

---

### Rezultati i zgjeruar i pipeline-it

Produkti final i këtij projekti nuk është më vetëm një dataset i përgatitur, por një bazë e plotë për analizë dhe modelim.

Rezultati final përfshin:

- një dataset të integruar, të pastruar, të validuar dhe të transformuar;
- një subset final tiparesh të përshtatshme për modelim;
- një model supervised `CatBoostRegressor` për parashikimin e `PM2.5`;
- një model supervised `LightGBM` për benchmark dhe analizë me lag features;
- një model supervised `SARIMAX` për forecast kohor të interpretueshëm statistikisht;
- një model unsupervised `HDBSCAN` për clustering dhe outlier analysis;
- një model unsupervised `Gaussian Mixture` për identifikimin probabilistik të regjimeve mjedisore;
- artefakte të metrikave, parashikimeve, cluster-ëve dhe rëndësisë së tipareve;
- si dhe vizualizime interaktive për interpretim më të qartë të rezultateve.

Kjo do të thotë se pipeline-i i ndërtuar në këtë projekt tashmë përbën jo vetëm një proces të përgatitjes së të dhënave, por edhe një bazë funksionale për krahasim modelesh, analiza të mëtejshme dhe zgjerim në faza të ardhshme.

---

---

## Zgjerime në vazhdim

Në vazhdim, ky projekt mund të zgjerohet me:

- modele decision trees për parashikimin e `PM2.5`,
- krahasim modelesh si Random Forest, XGBoost,
- validim temporal `train/validation/test split` sipas kohës,
- analizë më të avancuar të rëndësisë së tipareve,
- krahasim të performancës para dhe pas feature selection.

---

## Anëtarët e grupit

- **Diellza Përvetica**
- **Fatjeta Gashi**
- **Festina Klinaku**

---

## Acknowledgments

- Universiteti i Prishtinës
- Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike
- Dr. Sc. Mërgim H. Hoti
- Burimet publike dhe institucionale të përdorura për ndërtimin e dataset-eve hyrëse
- Të gjithë anëtarët e grupit që kontribuan në ndërtimin e pipeline-it
