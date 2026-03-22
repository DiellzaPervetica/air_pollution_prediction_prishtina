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
      <p><strong>Profesor:</strong> Dr. Sc. Mërgim H. Hoti</p>
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
3. [Burimet e të dhënave](#burimet-e-të-dhënave)
4. [Përshkrimi i dataset-eve hyrëse](#përshkrimi-i-dataset-eve-hyrëse)
5. [Struktura e repository-t](#struktura-e-repository-t)
6. [Topologjia e pipeline-it](#topologjia-e-pipeline-it)
7. [Përshkrimi i detajuar i çdo skripte](#përshkrimi-i-detajuar-i-çdo-skripte)
   - [Data collection](#data-collection)
   - [Integration](#integration)
   - [Distinct values](#distinct-values)
   - [Data cleaning](#data-cleaning)
   - [Feature engineering](#feature-engineering)
   - [Preprocessing](#preprocessing)
8. [Artefaktet dhe output-et e krijuara](#artefaktet-dhe-output-et-e-krijuara)
9. [Vizualizimet e gjeneruara](#vizualizimet-e-gjeneruara)
10. [Teknikat e zbatuara dhe lidhja me lëndën](#teknikat-e-zbatuara-dhe-lidhja-me-lëndën)
11. [Ekzekutimi i projektit](#ekzekutimi-i-projektit)
12. [Rezultati final i pipeline-it](#rezultati-final-i-pipeline-it)
13. [Zgjerime në vazhdim](#zgjerime-në-vazhdim)
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

---

## Qëllimi i punimit

Qëllimi kryesor i këtij projekti është të ndërtojë një dataset të pastër dhe analitikisht të qëndrueshëm për të studiuar marrëdhëniet ndërmjet:

- prodhimit të energjisë elektrike,
- kushteve meteorologjike,
- dhe ndotësve atmosferikë në Prishtinë,

me fokus të veçantë në përdorimin e këtyre të dhënave për parashikimin e `PM2.5`.

Objektivat kryesore janë:

- të integrohen burime heterogjene të të dhënave në një bosht të përbashkët kohor;
- të kontrollohet cilësia e të dhënave dhe të korrigjohen vlera të pasakta;
- të trajtohen mungesat pa humbur informacion të vlefshëm;
- të krijohen tipare të reja kohore, meteorologjike dhe ndërvepruese;
- të zbutet ndikimi i outlier-ave dhe shpërndarjeve shumë të shtrembëruara;
- të standardizohet dataset-i për përdorim në modele statistikore dhe machine learning;
- të eliminohet multikolineariteti i tepërt përmes VIF-based feature selection.

---

## Burimet e të dhënave

Ky projekt bazohet në tre burime kryesore të të dhënave:

### 1. Prodhimi i energjisë elektrike nga termocentralet e Kosovës
Dataset-i përmban prodhimin orar të njësive energjetike:
- `A3_MW`
- `A4_MW`
- `A5_MW`
- `B1_MW`
- `B2_MW`

Nga këto është ndërtuar edhe:
- `total_generation_mw`

Të dhënat janë marrë nga KOSTT dhe janë harmonizuar në nivel orar.

### 2. Të dhënat meteorologjike për Prishtinën
Dataset-i meteorologjik përmban atribute si:
- temperatura,
- reshjet,
- bora,
- lagështia relative,
- drejtimi i erës,
- shpejtësia e erës.

Këto të dhëna janë përdorur për të modeluar kushtet atmosferike që ndikojnë në përhapjen ose stagnimin e ndotjes.

### 3. Të dhënat e ndotjes së ajrit në Prishtinë
Dataset-i i cilësisë së ajrit përmban matje të ndotësve:
- `co`
- `no2`
- `o3`
- `pm10`
- `pm25`
- `so2`

Këto të dhëna janë mbledhur dhe konsoliduar për Prishtinën përmes burimeve të tipit OpenAQ / arkivave përkatëse / notebook-ut të kolektimit të përdorur në projekt.

### Shtrirja kohore
Burimet hyrëse mbulojnë periudhën 2023–2026. Megjithatë, dataset-i i integruar final ruan vetëm intervalin ku të tre burimet kanë mbulim të përbashkët orar, prandaj output-i i parë i integruar ruhet si:

- `1A_merged_data_hourly_2023_2025.csv`

Kjo e bën integrimin kohor të saktë dhe shmang boshllëqet e krijuara nga mungesa e përbashkët midis burimeve.

### Dataset-i i integruar

Pas bashkimit (`merge`) të tre burimeve me `inner join`, dataset-i final përmban vetëm intervalin e përbashkët kohor:

- Numri i rreshtave: **9,370**
- Numri i kolonave: **22**
- Numri total i vlerave: **206,140**
- Intervali kohor: **2023-08-01 → 2025-11-27**

- Reduktimi i numrit të rreshtave është rezultat i sinkronizimit strikt kohor ndërmjet burimeve, ku ruhen vetëm momentet për të cilat ekzistojnë të dhëna në të tre dataset-et.
---

## Përshkrimi i dataset-eve hyrëse

Pipeline-i përdor tre skedarë bruto të ruajtur në `data/raw/`:

- `prishtina_air_quality_2023_2025.csv`
- `prishtina_weather_2023_2026.csv`
- `prishtina_energy_production_2023_2026.csv`

### Dataset-i i ndotjes së ajrit
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

### Dataset-i meteorologjik
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

### Dataset-i i energjisë
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

## Struktura e repository-t

```text
TEST/
│
├── app.py #Vizualizimi i tere projektit
│
├── src/
│   ├── data_cleaning/
│   │   ├── 2A_datetime_and_duplicates.py
│   │   ├── 2B_data_quality_cleaning.py
│   │   ├── 2C_missing_values_handling.py
│   │   └── 2D_validate_final_dataset.py
│   │
│   ├── data_collection/
│   │   ├── get_kosova_air_quality_data.ps1
│   │   └── get_prishtina_air_quality_data.ipynb
│   │
│   ├── distinct_values/
│   │   └── 1B_distinct_values.py
│   │
│   ├── feature_engineering/
│   │   ├── 3A_target_analysis.py
│   │   └── 3B_feature_engineering.py
│   │
│   ├── integration/
│   │   └── 1A_merge_data.py
│   │
│   └── preprocessing/
│       ├── 4A_outlier_treatment.py
│       ├── 4B_skewness_treatment.py
│       ├── 4C_visualization_before_after.py
│       ├── 4D_feature_scaling.py
│       └── 4E_feature_selection.py
│
├── data/
│   ├── raw/
│   ├── 1B_distinct_values/
│   ├── 1A_merged_data_hourly_2023_2025.csv
│   ├── 2A_cleaned_no_duplicates.csv
│   ├── 2B_quality_checked.csv
│   ├── 2C_missing_values_handled.csv
│   ├── 2D_validated_final_dataset.csv
│   ├── 3B_engineered_dataset.csv
│   ├── 4A_outliers_handled.csv
│   ├── 4B_skewness_handled.csv
│   ├── 4D_scaled_dataset.csv
│   └── 4E_selected_dataset.csv
│
├── models/
│   └── scaler.pkl
│
├── pictures/
│   ├── pollutant_correlation_heatmap.png
│   ├── pollutant_vs_predictors_heatmap.png
│   └── 4C_visualization_before_after/
│       ├── pm25_distribution_comparison.png
│       ├── pollution_stagnation_index_distribution_comparison.png
│       ├── rain_mm_distribution_comparison.png
│       ├── temp_wind_interact_distribution_comparison.png
│       └── total_generation_mw_distribution_comparison.png
│
└── README.md
```

---

## Topologjia e pipeline-it

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

## Përshkrimi i detajuar i çdo skripte

## App.py - Dashboard

- Ky projekt përfshin gjithashtu një dashboard interaktiv të ndërtuar me Streamlit, i cili shërben si një simulator vizual për eksplorimin në kohë reale të ndikimit që kanë prodhimi i termocentraleve dhe kushtet meteorologjike në ndotjen e ajrit në Prishtinë. Përmes këtij vizualizimi, përdoruesi mund të ndryshojë në mënyrë dinamike parametrat e prodhimit energjetik, temperaturës, reshjeve, lagështisë dhe erës, dhe të vëzhgojë menjëherë se si këto ndryshime reflektohen në nivelet e ndotësve kryesorë atmosferikë, veçanërisht te PM2.5. Dashboard-i është konceptuar si një komponent interaktiv dhe intuitiv që e bën analizën më të kuptueshme, më eksploruese dhe më afër një skenari simulues të botës reale.

![Streamlit Dashboard](img.png)

## Data collection

### `get_kosova_air_quality_data.ps1`
Ky skript PowerShell përdoret për shkarkimin e të dhënave arkivore nga OpenAQ për disa `location IDs` të lidhura me Prishtinën ose pikat përkatëse të matjes.

#### Çfarë bën skripta
- krijon folder-in bazë të ruajtjes në disk,
- iteron mbi një listë `location IDs`,
- për secilin lokacion përdor komandën `aws s3 cp` për të shkarkuar skedarët `.csv.gz` nga arkiva publike e OpenAQ,
- ruan të dhënat në nënfolderë të ndarë sipas `location ID`.

#### Qëllimi
Ky hap siguron mbledhjen e të dhënave bruto të ndotjes / matjeve për përpunim të mëtejshëm.

#### Lokacionet e përdorura
Në versionin aktual përdoren:
- `2536`
- `7674`
- `7931`
- `7933`
- `9337`

#### Output
Skedarët bruto ruhen lokalisht në strukturë të ndarë sipas lokacionit.

---

### `get_prishtina_air_quality_data.ipynb`
Ky notebook shërben si mjedis interaktiv për mbledhje, eksplorim, filtrime dhe/ose konsolidim të të dhënave të cilësisë së ajrit për Prishtinën.

Meqë logjika e plotë e notebook-ut nuk është përfshirë këtu në README, roli i tij në projekt është:
- të ndihmojë në eksplorimin fillestar të të dhënave,
- të përgatisë ose eksportojë skedarët bruto/finalë të përdorur më pas në pipeline,
- të shërbejë si hap ndërmjetës midis burimeve online dhe CSV-ve në `data/raw/`.

---

## Integration

### `1A_merge_data.py`
Ky është hapi themelor i integrimit të të tre burimeve.

#### Input
- `data/raw/prishtina_air_quality_2023_2025.csv`
- `data/raw/prishtina_weather_2023_2026.csv`
- `data/raw/prishtina_energy_production_2023_2026.csv`

#### Hapat kryesorë
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
8. Harmonizon timezone-in e ndotjes dhe motit në `Europe/Belgrade`, pastaj i kthen në naive timestamps.

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
14. Krijon kolonat:
    - `date`
    - `hour`
    - `interval_start`
<img width="431" height="94" alt="{AA095FE6-7145-4932-98A4-BCCD0F0B1ACA}" src="https://github.com/user-attachments/assets/9cac6b45-b4fd-47a2-b479-650faa2d1d9f" />

#### Output
- `data/1A_merged_data_hourly_2023_2025.csv`

#### Roli në pipeline
Ky skript krijon dataset-in e parë të integruar orar, që shërben si bazë për të gjitha hapat pasues.

---

## Distinct values

### `1B_distinct_values.py`
Ky skript bën profilizimin e vlerave unike për një grup kolonash kryesore.

#### Input
- `data/1A_merged_data_hourly_2023_2025.csv`

#### Kolonat e përfshira
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


#### Çfarë bën
- lexon dataset-in e integruar,

<img width="428" height="126" alt="{85DD1928-3765-4E4A-B0D3-D437772217AC}" src="https://github.com/user-attachments/assets/012286f2-7b62-4f35-90db-f70fb9c366c6" />

- për secilën kolonë nxjerr vlerat unike jo-null,
- i rendit,
- dhe i ruan si CSV të ndarë në folderin `data/1B_distinct_values/`.

<img width="523" height="140" alt="{1410133E-14B9-47EE-8AA0-816CBF5B5718}" src="https://github.com/user-attachments/assets/a5667111-5910-4add-9ea8-036b7ce44bf7" />


#### Output
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


#### Roli ne pipeline
Ky hap mbështet eksplorimin fillestar të shpërndarjeve dhe kontrollin e domenit të vlerave.

---

## Data cleaning

### `2A_datetime_and_duplicates.py`
Ky skript kryen pastrimin fillestar të dimensionit kohor dhe duplikateve.

#### Input
- `data/1A_merged_data_hourly_2023_2025.csv`

#### Çarë bën
- konverton `datetime` në format korrekt,
- heq rreshtat ku `datetime` është invalid,
- rendit dataset-in sipas kohës,

<img width="495" height="33" alt="{7CEB88CC-989B-436C-8FA7-0419144A38ED}" src="https://github.com/user-attachments/assets/cec00aab-5620-4c60-9428-4f12eb715584" />

- numëron duplikatet,
- heq duplikatet e plota.

<img width="308" height="93" alt="{1DBC8645-552B-4E2F-A0CB-606E6BD3F65A}" src="https://github.com/user-attachments/assets/50220b1f-63cd-4f8e-a1ef-962ad42637eb" />


#### Output
- `data/2A_cleaned_no_duplicates.csv`

#### Roli ne pipeline
Siguron që dataset-i i integruar të ketë rend kronologjik korrekt dhe të mos ketë rreshta të përsëritur.

---

### `2B_data_quality_cleaning.py`
Ky skript zbaton rregulla të cilësisë së të dhënave.

#### Input
- `data/2A_cleaned_no_duplicates.csv`

#### Cfarë bën
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


#### Output
- `data/2B_quality_checked.csv`

#### Roli në pipeline
Ky hap vendos validim fizik dhe konsistencë numerike mbi të dhënat.

---

### `2C_missing_values_handling.py`
Ky skript trajton vlerat mungesë.

#### Input
- `data/2B_quality_checked.csv`

#### Strategjia e trajtimit
- `pm10` dhe `pm25`: plotësohen me `backfill`
- `co`, `no2`, `o3`, `so2`: plotësohen me `forward fill`
- në fund aplikohet kombinimi `ffill().bfill()` për gjithë dataset-in

#### Çfarë bën
- llogarit mungesat për kolonë dhe përqindjen e tyre,

<img width="486" height="55" alt="{01A3889B-10DA-4CF6-B7CB-E035F8E86192}" src="https://github.com/user-attachments/assets/928a6ed8-7b39-4275-a0f9-2e7ab8a9ee39" />

- raporton sa vlera janë plotësuar për secilin ndotës,

<img width="436" height="38" alt="{CE119CE1-8023-47EF-9506-C10DD4FDF390}" src="https://github.com/user-attachments/assets/4b7ee0b4-ddf2-4a0f-b0f4-05e0661173b7" />

- plotëson vlerat mungesë sipas logjikës së përcaktuar,

<img width="303" height="129" alt="{3540B4D7-B0C9-4BCB-AEF1-1243168EF91D}" src="https://github.com/user-attachments/assets/7ed3034b-734f-434e-bea2-6701b38ef879" />

<img width="303" height="129" alt="image" src="https://github.com/user-attachments/assets/41ba9138-c6ac-4886-8df8-5fabeab93f7c" />

- verifikon sa `NULL` mbeten në fund.

<img width="303" height="40" alt="{73B7F4D2-33C2-4F4A-A6D8-15DA761D7F8F}" src="https://github.com/user-attachments/assets/b9de8d37-492e-48c5-b67d-1d8a163e05f0" />


#### Output
- `data/2C_missing_values_handled.csv`

#### Roli në pipeline
Ky hap shmang humbjen e rreshtave dhe prodhon një dataset të plotë për analizat pasuese.

---

### `2D_validate_final_dataset.py`
Ky skript bën validimin final të dataset-it pas trajtimit të mungesave.

#### Input
- `data/2C_missing_values_handled.csv`

#### Çfarë bën
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


#### Output
- `data/2D_validated_final_dataset.csv`

#### Roli në pipeline
Ky është dataset-i final i pastruar dhe validuar, mbi të cilin kryhen analiza dhe inxhinierim tiparesh.

---

## Feature engineering

### `3A_target_analysis.py`
Ky skript kryen analizën fillestare të target-it dhe marrëdhënieve të tij me tiparet shpjeguese.

#### Input
- `data/2D_validated_final_dataset.csv`

#### Çfarë bën
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

#### Output
- `pictures/pollutant_vs_predictors_heatmap.png`
- `pictures/pollutant_correlation_heatmap.png`

#### Roli në pipeline
Ky hap ndihmon në identifikimin e lidhjeve lineare fillestare dhe në justifikimin e tipareve të përdorura më pas në feature engineering.

---

### `3B_feature_engineering.py`
Ky skript ndërton dataset-in e pasuruar me tipare të reja.

#### Input
- `data/2D_validated_final_dataset.csv`

#### Target
- `pm25`

#### Çfarë bën

##### 1. Përgatitje kohore
- konverton `datetime`,
- rendit dataset-in kronologjikisht,
- nxjerr:
  - `hour`
  - `day_of_week`
  - `month`

<img width="507" height="93" alt="{9D5E10B1-7451-40A8-BA92-01DE19B074E0}" src="https://github.com/user-attachments/assets/382465fc-9cae-4af7-925a-1ff1dc0ae6a1" />

##### 2. Encodim ciklik
Krijon:
- `hour_sin`
- `hour_cos`
- `month_sin`
- `month_cos`

<img width="388" height="68" alt="{027EC64A-7F94-4673-8856-A2FA95B7FD55}" src="https://github.com/user-attachments/assets/e6ef76dd-d39e-4241-aadf-313f80d33cb4" />

Qëllimi është të përfaqësojë natyrën ciklike të orës dhe muajit.

##### 3. Lag features
Për kolonat:
- `total_generation_mw`
- `wind_speed_10m (km/h)`
- `temperature_2m (°C)`

krijohen lag-e:
- `lag_1h`
- `lag_3h`
- `lag_6h`

<img width="611" height="128" alt="{0129CD0F-43C3-46C7-857F-CC79A2E4E235}" src="https://github.com/user-attachments/assets/7f5885ea-3b9b-41b0-a1f2-24b55c428940" />

##### 4. Rolling features
Krijohen:
- `total_gen_rolling_sum_12h`
- `total_gen_rolling_sum_24h`

<img width="604" height="34" alt="{84F7C7EB-928A-4284-92DF-77244C86351B}" src="https://github.com/user-attachments/assets/8cd91bcd-5ad6-4c38-b1e7-a03d92793557" />

##### 5. Interaction features
Krijohen:
- `temp_wind_interact`
- `generation_humidity_interact`

<img width="595" height="41" alt="{CB1FE4F3-C35F-4AB7-8E84-B5113E104D46}" src="https://github.com/user-attachments/assets/161bea77-ad84-4b1c-a731-d92b75a50321" />

##### 6. Stagnation proxy
Krijohet:
- `pollution_stagnation_index = total_generation_mw / (wind_speed + 0.1)`

Ky indikator përpiqet të përfaqësojë situatat kur ka prodhim të lartë dhe erë të ulët, pra kushte më të favorshme për grumbullim ndotjesh.

<img width="593" height="31" alt="{7736A968-7617-4B05-AB85-04C693E45840}" src="https://github.com/user-attachments/assets/0b615ce4-1d7f-4258-bfb9-1d04e49561ac" />

##### 7. Wind vector decomposition
Nga shpejtësia dhe drejtimi i erës krijohen:
- `wind_x_vector`
- `wind_y_vector`

<img width="322" height="69" alt="{9EE0FD41-466C-406C-A328-75084CFF86E6}" src="https://github.com/user-attachments/assets/69a803b2-5af1-4a22-beb9-808ec06a6aeb" />

##### 8. Heqja e rreshtave me `NaN`
Pas krijimit të lag-eve dhe rolling windows hiqen rreshtat fillestarë që mbeten pa vlera të plota.

<img width="252" height="34" alt="{9214D524-77A2-425F-88FE-1406798AAE8D}" src="https://github.com/user-attachments/assets/367b4a97-7563-4864-ab33-e71c1d7bd6ea" />


#### Output
- `data/3B_engineered_dataset.csv`

#### Roli në pipeline
Ky është dataset-i i parë i pasuruar me tipare që modelojnë dinamikat kohore, ndikimet meteorologjike dhe ndërveprimet me prodhimin e energjisë.

---

## Preprocessing

### `4A_outlier_treatment.py`
Ky skript trajton outlier-at me quantile capping.

#### Input
- `data/3B_engineered_dataset.csv`

#### Strategjia
Për secilën kolonë numerike kandidate:
- kufiri i poshtëm = quantile `0.1%`
- kufiri i sipërm = quantile `99%`

Vlerat jashtë këtij intervali nuk fshihen, por priten në kufijtë përkatës.

#### Kolonat e përjashtuara
- `datetime`
- `date`
- disa tipare ciklike dhe vektorë strukturorë si:
  - `hour_sin`
  - `hour_cos`
  - `month_sin`
  - `month_cos`
  - `wind_x_vector`
  - `wind_y_vector`

#### Çfarë bën
- identifikon kolonat numerike kandidate,

<img width="" height="100" alt="image" src="https://github.com/user-attachments/assets/1277d0ad-20f8-4edd-aa83-7c3518c640b5" />

- llogarit kufijtë e poshtëm dhe të sipërm,

<img width="333" height="264" alt="{7637B515-ECC8-42F7-93E4-F51A36886583}" src="https://github.com/user-attachments/assets/5dc44eae-bd83-4230-8c0c-96b366d56d3f" />

- numëron sa vlera u cap-en në secilin krah,

<img width="300" height="64" alt="{868F8310-591A-4475-A72B-6B3610709F19}" src="https://github.com/user-attachments/assets/cc9dd492-043c-4cf0-8eea-6e39a24f3fd0" />

- krijon një raport për tiparet me më shumë vlera të kufizuara.

<img width="304" height="92" alt="{5C06F89A-7C0E-4C98-B3D6-7EB360549105}" src="https://github.com/user-attachments/assets/f774921a-69c9-493e-bb9d-55b0fe23b267" />

#### Output
- `data/4A_outliers_handled.csv`

#### Roli në pipeline
Ky hap redukton ndikimin e vlerave ekstreme pa humbur rreshta.

---

### `4B_skewness_treatment.py`
Ky skript trajton shtrembërimin e shpërndarjes së kolonave numerike.

#### Input
- `data/4A_outliers_handled.csv`

#### Strategjia
Për secilën kolonë numerike:
- llogaritet skewness,
- nëse `|skew| > 1.0`, zbatohet transformim.

#### Llojet e transformimit
- nëse kolona ka vetëm vlera jo-negative:
  - përdoret `log1p`
- ndryshe:
  - përdoret `PowerTransformer(method="yeo-johnson")`

#### Çfarë bën
- krahason skewness para dhe pas transformimit,

<img width="293" height="140" alt="{ACF2D39A-7132-44DE-94FD-02FBADE7EFE2}" src="https://github.com/user-attachments/assets/52ef2624-8f11-497a-a0c8-219132acfe5e" />

- ruan metodën e përdorur për secilën kolonë,

<img width="600" height="283" alt="{54D65469-B661-45E2-8812-109DCE98FE9B}" src="https://github.com/user-attachments/assets/3de4131c-acf9-4a06-80d8-beb30e751223" />

- raporton mean absolute skewness dhe median absolute skewness para/pas.

<img width="" height="86" alt="{27116D5F-1372-4BBB-8835-D8036D487641}" src="https://github.com/user-attachments/assets/33186496-a0f3-45a0-a447-d9ac72219563" />


#### Output
- `data/4B_skewness_handled.csv`

#### Roli në pipeline
Ky hap i bën shpërndarjet më të përshtatshme për standardizim, analiza lineare dhe modele machine learning.

---

### `4C_visualization_before_after.py`
Ky skript gjeneron histogramat krahasuese para dhe pas trajtimit të outlier-ave dhe skewness.

#### Input
- `data/3B_engineered_dataset.csv`
- `data/4A_outliers_handled.csv`
- `data/4B_skewness_handled.csv`

#### Tiparet e vizualizuara
- `pm25`
- `total_generation_mw`
- `pollution_stagnation_index`
- `rain (mm)`
- `temp_wind_interact`

#### Çfarë bën
Për secilin atribut:
- vizaton tre histogramë në të njëjtën figurë:
  - para trajtimit,
  - pas trajtimit të outlier-ave,
  - pas trajtimit të skewness.

#### Output
Folderi:
- `pictures/4C_visualization_before_after/`

me figurat:

#### PM2.5 Distribution Comparison
![PM2.5](pictures/4C_visualization_before_after/pm25_distribution_comparison.png)

#### Total Generation MW Distribution Comparison
![Total Generation](pictures/4C_visualization_before_after/total_generation_mw_distribution_comparison.png)

#### Pollution Stagnation Index Distribution Comparison
![Stagnation](pictures/4C_visualization_before_after/pollution_stagnation_index_distribution_comparison.png)

#### Rain (mm) Distribution Comparison
![Rain](pictures/4C_visualization_before_after/rain_mm_distribution_comparison.png)

#### Temperature-Wind Interaction Distribution Comparison
![Temp Wind](pictures/4C_visualization_before_after/temp_wind_interact_distribution_comparison.png)

#### Roli ne pipeline
Ky hap dokumenton vizualisht efektin e transformimeve statistikore.

---

### `4D_feature_scaling.py`
Ky skript standardizon të gjitha kolonat numerike.

#### Input
- `data/4B_skewness_handled.csv`

#### Çfarë bën
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

#### Output
- `data/4D_scaled_dataset.csv`
- `models/scaler.pkl`

#### Roli në pipeline
Ky hap siguron që tiparet numerike të jenë në të njëjtën shkallë dhe gati për feature selection ose modelim.

---

### `4E_feature_selection.py`
Ky skript kryen reduktimin final të tipareve.

#### Input
- `data/4D_scaled_dataset.csv`

#### Target
- `pm25`

#### Strategjia e seleksionimit

##### 1. Heqje manuale e kolonave jo të dëshiruara
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

##### 2. Heqje e kolonave konstante ose pothuajse konstante
- kolona me vetëm 1 vlerë unike
- kolona me devijim standard pothuajse zero

<img width="433" height="108" alt="{4008EF22-1994-45BF-999C-9B987BC2C534}" src="https://github.com/user-attachments/assets/ee46b9fb-c185-4479-baff-13c2531c6685" />

##### 3. VIF-based elimination
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

##### 4. Raportim
Në fund raportohet:
- madhësia e dataset-it fillestar,
- madhësia e dataset-it final,
- numri i tipareve finale,
- tiparet e mbajtura, të renditura sipas korrelacionit absolut me `pm25`.

<img width="506" height="151" alt="{7C2D8392-1353-40D2-937B-7035E866EA08}" src="https://github.com/user-attachments/assets/3e56b2a1-22dc-4f9f-843c-b03bd3c7eaee" />

#### Output
- `data/4E_selected_dataset.csv`

#### Roli në pipeline
Ky është dataset-i final i reduktuar, i përgatitur për modelim statistikor ose machine learning me target `pm25`.

---

## Artefaktet dhe output-et e krijuara

### Dataset-et e ruajtura ne `data/`
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

### Artefakte shtesë
- `models/scaler.pkl`  
  Objekti i `StandardScaler` për ripërdorim në inferencë ose pipeline të mëtejshme.

- `data/1B_distinct_values/`  
  Folder me vlera unike për atributet kryesore.

---

## Vizualizimet e gjeneruara

### 1. Heatmap-at nga analiza fillestare
#### `pictures/pollutant_vs_predictors_heatmap.png`
Paraqet korrelacionin ndërmjet ndotësve dhe tipareve të energjisë + motit.

#### `pictures/pollutant_correlation_heatmap.png`
Paraqet korrelacionin ndërmjet vetë ndotësve atmosferikë.

### 2. Histogramat krahasuese para/pas
Folderi `pictures/4C_visualization_before_after/` përmban figura që krahasojnë shpërndarjen:
- para trajtimit,
- pas trajtimit të outlier-ave,
- pas trajtimit të skewness.

#### Figurat aktuale
- `pm25_distribution_comparison.png`
- `pollution_stagnation_index_distribution_comparison.png`
- `rain_mm_distribution_comparison.png`
- `temp_wind_interact_distribution_comparison.png`
- `total_generation_mw_distribution_comparison.png`

### Figurat e projektit

#### Pollutant vs Predictors Heatmap
![Pollutant vs Predictors](pictures/pollutant_vs_predictors_heatmap.png)

#### Pollutant Correlation Heatmap
![Pollutant Correlation](pictures/pollutant_correlation_heatmap.png)

#### PM2.5 Distribution Comparison
![PM2.5](pictures/4C_visualization_before_after/pm25_distribution_comparison.png)

#### Total Generation MW Distribution Comparison
![Total Generation](pictures/4C_visualization_before_after/total_generation_mw_distribution_comparison.png)

#### Pollution Stagnation Index Distribution Comparison
![Stagnation](pictures/4C_visualization_before_after/pollution_stagnation_index_distribution_comparison.png)

#### Rain (mm) Distribution Comparison
![Rain](pictures/4C_visualization_before_after/rain_mm_distribution_comparison.png)

#### Temperature-Wind Interaction Distribution Comparison
![Temp Wind](pictures/4C_visualization_before_after/temp_wind_interact_distribution_comparison.png)

## Teknikat e zbatuara dhe lidhja me lëndën

Ky projekt përmbush në mënyrë të drejtpërdrejtë temat kryesore të lëndës “Machine Learning”.

### 1. Data collection
- Shkarkim dhe konsolidim i të dhënave nga burime të ndryshme.
- Përdorim i PowerShell, notebook-ut dhe CSV-ve bruto.

### 2. Data integration
- Bashkim i tre burimeve heterogjene mbi bosht kohor të përbashkët.
- Harmonizim i formateve të kohës dhe timezone.

### 3. Data cleaning
- Heqja e duplikateve.
- Korrigjimi i vlerave jo-logjike.
- Kufizim i vlerave fizike jashtë intervaleve të pranueshme.

### 4. Missing value handling
- Forward fill
- Backfill
- Plotësim i të dhënave pa heqje agresive të rreshtave

### 5. Validation
- Kontrolli fizik `PM2.5 <= PM10`
- Kontrolli i gaps kohore
- Kontrolli final i `NULL`

### 6. Exploratory data analysis
- Statistika përmbledhëse
- Matrica korrelacioni
- Heatmap-a për target-in dhe predictor-at

### 7. Feature engineering
- Encodim ciklik i kohës
- Lag features
- Rolling features
- Interaction terms
- Wind decomposition
- Domain-inspired stagnation index

### 8. Outlier handling
- Quantile capping me kufijtë `0.5%` dhe `99.5%`
- Qasje robuste pa fshirje të rreshtave

### 9. Skewness handling
- `log1p`
- `Yeo-Johnson`
- Krahasim para/pas me statistika dhe vizualizime

### 10. Scaling
- Standardizim i kolonave numerike me `StandardScaler`

### 11. Feature selection
- Heqje manuale e kolonave jorelevante ose problematike
- Heqje e kolonave konstante
- Reduktim i multikolinearitetit përmes `VIF`

---

## Ekzekutimi i projektit

### Parakushtet
- Python 3.10+ ose më i ri
- `pip`
- mjedis virtual i rekomanduar
- për skriptin PowerShell: qasje në `aws cli` nëse përdoret shkarkimi nga OpenAQ archive

### Instalimi i librarive
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

### Ekzekutimi i pipeline-it
Skriptat ekzekutohen sipas rendit logjik:

```bash
python src/integration/1A_merge_data.py
python src/distinct_values/1B_distinct_values.py

python src/data_cleaning/2A_datetime_and_duplicates.py
python src/data_cleaning/2B_data_quality_cleaning.py
python src/data_cleaning/2C_missing_values_handling.py
python src/data_cleaning/2D_validate_final_dataset.py

python src/feature_engineering/3A_target_analysis.py
python src/feature_engineering/3B_feature_engineering.py

python src/preprocessing/4A_outlier_treatment.py
python src/preprocessing/4B_skewness_treatment.py
python src/preprocessing/4C_visualization_before_after.py
python src/preprocessing/4D_feature_scaling.py
python src/preprocessing/4E_feature_selection.py
```

### Renditja e varësive
Çdo skript varet nga output-i i mëparshëm. Prandaj rekomandohet ekzekutimi në rend strikt.

---

## Rezultati final i pipeline-it

Produkti final i këtij projekti është:

- një dataset i integruar, i pastër dhe i validuar,
- një dataset i pasuruar me tipare domethënëse kohore dhe meteorologjike,
- një version i trajtuar për outlier-a dhe skewness,
- një version i standardizuar,
- dhe në fund një subset final tiparesh me multikolinearitet të reduktuar.

Dataset-i final:
- `data/4E_selected_dataset.csv`

është forma më e përshtatshme për:
- modelim prediktiv të `PM2.5`,
- regresion,
- krahasim modelesh machine learning,
- analiza statistikore të marrëdhënieve mes energjisë, motit dhe ndotjes.

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

