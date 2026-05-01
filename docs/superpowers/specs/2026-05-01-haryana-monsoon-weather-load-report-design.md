# Haryana Monsoon Weather-Load Impact Report — Design Spec
**Date:** 2026-05-01  
**Status:** Approved  
**Scope:** May-June 2022–2025, HARYANA, block-level weather impact on load

---

## 1. Goal

Generate a comprehensive report quantifying how temperature, humidity, rain, showers, and wind affect Haryana's 96-block load profile during May-June (pre-monsoon and active monsoon). Report covers 4 years (2022–2025) and is designed to inform forecast model calibration before the 2026 monsoon season.

Output formats: HTML (interactive), Excel (8 sheets), Jupyter Notebook (runnable), PDF (static).

---

## 2. Data Sources

| Source | Table | Date Range | Key Columns |
|---|---|---|---|
| Weather | `weather_mean` (MySQL) | 2022-01-01 → 2026-05-16 | temperature_2m, relative_humidity_2m, precipitation, rain, showers, wind_speed_10m, cloud_cover, sunshine_duration, direct_radiation |
| Load | `sldc` (MySQL) | 2022-01-01 → 2026-05-01 | load_mw |
| Resolution | Both | — | 96 blocks/day (15-min intervals) |
| Filter | Both | MONTH IN (5,6), state='HARYANA' | — |

A shared `data_loader.py` queries MySQL once and writes two parquet cache files:
- `reports/cache/may_jun_weather.parquet`
- `reports/cache/may_jun_load.parquet`

All 4 agents read from these parquets independently (no DB contention during parallel execution).

---

## 3. Agent Architecture

4 agents run in parallel, split by time-of-day. Each agent analyses its block window across all 4 years (2022–2025) and saves section outputs. Agent 4 additionally runs final synthesis and report assembly.

| Agent | Period | Blocks | Monsoon Relevance |
|---|---|---|---|
| Agent 1 | Night 00:00–06:00 | 1–24 | Nocturnal humidity-driven AC load, baseline cooling floor |
| Agent 2 | Morning 06:00–12:00 | 25–48 | Temperature ramp, morning industrial load, sunrise radiation |
| Agent 3 | Afternoon 12:00–18:00 | 49–72 | Peak load, max temp impact, pre-monsoon storm onset (3–5 PM) |
| Agent 4 | Evening 18:00–24:00 | 73–96 | Storm crash events, wind gust dip, evening cool-down + synthesis |

### Section outputs per agent (saved to `reports/sections/`)
- `section_{period}_data.parquet` — regression coefficients, regime labels, storm profiles
- `section_{period}_charts.json` — Plotly chart specs
- `section_{period}_excel.xlsx` — tables for that time window

---

## 4. Analysis Methodology

Each agent runs the same 4-step pipeline on its block window.

### Step 1 — Weather Regime Clustering (K-means, k=4)
Features: `(avg_temp, avg_humidity, total_precip, avg_wind)` per day per window.

| Regime | Conditions | Expected Load Effect |
|---|---|---|
| Hot-Dry | temp >38°C, humidity <30%, precip=0 | Max AC load, strong temp correlation |
| Pre-Monsoon Humid | temp 34–38°C, humidity 50–70%, precip <2mm | High AC + humidity stress |
| Active Monsoon Rain | precip 5–15mm, cloud >70% | Moderate load suppression 300–800 MW |
| Storm Event | precip >15mm OR wind_speed >35 km/h | Extreme load crash, excluded from regression |

Storm Event days are extracted separately and profiled as an independent section.

### Step 2 — Block-wise OLS Regression (non-storm days only)
Per block (within the agent's window), per year, fit:

```
load_mw = β₀ + β₁·temperature + β₂·humidity + β₃·precipitation
        + β₄·showers + β₅·wind_speed + β₆·cloud_cover + ε
```

Outputs: MW-per-unit coefficients at each block. Example: "Block 60 (15:00): 1°C = +142 MW, 1mm rain = −85 MW".

Report R² per block to flag low-confidence blocks.

### Step 3 — Storm Event Profiles
For Storm Event days:
- Average load curve (96 blocks) vs. clear-day average — shows MW suppression profile
- Storm onset block detection (first block where load drops >300 MW vs baseline)
- Duration: how many blocks the suppression persists
- Recovery: blocks to return to normal load level

### Step 4 — Year-over-Year Coefficient Drift
For key blocks (peak: 52–72, evening: 73–84), plot how each β coefficient changed 2022→2023→2024→2025.
- Rising temperature coefficient = growing AC penetration
- Changing rain coefficient = shifting monsoon intensity or load composition

---

## 5. Output Specification

### 5.1 HTML Report (`reports/haryana_monsoon_report.html`)
Single self-contained file with embedded Plotly charts. Structure:

- **Tab 0 — Executive Summary**: key numbers, monsoon onset dates, worst storm events
- **Tab 1 — Night (00:00–06:00)**: Agent 1 results
- **Tab 2 — Morning (06:00–12:00)**: Agent 2 results
- **Tab 3 — Afternoon (12:00–18:00)**: Agent 3 results
- **Tab 4 — Evening (18:00–24:00)**: Agent 4 results

Each tab contains:
1. Block-wise coefficient heatmap (96 blocks × 6 weather variables)
2. Regime distribution calendar (May-June day-by-day, per year)
3. Storm suppression profile (load curve storm vs clear)
4. YoY sensitivity drift line chart

### 5.2 Excel Workbook (`reports/haryana_monsoon_report.xlsx`)
8 sheets:
1. Summary — top-level numbers
2. Night_Coefficients
3. Morning_Coefficients
4. Afternoon_Coefficients
5. Evening_Coefficients
6. Storm_Events — all storm days with onset block, max suppression MW, duration
7. YoY_Drift — coefficient evolution table
8. Raw_Data — joined weather + load for May-June 2022–2025

### 5.3 Jupyter Notebook (`reports/haryana_monsoon_report.ipynb`)
Pre-executed notebook with all analysis cells inline. Users can re-run to update with new data.

### 5.4 PDF (`reports/haryana_monsoon_report.pdf`)
Static export via `matplotlib` multi-page PDF (most reliable on Windows). If `weasyprint` is available it is used instead for richer styling.

---

## 6. Key Charts

| Chart | Type | Dimensions |
|---|---|---|
| Block-wise coefficient heatmap | Plotly heatmap | 96 blocks × 6 variables, color = MW/unit |
| Regime distribution calendar | Plotly heatmap | Days × Years, color = regime |
| Storm suppression profile | Plotly line | 96 blocks, storm vs clear, per year |
| YoY sensitivity drift | Plotly line | 2022–2025, per weather variable, at peak blocks |
| Monsoon onset detection | Plotly scatter | rolling 5-day precip, onset markers per year |

---

## 7. File Structure

```
reports/
├── cache/
│   ├── may_jun_weather.parquet
│   └── may_jun_load.parquet
├── sections/
│   ├── section_night_data.parquet
│   ├── section_night_charts.json
│   ├── section_night_excel.xlsx
│   ├── section_morning_*.{parquet,json,xlsx}
│   ├── section_afternoon_*.{parquet,json,xlsx}
│   └── section_evening_*.{parquet,json,xlsx}
├── haryana_monsoon_report.html
├── haryana_monsoon_report.xlsx
├── haryana_monsoon_report.ipynb
└── haryana_monsoon_report.pdf

scripts/
├── data_loader.py          # MySQL → parquet cache
├── agent_analysis.py       # shared analysis functions (regression, clustering, storm detection)
├── agent_night.py          # Agent 1: blocks 1–24
├── agent_morning.py        # Agent 2: blocks 25–48
├── agent_afternoon.py      # Agent 3: blocks 49–72
├── agent_evening.py        # Agent 4: blocks 73–96 + synthesis
└── run_report.py           # Orchestrator: runs data_loader, then dispatches 4 agents in parallel
```

---

## 8. Dependencies

```
pandas, numpy, scikit-learn (KMeans, LinearRegression), plotly, openpyxl,
nbformat, nbconvert, weasyprint (PDF), pymysql, pyarrow
```

---

## 9. Success Criteria

- All 4 output files generated without error
- Each block has a regression R² value reported (flag blocks with R² < 0.3)
- Storm events detected and profiled for all 4 years
- YoY drift chart shows 2022, 2023, 2024, 2025 as distinct series
- Report runnable end-to-end via `python scripts/run_report.py`
