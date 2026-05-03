# Haryana Monsoon Weather-Load Impact Report — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a 4-output (HTML, Excel, Notebook, PDF) report quantifying block-level weather impact on Haryana load for May-June 2022–2025 using 4 parallel time-of-day agents.

**Architecture:** A shared `agent_analysis.py` library provides clustering, regression, storm detection, and charting. Four agent scripts (`agent_night`, `agent_morning`, `agent_afternoon`, `agent_evening`) each analyse blocks 1–24 / 25–48 / 49–72 / 73–96. Agents 1–3 run in parallel; Agent 4 runs last and assembles all outputs.

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn, plotly, kaleido, openpyxl, nbformat, pymysql, pyarrow, matplotlib, concurrent.futures

---

## Task 0: Setup — directories, dependencies, verify data

**Files:**
- Create: `scripts/` directory
- Create: `reports/cache/`, `reports/sections/` directories

- [ ] **Step 1: Create directory structure**

```bash
cd "c:/Users/RamanSharma/OneDrive - GNA-Energy/Desktop/RD"
mkdir -p scripts reports/cache reports/sections
```

- [ ] **Step 2: Install dependencies**

```bash
pip install pandas numpy scikit-learn plotly kaleido openpyxl nbformat nbconvert pymysql pyarrow matplotlib python-dotenv
```

- [ ] **Step 3: Verify data availability**

```python
# Run this interactively to confirm row counts
import sys; sys.path.insert(0, '.')
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB
import pymysql

conn = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
    password=MYSQL_PASSWORD, database=MYSQL_DB, charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor)
cur = conn.cursor()
cur.execute("SELECT COUNT(*) as n FROM weather_mean WHERE state='HARYANA' AND MONTH(date) IN (5,6) AND YEAR(date) BETWEEN 2022 AND 2025")
print("Weather rows:", cur.fetchone())  # expect ~23,424 (61+61+61+61 days * 96 blocks)
cur.execute("SELECT COUNT(*) as n FROM sldc WHERE state='HARYANA' AND MONTH(date) IN (5,6) AND YEAR(date) BETWEEN 2022 AND 2025")
print("Load rows:", cur.fetchone())     # expect ~23,424
conn.close()
```

Expected output:
```
Weather rows: {'n': 23424}
Load rows: {'n': 23424}
```

- [ ] **Step 4: Commit directory structure**

```bash
git add scripts/ reports/
git commit -m "feat: scaffold directories for monsoon report"
```

---

## Task 1: Data Loader — MySQL → parquet cache

**Files:**
- Create: `scripts/data_loader.py`

- [ ] **Step 1: Write `scripts/data_loader.py`**

```python
"""
data_loader.py — Query MySQL weather_mean + sldc for HARYANA May-June 2022-2025.
Writes two parquet cache files read by all 4 agents.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import pymysql
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'cache')
WEATHER_PARQUET = os.path.join(CACHE_DIR, 'may_jun_weather.parquet')
LOAD_PARQUET    = os.path.join(CACHE_DIR, 'may_jun_load.parquet')


def _get_conn():
    return pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
        password=MYSQL_PASSWORD, database=MYSQL_DB,
        charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor,
    )


def load_weather() -> pd.DataFrame:
    conn = _get_conn()
    query = """
        SELECT
            date, time_block,
            temperature_2m      AS temperature,
            relative_humidity_2m AS humidity,
            precipitation, rain, showers,
            wind_speed_10m      AS wind_speed,
            cloud_cover,
            sunshine_duration,
            direct_radiation
        FROM weather_mean
        WHERE state = 'HARYANA'
          AND MONTH(date) IN (5, 6)
          AND YEAR(date) BETWEEN 2022 AND 2025
        ORDER BY date, time_block
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month
    return df


def load_sldc() -> pd.DataFrame:
    conn = _get_conn()
    query = """
        SELECT date, time_block, load_mw
        FROM sldc
        WHERE state = 'HARYANA'
          AND MONTH(date) IN (5, 6)
          AND YEAR(date) BETWEEN 2022 AND 2025
        ORDER BY date, time_block
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df


def build_cache(force: bool = False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not force and os.path.exists(WEATHER_PARQUET) and os.path.exists(LOAD_PARQUET):
        print("Cache already exists. Use force=True to rebuild.")
        return

    print("Loading weather from MySQL...")
    weather = load_weather()
    print(f"  {len(weather):,} weather rows")

    print("Loading load from MySQL...")
    load = load_sldc()
    print(f"  {len(load):,} load rows")

    # Join on date + time_block
    merged = weather.merge(load, on=['date', 'time_block'], how='inner')
    merged = merged.dropna(subset=['load_mw'])
    print(f"  {len(merged):,} joined rows after inner join")

    # Save separately so agents can read just what they need
    weather_cols = ['date','time_block','year','month','temperature','humidity',
                    'precipitation','rain','showers','wind_speed','cloud_cover',
                    'sunshine_duration','direct_radiation']
    merged[weather_cols].to_parquet(WEATHER_PARQUET, index=False)

    load_cols = ['date','time_block','year','month','load_mw']
    merged[load_cols].to_parquet(LOAD_PARQUET, index=False)

    print(f"Cache written:\n  {WEATHER_PARQUET}\n  {LOAD_PARQUET}")


if __name__ == '__main__':
    build_cache(force='--force' in sys.argv)
```

- [ ] **Step 2: Run data loader and verify outputs**

```bash
cd "c:/Users/RamanSharma/OneDrive - GNA-Energy/Desktop/RD"
python scripts/data_loader.py
```

Expected output:
```
Loading weather from MySQL...
  23424 weather rows
Loading load from MySQL...
  23424 load rows
  23xxx joined rows after inner join
Cache written:
  .../reports/cache/may_jun_weather.parquet
  .../reports/cache/may_jun_load.parquet
```

Then verify:
```python
import pandas as pd
w = pd.read_parquet('reports/cache/may_jun_weather.parquet')
l = pd.read_parquet('reports/cache/may_jun_load.parquet')
print(w.shape, w.columns.tolist())
print(l.shape, l['year'].unique())
assert set(w['year'].unique()) == {2022, 2023, 2024, 2025}
assert set(w['month'].unique()) == {5, 6}
print("OK")
```

- [ ] **Step 3: Commit**

```bash
git add scripts/data_loader.py
git commit -m "feat: data_loader — MySQL to parquet cache for monsoon report"
```

---

## Task 2: Shared Analysis Library

**Files:**
- Create: `scripts/agent_analysis.py`

- [ ] **Step 1: Write `scripts/agent_analysis.py`**

```python
"""
agent_analysis.py — Shared functions used by all 4 time-of-day agents.

Provides:
  - load_section_df()      : load + filter parquet to a block window
  - classify_regimes()     : K-means (k=4) weather regime per day
  - get_storm_days()       : days with precip>15mm OR wind>35 km/h
  - run_all_regressions()  : block-wise OLS per year, returns coeff_df
  - build_storm_profile()  : avg load curve storm vs clear day
  - detect_monsoon_onset() : rolling 5-day precip threshold per year
  - Charts: build_coeff_heatmap, build_regime_calendar,
            build_storm_chart, build_yoy_drift_chart, build_onset_chart
  - save_section_outputs() : write parquet + charts JSON + Excel
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import warnings
warnings.filterwarnings('ignore')

ROOT     = os.path.join(os.path.dirname(__file__), '..')
CACHE    = os.path.join(ROOT, 'reports', 'cache')
SECTIONS = os.path.join(ROOT, 'reports', 'sections')

WEATHER_FEATURES = ['temperature', 'humidity', 'precipitation',
                    'showers', 'wind_speed', 'cloud_cover']
YEARS = [2022, 2023, 2024, 2025]

REGIME_NAMES  = {0: 'Hot-Dry', 1: 'Pre-Monsoon Humid',
                 2: 'Active Monsoon Rain', 3: 'Storm Event'}
REGIME_COLORS = {0: '#e74c3c', 1: '#f39c12', 2: '#2980b9', 3: '#8e44ad'}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_section_df(block_start: int, block_end: int) -> pd.DataFrame:
    """Load joined weather+load parquet filtered to [block_start, block_end]."""
    w = pd.read_parquet(os.path.join(CACHE, 'may_jun_weather.parquet'))
    l = pd.read_parquet(os.path.join(CACHE, 'may_jun_load.parquet'))
    df = w.merge(l, on=['date', 'time_block', 'year', 'month'])
    df = df[(df['time_block'] >= block_start) & (df['time_block'] <= block_end)].copy()
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    return df.reset_index(drop=True)


# ── Weather regime clustering ─────────────────────────────────────────────────

def get_storm_days(df: pd.DataFrame) -> set:
    """Return set of date strings where precip>15mm OR wind>35 km/h (in window)."""
    daily = df.groupby('date_str').agg(
        total_precip=('precipitation', 'sum'),
        max_wind=('wind_speed', 'max'),
    )
    storm = daily[(daily['total_precip'] > 15) | (daily['max_wind'] > 35)]
    return set(storm.index)


def classify_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    K-means (k=4) on daily (avg_temp, avg_hum, total_precip, avg_wind).
    Returns daily DataFrame with columns: date_str, year, month, regime_id, regime_name.
    Storm days are always assigned regime 3 regardless of cluster.
    """
    storm_days = get_storm_days(df)
    daily = df.groupby(['date_str', 'year', 'month']).agg(
        avg_temp=('temperature', 'mean'),
        avg_hum=('humidity', 'mean'),
        total_precip=('precipitation', 'sum'),
        avg_wind=('wind_speed', 'mean'),
    ).reset_index()

    non_storm = daily[~daily['date_str'].isin(storm_days)].copy()
    features = non_storm[['avg_temp', 'avg_hum', 'total_precip', 'avg_wind']].fillna(0)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(features)
    non_storm['regime_id'] = km.labels_

    # Re-label clusters by temperature (0=Hot-Dry, 1=Pre-Mon Humid, 2=Rain)
    centers = pd.DataFrame(km.cluster_centers_,
                           columns=['avg_temp','avg_hum','total_precip','avg_wind'])
    temp_order = centers['avg_temp'].argsort().values[::-1]  # hottest first
    remap = {old: new for new, old in enumerate(temp_order)}
    non_storm['regime_id'] = non_storm['regime_id'].map(remap)

    storm_df = daily[daily['date_str'].isin(storm_days)].copy()
    storm_df['regime_id'] = 3

    result = pd.concat([non_storm, storm_df], ignore_index=True)
    result['regime_name'] = result['regime_id'].map(REGIME_NAMES)
    return result[['date_str', 'year', 'month', 'regime_id', 'regime_name',
                   'avg_temp', 'avg_hum', 'total_precip', 'avg_wind']]


# ── Block-wise OLS regression ─────────────────────────────────────────────────

def run_all_regressions(df: pd.DataFrame, storm_days: set) -> pd.DataFrame:
    """
    For each (block, year) in df (excluding storm days), fit OLS:
      load_mw ~ temperature + humidity + precipitation + showers + wind_speed + cloud_cover
    Returns DataFrame with columns:
      block, year, intercept, temperature, humidity, precipitation,
      showers, wind_speed, cloud_cover, r2, n_samples
    """
    df_clean = df[~df['date_str'].isin(storm_days)].copy()
    rows = []
    for year in YEARS:
        year_df = df_clean[df_clean['year'] == year]
        for block in sorted(df_clean['time_block'].unique()):
            bdf = year_df[year_df['time_block'] == block].dropna(
                subset=WEATHER_FEATURES + ['load_mw'])
            if len(bdf) < 10:
                continue
            X = bdf[WEATHER_FEATURES].values
            y = bdf['load_mw'].values
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            row = {'block': int(block), 'year': int(year),
                   'intercept': round(model.intercept_, 2), 'r2': round(r2, 3),
                   'n_samples': len(bdf)}
            for feat, coef in zip(WEATHER_FEATURES, model.coef_):
                row[feat] = round(coef, 3)
            rows.append(row)
    return pd.DataFrame(rows)


# ── Storm event profiles ──────────────────────────────────────────────────────

def build_storm_profile(df: pd.DataFrame, storm_days: set) -> pd.DataFrame:
    """
    Returns DataFrame with columns: block, year, storm_load_avg, clear_load_avg,
    suppression_mw, onset_block, duration_blocks, recovery_blocks.
    """
    results = []
    all_blocks = sorted(df['time_block'].unique())
    for year in YEARS:
        year_df = df[df['year'] == year]
        storm_df = year_df[year_df['date_str'].isin(storm_days)]
        clear_df  = year_df[~year_df['date_str'].isin(storm_days)]

        storm_curve = storm_df.groupby('time_block')['load_mw'].mean()
        clear_curve = clear_df.groupby('time_block')['load_mw'].mean()

        for blk in all_blocks:
            s = storm_curve.get(blk, np.nan)
            c = clear_curve.get(blk, np.nan)
            results.append({
                'block': blk, 'year': year,
                'storm_load_avg': round(s, 1) if not np.isnan(s) else None,
                'clear_load_avg': round(c, 1) if not np.isnan(c) else None,
                'suppression_mw': round(c - s, 1) if not (np.isnan(s) or np.isnan(c)) else None,
            })

    # Onset / duration per storm day
    onset_rows = []
    for day in storm_days:
        day_df = df[df['date_str'] == day].sort_values('time_block')
        if day_df.empty:
            continue
        year = int(day_df['year'].iloc[0])
        clear_mean = df[(df['year'] == year) &
                        (~df['date_str'].isin(storm_days))].groupby('time_block')['load_mw'].mean()
        day_load = day_df.set_index('time_block')['load_mw']
        drop = (clear_mean - day_load).dropna()
        onset_blocks = drop[drop > 300].index.tolist()
        onset_block   = int(onset_blocks[0]) if onset_blocks else None
        duration      = len(onset_blocks)
        recovery      = int(all_blocks[-1]) - (onset_block or 0) if onset_block else 0
        onset_rows.append({'date': day, 'year': year, 'onset_block': onset_block,
                           'max_suppression_mw': round(float(drop.max()), 1) if not drop.empty else 0,
                           'duration_blocks': duration, 'recovery_blocks': recovery})
    storm_events_df = pd.DataFrame(onset_rows)
    return pd.DataFrame(results), storm_events_df


# ── Monsoon onset detection ───────────────────────────────────────────────────

def detect_monsoon_onset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling 5-day total precip per year. Onset = first date where 5-day sum > 25mm.
    Returns DataFrame: year, onset_date, rolling_precip (daily series).
    """
    daily_precip = df.groupby(['year', 'date_str'])['precipitation'].sum().reset_index()
    rows = []
    for year in YEARS:
        ydf = daily_precip[daily_precip['year'] == year].sort_values('date_str')
        ydf = ydf.set_index('date_str')
        rolling = ydf['precipitation'].rolling(5, min_periods=3).sum()
        onset_dates = rolling[rolling > 25].index.tolist()
        onset_date  = onset_dates[0] if onset_dates else None
        for d, val in rolling.items():
            rows.append({'year': year, 'date': d, 'rolling_5d_precip': round(val, 1),
                         'is_onset': (d == onset_date)})
    return pd.DataFrame(rows)


# ── Block number → time string ────────────────────────────────────────────────

def block_to_time(block: int) -> str:
    h = int((block - 1) * 0.25)
    m = int(((block - 1) * 0.25 % 1) * 60)
    return f'{h:02d}:{m:02d}'


# ── Plotly charts ─────────────────────────────────────────────────────────────

def build_coeff_heatmap(coeff_df: pd.DataFrame, period_name: str) -> go.Figure:
    """96-block × 6-variable coefficient heatmap, one trace per year."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=[str(y) for y in YEARS],
                        shared_xaxes=True, shared_yaxes=True)
    for i, year in enumerate(YEARS):
        r, c = divmod(i, 2)
        ydf = coeff_df[coeff_df['year'] == year].sort_values('block')
        z = ydf[WEATHER_FEATURES].T.values
        xticks = [block_to_time(int(b)) for b in ydf['block']]
        fig.add_trace(go.Heatmap(
            z=z, x=xticks, y=WEATHER_FEATURES,
            colorscale='RdBu', zmid=0,
            colorbar=dict(title='MW/unit', len=0.45, x=0.46 if c == 0 else 1.0),
            showscale=(i == 0),
        ), row=r+1, col=c+1)
    fig.update_layout(
        title=f'Block-wise Weather Coefficients — {period_name} ({min(YEARS)}–{max(YEARS)})',
        height=600, template='plotly_dark',
    )
    return fig


def build_regime_calendar(regime_df: pd.DataFrame, period_name: str) -> go.Figure:
    """Day-level heatmap showing weather regime per year."""
    fig = make_subplots(rows=len(YEARS), cols=1, subplot_titles=[str(y) for y in YEARS],
                        shared_xaxes=True, vertical_spacing=0.04)
    for i, year in enumerate(YEARS):
        ydf = regime_df[regime_df['year'] == year].sort_values('date_str')
        fig.add_trace(go.Heatmap(
            z=[ydf['regime_id'].tolist()],
            x=ydf['date_str'].tolist(),
            colorscale=[[0,'#e74c3c'],[0.33,'#f39c12'],[0.67,'#2980b9'],[1,'#8e44ad']],
            zmin=0, zmax=3,
            showscale=(i == 0),
            colorbar=dict(
                tickvals=[0,1,2,3],
                ticktext=['Hot-Dry','Pre-Mon Humid','Active Rain','Storm'],
                len=0.25, y=1.0 - i*0.28,
            ),
        ), row=i+1, col=1)
    fig.update_layout(
        title=f'Weather Regime Calendar — {period_name}',
        height=500, template='plotly_dark',
    )
    return fig


def build_storm_chart(storm_profile_df: pd.DataFrame, period_name: str) -> go.Figure:
    """Storm vs clear load curve per year."""
    fig = go.Figure()
    colors = ['#e74c3c','#f39c12','#2980b9','#27ae60']
    for year, color in zip(YEARS, colors):
        ydf = storm_profile_df[storm_profile_df['year'] == year].sort_values('block')
        xticks = [block_to_time(int(b)) for b in ydf['block']]
        fig.add_trace(go.Scatter(x=xticks, y=ydf['clear_load_avg'],
                                  name=f'{year} Clear', line=dict(color=color, dash='dot')))
        fig.add_trace(go.Scatter(x=xticks, y=ydf['storm_load_avg'],
                                  name=f'{year} Storm', line=dict(color=color, dash='solid')))
    fig.update_layout(
        title=f'Storm vs Clear Load Profile — {period_name}',
        xaxis_title='Time', yaxis_title='Load (MW)',
        height=450, template='plotly_dark',
    )
    return fig


def build_yoy_drift_chart(coeff_df: pd.DataFrame, period_name: str) -> go.Figure:
    """Year-over-year drift of weather coefficients at peak blocks."""
    peak_blocks = coeff_df['block'].unique()
    avg_by_year = coeff_df.groupby('year')[WEATHER_FEATURES].mean().reset_index()
    fig = go.Figure()
    colors = ['#e74c3c','#f39c12','#2980b9','#27ae60','#9b59b6','#1abc9c']
    for feat, color in zip(WEATHER_FEATURES, colors):
        fig.add_trace(go.Scatter(
            x=avg_by_year['year'], y=avg_by_year[feat],
            name=feat, mode='lines+markers',
            line=dict(color=color),
        ))
    fig.update_layout(
        title=f'Year-over-Year Coefficient Drift — {period_name}',
        xaxis=dict(tickvals=YEARS),
        yaxis_title='Avg MW per unit',
        height=400, template='plotly_dark',
    )
    return fig


def build_onset_chart(onset_df: pd.DataFrame) -> go.Figure:
    """Rolling 5-day precipitation with monsoon onset markers."""
    fig = go.Figure()
    colors = ['#e74c3c','#f39c12','#2980b9','#27ae60']
    for year, color in zip(YEARS, colors):
        ydf = onset_df[onset_df['year'] == year]
        fig.add_trace(go.Scatter(
            x=ydf['date'], y=ydf['rolling_5d_precip'],
            name=str(year), line=dict(color=color),
        ))
        onset = ydf[ydf['is_onset']]
        if not onset.empty:
            fig.add_trace(go.Scatter(
                x=onset['date'], y=onset['rolling_5d_precip'],
                mode='markers', marker=dict(color=color, size=12, symbol='star'),
                name=f'{year} onset', showlegend=True,
            ))
    fig.add_hline(y=25, line_dash='dash', line_color='white',
                  annotation_text='Onset threshold (25mm)')
    fig.update_layout(
        title='Monsoon Onset Detection (Rolling 5-day Precipitation)',
        xaxis_title='Date', yaxis_title='5-day total precip (mm)',
        height=400, template='plotly_dark',
    )
    return fig


# ── Section output writer ─────────────────────────────────────────────────────

def save_section_outputs(
    period_name: str,
    coeff_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    storm_profile_df: pd.DataFrame,
    storm_events_df: pd.DataFrame,
    onset_df: pd.DataFrame,
    figures: dict,
):
    """
    Save:
      reports/sections/section_{period_name}_data.parquet
      reports/sections/section_{period_name}_charts.json
      reports/sections/section_{period_name}_excel.xlsx
    """
    os.makedirs(SECTIONS, exist_ok=True)
    base = os.path.join(SECTIONS, f'section_{period_name}')

    # Parquet — analysis data
    coeff_df['period'] = period_name
    regime_df['period'] = period_name
    storm_profile_df['period'] = period_name
    coeff_df.to_parquet(f'{base}_coeff.parquet', index=False)
    regime_df.to_parquet(f'{base}_regime.parquet', index=False)
    storm_profile_df.to_parquet(f'{base}_storm_profile.parquet', index=False)
    storm_events_df.to_parquet(f'{base}_storm_events.parquet', index=False)
    onset_df.to_parquet(f'{base}_onset.parquet', index=False)

    # Charts JSON
    charts_json = {name: pio.to_json(fig) for name, fig in figures.items()}
    with open(f'{base}_charts.json', 'w') as f:
        json.dump(charts_json, f)

    # Excel
    with pd.ExcelWriter(f'{base}_excel.xlsx', engine='openpyxl') as writer:
        coeff_df.to_excel(writer, sheet_name='Coefficients', index=False)
        regime_df.to_excel(writer, sheet_name='Regimes', index=False)
        storm_profile_df.to_excel(writer, sheet_name='Storm_Profile', index=False)
        storm_events_df.to_excel(writer, sheet_name='Storm_Events', index=False)

    print(f"[{period_name}] Section outputs saved to {SECTIONS}")
```

- [ ] **Step 2: Verify the module imports without error**

```bash
cd "c:/Users/RamanSharma/OneDrive - GNA-Energy/Desktop/RD"
python -c "from scripts.agent_analysis import *; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/agent_analysis.py
git commit -m "feat: agent_analysis — shared clustering, regression, charting library"
```

---

## Task 3: Agent Night (blocks 1–24, 00:00–06:00)

**Files:**
- Create: `scripts/agent_night.py`

- [ ] **Step 1: Write `scripts/agent_night.py`**

```python
"""
agent_night.py — Agent 1: blocks 1-24 (00:00-06:00)
Nocturnal humidity-driven AC load, baseline cooling floor.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.agent_analysis import (
    load_section_df, classify_regimes, get_storm_days,
    run_all_regressions, build_storm_profile, detect_monsoon_onset,
    build_coeff_heatmap, build_regime_calendar, build_storm_chart,
    build_yoy_drift_chart, build_onset_chart, save_section_outputs,
)

PERIOD     = 'night'
BLK_START  = 1
BLK_END    = 24


def main():
    print(f"[{PERIOD}] Loading data blocks {BLK_START}–{BLK_END}...")
    df = load_section_df(BLK_START, BLK_END)
    print(f"[{PERIOD}] {len(df):,} rows loaded")

    storm_days  = get_storm_days(df)
    print(f"[{PERIOD}] Storm days detected: {len(storm_days)}")

    regime_df   = classify_regimes(df)
    coeff_df    = run_all_regressions(df, storm_days)
    storm_profile_df, storm_events_df = build_storm_profile(df, storm_days)
    onset_df    = detect_monsoon_onset(df)

    figures = {
        'coeff_heatmap':   build_coeff_heatmap(coeff_df, PERIOD),
        'regime_calendar': build_regime_calendar(regime_df, PERIOD),
        'storm_profile':   build_storm_chart(storm_profile_df, PERIOD),
        'yoy_drift':       build_yoy_drift_chart(coeff_df, PERIOD),
        'onset':           build_onset_chart(onset_df),
    }

    save_section_outputs(PERIOD, coeff_df, regime_df, storm_profile_df,
                         storm_events_df, onset_df, figures)
    print(f"[{PERIOD}] Done.")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run and verify**

```bash
cd "c:/Users/RamanSharma/OneDrive - GNA-Energy/Desktop/RD"
python scripts/agent_night.py
```

Expected output:
```
[night] Loading data blocks 1–24...
[night] 5856 rows loaded
[night] Storm days detected: N
[night] Section outputs saved to .../reports/sections
[night] Done.
```

Then verify files exist:
```bash
ls reports/sections/section_night_*
```

- [ ] **Step 3: Commit**

```bash
git add scripts/agent_night.py
git commit -m "feat: agent_night — blocks 1-24 weather-load analysis"
```

---

## Task 4: Agent Morning (blocks 25–48, 06:00–12:00)

**Files:**
- Create: `scripts/agent_morning.py`

- [ ] **Step 1: Write `scripts/agent_morning.py`**

```python
"""
agent_morning.py — Agent 2: blocks 25-48 (06:00-12:00)
Temperature ramp, morning industrial load, sunrise radiation effect.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.agent_analysis import (
    load_section_df, classify_regimes, get_storm_days,
    run_all_regressions, build_storm_profile, detect_monsoon_onset,
    build_coeff_heatmap, build_regime_calendar, build_storm_chart,
    build_yoy_drift_chart, build_onset_chart, save_section_outputs,
)

PERIOD    = 'morning'
BLK_START = 25
BLK_END   = 48


def main():
    print(f"[{PERIOD}] Loading data blocks {BLK_START}–{BLK_END}...")
    df = load_section_df(BLK_START, BLK_END)
    print(f"[{PERIOD}] {len(df):,} rows loaded")

    storm_days = get_storm_days(df)
    print(f"[{PERIOD}] Storm days detected: {len(storm_days)}")

    regime_df        = classify_regimes(df)
    coeff_df         = run_all_regressions(df, storm_days)
    storm_profile_df, storm_events_df = build_storm_profile(df, storm_days)
    onset_df         = detect_monsoon_onset(df)

    figures = {
        'coeff_heatmap':   build_coeff_heatmap(coeff_df, PERIOD),
        'regime_calendar': build_regime_calendar(regime_df, PERIOD),
        'storm_profile':   build_storm_chart(storm_profile_df, PERIOD),
        'yoy_drift':       build_yoy_drift_chart(coeff_df, PERIOD),
        'onset':           build_onset_chart(onset_df),
    }

    save_section_outputs(PERIOD, coeff_df, regime_df, storm_profile_df,
                         storm_events_df, onset_df, figures)
    print(f"[{PERIOD}] Done.")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run and verify**

```bash
python scripts/agent_morning.py
```

Expected: `[morning] Done.` and files in `reports/sections/section_morning_*`

- [ ] **Step 3: Commit**

```bash
git add scripts/agent_morning.py
git commit -m "feat: agent_morning — blocks 25-48 weather-load analysis"
```

---

## Task 5: Agent Afternoon (blocks 49–72, 12:00–18:00)

**Files:**
- Create: `scripts/agent_afternoon.py`

- [ ] **Step 1: Write `scripts/agent_afternoon.py`**

```python
"""
agent_afternoon.py — Agent 3: blocks 49-72 (12:00-18:00)
Peak load, max temp impact, pre-monsoon storm onset (3-5 PM).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.agent_analysis import (
    load_section_df, classify_regimes, get_storm_days,
    run_all_regressions, build_storm_profile, detect_monsoon_onset,
    build_coeff_heatmap, build_regime_calendar, build_storm_chart,
    build_yoy_drift_chart, build_onset_chart, save_section_outputs,
)

PERIOD    = 'afternoon'
BLK_START = 49
BLK_END   = 72


def main():
    print(f"[{PERIOD}] Loading data blocks {BLK_START}–{BLK_END}...")
    df = load_section_df(BLK_START, BLK_END)
    print(f"[{PERIOD}] {len(df):,} rows loaded")

    storm_days = get_storm_days(df)
    print(f"[{PERIOD}] Storm days detected: {len(storm_days)}")

    regime_df        = classify_regimes(df)
    coeff_df         = run_all_regressions(df, storm_days)
    storm_profile_df, storm_events_df = build_storm_profile(df, storm_days)
    onset_df         = detect_monsoon_onset(df)

    figures = {
        'coeff_heatmap':   build_coeff_heatmap(coeff_df, PERIOD),
        'regime_calendar': build_regime_calendar(regime_df, PERIOD),
        'storm_profile':   build_storm_chart(storm_profile_df, PERIOD),
        'yoy_drift':       build_yoy_drift_chart(coeff_df, PERIOD),
        'onset':           build_onset_chart(onset_df),
    }

    save_section_outputs(PERIOD, coeff_df, regime_df, storm_profile_df,
                         storm_events_df, onset_df, figures)
    print(f"[{PERIOD}] Done.")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run and verify**

```bash
python scripts/agent_afternoon.py
```

Expected: `[afternoon] Done.` and files in `reports/sections/section_afternoon_*`

- [ ] **Step 3: Commit**

```bash
git add scripts/agent_afternoon.py
git commit -m "feat: agent_afternoon — blocks 49-72 weather-load analysis"
```

---

## Task 6: Agent Evening + Synthesis (blocks 73–96, 18:00–24:00)

**Files:**
- Create: `scripts/agent_evening.py`

This is the largest agent — it runs its own block analysis AND assembles all 4 sections into the 4 output files.

- [ ] **Step 1: Write `scripts/agent_evening.py`**

```python
"""
agent_evening.py — Agent 4: blocks 73-96 (18:00-24:00) + final synthesis.
Evening storm crash events, wind gust dip, evening cool-down.
Reads all 4 section outputs and builds: HTML, Excel, Notebook, PDF.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scripts.agent_analysis import (
    load_section_df, classify_regimes, get_storm_days,
    run_all_regressions, build_storm_profile, detect_monsoon_onset,
    build_coeff_heatmap, build_regime_calendar, build_storm_chart,
    build_yoy_drift_chart, build_onset_chart, save_section_outputs,
    SECTIONS, YEARS, WEATHER_FEATURES, block_to_time,
)

PERIOD    = 'evening'
BLK_START = 73
BLK_END   = 96
ROOT      = os.path.join(os.path.dirname(__file__), '..')
REPORTS   = os.path.join(ROOT, 'reports')
PERIODS   = ['night', 'morning', 'afternoon', 'evening']
PERIOD_LABELS = {
    'night':     'Night (00:00–06:00)',
    'morning':   'Morning (06:00–12:00)',
    'afternoon': 'Afternoon (12:00–18:00)',
    'evening':   'Evening (18:00–24:00)',
}


# ── Evening analysis ──────────────────────────────────────────────────────────

def run_evening_analysis():
    print(f"[{PERIOD}] Loading data blocks {BLK_START}–{BLK_END}...")
    df = load_section_df(BLK_START, BLK_END)
    print(f"[{PERIOD}] {len(df):,} rows loaded")

    storm_days = get_storm_days(df)
    print(f"[{PERIOD}] Storm days detected: {len(storm_days)}")

    regime_df        = classify_regimes(df)
    coeff_df         = run_all_regressions(df, storm_days)
    storm_profile_df, storm_events_df = build_storm_profile(df, storm_days)
    onset_df         = detect_monsoon_onset(df)

    figures = {
        'coeff_heatmap':   build_coeff_heatmap(coeff_df, PERIOD),
        'regime_calendar': build_regime_calendar(regime_df, PERIOD),
        'storm_profile':   build_storm_chart(storm_profile_df, PERIOD),
        'yoy_drift':       build_yoy_drift_chart(coeff_df, PERIOD),
        'onset':           build_onset_chart(onset_df),
    }
    save_section_outputs(PERIOD, coeff_df, regime_df, storm_profile_df,
                         storm_events_df, onset_df, figures)
    print(f"[{PERIOD}] Section analysis done.")


# ── Load all section data ─────────────────────────────────────────────────────

def load_all_sections() -> dict:
    data = {}
    for p in PERIODS:
        base = os.path.join(SECTIONS, f'section_{p}')
        data[p] = {
            'coeff':        pd.read_parquet(f'{base}_coeff.parquet'),
            'regime':       pd.read_parquet(f'{base}_regime.parquet'),
            'storm_profile':pd.read_parquet(f'{base}_storm_profile.parquet'),
            'storm_events': pd.read_parquet(f'{base}_storm_events.parquet'),
            'onset':        pd.read_parquet(f'{base}_onset.parquet'),
        }
        with open(f'{base}_charts.json') as f:
            data[p]['charts'] = json.load(f)
    return data


# ── HTML report ───────────────────────────────────────────────────────────────

def build_html_report(sections: dict) -> str:
    # Build summary stats
    all_storm_events = pd.concat(
        [v['storm_events'].assign(period=k) for k, v in sections.items()
         if not v['storm_events'].empty], ignore_index=True)
    all_onset = sections['afternoon']['onset']  # use afternoon for onset detection
    onset_summary = all_onset[all_onset['is_onset']].groupby('year')['date'].first()

    summary_html = '<h2>Executive Summary</h2><table border=1 cellpadding=6>'
    summary_html += '<tr><th>Year</th><th>Storm Days</th><th>Monsoon Onset</th><th>Worst Storm</th></tr>'
    for year in YEARS:
        storms = all_storm_events[all_storm_events['year'] == year] if not all_storm_events.empty else pd.DataFrame()
        n_storms = len(storms['date'].unique()) if not storms.empty else 0
        onset = onset_summary.get(year, 'Not detected')
        worst = storms.loc[storms['max_suppression_mw'].idxmax(), 'date'] if not storms.empty else 'N/A'
        summary_html += f'<tr><td>{year}</td><td>{n_storms}</td><td>{onset}</td><td>{worst}</td></tr>'
    summary_html += '</table>'

    # Tab JS/CSS
    tab_css = """
<style>
body { background:#1a1a2e; color:#eee; font-family:Arial,sans-serif; margin:20px; }
.tab { overflow:hidden; border-bottom:2px solid #444; margin-bottom:20px; }
.tab button { background:#2d2d44; color:#ccc; border:none; padding:12px 20px;
              cursor:pointer; font-size:14px; transition:0.3s; }
.tab button:hover, .tab button.active { background:#e74c3c; color:#fff; }
.tabcontent { display:none; }
.tabcontent.active { display:block; }
h2 { color:#e74c3c; } h3 { color:#f39c12; }
table { border-collapse:collapse; width:100%; }
th { background:#2d2d44; } td,th { padding:6px 12px; border:1px solid #444; }
</style>
<script>
function showTab(id) {
  document.querySelectorAll('.tabcontent').forEach(e => e.classList.remove('active'));
  document.querySelectorAll('.tab button').forEach(e => e.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  event.currentTarget.classList.add('active');
}
</script>
"""
    tab_buttons = '<div class="tab">'
    tab_buttons += '<button class="active" onclick="showTab(\'summary\')">Summary</button>'
    for p in PERIODS:
        label = PERIOD_LABELS[p].split(' ')[0]
        tab_buttons += f'<button onclick="showTab(\'{p}\')">{label}</button>'
    tab_buttons += '</div>'

    # Summary tab
    tabs_html = f'<div id="summary" class="tabcontent active">{summary_html}</div>'

    # Period tabs
    for p in PERIODS:
        charts = sections[p]['charts']
        coeff_df = sections[p]['coeff']
        low_r2 = coeff_df[coeff_df['r2'] < 0.3][['block','year','r2']].sort_values('r2')

        low_r2_html = ''
        if not low_r2.empty:
            low_r2_html = f'<p style="color:#f39c12;">⚠ {len(low_r2)} block-year combinations with R² &lt; 0.3 (low confidence)</p>'

        chart_divs = ''
        for chart_name, chart_json in charts.items():
            fig = pio.from_json(chart_json)
            chart_divs += pio.to_html(fig, full_html=False,
                                       include_plotlyjs=False, div_id=f'{p}_{chart_name}')

        tabs_html += f'''
<div id="{p}" class="tabcontent">
  <h2>{PERIOD_LABELS[p]}</h2>
  {low_r2_html}
  {chart_divs}
</div>'''

    plotlyjs = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
    html = f'<!DOCTYPE html><html><head><meta charset="utf-8">{plotlyjs}{tab_css}</head><body>'
    html += '<h1>Haryana Monsoon Weather-Load Impact Report (May-June 2022–2025)</h1>'
    html += tab_buttons + tabs_html + '</body></html>'
    return html


# ── Excel workbook ────────────────────────────────────────────────────────────

def build_excel_report(sections: dict, raw_df: pd.DataFrame):
    path = os.path.join(REPORTS, 'haryana_monsoon_report.xlsx')
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        # Sheet 1: Summary
        onset_rows = []
        for p in PERIODS:
            for year in YEARS:
                onset_day = sections[p]['onset']
                onset_f = onset_day[(onset_day['year']==year) & onset_day['is_onset']]
                storms = sections[p]['storm_events']
                storm_f = storms[storms['year']==year] if not storms.empty else pd.DataFrame()
                onset_rows.append({
                    'period': PERIOD_LABELS[p], 'year': year,
                    'monsoon_onset': onset_f['date'].iloc[0] if not onset_f.empty else None,
                    'storm_days': len(storm_f['date'].unique()) if not storm_f.empty else 0,
                })
        pd.DataFrame(onset_rows).to_excel(writer, sheet_name='Summary', index=False)

        # Sheets 2-5: Coefficients per period
        for p in PERIODS:
            sheet = p.capitalize() + '_Coefficients'
            sections[p]['coeff'].to_excel(writer, sheet_name=sheet, index=False)

        # Sheet 6: Storm Events
        all_events = pd.concat(
            [v['storm_events'].assign(period=k) for k, v in sections.items()
             if not v['storm_events'].empty], ignore_index=True)
        if not all_events.empty:
            all_events.to_excel(writer, sheet_name='Storm_Events', index=False)

        # Sheet 7: YoY Drift
        drift_rows = []
        for p in PERIODS:
            avg = sections[p]['coeff'].groupby('year')[WEATHER_FEATURES].mean().reset_index()
            avg['period'] = PERIOD_LABELS[p]
            drift_rows.append(avg)
        pd.concat(drift_rows).to_excel(writer, sheet_name='YoY_Drift', index=False)

        # Sheet 8: Raw Data (sample — full join)
        raw_df.head(5000).to_excel(writer, sheet_name='Raw_Data', index=False)

    print(f"Excel saved: {path}")


# ── Jupyter Notebook ──────────────────────────────────────────────────────────

def build_notebook():
    nb = new_notebook()
    cells = []

    cells.append(new_markdown_cell(
        '# Haryana Monsoon Weather-Load Impact Report\n'
        '## May-June 2022–2025 | Block-wise Analysis\n'
        'Re-run this notebook to regenerate the analysis with updated data.\n'
    ))
    cells.append(new_code_cell(
        'import sys, os\n'
        'sys.path.insert(0, os.path.abspath(".."))\n'
        'import pandas as pd\n'
        'from scripts.agent_analysis import *\n'
        'PERIODS = ["night","morning","afternoon","evening"]\n'
        'CACHE = "reports/cache"\n'
        'SECTIONS = "reports/sections"\n'
    ))
    cells.append(new_markdown_cell('## 1. Load cached data'))
    cells.append(new_code_cell(
        'w = pd.read_parquet(f"{CACHE}/may_jun_weather.parquet")\n'
        'l = pd.read_parquet(f"{CACHE}/may_jun_load.parquet")\n'
        'df = w.merge(l, on=["date","time_block","year","month"])\n'
        'print(f"Total rows: {len(df):,}")\n'
        'df.head(3)\n'
    ))

    for p, (bs, be) in zip(PERIODS, [(1,24),(25,48),(49,72),(73,96)]):
        cells.append(new_markdown_cell(f'## {PERIOD_LABELS[p]}'))
        cells.append(new_code_cell(
            f'section_df = load_section_df({bs}, {be})\n'
            f'storm_days = get_storm_days(section_df)\n'
            f'coeff_df = pd.read_parquet(f"{{SECTIONS}}/section_{p}_coeff.parquet")\n'
            f'regime_df = pd.read_parquet(f"{{SECTIONS}}/section_{p}_regime.parquet")\n'
            f'storm_profile_df = pd.read_parquet(f"{{SECTIONS}}/section_{p}_storm_profile.parquet")\n'
            f'print(f"[{p}] {bs}-{be}: " + str(len(section_df)) + " rows, " + str(len(storm_days)) + " storm days")\n'
            f'build_coeff_heatmap(coeff_df, "{p}").show()\n'
            f'build_regime_calendar(regime_df, "{p}").show()\n'
            f'build_storm_chart(storm_profile_df, "{p}").show()\n'
            f'build_yoy_drift_chart(coeff_df, "{p}").show()\n'
        ))

    nb.cells = cells
    path = os.path.join(REPORTS, 'haryana_monsoon_report.ipynb')
    with open(path, 'w') as f:
        nbformat.write(nb, f)
    print(f"Notebook saved: {path}")


# ── PDF report ────────────────────────────────────────────────────────────────

def build_pdf_report(sections: dict):
    path = os.path.join(REPORTS, 'haryana_monsoon_report.pdf')
    with PdfPages(path) as pdf:
        # Cover page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.text(0.5, 0.6, 'Haryana Monsoon Weather-Load\nImpact Report',
                ha='center', va='center', fontsize=28, fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.5, 0.4, 'May-June 2022–2025 | Block-level Analysis',
                ha='center', va='center', fontsize=16, color='gray',
                transform=ax.transAxes)
        ax.text(0.5, 0.25, 'Generated: 2026-05-01 | State: HARYANA',
                ha='center', va='center', fontsize=12, color='gray',
                transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # One page per period: coefficient bar chart
        for p in PERIODS:
            coeff_df = sections[p]['coeff']
            avg = coeff_df.groupby('year')[WEATHER_FEATURES].mean()
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            fig.suptitle(f'{PERIOD_LABELS[p]} — Weather Coefficients by Year', fontsize=14)
            for ax, feat in zip(axes.flat, WEATHER_FEATURES):
                ax.bar(avg.index.astype(str), avg[feat],
                       color=['#e74c3c','#f39c12','#2980b9','#27ae60'])
                ax.set_title(feat); ax.set_ylabel('Avg MW/unit')
                ax.axhline(0, color='black', linewidth=0.8)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Storm events summary page
        all_events = pd.concat(
            [v['storm_events'].assign(period=k) for k, v in sections.items()
             if not v['storm_events'].empty], ignore_index=True)
        if not all_events.empty:
            fig, ax = plt.subplots(figsize=(14, 6))
            for i, year in enumerate(YEARS):
                yev = all_events[all_events['year'] == year]
                ax.scatter([year]*len(yev), yev['max_suppression_mw'],
                           s=60, label=str(year), zorder=3)
            ax.set_xlabel('Year'); ax.set_ylabel('Max Suppression MW')
            ax.set_title('Storm Events — Max Load Suppression MW by Year')
            ax.legend(); ax.grid(True, alpha=0.3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"PDF saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Step 1: Evening analysis
    run_evening_analysis()

    # Step 2: Load all section data
    print("[synthesis] Loading all section outputs...")
    sections = load_all_sections()

    # Load raw data for Excel Raw_Data sheet
    import pyarrow.parquet as pq
    raw_w = pd.read_parquet(os.path.join(ROOT, 'reports', 'cache', 'may_jun_weather.parquet'))
    raw_l = pd.read_parquet(os.path.join(ROOT, 'reports', 'cache', 'may_jun_load.parquet'))
    raw_df = raw_w.merge(raw_l, on=['date','time_block','year','month'])

    os.makedirs(REPORTS, exist_ok=True)

    # Step 3: HTML
    print("[synthesis] Building HTML report...")
    html = build_html_report(sections)
    html_path = os.path.join(REPORTS, 'haryana_monsoon_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML saved: {html_path}")

    # Step 4: Excel
    print("[synthesis] Building Excel workbook...")
    build_excel_report(sections, raw_df)

    # Step 5: Notebook
    print("[synthesis] Building Jupyter notebook...")
    build_notebook()

    # Step 6: PDF
    print("[synthesis] Building PDF...")
    build_pdf_report(sections)

    print("\n[synthesis] All outputs complete:")
    for fname in ['haryana_monsoon_report.html','haryana_monsoon_report.xlsx',
                  'haryana_monsoon_report.ipynb','haryana_monsoon_report.pdf']:
        fpath = os.path.join(REPORTS, fname)
        size  = os.path.getsize(fpath) / 1024
        print(f"  {fname}  ({size:.0f} KB)")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/agent_evening.py
git commit -m "feat: agent_evening — blocks 73-96 + full report synthesis (HTML/Excel/Notebook/PDF)"
```

---

## Task 7: Orchestrator — run all 4 agents in parallel

**Files:**
- Create: `scripts/run_report.py`

- [ ] **Step 1: Write `scripts/run_report.py`**

```python
"""
run_report.py — Orchestrator.
1. Builds parquet cache (data_loader)
2. Runs Agents 1-3 in parallel (ThreadPoolExecutor)
3. Runs Agent 4 (evening + synthesis) after 1-3 complete
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.data_loader import build_cache
from scripts import agent_night, agent_morning, agent_afternoon, agent_evening


def run_agent(agent_module):
    name = agent_module.__name__.split('.')[-1]
    try:
        print(f"[orchestrator] Starting {name}...")
        agent_module.main()
        print(f"[orchestrator] {name} DONE")
        return name, None
    except Exception as e:
        print(f"[orchestrator] {name} FAILED: {e}")
        return name, e


def main():
    t0 = time.time()
    print("=" * 60)
    print("Haryana Monsoon Weather-Load Report — Full Run")
    print("=" * 60)

    # Phase 1: Build cache
    print("\n[Phase 1] Building data cache...")
    build_cache(force=False)

    # Phase 2: Run agents 1-3 in parallel
    print("\n[Phase 2] Running agents 1-3 in parallel...")
    parallel_agents = [agent_night, agent_morning, agent_afternoon]
    errors = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_agent, a): a for a in parallel_agents}
        for fut in as_completed(futures):
            name, err = fut.result()
            if err:
                errors.append((name, err))

    if errors:
        print(f"\n[orchestrator] ERROR: {len(errors)} agent(s) failed:")
        for name, err in errors:
            print(f"  {name}: {err}")
        sys.exit(1)

    # Phase 3: Agent 4 (evening + synthesis)
    print("\n[Phase 3] Running agent 4 (evening + synthesis)...")
    agent_evening.main()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Report complete in {elapsed:.1f}s")
    print(f"Outputs in: reports/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Add `scripts/__init__.py`**

```bash
touch scripts/__init__.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_report.py scripts/__init__.py
git commit -m "feat: run_report — parallel orchestrator for 4 monsoon analysis agents"
```

---

## Task 8: End-to-end run and verification

- [ ] **Step 1: Run the full pipeline**

```bash
cd "c:/Users/RamanSharma/OneDrive - GNA-Energy/Desktop/RD"
python scripts/run_report.py
```

Expected final output:
```
======================================================
Report complete in Xs
Outputs in: reports/
======================================================
```

- [ ] **Step 2: Verify all 4 output files exist and are non-empty**

```python
import os
REPORTS = 'reports'
files = [
    'haryana_monsoon_report.html',
    'haryana_monsoon_report.xlsx',
    'haryana_monsoon_report.ipynb',
    'haryana_monsoon_report.pdf',
]
for f in files:
    path = os.path.join(REPORTS, f)
    size = os.path.getsize(path) / 1024
    assert size > 10, f"{f} is too small ({size:.1f} KB)"
    print(f"  OK  {f}  ({size:.0f} KB)")
```

- [ ] **Step 3: Verify R² coverage**

```python
import pandas as pd, glob
coeff_files = glob.glob('reports/sections/section_*_coeff.parquet')
for f in coeff_files:
    df = pd.read_parquet(f)
    low_r2 = df[df['r2'] < 0.3]
    pct = len(low_r2) / len(df) * 100
    print(f"{os.path.basename(f)}: {pct:.1f}% blocks with R²<0.3")
```

- [ ] **Step 4: Open HTML report in browser and confirm 5 tabs load**

```bash
start reports/haryana_monsoon_report.html
```

Confirm: Summary, Night, Morning, Afternoon, Evening tabs all render charts.

- [ ] **Step 5: Final commit**

```bash
git add reports/
git commit -m "feat: generate haryana monsoon weather-load report (HTML/Excel/Notebook/PDF)"
```

---

## Self-Review

**Spec coverage check:**
- ✅ MySQL → parquet cache (Task 1)
- ✅ 4 agents by time-of-day (Tasks 3–6)
- ✅ K-means regime clustering (agent_analysis.py: `classify_regimes`)
- ✅ Storm day detection >15mm or >35 km/h (agent_analysis.py: `get_storm_days`)
- ✅ Block-wise OLS regression (agent_analysis.py: `run_all_regressions`)
- ✅ Storm event profiles with onset/duration/recovery (agent_analysis.py: `build_storm_profile`)
- ✅ YoY coefficient drift (agent_analysis.py: `build_yoy_drift_chart`)
- ✅ Monsoon onset detection rolling 5-day precip (agent_analysis.py: `detect_monsoon_onset`)
- ✅ R² flagging for low-confidence blocks (Task 8 step 3)
- ✅ HTML (tabbed, 5 tabs, Plotly) (agent_evening.py: `build_html_report`)
- ✅ Excel (8 sheets) (agent_evening.py: `build_excel_report`)
- ✅ Jupyter notebook (agent_evening.py: `build_notebook`)
- ✅ PDF via matplotlib PdfPages (agent_evening.py: `build_pdf_report`)
- ✅ Parallel execution agents 1-3 (run_report.py: ThreadPoolExecutor)
- ✅ Agent 4 synthesises after 1-3 complete (run_report.py phase ordering)
- ✅ All 4 years 2022–2025 (YEARS constant in agent_analysis.py)
- ✅ HARYANA, May-June filter (data_loader.py SQL WHERE clause)

**No gaps found.**
