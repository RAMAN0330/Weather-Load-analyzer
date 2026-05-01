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
    K-means (k=3 non-storm + forced storm=3) on daily features.
    Returns daily DataFrame: date_str, year, month, regime_id, regime_name.
    Storm days are always assigned regime 3.
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

    n_clusters = min(3, len(non_storm))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(features)
    non_storm = non_storm.copy()
    non_storm['regime_id'] = km.labels_

    # Re-label clusters by temperature descending (0=Hot-Dry, 1=Pre-Mon Humid, 2=Rain)
    centers = pd.DataFrame(km.cluster_centers_,
                           columns=['avg_temp', 'avg_hum', 'total_precip', 'avg_wind'])
    temp_order = centers['avg_temp'].argsort().values[::-1]
    remap = {int(old): int(new) for new, old in enumerate(temp_order)}
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
    For each (block, year) excluding storm days, fit OLS:
      load_mw ~ temperature + humidity + precipitation + showers + wind_speed + cloud_cover
    Returns DataFrame with coefficients, r2, n_samples per block-year.
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

def build_storm_profile(df: pd.DataFrame, storm_days: set):
    """
    Returns (storm_profile_df, storm_events_df).
    storm_profile_df: block, year, storm_load_avg, clear_load_avg, suppression_mw
    storm_events_df:  date, year, onset_block, max_suppression_mw, duration_blocks, recovery_blocks
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
                'storm_load_avg': round(float(s), 1) if not np.isnan(s) else None,
                'clear_load_avg': round(float(c), 1) if not np.isnan(c) else None,
                'suppression_mw': round(float(c - s), 1) if not (np.isnan(s) or np.isnan(c)) else None,
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
    storm_events_df = pd.DataFrame(onset_rows) if onset_rows else pd.DataFrame(
        columns=['date','year','onset_block','max_suppression_mw','duration_blocks','recovery_blocks'])
    return pd.DataFrame(results), storm_events_df


# ── Monsoon onset detection ───────────────────────────────────────────────────

def detect_monsoon_onset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling 5-day total precip per year. Onset = first date where 5-day sum > 25mm.
    Returns DataFrame: year, date, rolling_5d_precip, is_onset.
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
            rows.append({'year': year, 'date': d, 'rolling_5d_precip': round(float(val), 1),
                         'is_onset': (d == onset_date)})
    return pd.DataFrame(rows)


# ── Block number → time string ────────────────────────────────────────────────

def block_to_time(block: int) -> str:
    h = int((block - 1) * 0.25)
    m = int(((block - 1) * 0.25 % 1) * 60)
    return f'{h:02d}:{m:02d}'


# ── Plotly charts ─────────────────────────────────────────────────────────────

def build_coeff_heatmap(coeff_df: pd.DataFrame, period_name: str) -> go.Figure:
    """96-block × 6-variable coefficient heatmap, one subplot per year."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=[str(y) for y in YEARS],
                        shared_xaxes=True, shared_yaxes=True)
    for i, year in enumerate(YEARS):
        r, c = divmod(i, 2)
        ydf = coeff_df[coeff_df['year'] == year].sort_values('block')
        if ydf.empty:
            continue
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
    """Year-over-year drift of average weather coefficients."""
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
      reports/sections/section_{period_name}_coeff.parquet
      reports/sections/section_{period_name}_regime.parquet
      reports/sections/section_{period_name}_storm_profile.parquet
      reports/sections/section_{period_name}_storm_events.parquet
      reports/sections/section_{period_name}_onset.parquet
      reports/sections/section_{period_name}_charts.json
      reports/sections/section_{period_name}_excel.xlsx
    """
    os.makedirs(SECTIONS, exist_ok=True)
    base = os.path.join(SECTIONS, f'section_{period_name}')

    coeff_df = coeff_df.copy(); coeff_df['period'] = period_name
    regime_df = regime_df.copy(); regime_df['period'] = period_name
    storm_profile_df = storm_profile_df.copy(); storm_profile_df['period'] = period_name

    coeff_df.to_parquet(f'{base}_coeff.parquet', index=False)
    regime_df.to_parquet(f'{base}_regime.parquet', index=False)
    storm_profile_df.to_parquet(f'{base}_storm_profile.parquet', index=False)
    storm_events_df.to_parquet(f'{base}_storm_events.parquet', index=False)
    onset_df.to_parquet(f'{base}_onset.parquet', index=False)

    charts_json = {name: pio.to_json(fig) for name, fig in figures.items()}
    with open(f'{base}_charts.json', 'w') as f:
        json.dump(charts_json, f)

    with pd.ExcelWriter(f'{base}_excel.xlsx', engine='openpyxl') as writer:
        coeff_df.to_excel(writer, sheet_name='Coefficients', index=False)
        regime_df.to_excel(writer, sheet_name='Regimes', index=False)
        storm_profile_df.to_excel(writer, sheet_name='Storm_Profile', index=False)
        storm_events_df.to_excel(writer, sheet_name='Storm_Events', index=False)

    print(f"[{period_name}] Section outputs saved to {SECTIONS}")
