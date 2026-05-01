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
            'coeff':         pd.read_parquet(f'{base}_coeff.parquet'),
            'regime':        pd.read_parquet(f'{base}_regime.parquet'),
            'storm_profile': pd.read_parquet(f'{base}_storm_profile.parquet'),
            'storm_events':  pd.read_parquet(f'{base}_storm_events.parquet'),
            'onset':         pd.read_parquet(f'{base}_onset.parquet'),
        }
        with open(f'{base}_charts.json') as f:
            data[p]['charts'] = json.load(f)
    return data


# ── HTML report ───────────────────────────────────────────────────────────────

def build_html_report(sections: dict) -> str:
    all_storm_events_list = [
        v['storm_events'].assign(period=k) for k, v in sections.items()
        if not v['storm_events'].empty
    ]
    all_storm_events = pd.concat(all_storm_events_list, ignore_index=True) if all_storm_events_list else pd.DataFrame()
    all_onset = sections['afternoon']['onset']
    onset_summary = all_onset[all_onset['is_onset']].groupby('year')['date'].first()

    summary_html = '<h2>Executive Summary</h2><table border=1 cellpadding=6>'
    summary_html += '<tr><th>Year</th><th>Storm Days</th><th>Monsoon Onset</th><th>Worst Storm Date</th></tr>'
    for year in YEARS:
        storms = all_storm_events[all_storm_events['year'] == year] if not all_storm_events.empty else pd.DataFrame()
        n_storms = len(storms['date'].unique()) if not storms.empty else 0
        onset = onset_summary.get(year, 'Not detected')
        if not storms.empty and 'max_suppression_mw' in storms.columns:
            worst = storms.loc[storms['max_suppression_mw'].idxmax(), 'date']
        else:
            worst = 'N/A'
        summary_html += f'<tr><td>{year}</td><td>{n_storms}</td><td>{onset}</td><td>{worst}</td></tr>'
    summary_html += '</table>'

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

    tabs_html = f'<div id="summary" class="tabcontent active">{summary_html}</div>'

    for p in PERIODS:
        charts = sections[p]['charts']
        coeff_df = sections[p]['coeff']
        low_r2 = coeff_df[coeff_df['r2'] < 0.3][['block','year','r2']].sort_values('r2')

        low_r2_html = ''
        if not low_r2.empty:
            low_r2_html = (f'<p style="color:#f39c12;">&#9888; {len(low_r2)} block-year '
                           f'combinations with R&sup2; &lt; 0.3 (low confidence)</p>')

        chart_divs = ''
        for chart_name, chart_json in charts.items():
            fig = pio.from_json(chart_json)
            chart_divs += pio.to_html(fig, full_html=False,
                                       include_plotlyjs=False,
                                       div_id=f'{p}_{chart_name}')

        tabs_html += f'''
<div id="{p}" class="tabcontent">
  <h2>{PERIOD_LABELS[p]}</h2>
  {low_r2_html}
  {chart_divs}
</div>'''

    plotlyjs = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
    html = f'<!DOCTYPE html><html><head><meta charset="utf-8">{plotlyjs}{tab_css}</head><body>'
    html += '<h1>Haryana Monsoon Weather-Load Impact Report (May-June 2022&#8211;2025)</h1>'
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
                onset_f = onset_day[(onset_day['year'] == year) & onset_day['is_onset']]
                storms = sections[p]['storm_events']
                storm_f = storms[storms['year'] == year] if not storms.empty else pd.DataFrame()
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
        all_events_list = [
            v['storm_events'].assign(period=k) for k, v in sections.items()
            if not v['storm_events'].empty
        ]
        if all_events_list:
            all_events = pd.concat(all_events_list, ignore_index=True)
            all_events.to_excel(writer, sheet_name='Storm_Events', index=False)

        # Sheet 7: YoY Drift
        drift_rows = []
        for p in PERIODS:
            avg = sections[p]['coeff'].groupby('year')[WEATHER_FEATURES].mean().reset_index()
            avg['period'] = PERIOD_LABELS[p]
            drift_rows.append(avg)
        pd.concat(drift_rows).to_excel(writer, sheet_name='YoY_Drift', index=False)

        # Sheet 8: Raw Data (first 5000 rows)
        raw_df.head(5000).to_excel(writer, sheet_name='Raw_Data', index=False)

    print(f"Excel saved: {path}")


# ── Jupyter Notebook ──────────────────────────────────────────────────────────

def build_notebook():
    nb = new_notebook()
    cells = []

    cells.append(new_markdown_cell(
        '# Haryana Monsoon Weather-Load Impact Report\n'
        '## May-June 2022-2025 | Block-wise Analysis\n'
        'Re-run this notebook to regenerate the analysis with updated data.\n'
    ))
    cells.append(new_code_cell(
        'import sys, os\n'
        'sys.path.insert(0, os.path.abspath("."))\n'
        'import pandas as pd\n'
        'from scripts.agent_analysis import *\n'
        'CACHE = "reports/cache"\n'
        'SECTIONS_DIR = "reports/sections"\n'
    ))
    cells.append(new_markdown_cell('## 1. Load cached data'))
    cells.append(new_code_cell(
        'w = pd.read_parquet(f"{CACHE}/may_jun_weather.parquet")\n'
        'l = pd.read_parquet(f"{CACHE}/may_jun_load.parquet")\n'
        'df = w.merge(l, on=["date","time_block","year","month"])\n'
        'print(f"Total rows: {len(df):,}")\n'
        'df.head(3)\n'
    ))

    period_blocks = [('night',1,24), ('morning',25,48), ('afternoon',49,72), ('evening',73,96)]
    for p, bs, be in period_blocks:
        cells.append(new_markdown_cell(f'## {PERIOD_LABELS[p]}'))
        cells.append(new_code_cell(
            f'section_df = load_section_df({bs}, {be})\n'
            f'storm_days = get_storm_days(section_df)\n'
            f'coeff_df = pd.read_parquet(f"{{SECTIONS_DIR}}/section_{p}_coeff.parquet")\n'
            f'regime_df = pd.read_parquet(f"{{SECTIONS_DIR}}/section_{p}_regime.parquet")\n'
            f'storm_profile_df = pd.read_parquet(f"{{SECTIONS_DIR}}/section_{p}_storm_profile.parquet")\n'
            f'print(f"[{p}] {bs}-{be}: {{len(section_df)}} rows, {{len(storm_days)}} storm days")\n'
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
        ax.text(0.5, 0.4, 'May-June 2022-2025 | Block-level Analysis',
                ha='center', va='center', fontsize=16, color='gray',
                transform=ax.transAxes)
        ax.text(0.5, 0.25, 'Generated: 2026-05-01 | State: HARYANA',
                ha='center', va='center', fontsize=12, color='gray',
                transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # One page per period: coefficient bar charts
        for p in PERIODS:
            coeff_df = sections[p]['coeff']
            avg = coeff_df.groupby('year')[WEATHER_FEATURES].mean()
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            fig.suptitle(f'{PERIOD_LABELS[p]} — Weather Coefficients by Year', fontsize=14)
            for ax, feat in zip(axes.flat, WEATHER_FEATURES):
                ax.bar(avg.index.astype(str), avg[feat],
                       color=['#e74c3c','#f39c12','#2980b9','#27ae60'])
                ax.set_title(feat)
                ax.set_ylabel('Avg MW/unit')
                ax.axhline(0, color='black', linewidth=0.8)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Storm events summary page
        all_events_list = [
            v['storm_events'].assign(period=k) for k, v in sections.items()
            if not v['storm_events'].empty
        ]
        if all_events_list:
            all_events = pd.concat(all_events_list, ignore_index=True)
            fig, ax = plt.subplots(figsize=(14, 6))
            for i, year in enumerate(YEARS):
                yev = all_events[all_events['year'] == year]
                if not yev.empty and 'max_suppression_mw' in yev.columns:
                    ax.scatter([year]*len(yev), yev['max_suppression_mw'],
                               s=60, label=str(year), zorder=3)
            ax.set_xlabel('Year')
            ax.set_ylabel('Max Suppression MW')
            ax.set_title('Storm Events — Max Load Suppression MW by Year')
            ax.legend()
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"PDF saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_evening_analysis()

    print("[synthesis] Loading all section outputs...")
    sections = load_all_sections()

    raw_w = pd.read_parquet(os.path.join(ROOT, 'reports', 'cache', 'may_jun_weather.parquet'))
    raw_l = pd.read_parquet(os.path.join(ROOT, 'reports', 'cache', 'may_jun_load.parquet'))
    raw_df = raw_w.merge(raw_l, on=['date','time_block','year','month'])

    os.makedirs(REPORTS, exist_ok=True)

    print("[synthesis] Building HTML report...")
    html = build_html_report(sections)
    html_path = os.path.join(REPORTS, 'haryana_monsoon_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML saved: {html_path}")

    print("[synthesis] Building Excel workbook...")
    build_excel_report(sections, raw_df)

    print("[synthesis] Building Jupyter notebook...")
    build_notebook()

    print("[synthesis] Building PDF...")
    build_pdf_report(sections)

    print("\n[synthesis] All outputs complete:")
    for fname in ['haryana_monsoon_report.html','haryana_monsoon_report.xlsx',
                  'haryana_monsoon_report.ipynb','haryana_monsoon_report.pdf']:
        fpath = os.path.join(REPORTS, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / 1024
            print(f"  {fname}  ({size:.0f} KB)")


if __name__ == '__main__':
    main()
