"""
Backtest T+1 and T+2 forecasts for May–June 2025 (Haryana).

Uses the cached parquet files in reports/cache/ as the data source so no
live database connection is needed.

Run from the project root:
    python backend/backtest_may_jun_2025.py

Output:
    - Console table: per-date T+1 and T+2 MAPE
    - reports/backtest_may_jun_2025.csv  (machine-readable)
    - reports/backtest_may_jun_2025.xlsx (formatted with summary sheet)
"""

import sys, os, datetime, logging
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backtest")

ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(ROOT, "reports", "cache")
OUT_DIR = os.path.join(ROOT, "reports")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_full_df() -> pd.DataFrame:
    load_path = os.path.join(CACHE_DIR, "may_jun_load.parquet")
    wx_path   = os.path.join(CACHE_DIR, "may_jun_weather.parquet")
    load = pd.read_parquet(load_path)
    wx   = pd.read_parquet(wx_path)
    load["date"] = pd.to_datetime(load["date"]).dt.strftime("%Y-%m-%d")
    wx["date"]   = pd.to_datetime(wx["date"]).dt.strftime("%Y-%m-%d")
    df = pd.merge(load, wx, on=["date", "time_block", "year", "month"], how="inner")
    df = df.rename(columns={"load_mw": "total_drawal", "wind_speed": "wind_speed_10m"})
    df = df.sort_values(["date", "time_block"]).reset_index(drop=True)
    logger.info("Loaded %d rows, %d dates", len(df), df["date"].nunique())
    return df


# ── MAPE helper ───────────────────────────────────────────────────────────────

def _mape(actual: np.ndarray, forecast: np.ndarray, mask: np.ndarray = None) -> float:
    if mask is None:
        mask = (actual > 50) & np.isfinite(actual) & np.isfinite(forecast)
    if mask.sum() == 0:
        return float("nan")
    a, f = actual[mask], forecast[mask]
    return float(np.mean(np.abs(f - a) / np.maximum(a, 1.0)) * 100)


def _extract_96(result: dict, key: str = "forecast") -> np.ndarray | None:
    """Pull a 96-block vector from pipeline result series dict."""
    series = result.get("series", {}) if isinstance(result, dict) else {}
    vec = series.get(key) or result.get(key)
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=float)
    return arr if len(arr) == 96 else None


# ── Per-date runner ───────────────────────────────────────────────────────────

def run_date(df: pd.DataFrame, target_date: str, run_t2: bool = True) -> dict:
    from short_term_pipeline import run_short_term_pipeline, run_t2_pipeline

    cfg = {"region": "haryana", "debug_stages": False}
    row = {"date": target_date}

    # actual load for this date (96 blocks)
    tgt = df[df["date"] == target_date].sort_values("time_block")
    actual = tgt["total_drawal"].to_numpy(dtype=float)
    if len(actual) != 96:
        row.update(t1_mape=None, t2_mape=None, error="incomplete actuals")
        return row

    # ── T+1 ──────────────────────────────────────────────────────────────────
    try:
        r1 = run_short_term_pipeline(df, target_date, actual_blocks=0, config=cfg)
        fc1 = _extract_96(r1, "forecast")
        if fc1 is not None:
            row["t1_mape"] = round(_mape(actual, fc1), 3)
            row["t1_acc"]  = round(100 - row["t1_mape"], 3)
            row["t1_peak_mape"] = round(_mape(actual[47:72], fc1[47:72]), 3)  # blocks 48-72
        else:
            row.update(t1_mape=None, t1_acc=None, t1_peak_mape=None, t1_error="no forecast vector")
    except Exception as e:
        row.update(t1_mape=None, t1_acc=None, t1_peak_mape=None, t1_error=str(e)[:120])

    # ── T+2 ──────────────────────────────────────────────────────────────────
    if run_t2:
        # T+2 forecast is issued one day before target_date
        t1_date = (datetime.date.fromisoformat(target_date) - datetime.timedelta(days=1)).isoformat()
        if t1_date not in df["date"].values:
            row.update(t2_mape=None, t2_acc=None, t2_peak_mape=None, t2_error="no prior day data")
        else:
            try:
                r2 = run_t2_pipeline(df, t1_date=t1_date, config=cfg)
                fc2 = _extract_96(r2, "forecast")
                if fc2 is None:
                    # try alternate key
                    fc2 = _extract_96(r2, "t2_forecast")
                if fc2 is not None:
                    row["t2_mape"] = round(_mape(actual, fc2), 3)
                    row["t2_acc"]  = round(100 - row["t2_mape"], 3)
                    row["t2_peak_mape"] = round(_mape(actual[47:72], fc2[47:72]), 3)
                else:
                    row.update(t2_mape=None, t2_acc=None, t2_peak_mape=None, t2_error="no T+2 forecast vector")
            except Exception as e:
                row.update(t2_mape=None, t2_acc=None, t2_peak_mape=None, t2_error=str(e)[:120])
    else:
        row.update(t2_mape=None, t2_acc=None, t2_peak_mape=None)

    # metadata
    row["avg_temp"]  = round(float(tgt["temperature"].mean()), 1) if "temperature" in tgt.columns else None
    row["peak_load"] = round(float(actual.max()), 1)
    dow = datetime.date.fromisoformat(target_date).strftime("%A")
    row["day_of_week"] = dow
    row["is_weekend"] = dow in ("Saturday", "Sunday")
    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    df = load_full_df()

    # Filter to May–June 2025 test dates; need all prior data for context window
    all_dates_2025 = sorted(df[df["date"].str.startswith("2025")]["date"].unique())
    test_dates = [d for d in all_dates_2025 if "2025-05" <= d <= "2025-06-30"]
    print(f"Running backtest on {len(test_dates)} dates (May–Jun 2025) …\n")

    results = []
    for i, date in enumerate(test_dates):
        sys.stdout.write(f"\r  {i+1:02d}/{len(test_dates)}  {date} …        ")
        sys.stdout.flush()
        row = run_date(df, date, run_t2=True)
        results.append(row)
        t1 = f"{row.get('t1_mape', '?'):>6}" if row.get('t1_mape') is not None else "  ERR"
        t2 = f"{row.get('t2_mape', '?'):>6}" if row.get('t2_mape') is not None else "  ERR"
        sys.stdout.write(f"\r  {i+1:02d}/{len(test_dates)}  {date}  T+1={t1}%  T+2={t2}%\n")
        sys.stdout.flush()

    results_df = pd.DataFrame(results)

    # ── Summary stats ─────────────────────────────────────────────────────────
    def _stats(col):
        s = results_df[col].dropna()
        if s.empty:
            return {}
        return {
            "mean":   round(s.mean(), 3),
            "median": round(s.median(), 3),
            "min":    round(s.min(), 3),
            "max":    round(s.max(), 3),
            "p90":    round(s.quantile(0.9), 3),
            "days_under_3pct": int((s < 3).sum()),
            "days_under_5pct": int((s < 5).sum()),
            "n":      int(s.count()),
        }

    t1_stats = _stats("t1_mape")
    t2_stats = _stats("t2_mape")

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY — May–June 2025 (Haryana)")
    print("=" * 60)
    print(f"{'Metric':<25} {'T+1':>8} {'T+2':>8}")
    print("-" * 43)
    for k in ("mean", "median", "min", "max", "p90", "days_under_3pct", "days_under_5pct", "n"):
        v1 = t1_stats.get(k, "—")
        v2 = t2_stats.get(k, "—")
        unit = "%" if "days" not in k and k != "n" else ""
        print(f"  {k:<23} {str(v1)+unit:>8} {str(v2)+unit:>8}")

    # Weekend vs weekday
    for subset, label in [("is_weekend == True", "Weekend"), ("is_weekend == False", "Weekday")]:
        sub = results_df.query(subset)
        t1m = round(sub["t1_mape"].mean(), 3) if "t1_mape" in sub and sub["t1_mape"].notna().any() else "—"
        t2m = round(sub["t2_mape"].mean(), 3) if "t2_mape" in sub and sub["t2_mape"].notna().any() else "—"
        print(f"  {label+' avg MAPE':<23} {str(t1m)+'%':>8} {str(t2m)+'%':>8}")

    print("=" * 60)

    # ── Save outputs ──────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "backtest_may_jun_2025.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")

    try:
        import openpyxl
        xlsx_path = os.path.join(OUT_DIR, "backtest_may_jun_2025.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
            # Detail sheet
            results_df.to_excel(xw, sheet_name="Per-Date Results", index=False)

            # Summary sheet
            summary_rows = []
            for k in ("mean", "median", "min", "max", "p90", "days_under_3pct", "days_under_5pct", "n"):
                summary_rows.append({"Metric": k, "T+1": t1_stats.get(k), "T+2": t2_stats.get(k)})
            pd.DataFrame(summary_rows).to_excel(xw, sheet_name="Summary", index=False)

            # Highlight MAPE columns
            ws = xw.sheets["Per-Date Results"]
            from openpyxl.styles import PatternFill, Font
            header = {cell.value: cell.column for cell in ws[1]}
            for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
                for col_name, col_idx in header.items():
                    if col_name in ("t1_mape", "t2_mape"):
                        cell = row[col_idx - 1]
                        if cell.value is not None:
                            try:
                                v = float(cell.value)
                                if v < 3:
                                    cell.fill = PatternFill("solid", fgColor="C6EFCE")
                                elif v < 5:
                                    cell.fill = PatternFill("solid", fgColor="FFEB9C")
                                else:
                                    cell.fill = PatternFill("solid", fgColor="FFC7CE")
                            except (TypeError, ValueError):
                                pass

        print(f"Excel saved: {xlsx_path}")
    except ImportError:
        print("(openpyxl not installed — Excel skipped)")

    return results_df


if __name__ == "__main__":
    main()
