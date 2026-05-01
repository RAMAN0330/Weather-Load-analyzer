"""
Backtest the rebuilt T+1/T+2 pipeline against Haryana_mape.xlsx.

For each date listed in the xlsx (Apr 1–17, 2026), we:
  1. Use only the historical data strictly BEFORE that date as input.
  2. Produce T+1 (the date itself) and T+2 (next day) forecasts.
  3. Compare against the actuals stored in the same xlsx.
  4. Print per-day accuracy and the delta vs the xlsx benchmark columns
     (T+1_Accuracy_%, T+2_Accuracy_%).

Usage (from repo root):
    python scripts/backtest_haryana.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.short_term_pipeline import (  # noqa: E402
    run_short_term_pipeline,
    run_t2_pipeline,
)

XLSX_PATH = ROOT / "Haryana_mape.xlsx"


def _pick_history_csv() -> Path:
    """Use whichever Haryana CSV covers the most dates needed by the xlsx."""
    candidates = [ROOT / "final_data.csv", ROOT / "final_data_haryana.csv"]
    best = None
    best_max = pd.Timestamp.min
    for p in candidates:
        if not p.exists():
            continue
        try:
            head = pd.read_csv(p, usecols=["date"])
            mx = pd.to_datetime(head["date"], errors="coerce").max()
            if pd.notna(mx) and mx > best_max:
                best_max = mx
                best = p
        except Exception:
            continue
    if best is None:
        raise FileNotFoundError("No history CSV found.")
    print(f"[history] using {best.name} (max date {best_max.date()})")
    return best


def _load_history() -> pd.DataFrame:
    df = pd.read_csv(_pick_history_csv())
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date", "time_block", "total_drawal"]).reset_index(drop=True)
    df["time_block"] = pd.to_numeric(df["time_block"], errors="coerce").astype(int)
    df["total_drawal"] = pd.to_numeric(df["total_drawal"], errors="coerce")
    return df


def _load_xlsx() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(XLSX_PATH, sheet_name="Sheet1")
    block_df = raw[["date", "time_block", "Actual", "T+1", "T+2"]].copy()
    block_df["date"] = pd.to_datetime(block_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    block_df = block_df.dropna(subset=["date", "time_block"]).reset_index(drop=True)
    block_df["time_block"] = block_df["time_block"].astype(int)

    bench_cols = {"Date": "date", "T+1_Accuracy_%": "bench_t1", "T+2_Accuracy_%": "bench_t2"}
    bench = raw[list(bench_cols)].rename(columns=bench_cols)
    bench["date"] = pd.to_datetime(bench["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    bench = bench.dropna(subset=["date"]).reset_index(drop=True)
    return block_df, bench


def _accuracy(actual: np.ndarray, pred: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = (actual > 0) & np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return float("nan")
    return float(100.0 - np.mean(np.abs(pred[mask] - actual[mask]) / actual[mask]) * 100.0)


def _series_from_result(result, key="forecast") -> np.ndarray:
    if not isinstance(result, dict):
        return np.full(96, np.nan)
    series = result.get("series") or {}
    arr = series.get(key)
    if arr is None:
        return np.full(96, np.nan)
    a = np.asarray(arr, dtype=float)
    if a.size != 96:
        out = np.full(96, np.nan)
        out[: min(96, a.size)] = a[:96]
        return out
    return a


def main() -> None:
    history = _load_history()
    block_df, bench = _load_xlsx()

    rows = []
    for target_date in sorted(block_df["date"].unique()):
        target_ts = pd.Timestamp(target_date)
        hist_slice = history[pd.to_datetime(history["date"]) < target_ts].copy()
        if hist_slice.empty:
            print(f"[skip] {target_date}: no history before this date")
            continue

        # T+1 == target_date
        try:
            t1_res = run_short_term_pipeline(
                df=hist_slice,
                target_date=target_date,
                actual_blocks=0,
                config={"region": "haryana"},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[err  ] {target_date} T+1: {exc}")
            continue
        t1_pred = _series_from_result(t1_res)

        # T+2 == target_date + 1
        try:
            t2_res = run_t2_pipeline(
                df=hist_slice,
                t1_date=target_date,
                config={"region": "haryana", "candidate_lookback_days": 60},
                region="haryana",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[err  ] {target_date} T+2: {exc}")
            t2_pred = np.full(96, np.nan)
        else:
            t2_pred = _series_from_result(t2_res)

        # Actuals: T+1 actuals come from this row's "Actual" column.
        t1_actual = (
            block_df[block_df["date"] == target_date]
            .sort_values("time_block")["Actual"]
            .to_numpy(dtype=float)
        )
        # T+2 actuals come from the NEXT date's "Actual" column.
        next_date = (target_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        t2_actual_df = block_df[block_df["date"] == next_date].sort_values("time_block")
        t2_actual = t2_actual_df["Actual"].to_numpy(dtype=float) if not t2_actual_df.empty else np.full(96, np.nan)

        new_t1_acc = _accuracy(t1_actual, t1_pred) if t1_actual.size == 96 else float("nan")
        new_t2_acc = _accuracy(t2_actual, t2_pred) if t2_actual.size == 96 else float("nan")

        b = bench[bench["date"] == target_date]
        old_t1 = float(b["bench_t1"].iloc[0]) if not b.empty else float("nan")
        old_t2 = float(b["bench_t2"].iloc[0]) if not b.empty else float("nan")

        rows.append(
            {
                "date": target_date,
                "old_t1": old_t1,
                "new_t1": new_t1_acc,
                "d_t1": new_t1_acc - old_t1 if np.isfinite(new_t1_acc) and np.isfinite(old_t1) else float("nan"),
                "old_t2": old_t2,
                "new_t2": new_t2_acc,
                "d_t2": new_t2_acc - old_t2 if np.isfinite(new_t2_acc) and np.isfinite(old_t2) else float("nan"),
            }
        )

    if not rows:
        print("No rows produced — check inputs.")
        return

    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
    print()
    print(
        f"Mean T+1: old={out['old_t1'].mean():.2f}  new={out['new_t1'].mean():.2f}  "
        f"Δ={out['d_t1'].mean():+.2f}"
    )
    print(
        f"Mean T+2: old={out['old_t2'].mean():.2f}  new={out['new_t2'].mean():.2f}  "
        f"Δ={out['d_t2'].mean():+.2f}"
    )
    out.to_csv(ROOT / "exports" / "haryana_backtest.csv", index=False)
    print(f"\nWritten: {ROOT / 'exports' / 'haryana_backtest.csv'}")


if __name__ == "__main__":
    main()
