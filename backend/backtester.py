"""
Rolling Backtest Framework for Load Forecasting Pipeline.

Runs the forecast pipeline as if it were day-ahead for each of the last N days,
computes block-level MAPE, RMSE, peak-hour accuracy, and energy error.
Results are stratified by season, day-type, weather regime, and time-of-day.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


def run_backtest(
    df: pd.DataFrame,
    pipeline_fn,
    n_days: int = 90,
    actual_blocks_sim: int = 36,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Rolling backtest: for each of the last n_days, run pipeline as day-ahead
    with actual_blocks_sim known blocks and compare forecast vs actual.

    Args:
        df: Full historical DataFrame with date, time_block, total_drawal, weather cols
        pipeline_fn: Callable(df, target_date, config) -> dict with series.forecast, series.actual
        n_days: Number of days to backtest
        actual_blocks_sim: Simulated number of known blocks (e.g., 36 = 09:00)
        config: Pipeline config override

    Returns:
        Dict with daily_results, summary_metrics, stratified_metrics
    """
    dates = sorted(df["date"].unique())
    if len(dates) < n_days + 15:
        n_days = max(len(dates) - 15, 7)

    test_dates = dates[-n_days:]
    daily_results = []

    for target_date in test_dates:
        try:
            result = pipeline_fn(df, target_date, config=config or {})
            if not isinstance(result, dict):
                continue

            series = result.get("series", {})
            forecast = np.asarray(series.get("forecast", []), dtype=float)
            actual = np.asarray(series.get("actual", []), dtype=float)

            if len(forecast) != 96 or len(actual) != 96:
                continue

            # Only evaluate forecast blocks (not known blocks)
            fc_window = forecast[actual_blocks_sim:]
            ac_window = actual[actual_blocks_sim:]

            # Filter out zero/nan actuals
            valid = (ac_window > 10) & np.isfinite(ac_window) & np.isfinite(fc_window)
            if valid.sum() < 10:
                continue

            fc_valid = fc_window[valid]
            ac_valid = ac_window[valid]

            mape = float(np.mean(np.abs(fc_valid - ac_valid) / np.maximum(ac_valid, 1.0)) * 100)
            rmse = float(np.sqrt(np.mean((fc_valid - ac_valid) ** 2)))
            energy_fc = float(np.sum(fc_valid) * 0.25)  # MWh
            energy_ac = float(np.sum(ac_valid) * 0.25)
            energy_error_pct = float((energy_fc - energy_ac) / max(energy_ac, 1.0) * 100)

            # Peak hour accuracy (blocks 48-72 = 12:00-18:00)
            peak_start = max(48 - actual_blocks_sim, 0)
            peak_end = min(72 - actual_blocks_sim, len(fc_window))
            if peak_end > peak_start:
                peak_fc = fc_window[peak_start:peak_end]
                peak_ac = ac_window[peak_start:peak_end]
                peak_valid = (peak_ac > 10) & np.isfinite(peak_ac) & np.isfinite(peak_fc)
                peak_mape = float(np.mean(np.abs(peak_fc[peak_valid] - peak_ac[peak_valid]) / np.maximum(peak_ac[peak_valid], 1.0)) * 100) if peak_valid.sum() > 0 else None
            else:
                peak_mape = None

            # Classify day
            dt = pd.to_datetime(target_date)
            dow = dt.weekday()
            day_type = "Weekend" if dow >= 5 else "Weekday"
            month = dt.month
            if month in (3, 4, 5):
                season_label = "Summer"
            elif month in (6, 7, 8, 9):
                season_label = "Monsoon"
            elif month in (10, 11):
                season_label = "Post-Monsoon"
            else:
                season_label = "Winter"

            # Temperature regime
            temp_col = None
            for c in ["temperature_2m", "temperature", "temp"]:
                if c in df.columns:
                    temp_col = c
                    break
            avg_temp = None
            if temp_col:
                day_temps = df[df["date"] == target_date][temp_col]
                avg_temp = float(day_temps.mean()) if len(day_temps) > 0 else None

            temp_regime = "Unknown"
            if avg_temp is not None:
                if avg_temp >= 38:
                    temp_regime = "Hot"
                elif avg_temp >= 25:
                    temp_regime = "Warm"
                elif avg_temp >= 15:
                    temp_regime = "Mild"
                else:
                    temp_regime = "Cold"

            daily_results.append({
                "date": str(target_date),
                "mape": round(mape, 2),
                "rmse": round(rmse, 1),
                "energy_error_pct": round(energy_error_pct, 2),
                "peak_mape": round(peak_mape, 2) if peak_mape is not None else None,
                "day_type": day_type,
                "season": season_label,
                "temp_regime": temp_regime,
                "avg_temp": round(avg_temp, 1) if avg_temp is not None else None,
            })

        except Exception as e:
            logger.warning(f"Backtest failed for {target_date}: {e}")
            continue

    if not daily_results:
        return {"daily_results": [], "summary": {}, "stratified": {}}

    results_df = pd.DataFrame(daily_results)

    # Overall summary
    summary = {
        "n_days": len(results_df),
        "mape_mean": round(float(results_df["mape"].mean()), 2),
        "mape_median": round(float(results_df["mape"].median()), 2),
        "mape_p90": round(float(results_df["mape"].quantile(0.9)), 2),
        "rmse_mean": round(float(results_df["rmse"].mean()), 1),
        "energy_error_mean_pct": round(float(results_df["energy_error_pct"].mean()), 2),
        "peak_mape_mean": round(float(results_df["peak_mape"].dropna().mean()), 2) if results_df["peak_mape"].notna().sum() > 0 else None,
        "days_under_3pct": int((results_df["mape"] < 3.0).sum()),
        "days_under_5pct": int((results_df["mape"] < 5.0).sum()),
    }

    # Stratified metrics
    stratified = {}
    for group_col in ["season", "day_type", "temp_regime"]:
        group_stats = {}
        for name, grp in results_df.groupby(group_col):
            group_stats[str(name)] = {
                "n": len(grp),
                "mape_mean": round(float(grp["mape"].mean()), 2),
                "mape_median": round(float(grp["mape"].median()), 2),
                "rmse_mean": round(float(grp["rmse"].mean()), 1),
                "peak_mape_mean": round(float(grp["peak_mape"].dropna().mean()), 2) if grp["peak_mape"].notna().sum() > 0 else None,
            }
        stratified[group_col] = group_stats

    # Time-of-day breakdown (morning ramp, peak, valley)
    # This would require block-level storage, simplified here
    stratified["time_of_day"] = {
        "morning_ramp": {"blocks": "24-40", "note": "06:00-10:00"},
        "peak": {"blocks": "48-72", "note": "12:00-18:00"},
        "evening": {"blocks": "72-88", "note": "18:00-22:00"},
        "valley": {"blocks": "0-24", "note": "00:00-06:00"},
    }

    return {
        "daily_results": daily_results,
        "summary": summary,
        "stratified": stratified,
        "config": {
            "n_days": n_days,
            "actual_blocks_sim": actual_blocks_sim,
        }
    }


def compare_models(
    df: pd.DataFrame,
    champion_fn,
    challenger_fn,
    n_days: int = 14,
    actual_blocks_sim: int = 36,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Champion-challenger comparison: run both models on same dates and compare.
    """
    champion_results = run_backtest(df, champion_fn, n_days, actual_blocks_sim, config)
    challenger_results = run_backtest(df, challenger_fn, n_days, actual_blocks_sim, config)

    c_summary = champion_results.get("summary", {})
    ch_summary = challenger_results.get("summary", {})

    comparison = {
        "champion": c_summary,
        "challenger": ch_summary,
        "challenger_wins": ch_summary.get("mape_mean", 999) < c_summary.get("mape_mean", 999),
        "mape_improvement": round(c_summary.get("mape_mean", 0) - ch_summary.get("mape_mean", 0), 2),
        "recommendation": "promote_challenger" if ch_summary.get("mape_mean", 999) < c_summary.get("mape_mean", 999) else "keep_champion",
    }

    return {
        "champion": champion_results,
        "challenger": challenger_results,
        "comparison": comparison,
    }
