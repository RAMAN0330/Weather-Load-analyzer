from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

BLOCK_COUNT = 96


@dataclass
class DailyLoadPanel:
    daily_matrix: pd.DataFrame
    context: pd.DataFrame


def _coerce_daily_blocks(day_df: pd.DataFrame) -> pd.Series:
    work = day_df.copy()
    work["time_block"] = pd.to_numeric(work.get("time_block"), errors="coerce")
    work["total_drawal"] = pd.to_numeric(work.get("total_drawal"), errors="coerce")
    work = work.dropna(subset=["time_block"])
    work["time_block"] = work["time_block"].astype(int)
    work = work[work["time_block"].between(1, BLOCK_COUNT)]
    if work.empty:
        return pd.Series(np.nan, index=range(1, BLOCK_COUNT + 1), dtype=float)

    by_block = (
        work.groupby("time_block")["total_drawal"]
        .mean()
        .reindex(range(1, BLOCK_COUNT + 1))
        .astype(float)
    )
    by_block = by_block.interpolate(limit_direction="both").ffill().bfill()
    return by_block


def build_daily_load_panel(df: pd.DataFrame) -> DailyLoadPanel:
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty")
    if "date" not in df.columns or "time_block" not in df.columns or "total_drawal" not in df.columns:
        raise ValueError("Expected columns: date, time_block, total_drawal")

    work = df.copy()
    work["date"] = work["date"].astype(str)
    work = work[work["date"].notna()]
    dates: List[str] = sorted(work["date"].unique().tolist())
    if not dates:
        raise ValueError("No valid dates found in dataframe")

    rows = []
    for date_str in dates:
        vec = _coerce_daily_blocks(work[work["date"] == date_str])
        rows.append(vec.to_numpy(dtype=float))

    daily_matrix = pd.DataFrame(
        np.vstack(rows),
        index=pd.Index(dates, name="date"),
        columns=list(range(1, BLOCK_COUNT + 1)),
    )
    daily_mean = daily_matrix.mean(axis=1)
    daily_std = daily_matrix.std(axis=1).fillna(0.0)
    daily_var = daily_matrix.var(axis=1).fillna(0.0)
    daily_range = (daily_matrix.max(axis=1) - daily_matrix.min(axis=1)).fillna(0.0)

    context = pd.DataFrame(index=daily_matrix.index)
    context["date"] = daily_matrix.index.astype(str)
    context["day_of_week"] = pd.to_datetime(context["date"]).dt.dayofweek.astype(int)
    context["is_weekend"] = (context["day_of_week"] >= 5).astype(int)
    context["daily_mean"] = daily_mean.to_numpy(dtype=float)
    context["daily_std"] = daily_std.to_numpy(dtype=float)
    context["daily_var"] = daily_var.to_numpy(dtype=float)
    context["daily_range"] = daily_range.to_numpy(dtype=float)
    context["recent_mean_3d"] = daily_mean.rolling(3, min_periods=1).mean().to_numpy(dtype=float)
    context["recent_mean_7d"] = daily_mean.rolling(7, min_periods=1).mean().to_numpy(dtype=float)
    context["recent_var_3d"] = daily_var.rolling(3, min_periods=1).mean().to_numpy(dtype=float)
    context["recent_var_7d"] = daily_var.rolling(7, min_periods=1).mean().to_numpy(dtype=float)

    vol_threshold = float(np.nanquantile(context["daily_var"].to_numpy(dtype=float), 0.75)) if len(context) else 0.0
    context["is_high_volatility"] = (context["daily_var"] >= max(vol_threshold, 1e-6)).astype(int)
    context["volatility_zscore"] = (
        (context["daily_var"] - context["daily_var"].mean())
        / max(float(context["daily_var"].std(ddof=0)), 1e-6)
    )
    context["load_ratio_1d"] = context["daily_mean"] / np.maximum(context["daily_mean"].shift(1).fillna(context["daily_mean"]), 1.0)
    context["load_ratio_7d"] = context["daily_mean"] / np.maximum(context["daily_mean"].shift(7).fillna(context["daily_mean"]), 1.0)
    context = context.reset_index(drop=True)

    return DailyLoadPanel(daily_matrix=daily_matrix, context=context)
