from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .data import BLOCK_COUNT, DailyLoadPanel


@dataclass
class HybridSequenceDataset:
    X_seq: np.ndarray
    y_curve: np.ndarray
    y_scale: np.ndarray
    dates: List[str]
    gating_features: np.ndarray
    gating_frame: pd.DataFrame
    recent_actuals: np.ndarray


def _normalize_curve(curve: np.ndarray) -> tuple[np.ndarray, float]:
    vec = np.asarray(curve, dtype=float).reshape(-1)
    scale = float(np.mean(vec))
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    return vec / scale, scale


def build_sequence_dataset(
    panel: DailyLoadPanel,
    lookback_days: int = 7,
) -> HybridSequenceDataset:
    matrix = panel.daily_matrix.copy()
    ctx = panel.context.copy()
    n_days = len(matrix)
    if n_days <= lookback_days:
        raise ValueError("Insufficient history for sequence dataset")

    X_seq = []
    y_curve = []
    y_scale = []
    dates: List[str] = []
    gating_rows: List[Dict[str, float]] = []
    recent_actuals = []

    for idx in range(lookback_days, n_days):
        hist = matrix.iloc[idx - lookback_days:idx].to_numpy(dtype=float)
        target = matrix.iloc[idx].to_numpy(dtype=float)
        hist_norm = []
        for day_curve in hist:
            norm_curve, _ = _normalize_curve(day_curve)
            hist_norm.append(norm_curve)
        target_norm, target_scale = _normalize_curve(target)

        ctx_row = ctx.iloc[idx].copy()
        angle = (2.0 * np.pi * float(ctx_row["day_of_week"])) / 7.0
        gating_rows.append(
            {
                "is_weekend": float(ctx_row["is_weekend"]),
                "is_high_volatility": float(ctx_row["is_high_volatility"]),
                "volatility_zscore": float(ctx_row["volatility_zscore"]),
                "recent_var_3d": float(ctx_row["recent_var_3d"]),
                "recent_var_7d": float(ctx_row["recent_var_7d"]),
                "load_ratio_1d": float(ctx_row["load_ratio_1d"]),
                "load_ratio_7d": float(ctx_row["load_ratio_7d"]),
                "dow_sin": float(np.sin(angle)),
                "dow_cos": float(np.cos(angle)),
            }
        )

        X_seq.append(np.asarray(hist_norm, dtype=float))
        y_curve.append(target_norm)
        y_scale.append(target_scale)
        dates.append(str(matrix.index[idx]))
        recent_actuals.append(hist[-1])

    gating_frame = pd.DataFrame(gating_rows)
    return HybridSequenceDataset(
        X_seq=np.asarray(X_seq, dtype=float),
        y_curve=np.asarray(y_curve, dtype=float),
        y_scale=np.asarray(y_scale, dtype=float),
        dates=dates,
        gating_features=gating_frame.to_numpy(dtype=float),
        gating_frame=gating_frame,
        recent_actuals=np.asarray(recent_actuals, dtype=float),
    )


def build_target_features(
    panel: DailyLoadPanel,
    target_date: str,
    lookback_days: int = 7,
) -> Dict[str, np.ndarray]:
    matrix = panel.daily_matrix.copy()
    ctx = panel.context.copy()
    if target_date not in matrix.index:
        raise ValueError(f"Target date {target_date} not found in daily matrix")
    idx = matrix.index.get_loc(target_date)
    if int(idx) < int(lookback_days):
        raise ValueError("Insufficient history before target date")

    hist = matrix.iloc[idx - lookback_days:idx].to_numpy(dtype=float)
    hist_norm = []
    for day_curve in hist:
        norm_curve, _ = _normalize_curve(day_curve)
        hist_norm.append(norm_curve)

    ctx_row = ctx.iloc[idx].copy()
    angle = (2.0 * np.pi * float(ctx_row["day_of_week"])) / 7.0
    gating = np.asarray(
        [
            float(ctx_row["is_weekend"]),
            float(ctx_row["is_high_volatility"]),
            float(ctx_row["volatility_zscore"]),
            float(ctx_row["recent_var_3d"]),
            float(ctx_row["recent_var_7d"]),
            float(ctx_row["load_ratio_1d"]),
            float(ctx_row["load_ratio_7d"]),
            float(np.sin(angle)),
            float(np.cos(angle)),
        ],
        dtype=float,
    )
    return {
        "X_seq": np.asarray(hist_norm, dtype=float),
        "recent_actual": hist[-1].astype(float),
        "gating_features": gating,
        "context": ctx_row.to_dict(),
        "target_curve": matrix.loc[target_date].to_numpy(dtype=float),
    }
