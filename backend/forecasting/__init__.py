"""Forecast Engine v3 — weather-aware day-ahead (T+1 / T+2) load forecasting.

Modules
-------
physics   Vectorised thermodynamic transforms (wet-bulb, dew point, CDH).
spatial   District → state aggregation (demand-share weighting, extremes).
quality   Input data-quality checks and the forecast validation gate.
features  Origin-safe supervised design matrix (vectorised by matrix shifting).
baselines Seasonal-naive and day-type matched recent-day baselines.
models    LightGBM quantile models (direct / residual) and baseline wrappers.
metrics   Vectorised accuracy metrics (WAPE, pinball, coverage, ...).
data      Pipeline-API data access with a TTL'd on-disk cache.
service   ``ForecastService`` orchestration used by the FastAPI router.

Contract: ``docs/forecast-v3-api.md``.
"""
from __future__ import annotations

MODEL_VERSION: str = "v3.0.0"
FEATURE_VERSION: str = "wx-f1"

BLOCKS_PER_DAY: int = 96
QUANTILES: tuple[float, float, float] = (0.1, 0.5, 0.9)

__all__ = ["MODEL_VERSION", "FEATURE_VERSION", "BLOCKS_PER_DAY", "QUANTILES"]
