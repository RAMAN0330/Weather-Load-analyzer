"""Forecast Engine v3 orchestration (contract: ``docs/forecast-v3-api.md``).

``ForecastService`` wires data access → spatial weather features → origin-safe
design matrix → model → validation gate → contract-shaped dict. The data loader
is injectable (``ForecastService(loader=...)``) so tests run on synthetic data.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import BLOCKS_PER_DAY, FEATURE_VERSION, MODEL_VERSION
from .data import CACHE_DIR, DataSourceError, LoaderFn, load_region_frames
from .features import ALL_GROUPS, build_design, build_load_matrix, feature_columns, normalise_load_frame
from .metrics import compute_metrics, daily_metrics
from .models import LGBM_MODELS, MODEL_SPECS, make_model, reference_baseline
from .quality import (
    DEFAULT_RAMP_THRESHOLD_MW,
    data_quality_report,
    validate_forecast_blocks,
    validate_load,
    validate_weather,
    worst_status,
)
from .spatial import SpatialConfig, build_state_weather_features

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
MAX_BACKTEST_DAYS: int = 62
REFIT_EVERY_DAYS: int = 7
LAG_PAD_DAYS: int = 21  # extra history for lag-14 / rolling-7 features
QUALITY_WINDOW_DAYS: int = 28
WEATHER_NOTE: str = (
    "Historical weather is the stored (possibly observed) value, not the forecast "
    "issued at origin; accuracy may be optimistic."
)
_MODEL_CACHE_MAX: int = 32


class InsufficientHistory(ValueError):
    """Fewer than the required days of load history before the origin."""


def _r(x: Any, nd: int = 1) -> float | None:
    """Round to ``nd`` decimals; NaN/inf/None → None (JSON null, never 0)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _nanagg(fn: Callable, a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    return float(fn(a[np.isfinite(a)])) if np.isfinite(a).any() else float("nan")


def _block_time(block: int) -> str:
    m = (block - 1) * 15
    return f"{m // 60:02d}:{m % 60:02d}"


def _region_prefix(region: str) -> str:
    r = str(region).strip().lower()
    return "hry" if r in {"haryana", "hr", "hry"} else (r.replace(" ", "")[:3] or "reg")


def _day(d: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


class ForecastService:
    """Train-on-demand quantile forecaster with backtesting and data-quality checks.

    Parameters
    ----------
    loader : callable ``(region, from_date, to_date) -> (load_df, weather_long)``;
        defaults to :func:`data.load_region_frames` (pipeline API + disk cache).
    records_dir : where immutable forecast JSON records are written
        (``None`` → ``$FORECAST_RECORDS_DIR`` or ``backend/.cache/forecasting/records``;
        ``False`` disables).
    feature_groups : feature groups used by the LightGBM models (ablation).
    """

    def __init__(
        self,
        loader: LoaderFn | None = None,
        records_dir: str | Path | None | bool = None,
        feature_groups: Iterable[str] = ALL_GROUPS,
        train_days: int = 730,
        min_history_days: int = 28,
        ramp_threshold_mw: float = DEFAULT_RAMP_THRESHOLD_MW,
        spatial_config: SpatialConfig = SpatialConfig(),
        today_fn: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        self._loader: LoaderFn = loader or load_region_frames
        if records_dir is False:
            self._records_dir: Path | None = None
        else:
            default_dir = os.getenv("FORECAST_RECORDS_DIR") or CACHE_DIR / "records"
            self._records_dir = Path(records_dir) if records_dir else Path(default_dir)
        self.feature_groups: tuple[str, ...] = tuple(feature_groups)
        self.feature_cols: list[str] = feature_columns(self.feature_groups)
        self.train_days = int(train_days)
        self.min_history_days = int(min_history_days)
        self.ramp_threshold_mw = float(ramp_threshold_mw)
        self.spatial_config = spatial_config
        self._today_fn = today_fn or (lambda: pd.Timestamp(datetime.now(IST).date()))
        self._models: OrderedDict[tuple, Any] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ data
    def _fetch(self, region: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            load_df, wx_long = self._loader(region, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        except (DataSourceError, InsufficientHistory):
            raise
        except Exception as exc:
            raise DataSourceError(f"Data source failed: {type(exc).__name__}: {exc}") from exc
        if load_df is None:
            raise DataSourceError("Data source returned no load frame.")
        return load_df, (wx_long if wx_long is not None else pd.DataFrame())

    @staticmethod
    def _grid_start(start: pd.Timestamp, lf: pd.DataFrame, wx_long: pd.DataFrame) -> pd.Timestamp:
        """Earliest grid day: the requested start, trimmed to the first observed date.

        Rows before the first observation would be all-NaN, so trimming changes
        no feature value but keeps the (time × district) matrices small.
        """
        firsts = [lf["date"].min()] if not lf.empty else []
        if wx_long is not None and not wx_long.empty and "date" in wx_long.columns:
            firsts.append(pd.to_datetime(wx_long["date"]).min().normalize())
        firsts = [f for f in firsts if pd.notna(f)]
        return max(start, min(firsts)) if firsts else start

    def _history_days(self, lf: pd.DataFrame, start: pd.Timestamp, origin: pd.Timestamp) -> int:
        h = lf[(lf["date"] >= start) & (lf["date"] <= origin) & np.isfinite(lf["load"].to_numpy(dtype=np.float64))]
        return int(h["date"].nunique())

    # ---------------------------------------------------------------- models
    def _fit(self, key: str, train: pd.DataFrame):
        return make_model(key).fit(train, self.feature_cols)

    def _get_model(self, cache_key: tuple, key: str, train_fn: Callable[[], pd.DataFrame]):
        with self._lock:
            if cache_key in self._models:
                self._models.move_to_end(cache_key)
                return self._models[cache_key]
        model = self._fit(key, train_fn())
        with self._lock:
            self._models[cache_key] = model
            while len(self._models) > _MODEL_CACHE_MAX:
                self._models.popitem(last=False)
        return model

    @staticmethod
    def _validate_args(horizon: int, model: str) -> None:
        if int(horizon) not in (1, 2):
            raise ValueError("horizon must be 1 or 2")
        if model not in MODEL_SPECS:
            raise ValueError(f"model must be one of {sorted(MODEL_SPECS)}")

    def _train_rows(self, design: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        d = design["date"]
        m = (d <= cutoff) & (d > cutoff - pd.Timedelta(days=self.train_days)) & np.isfinite(design["load"].to_numpy(dtype=np.float64))
        return design.loc[m]

    # -------------------------------------------------------------- forecast
    def prepare(self, region: str, target_date: str | pd.Timestamp, horizon: int) -> dict[str, Any]:
        """Fetch inputs and build the origin-safe design for one target date.

        Exposed for diagnostics and leakage tests. Load rows dated after the
        origin are discarded before any feature is computed.
        """
        D = _day(target_date)
        h = int(horizon)
        O = D - pd.Timedelta(days=h)
        start = O - pd.Timedelta(days=self.train_days + LAG_PAD_DAYS)
        load_raw, wx_long = self._fetch(region, start, D)
        lf = normalise_load_frame(load_raw)
        history_days = self._history_days(lf, start, O)
        if history_days < self.min_history_days:
            raise InsufficientHistory(
                f"Only {history_days} day(s) of load history before origin {O:%Y-%m-%d}; "
                f"at least {self.min_history_days} required."
            )
        days = pd.date_range(min(self._grid_start(start, lf, wx_long), O), D, freq="D", name="date")
        wx = build_state_weather_features(wx_long, region, self.spatial_config, days=days)
        hist = lf[lf["date"] <= O]
        lm = build_load_matrix(hist, days, until=O)
        design = build_design(lm, wx, h, groups=self.feature_groups)
        target_rows = design.loc[design["date"] == D].reset_index(drop=True)
        actual = (
            lf[lf["date"] == D].groupby("block")["load"].mean().reindex(range(1, BLOCKS_PER_DAY + 1)).to_numpy(dtype=np.float64)
        )
        hist_raw = load_raw
        if "date" in load_raw.columns:
            hist_raw = load_raw.loc[pd.to_datetime(load_raw["date"]).dt.normalize() <= O]
            hist_raw.attrs = dict(load_raw.attrs)
        return {
            "D": D, "O": O, "h": h, "start": start, "days": days, "wx": wx, "wx_long": wx_long,
            "lm": lm, "design": design, "target_rows": target_rows, "actual": actual,
            "history_days": history_days, "hist_raw": hist_raw,
        }

    def forecast(
        self,
        region: str,
        target_date: str | pd.Timestamp,
        horizon: int = 1,
        model: str = "lgbm_residual",
    ) -> dict[str, Any]:
        """Day-ahead quantile forecast for ``target_date`` (contract shape)."""
        self._validate_args(horizon, model)
        ctx = self.prepare(region, target_date, horizon)
        D, O, h = ctx["D"], ctx["O"], ctx["h"]
        design, rows = ctx["design"], ctx["target_rows"]

        cache_key = (region.lower(), O, h, model, FEATURE_VERSION, self.feature_groups, self.train_days)
        fitted = self._get_model(cache_key, model, lambda: self._train_rows(design, O))
        q = fitted.predict(rows)
        baseline = reference_baseline(rows)
        actual = ctx["actual"]

        blocks = [
            {
                "block": b,
                "time": _block_time(b),
                "p10": _r(q[b - 1, 0]),
                "p50": _r(q[b - 1, 1]),
                "p90": _r(q[b - 1, 2]),
                "baseline": _r(baseline[b - 1]),
                "actual": _r(actual[b - 1]),
            }
            for b in range(1, BLOCKS_PER_DAY + 1)
        ]
        gate_status, gate_warn = validate_forecast_blocks(blocks, self.ramp_threshold_mw)
        quality = self._forecast_quality(region, ctx, gate_status, gate_warn, fitted)

        digest = hashlib.sha1()
        digest.update(f"{region.lower()}|{D:%Y-%m-%d}|{h}|{model}|{MODEL_VERSION}|{FEATURE_VERSION}|{self.feature_groups}".encode())
        digest.update(np.ascontiguousarray(ctx["lm"].values).tobytes())
        digest.update(np.ascontiguousarray(ctx["wx"].loc[D].to_numpy(dtype=np.float64)).tobytes())
        forecast_id = f"{_region_prefix(region)}-{D:%Y%m%d}-h{h}-{digest.hexdigest()[:8]}"

        result = {
            "forecast_id": forecast_id,
            "region": region,
            "target_date": D.strftime("%Y-%m-%d"),
            "origin_date": O.strftime("%Y-%m-%d"),
            "horizon": h,
            "model": model,
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": quality["status"],
            "blocks": blocks,
            "summary": self._summary(q),
            "drivers": self._drivers(ctx["wx"], D),
            "quality": quality,
            "feature_importance": fitted.feature_importance(12),
        }
        self._write_record(result)
        return result

    def _forecast_quality(self, region: str, ctx: dict, gate_status: str, gate_warn: list[str], fitted) -> dict[str, Any]:
        D, O = ctx["D"], ctx["O"]
        q_from = O - pd.Timedelta(days=QUALITY_WINDOW_DAYS - 1)
        load_sec, load_status, load_warn = validate_load(ctx["hist_raw"], q_from, O)
        wx_sec, wx_status, wx_warn = validate_weather(ctx["wx_long"], D, D, region)
        cov = ctx["wx"].loc[D, "wx_weight_coverage"].to_numpy(dtype=np.float64)
        warnings_ = [*load_warn, *wx_warn, *gate_warn]
        if getattr(fitted, "backend", "") == "sklearn_hgb":
            warnings_.append("LightGBM unavailable; used sklearn HistGradientBoosting quantile fallback.")
        if ctx["history_days"] < 365:
            warnings_.append(f"Only {ctx['history_days']} days of history; yearly seasonality not learned.")
        return {
            "status": worst_status(load_status, wx_status, gate_status),
            "weather_coverage_pct": _r(wx_sec["coverage_pct"]) or 0.0,
            "load_share_coverage_pct": _r(np.nan_to_num(cov, nan=0.0).mean() * 100.0) or 0.0,
            "history_days": int(ctx["history_days"]),
            "missing_load_blocks": int(load_sec["expected_blocks"] - load_sec["present_blocks"]),
            "warnings": warnings_,
        }

    @staticmethod
    def _summary(q: np.ndarray) -> dict[str, Any]:
        p50 = q[:, 1]
        if not np.isfinite(p50).all():
            return {k: None for k in ("energy_mwh", "peak_mw", "peak_block", "min_mw", "min_block", "mean_band_width_mw", "max_ramp_mw", "max_ramp_block")}
        ramps = np.abs(np.diff(p50))
        r_idx = int(np.argmax(ramps))
        return {
            "energy_mwh": _r(p50.sum() * 0.25),
            "peak_mw": _r(p50.max()),
            "peak_block": int(np.argmax(p50)) + 1,
            "min_mw": _r(p50.min()),
            "min_block": int(np.argmin(p50)) + 1,
            "mean_band_width_mw": _r(np.mean(q[:, 2] - q[:, 0])),
            "max_ramp_mw": _r(ramps[r_idx]),
            "max_ramp_block": r_idx + 2,
        }

    @staticmethod
    def _drivers(wx: pd.DataFrame, D: pd.Timestamp) -> dict[str, float | None]:
        day = wx.loc[D]
        col = lambda c: day[c].to_numpy(dtype=np.float64)  # noqa: E731
        precip = col("precip_wmean")
        return {
            "temp_weighted_max": _r(_nanagg(np.max, col("temp_wmean")), 2),
            "temp_p90_max": _r(_nanagg(np.max, col("temp_p90")), 2),
            "wet_bulb_weighted_max": _r(_nanagg(np.max, col("wb_wmean")), 2),
            "wet_bulb_p90_max": _r(_nanagg(np.max, col("wb_p90")), 2),
            "cdh_24h_end": _r(col("cdh24_wmean")[-1], 2),
            "night_min_temp": _r(col("night_min_temp")[0], 2),
            "hot_share_peak": _r(_nanagg(np.max, col("hot_share")), 3),
            "precip_total_mm": _r(_nanagg(np.sum, precip), 2),
        }

    def _write_record(self, result: dict[str, Any]) -> None:
        """Persist an immutable JSON record (never overwrites an existing id)."""
        if self._records_dir is None:
            return
        try:
            self._records_dir.mkdir(parents=True, exist_ok=True)
            path = self._records_dir / f"{result['forecast_id']}.json"
            with open(path, "x", encoding="utf-8") as fh:
                json.dump(result, fh, separators=(",", ":"))
        except FileExistsError:
            pass
        except OSError as exc:
            logger.warning("could not write forecast record: %s", exc)

    # -------------------------------------------------------------- backtest
    def backtest(
        self,
        region: str,
        date_from: str | pd.Timestamp,
        date_to: str | pd.Timestamp,
        horizons: Sequence[int] = (1, 2),
        models: Sequence[str] = ("seasonal_naive", "recent_day", "lgbm_residual"),
    ) -> dict[str, Any]:
        """Rolling-origin backtest (≤ 62 target dates).

        Features for each target use only data ≤ its own origin (structural, via
        row shifting). LightGBM models are refit weekly: a target with origin
        ``O`` uses the model fit on targets dated ≤ ``fit_origin`` where
        ``fit_origin`` is the latest weekly anchor ≤ ``O``.
        """
        horizons = sorted({int(h) for h in horizons})
        for h in horizons:
            self._validate_args(h, "seasonal_naive")
        for m in models:
            self._validate_args(1, m)
        f, t = _day(date_from), _day(date_to)
        if t < f:
            raise ValueError("date_to must be on or after date_from")
        targets = pd.date_range(f, t, freq="D")[:MAX_BACKTEST_DAYS]
        t = targets[-1]
        anchor0 = targets[0] - pd.Timedelta(days=max(horizons))
        start = anchor0 - pd.Timedelta(days=self.train_days + LAG_PAD_DAYS)
        load_raw, wx_long = self._fetch(region, start, t)
        lf = normalise_load_frame(load_raw)
        days = pd.date_range(min(self._grid_start(start, lf, wx_long), anchor0), t, freq="D", name="date")
        wx = build_state_weather_features(wx_long, region, self.spatial_config, days=days)
        lm = build_load_matrix(lf, days)
        day_has = np.isfinite(lm.values).any(axis=1)
        hist_count = np.cumsum(day_has)  # days with data from start up to row i
        t_rows = days.get_indexer(targets)

        results: list[dict[str, Any]] = []
        daily: list[dict[str, Any]] = []
        any_valid = False
        for h in horizons:
            design = build_design(lm, wx, h, groups=self.feature_groups)
            o_rows = t_rows - h
            ok = hist_count[o_rows] >= self.min_history_days
            if not ok.any():
                continue
            any_valid = True
            v_targets = targets[ok]
            origins = v_targets - pd.Timedelta(days=h)
            anchor = targets[0] - pd.Timedelta(days=h)
            steps = ((origins - anchor).days // REFIT_EVERY_DAYS).to_numpy()
            fit_origins = anchor + pd.to_timedelta(steps * REFIT_EVERY_DAYS, unit="D")
            tgt_design = design.loc[design["date"].isin(v_targets)].reset_index(drop=True)
            actual = lm.values[t_rows[ok]]
            for key in models:
                preds = np.full((len(v_targets), BLOCKS_PER_DAY, 3), np.nan)
                for fo in pd.unique(fit_origins):
                    sel = np.asarray(fit_origins == fo)
                    cache_key = (region.lower(), pd.Timestamp(fo), h, key, FEATURE_VERSION, self.feature_groups, self.train_days, "bt")
                    model = self._get_model(cache_key, key, lambda fo=fo: self._train_rows(design, pd.Timestamp(fo)))
                    rows = tgt_design.loc[tgt_design["date"].isin(v_targets[sel])]
                    preds[sel] = model.predict(rows).reshape(int(sel.sum()), BLOCKS_PER_DAY, 3)
                has_actual = np.isfinite(actual).any(axis=1)
                metrics = compute_metrics(actual, preds[..., 1], preds[..., 0], preds[..., 2])
                results.append({"model": key, "horizon": h, "n_days": int(has_actual.sum()), "metrics": metrics})
                dm = daily_metrics(actual, preds[..., 1])
                daily.extend(
                    {
                        "model": key,
                        "horizon": h,
                        "date": d.strftime("%Y-%m-%d"),
                        "wape": _r(dm["wape"][i], 3),
                        "mae": _r(dm["mae"][i], 2),
                        "peak_error_mw": _r(dm["peak_error_mw"][i], 2),
                    }
                    for i, d in enumerate(v_targets)
                    if has_actual[i]
                )
        if not any_valid:
            raise InsufficientHistory(
                f"No target date in {f:%Y-%m-%d}..{t:%Y-%m-%d} has {self.min_history_days} days of history before its origin."
            )
        return {
            "region": region,
            "date_from": targets[0].strftime("%Y-%m-%d"),
            "date_to": t.strftime("%Y-%m-%d"),
            "results": results,
            "daily": daily,
            "weather_note": WEATHER_NOTE,
        }

    # ---------------------------------------------------------- data quality
    def data_quality(self, region: str, days: int = 30, to_date: str | pd.Timestamp | None = None) -> dict[str, Any]:
        """Input quality over the ``days`` dates ending yesterday (IST) or ``to_date``."""
        days = int(days)
        if days < 1:
            raise ValueError("days must be ≥ 1")
        to = _day(to_date) if to_date is not None else self._today_fn() - pd.Timedelta(days=1)
        frm = to - pd.Timedelta(days=days - 1)
        load_raw, wx_long = self._fetch(region, frm, to)
        return data_quality_report(region, load_raw, wx_long, frm, to)

    # ---------------------------------------------------------------- models
    @staticmethod
    def models() -> dict[str, Any]:
        return {
            "models": [{"key": k, **spec} for k, spec in MODEL_SPECS.items()],
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
        }


__all__ = ["ForecastService", "InsufficientHistory", "DataSourceError", "MAX_BACKTEST_DAYS", "WEATHER_NOTE", "LGBM_MODELS"]
