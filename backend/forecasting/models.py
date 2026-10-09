"""Forecast models: LightGBM quantile (direct / residual) and baselines.

All models share one interface::

    m = make_model(key)
    m.fit(train_design, feature_cols)   # rows with known target, date ≤ origin
    q = m.predict(design_rows)          # (n, 3) array of p10, p50, p90

Quantile outputs are made non-crossing by sorting along the quantile axis.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import QUANTILES

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "lgbm_residual": {"label": "LightGBM residual", "quantiles": True, "default": True},
    "lgbm_direct": {"label": "LightGBM direct", "quantiles": True, "default": False},
    "recent_day": {"label": "Recent day (day-type matched)", "quantiles": False, "default": False},
    "seasonal_naive": {"label": "Seasonal naive (D-7)", "quantiles": False, "default": False},
}
LGBM_MODELS: frozenset[str] = frozenset({"lgbm_residual", "lgbm_direct"})

LGBM_PARAMS: dict[str, Any] = {
    "objective": "quantile",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": max(1, min(4, os.cpu_count() or 1)),
    "verbose": -1,
    "deterministic": True,
    "force_row_wise": True,
}
EARLY_STOPPING_ROUNDS: int = 50
VALIDATION_DAYS: int = 14
MIN_TRAIN_DAYS_FOR_ES: int = 21
BAND_WINDOW_DAYS: int = 56
DEFAULT_BAND: tuple[float, float] = (-0.05, 0.05)


def _import_lightgbm():
    """Import hook (monkeypatchable in tests); returns the module or None."""
    try:
        import lightgbm  # noqa: WPS433

        return lightgbm
    except Exception:  # pragma: no cover - depends on environment
        return None


def reference_baseline(df: pd.DataFrame) -> NDArray[np.float64]:
    """Recent-day baseline with D−7 then D−h fallbacks (origin-safe)."""
    base = df["recent_day_baseline"].to_numpy(dtype=np.float64)
    for col in ("lag_7d_load", "lag_h_load"):
        alt = df[col].to_numpy(dtype=np.float64)
        base = np.where(np.isfinite(base), base, alt)
    return base


def _ordered(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Enforce p10 ≤ p50 ≤ p90 (NaN-safe sort) and non-negativity."""
    return np.maximum(np.sort(q, axis=1), 0.0)


@dataclass
class BaselineModel:
    """Point baseline with an empirical relative-error band (p10/p90)."""

    key: str
    band: tuple[float, float] = DEFAULT_BAND
    backend: str = "baseline"

    def point(self, df: pd.DataFrame) -> NDArray[np.float64]:
        rd = reference_baseline(df)
        if self.key == "seasonal_naive":
            sn = df["lag_7d_load"].to_numpy(dtype=np.float64)
            return np.where(np.isfinite(sn), sn, rd)
        return rd

    def fit(self, train: pd.DataFrame, feature_cols: Sequence[str] | None = None) -> "BaselineModel":
        """Estimate the band from relative errors over the last 56 training days."""
        if train.empty:
            return self
        cutoff = train["date"].max() - pd.Timedelta(days=BAND_WINDOW_DAYS)
        recent = train[train["date"] > cutoff]
        y = recent["load"].to_numpy(dtype=np.float64)
        p = self.point(recent)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = y / p - 1.0
        r = r[np.isfinite(r)]
        if r.size >= 7 * 96:
            self.band = (float(np.quantile(r, 0.1)), float(np.quantile(r, 0.9)))
        return self

    def predict(self, df: pd.DataFrame) -> NDArray[np.float64]:
        p = self.point(df)
        q = np.stack([p * (1.0 + self.band[0]), p, p * (1.0 + self.band[1])], axis=1)
        return _ordered(q)

    def feature_importance(self, top: int = 12) -> list[dict[str, float]]:
        return []


@dataclass
class QuantileGBM:
    """Three quantile regressors (α = 0.1, 0.5, 0.9) on the design matrix.

    ``residual=True`` learns ``load − reference_baseline`` and adds the
    baseline back at prediction time.
    """

    key: str
    residual: bool
    params: dict[str, Any] = field(default_factory=lambda: dict(LGBM_PARAMS))
    backend: str = "lightgbm"
    feature_cols: list[str] = field(default_factory=list)
    models: list[Any] = field(default_factory=list)
    best_iterations: list[int] = field(default_factory=list)

    def _target(self, df: pd.DataFrame) -> NDArray[np.float64]:
        y = df["load"].to_numpy(dtype=np.float64)
        return y - reference_baseline(df) if self.residual else y

    def _new(self, lgb, alpha: float, n_estimators: int | None = None):
        if lgb is not None:
            p = dict(self.params, alpha=alpha)
            if n_estimators is not None:
                p["n_estimators"] = n_estimators
            return lgb.LGBMRegressor(**p)
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=alpha,
            learning_rate=self.params.get("learning_rate", 0.05),
            max_iter=n_estimators or 300,
            max_leaf_nodes=self.params.get("num_leaves", 31),
            min_samples_leaf=self.params.get("min_child_samples", 40),
            early_stopping=False,
            random_state=self.params.get("random_state", 42),
        )

    def fit(self, train: pd.DataFrame, feature_cols: Sequence[str]) -> "QuantileGBM":
        """Fit with early stopping on the chronologically last 14 training days,
        then refit on all rows with the selected iteration count."""
        lgb = _import_lightgbm()
        self.backend = "lightgbm" if lgb is not None else "sklearn_hgb"
        self.feature_cols = list(feature_cols)
        y = self._target(train)
        keep = np.isfinite(y)
        if keep.sum() == 0:
            raise ValueError("no finite training targets")
        X = train.loc[keep, self.feature_cols].astype(np.float64)
        y = y[keep]
        dates = train.loc[keep, "date"].to_numpy()
        uniq = np.unique(dates)
        use_es = lgb is not None and len(uniq) >= MIN_TRAIN_DAYS_FOR_ES + VALIDATION_DAYS
        self.models, self.best_iterations = [], []
        for alpha in QUANTILES:
            if use_es:
                split = uniq[-VALIDATION_DAYS]
                tr, va = dates < split, dates >= split
                probe = self._new(lgb, alpha)
                probe.fit(
                    X[tr],
                    y[tr],
                    eval_set=[(X[va], y[va])],
                    eval_metric="quantile",
                    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
                )
                best = int(probe.best_iteration_ or self.params["n_estimators"])
                n_est = max(best, 50)
            else:
                n_est = 200
            model = self._new(lgb, alpha, n_estimators=n_est)
            model.fit(X, y)
            self.models.append(model)
            self.best_iterations.append(n_est)
        return self

    def predict(self, df: pd.DataFrame) -> NDArray[np.float64]:
        X = df[self.feature_cols].astype(np.float64)
        q = np.column_stack([m.predict(X) for m in self.models])
        if self.residual:
            q = q + reference_baseline(df)[:, None]
        return _ordered(q)

    def feature_importance(self, top: int = 12) -> list[dict[str, float]]:
        """Gain share (%) of the median model, top ``top`` features."""
        if not self.models or self.backend != "lightgbm":
            return []
        gain = np.asarray(self.models[1].booster_.feature_importance(importance_type="gain"), dtype=np.float64)
        total = gain.sum()
        if total <= 0:
            return []
        order = np.argsort(-gain)[:top]
        return [{"feature": self.feature_cols[i], "gain_pct": round(float(gain[i] / total * 100.0), 2)} for i in order]


def make_model(key: str) -> BaselineModel | QuantileGBM:
    """Factory for a model key from :data:`MODEL_SPECS`."""
    if key not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{key}'. Allowed: {sorted(MODEL_SPECS)}")
    if key in LGBM_MODELS:
        return QuantileGBM(key=key, residual=(key == "lgbm_residual"))
    return BaselineModel(key=key)


__all__ = [
    "MODEL_SPECS",
    "LGBM_MODELS",
    "LGBM_PARAMS",
    "BaselineModel",
    "QuantileGBM",
    "make_model",
    "reference_baseline",
]
