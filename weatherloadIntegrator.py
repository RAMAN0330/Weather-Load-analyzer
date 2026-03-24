import warnings
from typing import Optional, List
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Canonical feature definitions ──────────────────────────────────
# Maps canonical output name → (possible raw column names, expected correlation sign with load)
FEATURE_SPEC = {
    "temperature":      (["apparent_temperature", "temperature_2m", "temperature"], +1),
    "humidity":         (["relative_humidity_2m", "relativehumidity_2m", "humidity"], -1),
    "precipitation":    (["precipitation", "rain"],                                  -1),
    "cloud_cover":      (["cloud_cover"],                                            -1),
    "cloud_cover_low":  (["cloud_cover_low"],                                        -1),
    "sunshine_duration": (["sunshine_duration"],                                      +1),
    "direct_radiation": (["direct_radiation", "direct_radiation_instant"],            +1),
    "wind_speed_10m":   (["wind_speed_10m"],                                         -1),
}


class WeatherLoadIntegrator:
    """Integrates multi-location weather data into state-level features
    using correlation-based location weighting — one pivot per feature."""

    def __init__(self, weather_df: pd.DataFrame, load_series: pd.Series):
        self.weather_df = weather_df
        self.load_series = load_series

    # ── helpers ────────────────────────────────────────────────────
    def _resolve_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure a Datetime column exists."""
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "Datetime"})
            df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if "time_block" in df.columns:
                df["Datetime"] = df["date"] + pd.to_timedelta(
                    (df["time_block"] - 1) * 15, unit="m"
                )
            elif "time" in df.columns:
                df["Datetime"] = pd.to_datetime(
                    df["date"].astype(str) + " " + df["time"].astype(str),
                    errors="coerce",
                )
        return df

    def _resolve_columns(self, df: pd.DataFrame) -> tuple:
        """Map raw columns → canonical names.  Returns (df, list_of_available_features)."""
        rename = {}
        available = []
        for canon, (alts, _sign) in FEATURE_SPEC.items():
            if canon in df.columns:
                available.append(canon)
                continue
            for alt in alts:
                if alt in df.columns:
                    rename[alt] = canon
                    available.append(canon)
                    break
        if rename:
            df = df.rename(columns=rename)
        return df, available

    @staticmethod
    def _robust_weights(
        corr: pd.Series,
        expected_sign: int = 1,
        shrink: float = 0.05,
    ) -> pd.Series:
        """Correlation → normalised location weights (fast softmax variant)."""
        c = corr.fillna(0) * expected_sign
        w = np.exp(c.clip(lower=-3, upper=3))          # bounded softmax
        w = w * (1 - shrink) + shrink / max(len(w), 1)  # shrinkage
        total = w.sum()
        return w / total if total > 0 else pd.Series(1 / max(len(w), 1), index=w.index)

    # ── main entry point ──────────────────────────────────────────
    def compute_state_level_features(self) -> pd.DataFrame:
        """Return a DataFrame indexed by Datetime with one column per weather feature,
        each collapsed to a single state-level value via correlation-weighted location average."""

        df = self.weather_df.copy()
        df = self._resolve_datetime(df)
        df, available = self._resolve_columns(df)

        if not available:
            raise ValueError(
                f"No usable weather features found.  Columns: {list(df.columns)}"
            )

        # Ensure location column
        if "location" not in df.columns:
            df["location"] = "single"

        # Coerce numerics once
        for col in available:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Dedup & keep only what we need
        keep_cols = ["Datetime", "location"] + available
        df = df[keep_cols].drop_duplicates(subset=["Datetime", "location"], keep="last")
        df = df.dropna(subset=["Datetime"])

        # Align load index
        load = self.load_series.copy()
        load.index = pd.to_datetime(load.index)

        # ── Seasonal weight recalibration ──────────────────────────
        # Correlation between weather stations and load shifts with seasons.
        # Summer weights differ from monsoon weights (rain stations matter more
        # in Jul-Sep, temperature stations dominate in Apr-Jun).
        # We compute per-month weights, falling back to global if a month
        # has too few data points.
        MIN_PTS_SEASONAL = 96 * 7  # ~1 week of 15-min data

        results = {}
        for feat in available:
            wide = df.pivot(index="Datetime", columns="location", values=feat)
            wide = wide.ffill().bfill()
            common = load.index.intersection(wide.index)
            expected_sign = FEATURE_SPEC[feat][1]

            # Global fallback weights
            if len(common) >= 50:
                global_corr = wide.loc[common].corrwith(load.loc[common])
                w_global = self._robust_weights(global_corr, expected_sign=expected_sign)
            else:
                w_global = pd.Series(1 / max(len(wide.columns), 1), index=wide.columns)
            w_global = w_global.reindex(wide.columns, fill_value=0)
            wsum = w_global.sum()
            if wsum > 0:
                w_global = w_global / wsum

            # Per-month weights (recalibrated seasonally)
            month_weights = {}
            if len(common) >= MIN_PTS_SEASONAL:
                months_in_data = pd.Series(common).dt.month.unique()
                for m in months_in_data:
                    mask = pd.Series(common).dt.month == m
                    cidx = pd.DatetimeIndex(pd.Series(common)[mask.values])
                    if len(cidx) < 50:
                        month_weights[m] = w_global
                        continue
                    mc = wide.loc[cidx].corrwith(load.loc[cidx])
                    mw = self._robust_weights(mc, expected_sign=expected_sign)
                    mw = mw.reindex(wide.columns, fill_value=0)
                    ms = mw.sum()
                    month_weights[m] = mw / ms if ms > 0 else w_global

            # Apply weights row-by-row using month lookup (vectorised per month)
            if month_weights:
                collapsed = pd.Series(np.nan, index=wide.index, dtype=float)
                for m, mw in month_weights.items():
                    mask = wide.index.month == m
                    if mask.any():
                        collapsed[mask] = wide.loc[mask].dot(mw)
                # Fill months without seasonal weights using global
                remaining = collapsed.isna()
                if remaining.any():
                    collapsed[remaining] = wide.loc[remaining].dot(w_global)
                results[feat] = collapsed
            else:
                results[feat] = wide.dot(w_global)

        state_exog = pd.DataFrame(results)
        state_exog.index.name = "Datetime"
        return state_exog
