import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV

try:
    from .india_intelligence import compute_india_adjustments
except ImportError:
    try:
        from india_intelligence import compute_india_adjustments
    except ImportError:
        compute_india_adjustments = None

try:
    from .hybrid import HybridForecastConfig, HybridForecastingEngine
except Exception:
    try:
        from hybrid import HybridForecastConfig, HybridForecastingEngine
    except Exception:
        HybridForecastConfig = None
        HybridForecastingEngine = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from .ml_baseline import compute_ml_baseline, blend_baselines
except ImportError:
    try:
        from ml_baseline import compute_ml_baseline, blend_baselines
    except ImportError:
        compute_ml_baseline = None
        blend_baselines = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    _TORCH_AVAILABLE = False


class NHiTSRegressor:
    """
    Lightweight N-HiTS-inspired residual MLP regressor (PyTorch).

    Designed to be a drop-in replacement for tree models in this pipeline:
    - fit(X, y, X_val=None, y_val=None)
    - predict(X) -> np.ndarray
    - exposes coef_ via a companion LinearRegression for explainability hooks.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_blocks: int = 4,
        dropout: float = 0.10,
        epochs: int = 120,
        batch_size: int = 512,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        patience: int = 18,
        min_epochs: int = 25,
        random_state: int = 42,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.n_blocks = int(n_blocks)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.min_epochs = int(min_epochs)
        self.random_state = int(random_state)
        self.device = device
        self.verbose = bool(verbose)

        self.n_features_in_: Optional[int] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self._x_scaler: Optional[StandardScaler] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._model: Any = None
        self._fallback: Optional[LinearRegression] = None

    def _resolve_device(self) -> str:
        if self.device:
            return str(self.device)
        return "cpu"

    def _ensure_model(self, input_dim: int) -> None:
        if not _TORCH_AVAILABLE:
            return
        if self._model is not None and self.n_features_in_ == int(input_dim):
            return

        class _Block(nn.Module):
            def __init__(self, in_dim: int, h_dim: int, layers: int, drop: float) -> None:
                super().__init__()
                seq = []
                d = int(in_dim)
                for _ in range(int(max(1, layers))):
                    seq.append(nn.Linear(d, int(h_dim)))
                    seq.append(nn.ReLU())
                    if float(drop) > 1e-9:
                        seq.append(nn.Dropout(float(drop)))
                    d = int(h_dim)
                self.mlp = nn.Sequential(*seq)
                self.backcast = nn.Linear(int(h_dim), int(in_dim))
                self.forecast = nn.Linear(int(h_dim), 1)

            def forward(self, x):
                h = self.mlp(x)
                return self.backcast(h), self.forecast(h)

        class _Net(nn.Module):
            def __init__(self, in_dim: int, h_dim: int, layers: int, blocks: int, drop: float) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([
                    _Block(in_dim=int(in_dim), h_dim=int(h_dim), layers=int(layers), drop=float(drop))
                    for _ in range(int(max(1, blocks)))
                ])

            def forward(self, x):
                residual = x
                out = 0.0
                for blk in self.blocks:
                    back, f = blk(residual)
                    residual = residual - back
                    out = out + f
                return out.squeeze(-1)

        self.n_features_in_ = int(input_dim)
        self._model = _Net(
            in_dim=int(input_dim),
            h_dim=int(self.hidden_dim),
            layers=int(max(1, self.n_layers)),
            blocks=int(max(1, self.n_blocks)),
            drop=float(max(0.0, self.dropout)),
        )
        self._model.to(self._resolve_device())

    def fit(self, X, y, X_val=None, y_val=None) -> "NHiTSRegressor":
        X_np = np.asarray(X, dtype=float)
        y_np = np.asarray(y, dtype=float).reshape(-1)
        if X_np.ndim != 2:
            raise ValueError("NHiTSRegressor.fit expects 2D X")
        if y_np.size != X_np.shape[0]:
            raise ValueError("X and y size mismatch")

        # Always compute a linear proxy for effects (used by downstream attribution).
        lin = LinearRegression()
        try:
            lin.fit(X_np, y_np)
            self.coef_ = np.asarray(lin.coef_, dtype=float).reshape(-1)
            self.intercept_ = float(lin.intercept_)
        except Exception:
            self.coef_ = np.zeros(X_np.shape[1], dtype=float)
            self.intercept_ = float(np.mean(y_np)) if y_np.size else 0.0

        if (not _TORCH_AVAILABLE) or X_np.shape[0] < 64:
            # Small-data (or no-torch) fallback for stability and speed.
            self._fallback = lin
            self._model = None
            self._x_scaler = None
            self._y_mean = float(np.mean(y_np)) if y_np.size else 0.0
            self._y_std = float(np.std(y_np)) if y_np.size else 1.0
            if not np.isfinite(self._y_std) or abs(self._y_std) <= 1e-9:
                self._y_std = 1.0
            return self

        # Scale inputs; scale target for stable NN training.
        self._x_scaler = StandardScaler()
        X_scaled = self._x_scaler.fit_transform(X_np).astype(np.float32, copy=False)
        self._y_mean = float(np.mean(y_np)) if y_np.size else 0.0
        self._y_std = float(np.std(y_np)) if y_np.size else 1.0
        if (not np.isfinite(self._y_std)) or abs(self._y_std) <= 1e-9:
            self._y_std = 1.0
        y_scaled = ((y_np - self._y_mean) / self._y_std).astype(np.float32, copy=False)

        Xv_scaled = None
        yv_scaled = None
        if X_val is not None and y_val is not None:
            Xv_np = np.asarray(X_val, dtype=float)
            yv_np = np.asarray(y_val, dtype=float).reshape(-1)
            if Xv_np.ndim == 2 and yv_np.size == Xv_np.shape[0] and Xv_np.size:
                Xv_scaled = self._x_scaler.transform(Xv_np).astype(np.float32, copy=False)
                yv_scaled = ((yv_np - self._y_mean) / self._y_std).astype(np.float32, copy=False)

        self._ensure_model(X_scaled.shape[1])
        if self._model is None:
            self._fallback = lin
            return self

        # Deterministic-ish training.
        torch.manual_seed(int(self.random_state))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.random_state))

        device = self._resolve_device()
        X_t = torch.from_numpy(X_scaled).to(device)
        y_t = torch.from_numpy(y_scaled).to(device)
        ds = TensorDataset(X_t, y_t)
        bsz = int(np.clip(int(self.batch_size), 32, max(32, X_scaled.shape[0])))
        loader = DataLoader(ds, batch_size=bsz, shuffle=True)

        loss_fn = nn.MSELoss()
        opt = torch.optim.AdamW(self._model.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))

        best_state = None
        best_val = None
        bad_epochs = 0
        max_epochs = int(np.clip(int(self.epochs), 10, 400))
        min_epochs = int(np.clip(int(self.min_epochs), 0, max_epochs))
        patience = int(np.clip(int(self.patience), 3, 80))

        for epoch in range(max_epochs):
            self._model.train()
            total = 0.0
            n_seen = 0
            for xb, yb in loader:
                opt.zero_grad(set_to_none=True)
                pred = self._model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu().item()) * int(xb.shape[0])
                n_seen += int(xb.shape[0])

            train_loss = total / max(n_seen, 1)
            val_loss = None

            if Xv_scaled is not None and yv_scaled is not None and Xv_scaled.size:
                self._model.eval()
                with torch.no_grad():
                    xv_t = torch.from_numpy(Xv_scaled).to(device)
                    yv_t = torch.from_numpy(yv_scaled).to(device)
                    pv = self._model(xv_t)
                    val_loss = float(loss_fn(pv, yv_t).detach().cpu().item())

                score = val_loss
                improved = (best_val is None) or (score < (best_val - 1e-6))
                if improved:
                    best_val = score
                    best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                if self.verbose and (epoch % 10 == 0 or epoch == max_epochs - 1):
                    logger.debug("[N-HiTS] epoch=%03d train=%.6f val=%.6f", epoch, train_loss, val_loss)

                if epoch + 1 >= min_epochs and bad_epochs >= patience:
                    break
            else:
                if self.verbose and (epoch % 10 == 0 or epoch == max_epochs - 1):
                    logger.debug("[N-HiTS] epoch=%03d train=%.6f", epoch, train_loss)

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._fallback = None
        return self

    def predict(self, X):
        X_np = np.asarray(X, dtype=float)
        if X_np.ndim != 2:
            raise ValueError("NHiTSRegressor.predict expects 2D X")
        if self._fallback is not None:
            return self._fallback.predict(X_np)
        if (not _TORCH_AVAILABLE) or self._model is None or self._x_scaler is None:
            # Last-resort safety.
            lr = LinearRegression()
            lr.fit(X_np, np.zeros(X_np.shape[0], dtype=float))
            return lr.predict(X_np)

        X_scaled = self._x_scaler.transform(X_np).astype(np.float32, copy=False)
        device = self._resolve_device()
        self._model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X_scaled).to(device)
            y_hat = self._model(x_t).detach().cpu().numpy().astype(float, copy=False).reshape(-1)
        return (y_hat * float(self._y_std)) + float(self._y_mean)

SEASON_ORDER = ["winter", "spring", "summer", "fall"]

SEASON_BY_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}

SEASONAL_WEATHER_INTERVALS = {
    "winter": {
        "temp_band": 2.5,
        "hum_band": 10.0,
        "cdd_base": 24.0, 
        "hdd_base": 18.0, 
    },
    "spring": {
        "temp_band": 3.0,
        "hum_band": 12.0,
        "cdd_base": 22.0,
        "hdd_base": 15.0,
    },
    "summer": {
        "temp_band": 3.5,
        "hum_band": 15.0,
        "cdd_base": 20.0, 
        "hdd_base": 12.0, 
    },
    "fall": {
        "temp_band": 3.0,
        "hum_band": 12.0,
        "cdd_base": 22.0,
        "hdd_base": 15.0,
    },
}

DEFAULT_CONFIG = {
    "candidate_lookback_days": 45,
    "similar_days_top_n": 5,
    "temp_band_c": 3.0,
    "humidity_band": 12.0,
    "require_rain_match": True,
    "similarity_weights": {"temp": 0.55, "humidity": 0.25, "rain": 0.2},
    "auto_similarity_from_data": True,
    "auto_require_rain_match": True,
    "similarity_calibration_days": 120,
    "similarity_weight_prior_blend": 0.35,
    "rain_match_min_days": 3,
    "rain_effect_threshold": 0.01,
    "baseline_window_candidates": [3, 5, 7, 10, 14],
    "baseline_backtest_days": 10,
    "weather_training_days": 180,
    "temp_asym_lookback_days": 180,
    "temp_asym_rolling_days": 7,
    "temp_asym_min_samples": 8,
    "hybrid_weights": {"alpha": 0.6, "beta": 0.4},
    "weather_weight_sensitivity": 0.25,
    "weather_weight_bounds": [0.2, 0.85],
    "baseline_weather_weight": 0.35,
    "trend_degree": 1,
    "trend_sign": 1.0,
    "pattern_weight": 0.12,
    "trend_weight": 0.12,
    "forecast_blend": {"hybrid": 0.20, "bias_applied": 0.80, "weather_baseline": 0.0},
    "forecast_shrinkage": {
        "offpeak_weight": 0.50,
        "peak_weight": 0.25,
        "known_blocks_bonus": 0.20,
        "peak_start_block": 48,
        "peak_end_block": 72,
    },
    "hybrid_ai": {
        "enabled": True,
        "lookback_days": 7,
        "min_history_days": 21,
        "training_days": 60,
        "blend_weight": 0.35,
        "min_blend_weight": 0.10,
        "max_blend_weight": 0.45,
        "sequence_epochs": 30,
        "random_state": 42,
    },
    "rain_coeffs": {"winter": 20.0, "spring": 18.0, "summer": 15.0, "fall": 18.0},
    "min_actual_blocks": 24,
    "max_actual_blocks": 96,
    "weather_tune": True,
    "weather_tune_iters": 10,
    "min_valid_load_mw": 100.0, # Minimum average load to consider a day valid for baseline
    "driver_sliders": {
        "temperature": 1.0,
        "humidity": 1.0,
        "precipitation": 1.0,
        "apparent_temperature": 1.0,
        "cloud_cover": 1.0,
        "sunshine_duration": 1.0,
        "direct_radiation": 1.0,
        "wind_speed_10m": 1.0,
        "weather": 1.0,
        "daytype": 1.0,
        "holiday": 1.0,
        "manual": 1.0
    },
    "selection": {"scope": "all"},
    "selection_smoothing": False,
    "manual_base": [0.0] * 96,
    "region": "punjab",
    "human_behaviour_weight": 0.35,
    "weather_divergence_threshold": 1.5,
    "calibrate_behaviour_from_data": True,
    "behaviour_calibration_days": 90,
    "behaviour_cyclic_harmonics": 4,
    "behaviour_cyclic_blend": 0.55,
    "behaviour_cyclic_smoothing": 5,
    "cdd_hdd_bases": {
        "north":   {"cdd": 24.0, "hdd": 15.0},
        "south":   {"cdd": 28.0, "hdd": 20.0},
        "west":    {"cdd": 26.0, "hdd": 18.0},
        "east":    {"cdd": 26.0, "hdd": 16.0},
        "central": {"cdd": 25.0, "hdd": 14.0},
    },
}

_IMPACT_MIN = -0.40
_IMPACT_MAX = 0.50
_BLOCK_COUNT = 96
logger = logging.getLogger(__name__)

SHORT_TERM_MODEL_FEATURES = [
    "temperature", "humidity", "precipitation",
    "CDD", "HDD", "temp_roll_3h", "temp_roll_6h", "wbgt",
    "cdh_24h", "cdh_48h", "cdh_72h",
    "apparent_temperature", "cloud_cover", "sunshine_duration",
    "direct_radiation", "wind_speed_10m",
    "solar_index", "solar_proxy_mw",
    "tb_sin", "tb_cos", "is_weekend", "season_idx",
    "hour_cos", "dow_sin", "dow_cos", "block_sin", "block_cos",
    "is_peak_hour", "is_night", "is_business_hour",
    "temp_squared", "temp_cubed", "temperature_sin",
    "lag_1", "lag_7", "lag_block_1", "lag_block_4",
    "rolling_4", "rolling_12",
    "load_lag_1d", "load_lag_2d", "load_lag_3d",
    "load_lag_7d", "load_rolling_7d", "load_trend_3d",
    "load_x_cdd", "load_x_humidity", "cdd_squared",
    "wbgt_x_load", "temp_momentum_3d",
]

# ── Indian State → Climate Region Mapping ──────────────────────────
INDIAN_STATE_REGIONS = {
    # North India (cold winters, hot summers, AC + heating heavy)
    "punjab": "north", "haryana": "north", "delhi": "north",
    "uttar pradesh": "north", "uttarakhand": "north",
    "himachal pradesh": "north", "jammu & kashmir": "north",
    # West India (hot & dry, heavy AC summers, moderate winters)
    "rajasthan": "west", "gujarat": "west", "maharashtra": "west", "goa": "west",
    # South India (tropical, year-round AC, monsoon patterns)
    "tamil nadu": "south", "kerala": "south", "karnataka": "south",
    "andhra pradesh": "south", "telangana": "south",
    # East India (humid, moderate temps, monsoon-sensitive)
    "west bengal": "east", "odisha": "east", "bihar": "east",
    "jharkhand": "east", "assam": "east",
    # Central India (agricultural load, extreme summers)
    "madhya pradesh": "central", "chhattisgarh": "central",
}

# ── Indian State → holidays library subdivision code ─────────────
INDIAN_STATE_HOLIDAY_SUBDIV = {
    "punjab": "PB", "haryana": "HR", "delhi": "DL",
    "uttar pradesh": "UP", "uttarakhand": "UK",
    "himachal pradesh": "HP", "jammu & kashmir": "JK",
    "rajasthan": "RJ", "gujarat": "GJ", "maharashtra": "MH", "goa": "GA",
    "tamil nadu": "TN", "kerala": "KL", "karnataka": "KA",
    "andhra pradesh": "AP", "telangana": "TS",
    "west bengal": "WB", "odisha": "OD", "bihar": "BR",
    "jharkhand": "JH", "assam": "AS",
    "madhya pradesh": "MP", "chhattisgarh": "CG",
}

# ── Human Behaviour Profiles ───────────────────────────────────────
# Keys: (climate_region, season, day_type)
# Values: list of (label, block_start, block_end, mw_boost) tuples
# Block ranges: 1-96 (15-min slots), e.g. block 1=00:00, 25=06:00,
# 37=09:00, 49=12:00, 61=15:00, 73=18:00, 85=21:00
HUMAN_BEHAVIOUR_PROFILES = {
    # ── NORTH ──
    ("north", "summer", "Weekday"): [
        ("AC + Industrial peak",       37, 72,  200),
        ("Evening cooking & lighting",  73, 84,  150),
        ("Late-night cool-down",        1, 20,  -80),
    ],
    ("north", "summer", "Weekend"): [
        ("Residential AC (late start)", 41, 76,  250),
        ("Evening leisure",             77, 88,  120),
        ("Night rest",                   1, 24,  -60),
    ],
    ("north", "winter", "Weekday"): [
        ("Morning heating demand",      21, 36,  200),
        ("Evening heating + lighting",  69, 84,  280),
        ("Midday mild",                 41, 60,  -40),
    ],
    ("north", "winter", "Weekend"): [
        ("Late morning heating",        25, 40,  220),
        ("Evening heating + cooking",   73, 88,  250),
        ("Night base",                   1, 20,  -30),
    ],
    ("north", "spring", "Weekday"): [
        ("Morning ramp",               29, 44,   80),
        ("Afternoon moderate",          49, 64,   60),
        ("Evening cooking",             73, 84,   90),
    ],
    ("north", "spring", "Weekend"): [
        ("Late morning activity",       33, 48,   70),
        ("Evening leisure",             73, 84,   80),
    ],
    ("north", "fall", "Weekday"): [
        ("Morning activity",            25, 40,  100),
        ("Evening early heating",       69, 84,  120),
    ],
    ("north", "fall", "Weekend"): [
        ("Morning leisure",             29, 44,   80),
        ("Evening gathering",           73, 88,  100),
    ],
    # ── WEST ──
    ("west", "summer", "Weekday"): [
        ("Extreme AC demand",           33, 76,  280),
        ("Evening industry wind-down",  77, 84,  100),
        ("Night rest",                   1, 20, -100),
    ],
    ("west", "summer", "Weekend"): [
        ("Residential AC all day",      37, 80,  300),
        ("Night cool-off",               1, 24,  -80),
    ],
    ("west", "winter", "Weekday"): [
        ("Mild morning ramp",           25, 40,   80),
        ("Afternoon steady",            45, 64,   50),
        ("Evening social + cooking",    73, 84,  100),
    ],
    ("west", "winter", "Weekend"): [
        ("Morning leisure",             29, 44,   70),
        ("Evening cooking",             73, 84,   90),
    ],
    ("west", "spring", "Weekday"): [
        ("Pre-summer AC start",         37, 68,  120),
        ("Evening moderate",            73, 84,   80),
    ],
    ("west", "spring", "Weekend"): [
        ("Daytime AC",                  41, 72,  100),
        ("Evening activity",            73, 84,   70),
    ],
    ("west", "fall", "Weekday"): [
        ("Post-monsoon moderate",       33, 64,   80),
        ("Evening demand",              73, 84,   90),
    ],
    ("west", "fall", "Weekend"): [
        ("Daytime moderate",            37, 68,   70),
        ("Evening activity",            73, 84,   80),
    ],
    # ── SOUTH ──
    ("south", "summer", "Weekday"): [
        ("Year-round AC",              33, 72,  180),
        ("Evening tropical load",       73, 84,  130),
        ("Night AC base",                1, 20,   40),
    ],
    ("south", "summer", "Weekend"): [
        ("Residential cooling",         37, 76,  200),
        ("Evening social",              77, 88,  110),
    ],
    ("south", "winter", "Weekday"): [
        ("Mild winter – no heating",    33, 68,   60),
        ("Evening cooking + lighting",  73, 84,  100),
    ],
    ("south", "winter", "Weekend"): [
        ("Moderate daytime",            37, 72,   50),
        ("Evening activity",            73, 84,   80),
    ],
    ("south", "spring", "Weekday"): [
        ("Pre-monsoon heat",            33, 72,  140),
        ("Evening demand",              73, 84,  100),
    ],
    ("south", "spring", "Weekend"): [
        ("Daytime cooling",             37, 76,  120),
        ("Evening leisure",             77, 84,   80),
    ],
    ("south", "fall", "Weekday"): [
        ("Monsoon retreat moderate",    33, 68,   80),
        ("Evening demand",              73, 84,   90),
    ],
    ("south", "fall", "Weekend"): [
        ("Moderate daytime",            37, 72,   70),
        ("Evening activity",            73, 84,   75),
    ],
    # ── EAST ──
    ("east", "summer", "Weekday"): [
        ("Humid heat AC",              33, 72,  160),
        ("Evening humidity surge",      73, 84,  120),
        ("Night base",                   1, 20,  -50),
    ],
    ("east", "summer", "Weekend"): [
        ("Residential cooling",         37, 76,  180),
        ("Evening social",              77, 88,  100),
    ],
    ("east", "winter", "Weekday"): [
        ("Moderate morning",            25, 40,   70),
        ("Evening cooking + warmth",    73, 84,  100),
    ],
    ("east", "winter", "Weekend"): [
        ("Morning leisure",             29, 44,   60),
        ("Evening gathering",           73, 84,   80),
    ],
    ("east", "spring", "Weekday"): [
        ("Pre-monsoon humid",           33, 68,  100),
        ("Evening demand",              73, 84,   80),
    ],
    ("east", "spring", "Weekend"): [
        ("Daytime moderate",            37, 72,   80),
        ("Evening activity",            73, 84,   70),
    ],
    ("east", "fall", "Weekday"): [
        ("Post-monsoon humid",          33, 68,   90),
        ("Evening Diwali-season",       73, 84,  100),
    ],
    ("east", "fall", "Weekend"): [
        ("Moderate daytime",            37, 72,   70),
        ("Evening social",              73, 84,   80),
    ],
    # ── CENTRAL ──
    ("central", "summer", "Weekday"): [
        ("Extreme heat + agri pumps",   29, 72,  240),
        ("Evening cooking & fans",      73, 84,  140),
        ("Night relief",                 1, 20,  -70),
    ],
    ("central", "summer", "Weekend"): [
        ("All-day heat demand",         33, 80,  260),
        ("Night cool-down",              1, 24,  -60),
    ],
    ("central", "winter", "Weekday"): [
        ("Morning chill",               21, 36,  100),
        ("Evening heating + cooking",   73, 84,  130),
    ],
    ("central", "winter", "Weekend"): [
        ("Morning warmth",              25, 40,   90),
        ("Evening gathering",           73, 84,  110),
    ],
    ("central", "spring", "Weekday"): [
        ("Pre-summer ramp",             33, 68,  100),
        ("Evening demand",              73, 84,   80),
    ],
    ("central", "spring", "Weekend"): [
        ("Daytime moderate",            37, 72,   80),
        ("Evening activity",            73, 84,   70),
    ],
    ("central", "fall", "Weekday"): [
        ("Post-monsoon agricultural",   29, 68,  110),
        ("Evening demand",              73, 84,   90),
    ],
    ("central", "fall", "Weekend"): [
        ("Moderate daytime",            37, 72,   80),
        ("Evening social",              73, 84,   80),
    ],
}

# ── Extended day-type profiles (Holiday, BridgeDay, Friday, Monday, etc.) ──
# Generated per (region, season) — holidays suppress load, bridge days lighter.
for _region in ("north", "west", "south", "east", "central"):
    for _season in ("summer", "winter", "spring", "fall"):
        # Get weekday profile as reference for scaling
        _wk_key = (_region, _season, "Weekday")
        _wk = HUMAN_BEHAVIOUR_PROFILES.get(_wk_key, [])
        _peak_mw = max((abs(e[3]) for e in _wk), default=100)

        HUMAN_BEHAVIOUR_PROFILES[(_region, _season, "Holiday")] = [
            ("Holiday suppression",         33, 72, int(-_peak_mw * 0.5)),
            ("Evening leisure bump",        73, 88, int(_peak_mw * 0.3)),
            ("Night rest",                   1, 20, int(-_peak_mw * 0.2)),
        ]
        HUMAN_BEHAVIOUR_PROFILES[(_region, _season, "Holiday-Fri")] = [
            ("Friday holiday + weekend start", 33, 72, int(-_peak_mw * 0.5)),
            ("Extended evening",            73, 92, int(_peak_mw * 0.35)),
        ]
        HUMAN_BEHAVIOUR_PROFILES[(_region, _season, "Holiday-Mon")] = [
            ("Slow Monday ramp",            20, 44, int(-_peak_mw * 0.3)),
            ("Holiday suppression",         45, 72, int(-_peak_mw * 0.4)),
            ("Evening recovery",            73, 84, int(_peak_mw * 0.2)),
        ]
        HUMAN_BEHAVIOUR_PROFILES[(_region, _season, "Holiday-MidWeek")] = [
            ("Mid-week holiday dip",        29, 72, int(-_peak_mw * 0.45)),
            ("Evening bump",                73, 84, int(_peak_mw * 0.25)),
        ]
        HUMAN_BEHAVIOUR_PROFILES[(_region, _season, "BridgeDay")] = [
            ("Bridge day light suppression", 33, 72, int(-_peak_mw * 0.25)),
            ("Evening normal",              73, 84, int(_peak_mw * 0.15)),
        ]
        HUMAN_BEHAVIOUR_PROFILES[(_region, _season, "LongWeekend")] = [
            ("Deep weekend rest",           33, 76, int(-_peak_mw * 0.35)),
            ("Evening leisure",             77, 88, int(_peak_mw * 0.2)),
        ]
        HUMAN_BEHAVIOUR_PROFILES[(_region, _season, "Friday")] = [
            ("Pre-weekend wind-down",       65, 76, int(-_peak_mw * 0.1)),
            ("Evening ramp-up",             77, 92, int(_peak_mw * 0.2)),
        ]
        HUMAN_BEHAVIOUR_PROFILES[(_region, _season, "Monday")] = [
            ("Monday morning recovery",     20, 36, int(-_peak_mw * 0.15)),
            ("Full activity ramp",          37, 72, int(_peak_mw * 0.1)),
        ]
del _region, _season, _wk_key, _wk, _peak_mw

# ── State-Wise Human Behaviour Profiles (v3.0) ────────────────────
# Override regional profiles when state-level granularity matters.
# Keys: (state_name, season, day_type) — only states that deviate
# significantly from their climate region average need entries here.
# The _human_behaviour_adjustment function checks these FIRST.
STATE_BEHAVIOUR_PROFILES: Dict[tuple, list] = {
    # ── DELHI (DL): Dense AC, metro rail, urban heat island ──
    ("delhi", "summer", "Weekday"): [
        ("Metro + AC peak",           37, 72,  400),
        ("Evening cooking + AC",      73, 84,  250),
        ("Night AC base (urban heat)",  1, 20,  -50),
    ],
    ("delhi", "summer", "Weekend"): [
        ("Residential AC all-day",    37, 76,  450),
        ("Evening leisure",           77, 88,  200),
        ("Night urban heat",           1, 24,  -30),
    ],
    ("delhi", "winter", "Weekday"): [
        ("Morning heating demand",    20, 36,  180),
        ("Evening heating + lighting", 69, 84,  280),
        ("Midday mild",               41, 60,  -40),
    ],
    # ── UTTAR PRADESH (UP): Agricultural pump 6-9AM, evening lighting ──
    ("uttar pradesh", "summer", "Weekday"): [
        ("Agri pump dawn load",       24, 36,  300),
        ("AC + industrial peak",      40, 68,  350),
        ("Evening lighting surge",    73, 84,  200),
        ("Night rest",                 1, 20,  -100),
    ],
    ("uttar pradesh", "summer", "Weekend"): [
        ("Agri pump (continues wknd)", 24, 36, 280),
        ("Residential AC",            40, 72,  300),
        ("Evening social",            73, 88,  180),
    ],
    # ── RAJASTHAN (RJ): Desert cooling, solar ramp compensation ──
    ("rajasthan", "summer", "Weekday"): [
        ("Extreme desert cooling",    38, 70,  400),
        ("Solar sunset compensation", 65, 76,  150),
        ("Evening demand",            73, 84,  200),
        ("Night cool desert",          1, 20,  -120),
    ],
    ("rajasthan", "summer", "Weekend"): [
        ("All-day desert AC",         36, 76,  420),
        ("Evening social",            77, 88,  180),
    ],
    # ── PUNJAB (PB): Paddy pump load Jun-Sep ──
    ("punjab", "summer", "Weekday"): [
        ("Paddy pump load",           24, 40,  250),
        ("AC + industrial",           42, 68,  280),
        ("Evening cooking",           73, 84,  160),
    ],
    ("punjab", "winter", "Weekday"): [
        ("Morning heating (Rabi)",    20, 36,  150),
        ("Evening heating + cooking", 69, 84,  250),
    ],
    # ── MAHARASHTRA (MH): Mumbai industrial + Pune IT ──
    ("maharashtra", "summer", "Weekday"): [
        ("Mumbai industrial base",    25, 36,  300),
        ("AC + IT corridor peak",     38, 70,  500),
        ("Evening residential",       73, 84,  250),
        ("Night industrial base",      1, 20,  -80),
    ],
    ("maharashtra", "summer", "Weekend"): [
        ("Residential AC dominant",   37, 76,  450),
        ("Evening social",            77, 88,  200),
    ],
    ("maharashtra", "fall", "Weekday"): [
        ("Post-monsoon humid",        33, 68,  250),
        ("Rain-day suppression",      73, 84, -150),
        ("Evening demand",            77, 88,  180),
    ],
    # ── GUJARAT (GJ): Textile mills, Jamnagar refinery ──
    ("gujarat", "summer", "Weekday"): [
        ("Textile + refinery base",   25, 36,  200),
        ("AC + industrial peak",      38, 68,  420),
        ("Evening demand",            73, 84,  180),
    ],
    # ── TAMIL NADU (TN): Chennai AC, Coimbatore textile ──
    ("tamil nadu", "summer", "Weekday"): [
        ("Textile continuous",        25, 36,  200),
        ("Chennai AC peak",           40, 72,  400),
        ("Evening tropical load",     73, 88,  250),
        ("Night AC base (tropical)",   1, 20,   40),
    ],
    ("tamil nadu", "summer", "Weekend"): [
        ("Residential cooling",       37, 76,  350),
        ("Evening social",            77, 88,  220),
    ],
    # ── KARNATAKA (KA): Bengaluru IT sustained 9AM-9PM ──
    ("karnataka", "summer", "Weekday"): [
        ("IT corridor sustained",     36, 84,  350),
        ("Morning ramp (IT start)",   33, 40,  200),
        ("Evening continued IT",      73, 84,  200),
    ],
    ("karnataka", "summer", "Weekend"): [
        ("Residential + IT weekend",  40, 76,  280),
        ("Evening Bengaluru social",  77, 84,  180),
    ],
    # ── TELANGANA (TS): Hyderabad pharma/IT 24h, high AC ──
    ("telangana", "summer", "Weekday"): [
        ("Pharma/IT 24h base",        1, 96,   80),
        ("AC peak overlay",           40, 68,  350),
        ("Evening residential",       73, 86,  220),
    ],
    ("telangana", "summer", "Weekend"): [
        ("Pharma continuous",          1, 96,   60),
        ("Residential AC",            40, 76,  300),
        ("Evening social",            77, 88,  180),
    ],
    # ── WEST BENGAL (WB): Kolkata urban heat island, jute ──
    ("west bengal", "summer", "Weekday"): [
        ("Kolkata heat island AC",    40, 72,  300),
        ("Jute industry dawn",        24, 36,  150),
        ("Humid evening surge",       73, 84,  200),
    ],
    ("west bengal", "summer", "Weekend"): [
        ("Residential humid AC",      40, 76,  280),
        ("Evening social",            77, 88,  180),
    ],
    # ── ODISHA (OR): Mining/steel, cyclone events ──
    ("odisha", "summer", "Weekday"): [
        ("Steel/mining base",         25, 72,  250),
        ("Agri pump morning",         24, 36,  180),
        ("Evening demand",            73, 84,  150),
    ],
    ("odisha", "fall", "Weekday"): [
        ("Cyclone suppression risk",  33, 72, -200),
        ("Post-event recovery",       73, 84,  100),
    ],
    # ── BIHAR (BR): Agricultural, low industrial, evening lighting ──
    ("bihar", "summer", "Weekday"): [
        ("Agri pump dawn",            24, 36,  200),
        ("Daytime moderate",          40, 66,  200),
        ("Evening lighting peak",     73, 84,  250),
        ("Night low base",             1, 20,  -100),
    ],
    # ── HIMACHAL PRADESH (HP): Hill station, heating dominant ──
    ("himachal pradesh", "winter", "Weekday"): [
        ("Heavy heating morning",     16, 36,  200),
        ("Heating sustained day",     37, 68,  100),
        ("Evening peak heating",      69, 84,  250),
    ],
    ("himachal pradesh", "summer", "Weekday"): [
        ("Tourism mild cooling",      44, 64,   80),
        ("Evening tourism activity",  73, 84,   60),
    ],
    # ── JAMMU & KASHMIR (JK): Electric heating extreme winter ──
    ("jammu & kashmir", "winter", "Weekday"): [
        ("Extreme electric heating",  14, 36,  220),
        ("Sustained daytime heating", 37, 68,  120),
        ("Evening peak heating",      69, 84,  280),
    ],
    # ── ASSAM (AS): Tea processing, oil refinery ──
    ("assam", "summer", "Weekday"): [
        ("Tea processing ramp",       28, 44,  100),
        ("Moderate humid AC",         46, 64,  120),
        ("Evening demand",            73, 84,   80),
    ],
    # ── CHHATTISGARH (CG): Steel/aluminium smelter, Korba ──
    ("chhattisgarh", "summer", "Weekday"): [
        ("Smelter continuous base",    1, 96,  100),
        ("Peak industrial + AC",      38, 68,  250),
        ("Evening demand",            73, 84,  150),
    ],
    ("chhattisgarh", "fall", "Weekday"): [
        ("Monsoon onset sharp drop",  33, 72, -220),
        ("Post-rain recovery",        73, 84,  100),
    ],
    # ── KERALA (KL): High humidity AC, fishing dawn ──
    ("kerala", "summer", "Weekday"): [
        ("Fishing industry dawn",     20, 32,   80),
        ("Humid AC demand",           44, 68,  160),
        ("Evening tropical",          73, 88,  180),
    ],
    # ── ANDHRA PRADESH (AP): Vizag industrial, aquaculture ──
    ("andhra pradesh", "summer", "Weekday"): [
        ("Aquaculture pump dawn",     22, 36,  200),
        ("Industrial + AC peak",      40, 70,  320),
        ("Evening demand",            73, 86,  150),
    ],
}


# ── Calibrated behaviour cache ───────────────────────────────────
_CALIBRATED_BEHAVIOUR: Dict[tuple, np.ndarray] = {}
_CALIBRATED_BEHAVIOUR_META: Dict[tuple, Dict[str, Any]] = {}


def _cyclic_block_distance(block: float, center: float, block_count: int = _BLOCK_COUNT) -> float:
    raw = abs(float(block) - float(center))
    return float(min(raw, float(block_count) - raw))


def _circular_smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    size = int(max(1, window))
    if size <= 1 or arr.size == 1:
        return arr.copy()
    if size % 2 == 0:
        size += 1
    radius = size // 2
    smoothed = np.zeros_like(arr, dtype=float)
    for shift in range(-radius, radius + 1):
        smoothed += np.roll(arr, shift)
    return smoothed / float((2 * radius) + 1)


def _build_cyclic_time_basis(blocks: np.ndarray, harmonics: int = 4) -> np.ndarray:
    blk = np.asarray(blocks, dtype=float).reshape(-1)
    angle = (2.0 * np.pi * (blk - 1.0)) / float(_BLOCK_COUNT)
    cols = [np.ones_like(angle)]
    for harmonic in range(1, int(max(1, harmonics)) + 1):
        cols.append(np.sin(harmonic * angle))
        cols.append(np.cos(harmonic * angle))
    return np.column_stack(cols)


def _fit_cyclic_behaviour_curve(
    blocks: np.ndarray,
    residuals: np.ndarray,
    weights: np.ndarray,
    harmonics: int = 4,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    blk = np.asarray(blocks, dtype=float).reshape(-1)
    y = np.asarray(residuals, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    mask = np.isfinite(blk) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if int(np.sum(mask)) < max(12, (2 * int(max(1, harmonics))) + 1):
        return np.zeros(_BLOCK_COUNT, dtype=float), {
            "method": "cyclic_fourier",
            "harmonics": int(max(1, harmonics)),
            "samples": int(np.sum(mask)),
            "fitted": False,
        }

    X = _build_cyclic_time_basis(blk[mask], harmonics=harmonics)
    sqrt_w = np.sqrt(np.clip(w[mask], 1e-9, None))
    Xw = X * sqrt_w[:, None]
    yw = y[mask] * sqrt_w
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    pred = _build_cyclic_time_basis(np.arange(1, _BLOCK_COUNT + 1), harmonics=harmonics) @ beta
    return np.asarray(pred, dtype=float), {
        "method": "cyclic_fourier",
        "harmonics": int(max(1, harmonics)),
        "samples": int(np.sum(mask)),
        "fitted": True,
    }


def _calibrate_behaviour_profiles(df: pd.DataFrame, region: str, cfg: dict) -> Dict[tuple, np.ndarray]:
    """Learn cyclic 96-block MW profiles from historical residuals per day-type."""
    cal_days = int(cfg.get("behaviour_calibration_days", 90))
    harmonics = int(max(1, cfg.get("behaviour_cyclic_harmonics", 4) or 4))
    cyclic_blend = float(np.clip(float(cfg.get("behaviour_cyclic_blend", 0.55) or 0.55), 0.0, 1.0))
    smooth_window = int(max(1, cfg.get("behaviour_cyclic_smoothing", 5) or 5))
    climate_region = INDIAN_STATE_REGIONS.get(region.lower().strip(), region.lower().strip())
    result: Dict[tuple, np.ndarray] = {}
    global _CALIBRATED_BEHAVIOUR_META
    _CALIBRATED_BEHAVIOUR_META = {}

    if df.empty or "total_drawal" not in df.columns or "date" not in df.columns:
        return result

    work = df.copy()
    work["date"] = work["date"].astype(str)
    dates_sorted = sorted(work["date"].unique())
    if cal_days > 0 and len(dates_sorted) > cal_days:
        keep = set(dates_sorted[-cal_days:])
        work = work[work["date"].isin(keep)]

    if work.empty or "time_block" not in work.columns:
        return result

    # Compute 7-day rolling baseline per block
    work = work.sort_values(["time_block", "date"])
    work["rolling_baseline"] = work.groupby("time_block")["total_drawal"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=3).mean()
    )
    work["residual"] = work["total_drawal"] - work["rolling_baseline"]
    work = work.dropna(subset=["residual"])

    # Recency weighting: 21-day half-life exponential decay (v3.0)
    max_date = pd.to_datetime(work["date"]).max()
    work["_age_days"] = (max_date - pd.to_datetime(work["date"])).dt.days
    work["_recency_weight"] = np.exp(-0.693 * work["_age_days"] / 21.0)

    if work.empty:
        return result

    # Compute holiday flags for grouping
    hf = _compute_holiday_flags(work, region=region)
    work["season"] = work["date"].apply(_season)

    # Build extended day type per row
    flag_cols = ["is_holiday", "bridge_day", "long_weekend", "holiday_before_weekend",
                 "holiday_after_weekend", "mid_week_holiday"]
    for col in flag_cols:
        if col in hf.columns:
            work[col] = hf[col].values

    def _row_day_type(row):
        flags = {c: int(row.get(c, 0)) for c in flag_cols}
        return _day_type_extended(row["date"], flags)

    work["ext_day_type"] = work.apply(_row_day_type, axis=1)

    # Group by (season, ext_day_type) and learn a cyclic profile from data.
    for (ssn, dt), grp in work.groupby(["season", "ext_day_type"]):
        if len(grp) < 3:
            continue

        def _weighted_mean(sub):
            w = sub["_recency_weight"].values
            r = sub["residual"].values
            return float(np.average(r, weights=w)) if len(w) > 0 and w.sum() > 0 else float(np.mean(r))

        profile = (
            grp.groupby("time_block")
            .apply(_weighted_mean, include_groups=False)
            .reindex(range(1, _BLOCK_COUNT + 1))
        )
        if profile.isna().all():
            continue

        empirical = profile.interpolate(limit_direction="both").ffill().bfill().to_numpy(dtype=float)
        cyclic_curve, cyclic_diag = _fit_cyclic_behaviour_curve(
            blocks=grp["time_block"].to_numpy(dtype=float),
            residuals=grp["residual"].to_numpy(dtype=float),
            weights=grp["_recency_weight"].to_numpy(dtype=float),
            harmonics=harmonics,
        )
        if not bool(cyclic_diag.get("fitted")):
            cyclic_curve = empirical.copy()

        day_count = int(grp["date"].astype(str).nunique()) if "date" in grp.columns else 0
        data_blend = float(np.clip(cyclic_blend + (0.20 if day_count < 8 else 0.0) - (0.10 if day_count >= 20 else 0.0), 0.25, 0.85))
        arr = (data_blend * cyclic_curve) + ((1.0 - data_blend) * empirical)
        arr = _circular_smooth(arr, window=smooth_window)

        state_peak = float(grp["total_drawal"].quantile(0.95)) if "total_drawal" in grp.columns else 5000.0
        clip_bound = max(0.08 * state_peak, 150.0)
        arr = np.clip(arr, -clip_bound, clip_bound)
        key = (climate_region, ssn, dt)
        result[key] = arr
        _CALIBRATED_BEHAVIOUR_META[key] = {
            "source": "calibrated_cyclic_data",
            "harmonics": harmonics,
            "day_count": day_count,
            "blend_weight_cyclic": round(float(data_blend), 3),
            "smoothing_window": smooth_window,
            "clip_bound_mw": round(float(clip_bound), 2),
            "fit": cyclic_diag,
        }

    return result


def _human_behaviour_adjustment(
    season: str,
    region: str,
    day_type: str,
    weight: float = 1.0,
) -> Tuple[np.ndarray, str]:
    """
    Compute a 96-block MW adjustment vector based on human behaviour
    patterns for the given season × region × day_type.

    Returns (adjustment_vector, profile_label).
    """
    state_name = region.lower().strip()
    climate_region = INDIAN_STATE_REGIONS.get(state_name, state_name)

    # Priority 1: Calibrated profiles from historical data
    cal_key = (climate_region, season, day_type)
    if _CALIBRATED_BEHAVIOUR and cal_key in _CALIBRATED_BEHAVIOUR:
        meta = _CALIBRATED_BEHAVIOUR_META.get(cal_key, {})
        harmonics = meta.get("harmonics")
        label = f"CalibratedCyclic({climate_region}/{season}/{day_type}"
        if harmonics is not None:
            label += f"/h{int(harmonics)}"
        label += ")"
        return _CALIBRATED_BEHAVIOUR[cal_key] * weight, label

    # Priority 2: State-specific hardcoded profiles (v3.0)
    state_key = (state_name, season, day_type)
    entries = STATE_BEHAVIOUR_PROFILES.get(state_key)
    if not entries:
        # Try state fallback to Weekday/Weekend
        for dt in ("Weekday", "Weekend"):
            state_fb = (state_name, season, dt)
            if state_fb in STATE_BEHAVIOUR_PROFILES:
                entries = STATE_BEHAVIOUR_PROFILES[state_fb]
                break

    # Priority 3: Climate region profiles (original)
    if not entries:
        key = (climate_region, season, day_type)
        entries = HUMAN_BEHAVIOUR_PROFILES.get(key)
    if not entries:
        for dt in ("Weekday", "Weekend"):
            fallback_key = (climate_region, season, dt)
            if fallback_key in HUMAN_BEHAVIOUR_PROFILES:
                entries = HUMAN_BEHAVIOUR_PROFILES[fallback_key]
                break
    if not entries:
        return np.zeros(96, dtype=float), "No profile"

    adjustment = np.zeros(96, dtype=float)
    labels = []

    for label, b_start, b_end, mw_boost in entries:
        labels.append(label)
        # Build a Gaussian-smoothed window to avoid step edges
        center = (b_start + b_end) / 2.0
        width = (b_end - b_start) / 2.0
        sigma = max(width / 2.5, 2.0)  # smooth falloff
        for b in range(96):
            block = b + 1  # 1-indexed
            dist = _cyclic_block_distance(block, center, block_count=_BLOCK_COUNT)
            gauss = np.exp(-0.5 * (dist / sigma) ** 2)
            adjustment[b] += mw_boost * gauss

    adjustment *= weight
    adjustment = _circular_smooth(adjustment, window=5)
    profile_label = " + ".join(labels)
    return adjustment, profile_label

def _safe_corr(a, b) -> float:
    """Spearman rank correlation (robust to outliers and non-linear relationships)."""
    from scipy.stats import spearmanr
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2 or b.size < 2:
        return 0.0
    if np.std(a) <= 1e-6 or np.std(b) <= 1e-6:
        return 0.0
    rho, _ = spearmanr(a, b)
    return 0.0 if np.isnan(rho) else float(rho)


def _normalize_similarity_weights(weights: Dict[str, float]) -> Dict[str, float]:
    vec = np.array([
        max(0.0, float(weights.get("temp", 0.0))),
        max(0.0, float(weights.get("humidity", 0.0))),
        max(0.0, float(weights.get("rain", 0.0))),
    ], dtype=float)
    if float(np.sum(vec)) <= 1e-9:
        vec = np.array([0.55, 0.25, 0.20], dtype=float)
    vec = vec / max(float(np.sum(vec)), 1e-9)
    return {
        "temp": float(vec[0]),
        "humidity": float(vec[1]),
        "rain": float(vec[2]),
    }


def _compute_similarity_profile_from_data(df: pd.DataFrame, target_date: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    prior = _normalize_similarity_weights(cfg.get("similarity_weights", {}))
    out = {
        "weights": prior,
        "require_rain_match": bool(cfg.get("require_rain_match", True)),
        "source": "default",
        "diagnostics": {},
    }

    required_cols = {"date", "temperature", "humidity", "precipitation", "total_drawal"}
    if df.empty or not required_cols.issubset(set(df.columns)):
        out["diagnostics"] = {"reason": "insufficient_columns_or_empty"}
        return out

    hist = df.copy()
    hist["date"] = hist["date"].astype(str)
    if target_date:
        hist = hist[hist["date"] < str(target_date)]
    if hist.empty:
        out["diagnostics"] = {"reason": "no_history_before_target"}
        return out

    calibration_days = int(cfg.get("similarity_calibration_days", cfg.get("weather_training_days", 120)) or 120)
    if calibration_days > 0:
        hist_dates = sorted(hist["date"].dropna().unique().tolist())
        keep_dates = set(hist_dates[-calibration_days:])
        hist = hist[hist["date"].isin(keep_dates)]
    if hist.empty:
        out["diagnostics"] = {"reason": "no_history_after_calibration_window"}
        return out

    for col in ("temperature", "humidity", "precipitation", "total_drawal"):
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
    hist = hist.dropna(subset=["total_drawal"])
    if hist.empty:
        out["diagnostics"] = {"reason": "no_numeric_history"}
        return out

    daily = (
        hist.groupby("date")
        .agg(
            temp_mean=("temperature", "mean"),
            hum_mean=("humidity", "mean"),
            rain_any=("precipitation", lambda x: float((x.fillna(0) > 0).any())),
            load_sum=("total_drawal", "sum"),
        )
        .reset_index()
    )
    daily = daily.dropna(subset=["load_sum"])
    if len(daily) < 10:
        out["diagnostics"] = {"reason": "too_few_days", "days": int(len(daily))}
        return out

    temp_signal = abs(_safe_corr(daily["temp_mean"].to_numpy(dtype=float), daily["load_sum"].to_numpy(dtype=float)))
    hum_signal = abs(_safe_corr(daily["hum_mean"].to_numpy(dtype=float), daily["load_sum"].to_numpy(dtype=float)))
    rain_signal = abs(_safe_corr(daily["rain_any"].to_numpy(dtype=float), daily["load_sum"].to_numpy(dtype=float)))
    data_vec = np.array([temp_signal, hum_signal, rain_signal], dtype=float)

    prior_blend = float(np.clip(float(cfg.get("similarity_weight_prior_blend", 0.35)), 0.0, 1.0))
    prior_vec = np.array([prior["temp"], prior["humidity"], prior["rain"]], dtype=float)
    if float(np.sum(data_vec)) > 1e-9:
        data_vec = data_vec / float(np.sum(data_vec))
        final_vec = ((1.0 - prior_blend) * data_vec) + (prior_blend * prior_vec)
        final_vec = np.clip(final_vec, 1e-6, None)
        final_vec = final_vec / float(np.sum(final_vec))
        out["weights"] = {
            "temp": float(final_vec[0]),
            "humidity": float(final_vec[1]),
            "rain": float(final_vec[2]),
        }
        out["source"] = "data_calibrated"
    else:
        out["weights"] = prior
        out["source"] = "default_no_signal"

    rain_any = (daily["rain_any"].to_numpy(dtype=float) > 0.5)
    rainy_days = int(np.sum(rain_any))
    dry_days = int(len(daily) - rainy_days)
    rain_rate = float(np.mean(rain_any)) if len(daily) else 0.0
    mean_load = float(np.mean(daily["load_sum"])) if len(daily) else 0.0
    mean_rain_load = float(np.mean(daily.loc[rain_any, "load_sum"])) if rainy_days > 0 else np.nan
    mean_dry_load = float(np.mean(daily.loc[~rain_any, "load_sum"])) if dry_days > 0 else np.nan
    if np.isfinite(mean_rain_load) and np.isfinite(mean_dry_load) and mean_load > 1e-6:
        rain_effect = float(abs(mean_rain_load - mean_dry_load) / mean_load)
    else:
        rain_effect = 0.0

    min_rain_days = int(max(1, int(cfg.get("rain_match_min_days", 3))))
    rain_effect_threshold = float(max(0.0, float(cfg.get("rain_effect_threshold", 0.01))))
    auto_rain = (
        rainy_days >= min_rain_days
        and dry_days >= min_rain_days
        and rain_effect >= rain_effect_threshold
        and (0.03 <= rain_rate <= 0.97)
    )

    if bool(cfg.get("auto_require_rain_match", True)):
        out["require_rain_match"] = bool(auto_rain)
    else:
        out["require_rain_match"] = bool(cfg.get("require_rain_match", True))

    out["diagnostics"] = {
        "days_used": int(len(daily)),
        "rainy_days": rainy_days,
        "dry_days": dry_days,
        "rain_rate": float(rain_rate),
        "rain_effect": float(rain_effect),
        "rain_effect_threshold": float(rain_effect_threshold),
        "signals": {
            "temp": float(temp_signal),
            "humidity": float(hum_signal),
            "rain": float(rain_signal),
        },
        "prior_blend": float(prior_blend),
    }
    return out


def _season(date_str: str) -> str:
    try:
        month = int(date_str.split("-")[1])
    except Exception:
        return "unknown"
    return SEASON_BY_MONTH.get(month, "unknown")


def _day_type(date_str: str) -> str:
    try:
        dt = pd.to_datetime(date_str)
    except Exception:
        return "unknown"
    return "Weekend" if dt.weekday() >= 5 else "Weekday"


def _day_type_extended(date_str: str, holiday_flags: dict = None) -> str:
    """Classify into one of 10 day types for behaviour lookup."""
    dt = pd.to_datetime(date_str)
    dow = dt.weekday()
    if holiday_flags:
        if holiday_flags.get("is_holiday"):
            if holiday_flags.get("holiday_before_weekend"):
                return "Holiday-Fri"
            if holiday_flags.get("holiday_after_weekend"):
                return "Holiday-Mon"
            if holiday_flags.get("mid_week_holiday"):
                return "Holiday-MidWeek"
            return "Holiday"
        if holiday_flags.get("bridge_day"):
            return "BridgeDay"
        if holiday_flags.get("long_weekend"):
            return "LongWeekend"
    if dow >= 5:
        return "Weekend"
    if dow == 4:
        return "Friday"
    if dow == 0:
        return "Monday"
    return "Weekday"


def _compute_holiday_flags(df: pd.DataFrame, region: str = "punjab") -> pd.DataFrame:
    """Add is_holiday and all transition flags using national + state-specific holidays."""
    out = df.copy()
    out["is_holiday"] = 0
    try:
        import holidays as _hol
        dates = pd.to_datetime(out["date"])
        years = dates.dt.year.unique().tolist()
        national = _hol.India(years=years)
        subdiv = INDIAN_STATE_HOLIDAY_SUBDIV.get(region.lower().strip(), "HR")
        state = _hol.India(subdiv=subdiv, years=years)
        combined_dates = set(national.keys()) | set(state.keys())
        out["is_holiday"] = dates.map(lambda d: int(d in combined_dates))
    except Exception:
        dates = pd.to_datetime(out["date"])
        combined_dates = set()

    holiday_set = set(dates[out["is_holiday"] == 1].unique()) if out["is_holiday"].any() else set()
    dow = dates.dt.weekday

    if holiday_set:
        out["holiday_on_weekday"] = dates.map(lambda d: int(d in holiday_set and d.weekday() < 5))
        out["holiday_before_weekend"] = dates.map(lambda d: int(d in holiday_set and d.weekday() == 4))
        out["holiday_after_weekend"] = dates.map(lambda d: int(d in holiday_set and d.weekday() == 0))
        out["bridge_day"] = dates.map(lambda d: int(
            d not in holiday_set and d.weekday() < 5 and (
                (d + pd.Timedelta(days=1)) in holiday_set or
                (d - pd.Timedelta(days=1)) in holiday_set
            )
        ))
        out["long_weekend"] = dates.map(lambda d: int(
            d.weekday() in (5, 6) and (
                (d - pd.Timedelta(days=1)) in holiday_set or
                (d + pd.Timedelta(days=1)) in holiday_set
            )
        ))
        out["post_weekend_holiday"] = dates.map(lambda d: int(d in holiday_set and d.weekday() == 0))
        out["mid_week_holiday"] = dates.map(lambda d: int(
            d in holiday_set and d.weekday() in (1, 2, 3)
        ))
        holidays_sorted = np.array(sorted(holiday_set))
        out["days_to_holiday"] = dates.map(
            lambda d: int((holidays_sorted[holidays_sorted >= d][0] - d).days)
            if (holidays_sorted >= d).any() else 999
        )
        out["days_since_holiday"] = dates.map(
            lambda d: int((d - holidays_sorted[holidays_sorted <= d][-1]).days)
            if (holidays_sorted <= d).any() else 999
        )
    else:
        for col in ["holiday_on_weekday", "holiday_before_weekend", "holiday_after_weekend",
                     "bridge_day", "long_weekend", "post_weekend_holiday", "mid_week_holiday"]:
            out[col] = 0
        out["days_to_holiday"] = 999
        out["days_since_holiday"] = 999

    return out


def _calendar_factor(df: pd.DataFrame, calendar_config: Optional[str],
                     holiday_flags: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """
    Returns discrete calendar impacts.
    Keys: "total", "day_type", "holiday", "transition"
    Supports both legacy string calendar_config AND auto-detect via holiday_flags.
    """
    base = {"total": np.ones(96, dtype=float), "day_type": np.ones(96, dtype=float),
            "holiday": np.ones(96, dtype=float), "transition": np.ones(96, dtype=float)}
    if df.empty:
        return base

    df = df.copy()
    df["day_type"] = df["date"].apply(_day_type)
    daily = df.groupby(["date", "day_type"])["total_drawal"].sum().reset_index()
    weekday_avg = daily[daily["day_type"] == "Weekday"]["total_drawal"].mean()
    weekend_avg = daily[daily["day_type"] == "Weekend"]["total_drawal"].mean()

    if not weekday_avg or weekday_avg < 1e-6:
        return base

    weekend_factor = float((weekend_avg / weekday_avg) if weekend_avg else 0.92)
    holiday_ratio = min(weekend_factor, 0.92)

    # ── Legacy string-based calendar_config (backwards compat) ──
    if calendar_config:
        cfg_str = calendar_config.strip().lower()
        if cfg_str == "weekend":
            base["total"] = np.full(96, weekend_factor, dtype=float)
            base["day_type"] = np.full(96, weekend_factor, dtype=float)
        elif cfg_str in ("holiday", "holiday_to_weekend", "weekend_to_holiday"):
            base["total"] = np.full(96, holiday_ratio, dtype=float)
            base["holiday"] = np.full(96, holiday_ratio, dtype=float)
        elif cfg_str == "weekend_to_weekday":
            trans = np.linspace(weekend_factor, 1.0, 96)
            base["total"] = trans
            base["transition"] = trans
        elif cfg_str == "weekday_to_weekend":
            trans = np.linspace(1.0, weekend_factor, 96)
            base["total"] = trans
            base["transition"] = trans
        return base

    # ── Auto-detect from holiday_flags ──
    if holiday_flags is None:
        return base

    hf = holiday_flags
    is_holiday = hf.get("is_holiday", 0)
    is_bridge = hf.get("bridge_day", 0)
    is_long_wknd = hf.get("long_weekend", 0)
    holiday_before_wknd = hf.get("holiday_before_weekend", 0)
    holiday_after_wknd = hf.get("holiday_after_weekend", 0)
    mid_week_hol = hf.get("mid_week_holiday", 0)
    days_to = hf.get("days_to_holiday", 999)
    days_since = hf.get("days_since_holiday", 999)

    if is_holiday:
        base["total"] = np.full(96, holiday_ratio)
        base["holiday"] = np.full(96, holiday_ratio)
        if holiday_before_wknd:
            evening_dip = np.ones(96)
            evening_dip[68:88] *= 0.95
            base["total"] = base["total"] * evening_dip
        if holiday_after_wknd:
            morning_slow = np.ones(96)
            morning_slow[20:40] = np.linspace(holiday_ratio, 1.0, 20)
            base["transition"] = morning_slow
    elif is_bridge:
        bridge_ratio = 0.5 * holiday_ratio + 0.5 * 1.0
        base["total"] = np.full(96, bridge_ratio)
        base["day_type"] = np.full(96, bridge_ratio)
    elif is_long_wknd:
        deep_weekend = weekend_factor * 0.97
        base["total"] = np.full(96, deep_weekend)
        base["day_type"] = np.full(96, deep_weekend)
    elif mid_week_hol:
        base["total"] = np.full(96, holiday_ratio)
        base["holiday"] = np.full(96, holiday_ratio)
    else:
        if days_to == 1:
            pre_hol = np.ones(96)
            pre_hol[72:96] *= 0.97
            base["total"] = pre_hol
            base["transition"] = pre_hol
        elif days_since == 1:
            post_hol = np.ones(96)
            post_hol[20:44] = np.linspace(0.96, 1.0, 24)
            base["total"] = post_hol
            base["transition"] = post_hol

    return base


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["season"] = out["date"].apply(_season)
    out["day_type"] = out["date"].apply(_day_type)
    out["is_weekend"] = out["day_type"].eq("Weekend").astype(int)
    return out


def _timeblock_features(series: pd.Series) -> pd.DataFrame:
    tb = series.astype(float)
    angle = 2 * np.pi * (tb - 1) / 96.0
    return pd.DataFrame({
        "tb_sin": np.sin(angle),
        "tb_cos": np.cos(angle),
    })


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical, categorical & weather-interaction temporal features."""
    if "time_block" not in df.columns:
        return df
    tb = df["time_block"].astype(float)
    hour = ((tb - 1) * 15 / 60).astype(int)
    dow = pd.to_datetime(df["date"]).dt.dayofweek if "date" in df.columns else pd.Series(0, index=df.index)

    # ── cyclical hour / day-of-week ──
    df["hour_cos"]  = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * dow / 7)
    df["block_sin"] = np.sin(2 * np.pi * (tb - 1.0) / _BLOCK_COUNT)
    df["block_cos"] = np.cos(2 * np.pi * (tb - 1.0) / _BLOCK_COUNT)

    # ── period flags ──
    df["is_peak_hour"]     = hour.isin([7, 8, 9, 18, 19, 20]).astype(int)
    df["is_night"]         = hour.isin([0, 1, 2, 3, 4, 5, 6]).astype(int)
    df["is_business_hour"] = ((hour >= 9) & (hour <= 17)).astype(int)

    # ── temperature × diurnal interaction ──
    if "temperature" in df.columns:
        t = df["temperature"]
        df["temp_squared"]     = t ** 2
        df["temp_cubed"]       = t ** 3
        df["temperature_sin"]  = t * np.sin(2 * np.pi * hour / 24)

    return df


def _get_short_term_feature_columns(data: pd.DataFrame) -> List[str]:
    return [feature for feature in SHORT_TERM_MODEL_FEATURES if feature in data.columns]


def _build_load_anchor(
    baseline_hist: np.ndarray,
    actual: Optional[np.ndarray] = None,
    actual_blocks: int = 0,
) -> np.ndarray:
    anchor = np.asarray(baseline_hist, dtype=float).reshape(-1).copy()
    if anchor.size < _BLOCK_COUNT:
        anchor = np.pad(anchor, (0, _BLOCK_COUNT - anchor.size), mode="edge")
    elif anchor.size > _BLOCK_COUNT:
        anchor = anchor[:_BLOCK_COUNT]

    actual_vec = np.asarray(actual if actual is not None else [], dtype=float).reshape(-1)
    observed = int(np.clip(actual_blocks, 0, min(anchor.size, actual_vec.size)))
    if observed > 0:
        anchor[:observed] = actual_vec[:observed]
    return anchor


def _prepare_inference_frame(
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    load_anchor: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    history = history_df.copy() if history_df is not None else pd.DataFrame()
    target = target_df.copy()

    if history.empty:
        history = pd.DataFrame()
    else:
        history["load_feature_source"] = pd.to_numeric(history.get("total_drawal"), errors="coerce")

    target["load_feature_source"] = pd.to_numeric(target.get("total_drawal"), errors="coerce")
    if load_anchor is not None:
        anchor = np.asarray(load_anchor, dtype=float).reshape(-1)
        if anchor.size < len(target):
            anchor = np.pad(anchor, (0, len(target) - anchor.size), mode="edge")
        elif anchor.size > len(target):
            anchor = anchor[:len(target)]
        target["load_feature_source"] = anchor

    target["_is_target_row"] = 1
    if history.empty:
        combined = target.copy()
    else:
        history["_is_target_row"] = 0
        union_cols = sorted(set(history.columns).union(target.columns))
        history = history.reindex(columns=union_cols)
        target = target.reindex(columns=union_cols)
        usable_cols = [
            col for col in union_cols
            if not (history[col].isna().all() and target[col].isna().all())
        ]
        hist_part = history[usable_cols].dropna(axis=1, how="all")
        tgt_part = target[usable_cols].dropna(axis=1, how="all")
        combined = pd.concat(
            [hist_part, tgt_part],
            ignore_index=True,
            sort=False,
        )
        combined = combined.reindex(columns=sorted(combined.columns))
    prepared = _prepare_training(combined, load_source_col="load_feature_source")
    return prepared[prepared["_is_target_row"] == 1].copy()


def _normalize_block_vector(values: Any, length: int = _BLOCK_COUNT, fill_value: float = 0.0) -> np.ndarray:
    vec = np.asarray(values if values is not None else [], dtype=float).reshape(-1)
    if vec.size == 0:
        return np.full(length, float(fill_value), dtype=float)
    if vec.size < length:
        pad_value = float(vec[-1]) if vec.size else float(fill_value)
        vec = np.pad(vec, (0, length - vec.size), mode="constant", constant_values=pad_value)
    elif vec.size > length:
        vec = vec[:length]
    return vec.astype(float, copy=False)


def _build_synthetic_forecast_day(
    template_df: pd.DataFrame,
    target_date: str,
    forecast: Any,
    zero_actuals: bool = False,
) -> pd.DataFrame:
    day = _coerce_day_to_96_blocks(template_df, fill_load=False)
    if day.empty:
        raise ValueError(f"Cannot synthesise day {target_date}: template day is empty.")

    synthetic = day.copy()
    synthetic["date"] = str(target_date)
    synthetic["time_block"] = np.arange(1, _BLOCK_COUNT + 1, dtype=int)
    synthetic["total_drawal"] = _normalize_block_vector(
        np.zeros(_BLOCK_COUNT, dtype=float) if zero_actuals else forecast,
        length=_BLOCK_COUNT,
        fill_value=0.0,
    )
    return synthetic


def _apply_seam_continuity(
    forecast: Any,
    previous_terminal_mw: float,
    window: int = 8,
) -> Tuple[np.ndarray, float, float]:
    adjusted = _normalize_block_vector(forecast, length=_BLOCK_COUNT, fill_value=0.0)
    gap_before = float(adjusted[0] - previous_terminal_mw)
    if adjusted.size == 0 or window <= 0:
        return adjusted, gap_before, gap_before

    taper = min(int(window), adjusted.size)
    if taper <= 1:
        adjusted[0] = float(previous_terminal_mw)
        return adjusted, gap_before, float(adjusted[0] - previous_terminal_mw)

    delta = float(previous_terminal_mw - adjusted[0])
    if abs(delta) < 1e-9:
        return adjusted, gap_before, gap_before

    taper_weights = 0.5 * (1.0 + np.cos(np.pi * np.arange(taper, dtype=float) / float(taper - 1)))
    adjusted[:taper] = adjusted[:taper] + (delta * taper_weights)
    gap_after = float(adjusted[0] - previous_terminal_mw)
    return adjusted, gap_before, gap_after


def _coerce_day_to_96_blocks(day_df: pd.DataFrame, fill_load: bool = False) -> pd.DataFrame:
    if day_df is None or day_df.empty:
        return day_df.copy() if isinstance(day_df, pd.DataFrame) else pd.DataFrame()

    work = day_df.copy()
    date_value = str(work["date"].astype(str).iloc[0]) if "date" in work.columns and not work.empty else None
    work["time_block"] = pd.to_numeric(work.get("time_block"), errors="coerce")
    work = work.dropna(subset=["time_block"])
    work["time_block"] = work["time_block"].astype(int)
    work = work[work["time_block"].between(1, _BLOCK_COUNT)]
    if work.empty:
        return work

    numeric_cols = [col for col in work.columns if col != "date"]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    grouped = work.groupby("time_block", as_index=False)[numeric_cols].mean()
    grouped = grouped.set_index("time_block").reindex(range(1, _BLOCK_COUNT + 1))
    grouped.index.name = "time_block"
    grouped = grouped.reset_index()
    if date_value is not None:
        grouped["date"] = date_value

    for col in grouped.columns:
        if col in {"date", "time_block", "total_drawal"}:
            continue
        grouped[col] = grouped[col].interpolate(limit_direction="both").ffill().bfill()

    if "total_drawal" in grouped.columns and fill_load:
        grouped["total_drawal"] = grouped["total_drawal"].interpolate(limit_direction="both").ffill().bfill()

    ordered_cols = [col for col in day_df.columns if col in grouped.columns]
    remaining_cols = [col for col in grouped.columns if col not in ordered_cols]
    return grouped[ordered_cols + remaining_cols]


def _prepare_training(df: pd.DataFrame, load_source_col: str = "total_drawal") -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)

    cols_to_clean = [
        "temperature", "humidity", "precipitation",
        "apparent_temperature", "cloud_cover", "sunshine_duration",
        "direct_radiation", "wind_speed_10m", "time_block",
    ]
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].interpolate(limit_direction="both").ffill().bfill().fillna(0.0)

    if "total_drawal" in df.columns:
        df["total_drawal"] = pd.to_numeric(df["total_drawal"], errors="coerce")

    if load_source_col in df.columns:
        load_source = pd.to_numeric(df[load_source_col], errors="coerce")
    elif "total_drawal" in df.columns:
        load_source = pd.to_numeric(df["total_drawal"], errors="coerce")
    else:
        load_source = pd.Series(0.0, index=df.index, dtype=float)

    if load_source_col == "total_drawal":
        load_source = load_source.interpolate(limit_direction="both").ffill().bfill()
    else:
        fallback = pd.to_numeric(df.get("total_drawal"), errors="coerce") if "total_drawal" in df.columns else pd.Series(np.nan, index=df.index)
        load_source = load_source.where(load_source.notna(), fallback)
        load_source = load_source.ffill().bfill()

    df["load_feature_source"] = load_source.fillna(0.0)
    df = df.sort_values(["date", "time_block"]).reset_index(drop=True)

    # --- V2 UPGRADE: Nonlinear & Lag Features ---
    if "temperature" in df.columns:
        # Regional CDD/HDD bases (default north)
        _cdd_base = 24.0
        _hdd_base = 15.0
        try:
            _bases = DEFAULT_CONFIG.get("cdd_hdd_bases", {})
            # Try to detect climate region from data context
            for _cr in ("north", "south", "west", "east", "central"):
                if _cr in _bases:
                    _cdd_base = _bases.get(_cr, {}).get("cdd", _cdd_base)
                    _hdd_base = _bases.get(_cr, {}).get("hdd", _hdd_base)
                    break  # Use first available; caller can override via config
        except Exception:
            pass
        df["CDD"] = (df["temperature"] - _cdd_base).clip(lower=0)
        df["HDD"] = (_hdd_base - df["temperature"]).clip(lower=0)
        # 3h and 6h rolling averages
        df["temp_roll_3h"] = df["temperature"].rolling(window=12, min_periods=1).mean()
        df["temp_roll_6h"] = df["temperature"].rolling(window=24, min_periods=1).mean()

        # ── Thermal inertia: accumulated cooling-degree-hours (CDH) ──
        # After 3 consecutive hot days buildings retain heat; evening load stays elevated.
        # 24h (1 day), 48h (2 day), 72h (3 day) rolling sum of CDD.
        df["cdh_24h"] = df["CDD"].rolling(window=96, min_periods=1).sum()
        df["cdh_48h"] = df["CDD"].rolling(window=192, min_periods=1).sum()
        df["cdh_72h"] = df["CDD"].rolling(window=288, min_periods=1).sum()

        # Wet-bulb globe temperature (WBGT) — better predictor of cooling load
        # in humid Indian summers than dry-bulb or apparent temperature.
        # Simplified Stull (2011) approximation: WBGT ≈ f(T, RH)
        if "humidity" in df.columns:
            T = df["temperature"].values
            RH = df["humidity"].clip(0, 100).values
            # Stull wet-bulb approximation
            wb = T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + \
                 np.arctan(T + RH) - np.arctan(RH - 1.676331) + \
                 0.00391838 * (RH ** 1.5) * np.arctan(0.023101 * RH) - 4.686035
            # WBGT ≈ 0.7 * wet_bulb + 0.2 * globe + 0.1 * dry_bulb
            # Simplified (no globe temp): WBGT ≈ 0.7 * wb + 0.3 * T
            df["wbgt"] = 0.7 * wb + 0.3 * T
        else:
            df["wbgt"] = df["temperature"]

    # ── Solar generation proxy ──
    # Odisha has growing rooftop solar. Grid drawal = demand - solar.
    # Cloud cover and radiation directly affect the residual the model predicts.
    # Estimated solar index: high radiation + low cloud → high solar output → lower grid load.
    if "direct_radiation" in df.columns and "cloud_cover" in df.columns:
        rad = pd.to_numeric(df["direct_radiation"], errors="coerce").fillna(0).clip(lower=0)
        cloud = pd.to_numeric(df["cloud_cover"], errors="coerce").fillna(50).clip(0, 100)
        # Normalised solar capacity factor (0-1 scale, ~peak at clear-sky noon)
        df["solar_index"] = (rad / max(rad.max(), 1)) * (1 - cloud / 100)
        # Estimated solar MW offset (rough proxy, scales with radiation)
        df["solar_proxy_mw"] = rad * (1 - cloud / 100) * 0.001  # arbitrary scaling
    elif "direct_radiation" in df.columns:
        rad = pd.to_numeric(df["direct_radiation"], errors="coerce").fillna(0).clip(lower=0)
        df["solar_index"] = rad / max(rad.max(), 1)
        df["solar_proxy_mw"] = rad * 0.001

    # Lagged load features (strongly predictive for short-term autocorrelation)
    if "load_feature_source" in df.columns:
        load_series = df["load_feature_source"]
        df["lag_1"] = load_series.shift(_BLOCK_COUNT)
        df["lag_7"] = load_series.shift(_BLOCK_COUNT * 7)
        df["lag_block_1"] = load_series.shift(1)
        df["lag_block_4"] = load_series.shift(4)
        df["rolling_4"] = load_series.shift(1).rolling(window=4, min_periods=1).mean()
        df["rolling_12"] = load_series.shift(1).rolling(window=12, min_periods=1).mean()

        df["load_lag_1d"] = df["lag_1"]
        df["load_lag_2d"] = load_series.shift(_BLOCK_COUNT * 2)
        df["load_lag_3d"] = load_series.shift(_BLOCK_COUNT * 3)
        df["load_lag_7d"] = df["lag_7"]
        df["load_rolling_7d"] = df.groupby("time_block")["load_feature_source"].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean()
        )
        # 3-day load trend: slope of last 3 days' daily average (v3.0)
        daily_avg = df.groupby("date")["load_feature_source"].transform("mean")
        df["load_trend_3d"] = (
            daily_avg - daily_avg.shift(_BLOCK_COUNT * 3)
        )

    # ── Multiplicative weather × load interaction features (v3.0) ────
    # Weather impact on load is fundamentally multiplicative — a 3°C rise
    # adds more MW when base load is 5000 vs 2000. These features let
    # tree models learn percentage-based weather sensitivity.
    if "total_drawal" in df.columns and "temperature" in df.columns:
        lag_1d = df.get("load_lag_1d", df["total_drawal"])
        roll_7d = df.get("load_rolling_7d", df["total_drawal"])
        cdd = df.get("CDD", pd.Series(0, index=df.index))
        hum = df.get("humidity", pd.Series(50, index=df.index))
        wbgt = df.get("wbgt", df["temperature"])

        df["load_x_cdd"] = lag_1d * cdd                    # AC impact scales with load
        df["load_x_humidity"] = lag_1d * (hum / 100.0)     # humid-heat compound
        df["cdd_squared"] = cdd ** 2                        # exponential AC response
        df["wbgt_x_load"] = wbgt * roll_7d                  # heat stress × load level
        # 3-day temperature momentum (thermal mass buildup)
        if "temperature" in df.columns:
            df["temp_momentum_3d"] = df["temperature"] - df["temperature"].shift(288).fillna(df["temperature"])

    df = _add_calendar(df)
    df = df.reset_index(drop=True)
    tb_feats = _timeblock_features(df["time_block"]).reset_index(drop=True)
    df = pd.concat([df, tb_feats], axis=1)
    df["season_idx"] = df["season"].apply(lambda s: SEASON_ORDER.index(s) if s in SEASON_ORDER else 0)
    df = _add_time_features(df)
    return df


def _best_baseline_window(df: pd.DataFrame, target_date: str, candidates: List[int], backtest_days: int, weather_weight: float = 0.0) -> Tuple[int, Optional[float]]:
    dates = sorted(df["date"].dropna().unique().tolist())
    if target_date not in dates:
        return candidates[0], None
    idx = dates.index(target_date)
    history = dates[:idx]
    if len(history) < 3:
        return candidates[0], None

    recent = history[-backtest_days:]
    best_w = candidates[0]
    best_mape = None
    best_score = None

    target_df = df[df["date"] == target_date]
    if not target_df.empty:
        tgt_weather = target_df[["temperature", "humidity", "precipitation"]].mean()
        target_weather_delta = float(tgt_weather.abs().mean())
    else:
        tgt_weather = None
        target_weather_delta = 0.0

    for w in candidates:
        errors = []
        for d in recent:
            di = history.index(d)
            start = max(0, di - w)
            window_days = history[start:di]
            if not window_days:
                continue
            baseline = df[df["date"].isin(window_days)].groupby("time_block")["total_drawal"].mean().reindex(range(1, 97))
            actual = df[df["date"] == d].sort_values("time_block")["total_drawal"].to_numpy()
            base_vals = baseline.to_numpy()
            if len(actual) != 96:
                continue
            mape = np.mean(np.abs(actual - base_vals) / np.maximum(actual, 1e-6)) * 100
            errors.append(mape)
        if not errors:
            continue
        avg_mape = float(np.mean(errors))
        weather_diff = 0.0
        if tgt_weather is not None:
            window_weather = df[df["date"].isin(history[max(0, idx - w):idx])][["temperature", "humidity", "precipitation"]].mean()
            weather_diff = float((tgt_weather - window_weather).abs().mean())
        delta_scale = min(1.0, target_weather_delta / 10.0) if target_weather_delta else 0.0
        weight = float(weather_weight) * (1.0 + delta_scale)
        score = avg_mape + (weight * weather_diff)
        if best_score is None or score < best_score:
            best_score = score
            best_mape = avg_mape
            best_w = w
    return best_w, best_mape


def _similar_day_baseline(df: pd.DataFrame, target_date: str, cfg: Dict) -> Tuple[np.ndarray, pd.DataFrame]:
    df = _add_calendar(df)
    dates = sorted(df["date"].dropna().unique().tolist())
    if target_date not in dates:
        # Find latest robust date
        target_date = dates[-1]
        for d in reversed(dates):
            day_df = df[df["date"] == d]
            if day_df["time_block"].nunique() >= 96 and day_df["total_drawal"].mean() > cfg.get("min_valid_load_mw", 100):
                target_date = d
                break
    idx = dates.index(target_date)
    candidates = dates[max(0, idx - cfg["candidate_lookback_days"]):idx]
    if not candidates:
        candidates = dates[:idx]

    target_df = df[df["date"] == target_date]
    target_season = target_df["season"].iloc[0] if not target_df.empty else "unknown"
    
    # Get seasonal intervals
    s_config = SEASONAL_WEATHER_INTERVALS.get(target_season, SEASONAL_WEATHER_INTERVALS["spring"])
    cdd_base = s_config["cdd_base"]
    hdd_base = s_config["hdd_base"]
    temp_band = s_config["temp_band"]
    hum_band = s_config["hum_band"]

    target_temp = target_df["temperature"].mean()
    target_hum = target_df["humidity"].mean()
    target_cdd = (target_df["temperature"] - cdd_base).clip(lower=0).mean()
    target_hdd = (hdd_base - target_df["temperature"]).clip(lower=0).mean()
    target_rain = (target_df["precipitation"] > 0).any()
    target_day_type = target_df["day_type"].iloc[0] if not target_df.empty else "unknown"

    df_prep = df.copy()
    df_prep["CDD"] = (df_prep["temperature"] - cdd_base).clip(lower=0)
    df_prep["HDD"] = (hdd_base - df_prep["temperature"]).clip(lower=0)

    summary = (
        df_prep[df_prep["date"].isin(candidates)]
        .groupby("date")
        .agg(
            temp_mean=("temperature", "mean"),
            hum_mean=("humidity", "mean"),
            cdd_mean=("CDD", "mean"),
            hdd_mean=("HDD", "mean"),
            rain_any=("precipitation", lambda x: float((x > 0).any())),
            season=("season", "first"),
            day_type=("day_type", "first"),
            avg_load=("total_drawal", "mean")
        )
        .reset_index()
    )

    # Filter out days with zero load
    summary = summary[summary["avg_load"] > cfg.get("min_valid_load_mw", 100)].copy()

    if summary.empty:
        baseline = target_df.sort_values("time_block")["total_drawal"].to_numpy()
        return baseline, summary

    summary["same_season"] = summary["season"].eq(target_season)
    summary["same_day_type"] = summary["day_type"].eq(target_day_type)
    summary["temp_diff"] = (summary["temp_mean"] - target_temp).abs()
    summary["hum_diff"] = (summary["hum_mean"] - target_hum).abs()
    summary["degree_day_diff"] = (summary["cdd_mean"] - target_cdd).abs() + (summary["hdd_mean"] - target_hdd).abs()
    summary["rain_match"] = summary["rain_any"].astype(bool).eq(bool(target_rain))

    filtered = summary[summary["same_season"] & summary["same_day_type"]]
    if cfg.get("require_rain_match", True):
        filtered = filtered[filtered["rain_match"]]
    filtered = filtered[filtered["temp_diff"] <= temp_band]

    # Progressive fallback: relax temp band if too few candidates (v3.0)
    if len(filtered) < 5:
        relaxed = summary[summary["same_season"] & summary["same_day_type"]]
        if cfg.get("require_rain_match", True):
            relaxed = relaxed[relaxed["rain_match"]]
        relaxed = relaxed[relaxed["temp_diff"] <= temp_band + 1.0]
        if len(relaxed) >= len(filtered):
            filtered = relaxed
    if len(filtered) < 3:
        relaxed2 = summary[summary["same_season"] & summary["same_day_type"]]
        relaxed2 = relaxed2[relaxed2["temp_diff"] <= temp_band + 2.0]
        if len(relaxed2) >= len(filtered):
            filtered = relaxed2
    if filtered.empty:
        filtered = summary.copy()

    temp_norm = filtered["temp_diff"] / max(temp_band, 1e-3)
    hum_norm = filtered["hum_diff"] / max(hum_band, 1e-3)
    dd_norm = filtered["degree_day_diff"] / 2.0 # Normalize degree day diff
    rain_penalty = (~filtered["rain_match"]).astype(int)

    w_temp = cfg["similarity_weights"]["temp"] * 0.7
    w_dd = cfg["similarity_weights"]["temp"] * 0.3
    w_hum = cfg["similarity_weights"]["humidity"]
    w_rain = cfg["similarity_weights"]["rain"]

    filtered["similarity_score"] = w_temp * temp_norm + w_dd * dd_norm + w_hum * hum_norm + w_rain * rain_penalty

    filtered = filtered.sort_values("similarity_score").head(cfg["similar_days_top_n"])
    selected_dates = filtered["date"].tolist()
    if not selected_dates:
        selected_dates = candidates[-1:]

    # Normalize similar days for load growth (User Fix #4)
    # Calculate recent 7-day average load level
    recent_dates = dates[max(0, idx - 7):idx]
    recent_load_level = 1.0
    if recent_dates:
        # Check if we have 'total_drawal' column
        recent_avg = df[df["date"].isin(recent_dates)]["total_drawal"].mean()
        if recent_avg > 100:
            recent_load_level = float(recent_avg)
    
    # Calculate similar days baseline
    similar_days_df = df[df["date"].isin(selected_dates)]
    
    # Scale each similar day
    scaled_stack = []
    for d in selected_dates:
        day_df = similar_days_df[similar_days_df["date"] == d]
        if day_df.empty: continue
        day_avg = day_df["total_drawal"].mean()
        scale = 1.0
        if day_avg > 100 and recent_load_level > 100:
             scale = recent_load_level / day_avg
             # Clamp scale to avoid extreme multipliers (e.g. 0.8 to 1.2)
             scale = np.clip(scale, 0.85, 1.15)
        
        vals = day_df.groupby("time_block")["total_drawal"].mean().reindex(range(1, 97)).ffill().bfill().to_numpy()
        scaled_stack.append(vals * scale)
        
    if not scaled_stack:
        return np.zeros(96), filtered

    baseline = np.mean(np.vstack(scaled_stack), axis=0)

    return baseline, filtered


def _build_weather_model() -> Any:
    if XGBRegressor is not None:
        return XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=4,
            verbosity=1,
            random_state=42,
        )
    if LGBMRegressor is not None:
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=64,
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            verbosity=1,
            random_state=42,
        )
    return LinearRegression()

def _time_based_split(data: pd.DataFrame, min_val_days: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "date" not in data.columns:
        return data, data
    dates = sorted(data["date"].dropna().unique().tolist())
    if len(dates) < min_val_days:
        return data, data
    val_days = min(7, max(1, len(dates) // 5))
    val_dates = set(dates[-val_days:])
    train = data[~data["date"].isin(val_dates)]
    val = data[data["date"].isin(val_dates)]
    if train.empty or val.empty:
        return data, data
    return train, val


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _tune_xgb(X_train, y_train, X_val, y_val, n_iter: int, quantile: float = 0.5) -> Any:
    best_model = None
    best_score = None
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        params = {
            "n_estimators": int(rng.integers(250, 800)),
            "max_depth": int(rng.integers(4, 9)),
            "learning_rate": float(rng.uniform(0.02, 0.12)),
            "subsample": float(rng.uniform(0.7, 0.95)),
            "colsample_bytree": float(rng.uniform(0.6, 0.95)),
            "min_child_weight": float(rng.uniform(1.0, 6.0)),
            "reg_alpha": float(rng.uniform(0.0, 0.5)),
            "reg_lambda": float(rng.uniform(0.8, 2.0)),
            "objective": "reg:quantileerror",
            "quantile_alpha": quantile,
            "n_jobs": 4,
            "random_state": 42,
        }
        model = XGBRegressor(**params)
        print(f"\n[ML Pipeline] XGBoost tuning iteration... Training epochs:")
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=100)
        preds = model.predict(X_val)
        score = _rmse(y_val, preds)
        if best_score is None or score < best_score:
            best_score = score
            best_model = model
    return best_model


def _tune_lgbm(X_train, y_train, X_val, y_val, n_iter: int, quantile: float = 0.5) -> Any:
    best_model = None
    best_score = None
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        params = {
            "n_estimators": int(rng.integers(250, 800)),
            "learning_rate": float(rng.uniform(0.02, 0.12)),
            "num_leaves": int(rng.integers(32, 128)),
            "max_depth": int(rng.integers(-1, 9)),
            "subsample": float(rng.uniform(0.7, 0.95)),
            "colsample_bytree": float(rng.uniform(0.6, 0.95)),
            "min_child_samples": int(rng.integers(10, 40)),
            "reg_alpha": float(rng.uniform(0.0, 0.5)),
            "reg_lambda": float(rng.uniform(0.8, 2.0)),
            "objective": "quantile",
            "alpha": quantile,
            "verbosity": -1,
            "random_state": 42,
        }
        model = LGBMRegressor(**params)
        print(f"\n[ML Pipeline] LightGBM tuning iteration... Training epochs:")
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
        preds = model.predict(X_val)
        score = _rmse(y_val, preds)
        if best_score is None or score < best_score:
            best_score = score
            best_model = model
    return best_model


class WeatherEnsemble:
    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights if weights else [1.0/len(models)] * len(models)
        
    def predict(self, X):
        if not self.models:
            return np.zeros(X.shape[0])
        preds = np.zeros(X.shape[0])
        total_weight = sum(self.weights)
        for model, w in zip(self.models, self.weights):
            preds += model.predict(X) * w
        return preds / total_weight

def _estimate_feature_effects(model: Any, data: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    y = data["total_drawal"].to_numpy()
    y_std = float(np.std(y)) if len(y) else 1.0
    
    # Handle Ensemble
    if isinstance(model, WeatherEnsemble):
        # Average effects across models
        agg_effects = {f: 0.0 for f in features}
        for sub_model, w in zip(model.models, model.weights):
            sub_effects = _estimate_feature_effects(sub_model, data, features)
            for f, val in sub_effects.items():
                agg_effects[f] += val * w
        # Normalize by total weight
        total_w = sum(model.weights)
        return {k: v / total_w for k, v in agg_effects.items()}

    effects = {}
    if hasattr(model, "coef_"):
        # Linear model: coef_ is in feature-units, scale to MW-per-unit
        for f, c in zip(features, model.coef_):
            effects[f] = float(c)
        return effects
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_, dtype=float)
    if importances is None or not np.isfinite(importances).any():
        return {f: 0.0 for f in features}
    importances = importances / max(importances.sum(), 1e-6)
    # For tree models, feature_importances_ is already a relative measure.
    # Use sign from correlation with load, but do NOT scale by y_std/x_std
    # (sparse features like rain have tiny x_std which inflates the ratio).
    for idx, f in enumerate(features):
        x = data[f].to_numpy()
        corr = _safe_corr(pd.Series(y), pd.Series(x))
        sign = 1.0 if corr >= 0 else -1.0
        effects[f] = float(sign * importances[idx])
    return effects


def _train_weather_model(df: pd.DataFrame, tune: bool = True, tune_iters: int = 12) -> Tuple[Any, pd.DataFrame, Dict[str, float]]:
    data = _prepare_training(df)
    features = _get_short_term_feature_columns(data)
    data = data.dropna(subset=features + ["total_drawal"])
    X = data[features]
    y = data["total_drawal"].to_numpy()
    model = None
    if tune and (XGBRegressor is not None or LGBMRegressor is not None):
        train_df, val_df = _time_based_split(data)
        X_train = train_df[features]
        y_train = train_df["total_drawal"].to_numpy()
        X_val = val_df[features]
        y_val = val_df["total_drawal"].to_numpy()
        
        models_with_rmse = []
        if XGBRegressor is not None:
            xgb_model = _tune_xgb(X_train, y_train, X_val, y_val, max(4, int(tune_iters)), quantile=0.5)
            xgb_rmse = _rmse(y_val, xgb_model.predict(X_val))
            models_with_rmse.append((xgb_model, xgb_rmse))
        if LGBMRegressor is not None:
            lgbm_model = _tune_lgbm(X_train, y_train, X_val, y_val, max(4, int(tune_iters)), quantile=0.5)
            lgbm_rmse = _rmse(y_val, lgbm_model.predict(X_val))
            models_with_rmse.append((lgbm_model, lgbm_rmse))

        # NHiTS removed from voting ensemble per ensemble_model_strategy.md:
        # it has no weather/calendar conditioning and drags T+2 accuracy.
        # XGBoost + LightGBM are the core ensemble; TiDE/PatchTST are planned replacements.

        if models_with_rmse:
            # RMSE-weighted ensemble (better models get more weight)
            models_list = [m for m, _ in models_with_rmse]
            total_inv_rmse = sum(1.0 / max(r, 1e-6) for _, r in models_with_rmse)
            weights = [(1.0 / max(r, 1e-6)) / total_inv_rmse for _, r in models_with_rmse]
            model = WeatherEnsemble(models_list, weights=weights)

    if model is None:
        model = _build_weather_model()
        print(f"\n[ML Pipeline] Training base weather model... Training epochs:")
        model.fit(X, y) # Default fallback without eval_set avoids shape issues
    effects = _estimate_feature_effects(model, data, features)
    return model, data, effects


def _weather_baseline(
    model: Any,
    history_df: pd.DataFrame,
    target_df: pd.DataFrame,
    effects: Dict[str, float],
    load_anchor: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    data = _prepare_inference_frame(history_df=history_df, target_df=target_df, load_anchor=load_anchor)
    features = _get_short_term_feature_columns(data)
    X = data.reindex(columns=features).copy()

    history_features = _prepare_training(history_df) if history_df is not None and not history_df.empty else pd.DataFrame()
    fill_values = {}
    if not history_features.empty:
        for feature in features:
            if feature in history_features.columns and history_features[feature].notna().any():
                fill_values[feature] = float(history_features[feature].median(skipna=True))

    load_cols = [
        "lag_1", "lag_7", "lag_block_1", "lag_block_4", "rolling_4", "rolling_12",
        "load_lag_1d", "load_lag_2d", "load_lag_3d", "load_lag_7d",
        "load_rolling_7d", "load_trend_3d", "load_x_cdd",
        "load_x_humidity", "wbgt_x_load",
    ]
    anchor_mean = float(np.nanmean(load_anchor)) if load_anchor is not None and len(load_anchor) else 0.0
    for col in load_cols:
        if col in X.columns:
            X[col] = X[col].fillna(fill_values.get(col, anchor_mean))

    for feature in features:
        if feature in X.columns:
            X[feature] = X[feature].fillna(fill_values.get(feature, 0.0))

    preds = model.predict(X)
    contributions = {
        "temperature": float(effects.get("temperature", 0.0)),
        "humidity": float(effects.get("humidity", 0.0)),
        "rain": float(effects.get("precipitation", 0.0)),
        "apparent_temp": float(effects.get("apparent_temperature", 0.0)),
        "cloud": float(effects.get("cloud_cover", 0.0)),
        "sunshine": float(effects.get("sunshine_duration", 0.0)),
        "radiation": float(effects.get("direct_radiation", 0.0)),
        "wind": float(effects.get("wind_speed_10m", 0.0)),
    }
    return preds, contributions


def _compute_weather_deviation(target_df: pd.DataFrame, baseline_df: pd.DataFrame) -> float:
    if target_df.empty or baseline_df.empty:
        return 0.0
    tgt = target_df["temperature"].mean()
    base = baseline_df["temperature"].mean()
    return float(abs(tgt - base))


def _seasonal_temp_prior(season: str) -> float:
    return {
        "summer": 0.0100,
        "spring": 0.0085,
        "fall": 0.0085,
        "winter": -0.0100,
    }.get(str(season), 0.0090)


def _clip_temp_coeff(value: float, season: str, prior: Optional[float] = None) -> float:
    season_prior = float(_seasonal_temp_prior(season))
    if abs(season_prior) < 1e-6:
        p = float(prior) if prior is not None else 0.0090
    else:
        p = season_prior
    v = float(np.clip(float(value), -0.03, 0.03))
    if p >= 0:
        return float(max(0.0005, v))
    return float(min(-0.0005, v))


def _robust_temp_slope(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], int]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    valid = np.isfinite(xx) & np.isfinite(yy) & (np.abs(xx) > 1e-6)
    xx = xx[valid]
    yy = yy[valid]
    n = int(xx.size)
    if n < 4:
        return None, n
    ratio = yy / xx
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size < 4:
        return None, int(ratio.size)
    q1, q3 = np.percentile(ratio, [25, 75])
    iqr = max(float(q3 - q1), 1e-6)
    lo = float(q1 - (2.0 * iqr))
    hi = float(q3 + (2.0 * iqr))
    trimmed = ratio[(ratio >= lo) & (ratio <= hi)]
    use = trimmed if trimmed.size >= 4 else ratio
    if use.size < 4:
        return None, int(use.size)
    return float(np.median(use)), int(use.size)


def _blend_temp_coeff(prior: float, estimate: Optional[float], sample_count: int, ridge: float) -> float:
    if estimate is None:
        return float(prior)
    w = float(np.clip(sample_count / max(sample_count + ridge, 1e-6), 0.0, 1.0))
    return float(((1.0 - w) * prior) + (w * float(estimate)))


def _learn_asymmetric_temp_profile(
    df: pd.DataFrame,
    target_date: str,
    season: str,
    day_type: str,
    lookback_days: int = 180,
    rolling_days: int = 7,
    min_block_samples: int = 8,
    prior_coeff: Optional[float] = None,
) -> Dict[str, Any]:
    prior = float(_seasonal_temp_prior(season) if prior_coeff is None else prior_coeff)
    default_up = np.full(96, prior, dtype=float)
    default_down = np.full(96, prior, dtype=float)
    default_samples = np.zeros(96, dtype=int)
    out = {
        "source": "seasonal_prior",
        "prior_coeff": float(prior),
        "global_increase_coeff": float(prior),
        "global_reduction_coeff": float(prior),
        "global_increase_samples": 0,
        "global_reduction_samples": 0,
        "block_increase_coeffs": default_up,
        "block_reduction_coeffs": default_down,
        "block_increase_samples": default_samples.copy(),
        "block_reduction_samples": default_samples.copy(),
        "diagnostics": {"reason": "fallback_prior"},
    }

    required_cols = {"date", "time_block", "temperature", "total_drawal"}
    if df.empty or not required_cols.issubset(set(df.columns)):
        out["diagnostics"] = {"reason": "missing_required_columns"}
        return out

    dates = sorted(df["date"].dropna().astype(str).unique().tolist())
    if target_date not in dates:
        out["diagnostics"] = {"reason": "target_not_in_dates"}
        return out
    idx = dates.index(target_date)
    hist_dates = dates[max(0, idx - max(lookback_days, 30)):idx]
    if not hist_dates:
        out["diagnostics"] = {"reason": "no_history_dates"}
        return out

    same_context = [d for d in hist_dates if (_season(d) == season and _day_type(d) == day_type)]
    if len(same_context) >= 28:
        selected_dates = same_context[-lookback_days:]
        context_mode = "season_daytype"
    else:
        same_daytype = [d for d in hist_dates if _day_type(d) == day_type]
        if len(same_daytype) >= 28:
            selected_dates = same_daytype[-lookback_days:]
            context_mode = "daytype"
        else:
            selected_dates = hist_dates[-lookback_days:]
            context_mode = "recent_all"

    hist = df[df["date"].astype(str).isin(set(selected_dates))][["date", "time_block", "temperature", "total_drawal"]].copy()
    if hist.empty:
        out["diagnostics"] = {"reason": "empty_history_slice"}
        return out

    hist["time_block"] = pd.to_numeric(hist["time_block"], errors="coerce")
    hist["temperature"] = pd.to_numeric(hist["temperature"], errors="coerce")
    hist["total_drawal"] = pd.to_numeric(hist["total_drawal"], errors="coerce")
    hist = hist.dropna(subset=["time_block", "temperature", "total_drawal"])
    hist["time_block"] = hist["time_block"].astype(int)
    hist = hist[(hist["time_block"] >= 1) & (hist["time_block"] <= 96)]
    if hist.empty:
        out["diagnostics"] = {"reason": "empty_after_numeric_clean"}
        return out

    date_order = {d: i for i, d in enumerate(selected_dates)}
    hist["date_idx"] = hist["date"].astype(str).map(date_order)
    hist = hist.dropna(subset=["date_idx"]).sort_values(["time_block", "date_idx"])
    if hist.empty:
        out["diagnostics"] = {"reason": "empty_after_date_order"}
        return out

    min_periods = max(3, int(min(rolling_days, 7)))
    hist["base_temp"] = (
        hist.groupby("time_block")["temperature"]
        .transform(lambda s: s.shift(1).rolling(rolling_days, min_periods=min_periods).mean())
    )
    hist["base_load"] = (
        hist.groupby("time_block")["total_drawal"]
        .transform(lambda s: s.shift(1).rolling(rolling_days, min_periods=min_periods).mean())
    )
    hist["temp_delta"] = hist["temperature"] - hist["base_temp"]
    hist["load_ratio"] = (hist["total_drawal"] / np.maximum(hist["base_load"], 1e-6)) - 1.0

    valid = (
        np.isfinite(hist["temp_delta"])
        & np.isfinite(hist["load_ratio"])
        & np.isfinite(hist["base_load"])
        & (hist["base_load"] > 1e-6)
        & (np.abs(hist["temp_delta"]) >= 0.15)
    )
    hist = hist[valid]
    if hist.empty:
        out["diagnostics"] = {"reason": "no_valid_delta_rows", "context_mode": context_mode}
        return out

    pos = hist[hist["temp_delta"] > 0]
    neg = hist[hist["temp_delta"] < 0]
    g_up_est, g_up_n = _robust_temp_slope(pos["temp_delta"].to_numpy(dtype=float), pos["load_ratio"].to_numpy(dtype=float))
    g_dn_est, g_dn_n = _robust_temp_slope(neg["temp_delta"].to_numpy(dtype=float), neg["load_ratio"].to_numpy(dtype=float))
    global_up = _clip_temp_coeff(_blend_temp_coeff(prior, g_up_est, g_up_n, ridge=60.0), season, prior=prior)
    global_down = _clip_temp_coeff(_blend_temp_coeff(prior, g_dn_est, g_dn_n, ridge=60.0), season, prior=prior)

    block_up = np.full(96, global_up, dtype=float)
    block_down = np.full(96, global_down, dtype=float)
    block_up_n = np.zeros(96, dtype=int)
    block_down_n = np.zeros(96, dtype=int)

    for b in range(1, 97):
        blk = hist[hist["time_block"] == b]
        if blk.empty:
            continue
        blk_pos = blk[blk["temp_delta"] > 0]
        blk_neg = blk[blk["temp_delta"] < 0]
        up_est, up_n = _robust_temp_slope(blk_pos["temp_delta"].to_numpy(dtype=float), blk_pos["load_ratio"].to_numpy(dtype=float))
        dn_est, dn_n = _robust_temp_slope(blk_neg["temp_delta"].to_numpy(dtype=float), blk_neg["load_ratio"].to_numpy(dtype=float))
        block_up_n[b - 1] = int(up_n)
        block_down_n[b - 1] = int(dn_n)

        if up_n >= int(min_block_samples):
            block_up[b - 1] = _clip_temp_coeff(_blend_temp_coeff(global_up, up_est, up_n, ridge=18.0), season, prior=prior)
        if dn_n >= int(min_block_samples):
            block_down[b - 1] = _clip_temp_coeff(_blend_temp_coeff(global_down, dn_est, dn_n, ridge=18.0), season, prior=prior)

    out.update({
        "source": "learned_history",
        "global_increase_coeff": float(global_up),
        "global_reduction_coeff": float(global_down),
        "global_increase_samples": int(g_up_n),
        "global_reduction_samples": int(g_dn_n),
        "block_increase_coeffs": block_up,
        "block_reduction_coeffs": block_down,
        "block_increase_samples": block_up_n,
        "block_reduction_samples": block_down_n,
        "diagnostics": {
            "context_mode": context_mode,
            "history_dates": int(len(selected_dates)),
            "valid_rows": int(len(hist)),
        },
    })
    return out


def _fit_trend(blocks: np.ndarray, residuals: np.ndarray, degree: int, eval_blocks: Optional[np.ndarray] = None) -> np.ndarray:
    target = eval_blocks if eval_blocks is not None else blocks
    if len(blocks) < 2:
        return np.zeros_like(target, dtype=float)
    deg = max(1, min(int(degree), 3))
    coeffs = np.polyfit(blocks, residuals, deg)
    return np.polyval(coeffs, target)


def _as_96_vector(values: Any, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    default = np.zeros(_BLOCK_COUNT, dtype=float) if fallback is None else np.asarray(fallback, dtype=float)
    arr = np.asarray(values if values is not None else default, dtype=float).reshape(-1)
    if arr.size == _BLOCK_COUNT:
        return arr
    if arr.size == 0:
        return default.copy()
    if arr.size == 1:
        return np.full(_BLOCK_COUNT, float(arr[0]), dtype=float)
    out = np.zeros(_BLOCK_COUNT, dtype=float)
    n = min(_BLOCK_COUNT, arr.size)
    out[:n] = arr[:n]
    if n < _BLOCK_COUNT:
        out[n:] = out[n - 1]
    return out


def _selection_mask(selection: Optional[Dict[str, Any]]) -> np.ndarray:
    sel = selection or {}
    scope = str(sel.get("scope", "all")).strip().lower()
    mask = np.zeros(_BLOCK_COUNT, dtype=float)
    if scope in ("all", "overall"):
        mask[:] = 1.0
        return mask
    if scope == "range":
        start = int(sel.get("start_block", sel.get("start", 1)) or 1)
        end = int(sel.get("end_block", sel.get("end", _BLOCK_COUNT)) or _BLOCK_COUNT)
        lo = max(1, min(start, end))
        hi = min(_BLOCK_COUNT, max(start, end))
        mask[lo - 1:hi] = 1.0
        return mask
    if scope in ("single", "block"):
        block = int(sel.get("block", sel.get("block_number", 1)) or 1)
        block = int(np.clip(block, 1, _BLOCK_COUNT))
        mask[block - 1] = 1.0
        return mask
    mask[:] = 1.0
    return mask


def _smoothed_selection_mask(mask: np.ndarray) -> np.ndarray:
    if mask.size != _BLOCK_COUNT:
        return mask
    smooth = mask.astype(float).copy()
    for i in range(_BLOCK_COUNT):
        left = mask[i - 1] if i > 0 else mask[i]
        right = mask[i + 1] if i < (_BLOCK_COUNT - 1) else mask[i]
        if (mask[i] == 0.0 and (left == 1.0 or right == 1.0)) or (mask[i] == 1.0 and (left == 0.0 or right == 0.0)):
            smooth[i] = 0.5
    return smooth


def build_block_driver_matrix(
    target_date: str,
    baseline_load: np.ndarray,
    weather_base: np.ndarray,
    daytype_base: np.ndarray,
    holiday_base: np.ndarray,
    manual_base: np.ndarray,
    temperature_base: Optional[np.ndarray] = None,
    humidity_base: Optional[np.ndarray] = None,
    precipitation_base: Optional[np.ndarray] = None,
    apparent_base: Optional[np.ndarray] = None,
    cloud_base: Optional[np.ndarray] = None,
    sun_base: Optional[np.ndarray] = None,
    radiation_base: Optional[np.ndarray] = None,
    wind_base: Optional[np.ndarray] = None,
    sliders: Optional[Dict[str, float]] = None,
    selection: Optional[Dict[str, Any]] = None,
    smooth_edges: bool = False,
) -> Dict[str, Any]:
    baseline_vec = _as_96_vector(baseline_load)
    w_base = _as_96_vector(weather_base)
    d_base = _as_96_vector(daytype_base)
    h_base = _as_96_vector(holiday_base)
    m_base = _as_96_vector(manual_base)
    t_base = _as_96_vector(temperature_base, fallback=np.zeros(_BLOCK_COUNT, dtype=float))
    hu_base = _as_96_vector(humidity_base, fallback=np.zeros(_BLOCK_COUNT, dtype=float))
    p_base = _as_96_vector(precipitation_base, fallback=np.zeros(_BLOCK_COUNT, dtype=float))
    a_base = _as_96_vector(apparent_base, fallback=np.zeros(_BLOCK_COUNT, dtype=float))
    c_base = _as_96_vector(cloud_base, fallback=np.zeros(_BLOCK_COUNT, dtype=float))
    s_base = _as_96_vector(sun_base, fallback=np.zeros(_BLOCK_COUNT, dtype=float))
    r_base = _as_96_vector(radiation_base, fallback=np.zeros(_BLOCK_COUNT, dtype=float))
    wi_base = _as_96_vector(wind_base, fallback=np.zeros(_BLOCK_COUNT, dtype=float))

    if (
        np.allclose(t_base, 0.0)
        and np.allclose(hu_base, 0.0)
        and np.allclose(p_base, 0.0)
        and np.allclose(a_base, 0.0)
        and np.allclose(c_base, 0.0)
        and np.allclose(s_base, 0.0)
        and np.allclose(r_base, 0.0)
        and np.allclose(wi_base, 0.0)
    ):
        # Backward compatibility: when weather is provided as a single base vector.
        t_base = w_base.copy()

    slider_cfg = sliders or {}
    s_w = float(slider_cfg.get("weather", slider_cfg.get("S_w", 1.0)))
    s_d = float(slider_cfg.get("daytype", slider_cfg.get("S_d", 1.0)))
    s_h = float(slider_cfg.get("holiday", slider_cfg.get("S_h", 1.0)))
    s_m = float(slider_cfg.get("manual", slider_cfg.get("S_m", 1.0)))
    s_temp = float(slider_cfg.get("temperature", slider_cfg.get("temp", 1.0)))
    s_hum = float(slider_cfg.get("humidity", 1.0))
    s_prec = float(slider_cfg.get("precipitation", slider_cfg.get("precip", 1.0)))
    s_apparent = float(slider_cfg.get("apparent_temperature", slider_cfg.get("apparent", 1.0)))
    s_cloud = float(slider_cfg.get("cloud_cover", slider_cfg.get("cloud", 1.0)))
    s_sun = float(slider_cfg.get("sunshine_duration", slider_cfg.get("sunshine", slider_cfg.get("sun", 1.0))))
    s_rad = float(slider_cfg.get("direct_radiation", slider_cfg.get("radiation", slider_cfg.get("rad", 1.0))))
    s_wind = float(slider_cfg.get("wind", slider_cfg.get("wind_speed_10m", slider_cfg.get("wind_speed", 1.0))))

    temperature_scaled = t_base * s_w * s_temp
    humidity_scaled = hu_base * s_w * s_hum
    precipitation_scaled = p_base * s_w * s_prec
    apparent_scaled = a_base * s_w * s_apparent
    cloud_scaled = c_base * s_w * s_cloud
    sun_scaled = s_base * s_w * s_sun
    rad_scaled = r_base * s_w * s_rad
    wind_scaled = wi_base * s_w * s_wind
    weather_scaled = (
        temperature_scaled
        + humidity_scaled
        + precipitation_scaled
        + apparent_scaled
        + cloud_scaled
        + sun_scaled
        + rad_scaled
        + wind_scaled
    )
    daytype_scaled = d_base * s_d
    holiday_scaled = h_base * s_h
    manual_scaled = m_base * s_m
    final_impact = np.clip(weather_scaled + daytype_scaled + holiday_scaled + manual_scaled, _IMPACT_MIN, _IMPACT_MAX)

    mask = _selection_mask(selection)
    if smooth_edges:
        mask = _smoothed_selection_mask(mask)

    final_load = baseline_vec * (1.0 + (final_impact * mask))

    matrix = pd.DataFrame({
        "date": [target_date] * _BLOCK_COUNT,
        "block": np.arange(1, _BLOCK_COUNT + 1, dtype=int),
        "weather_base": w_base,
        "temperature_base": t_base,
        "humidity_base": hu_base,
        "precipitation_base": p_base,
        "apparent_base": a_base,
        "cloud_base": c_base,
        "sun_base": s_base,
        "radiation_base": r_base,
        "wind_base": wi_base,
        "daytype_base": d_base,
        "holiday_base": h_base,
        "manual_base": m_base,
        "weather_scaled": weather_scaled,
        "temperature_scaled": temperature_scaled,
        "humidity_scaled": humidity_scaled,
        "precipitation_scaled": precipitation_scaled,
        "apparent_scaled": apparent_scaled,
        "cloud_scaled": cloud_scaled,
        "sun_scaled": sun_scaled,
        "rad_scaled": rad_scaled,
        "wind_scaled": wind_scaled,
        "daytype_scaled": daytype_scaled,
        "holiday_scaled": holiday_scaled,
        "manual_scaled": manual_scaled,
        "final_impact": final_impact,
    })

    return {
        "matrix_df": matrix,
        "selection_mask": mask,
        "final_load": final_load,
    }


def _resolve_wind_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("wind_speed_10m", "wind_speed", "wind", "windspeed", "wind_mps"):
        if col in df.columns:
            return col
    return None


def _resolve_holiday_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("is_holiday", "holiday", "holiday_flag", "holiday_ind"):
        if col in df.columns:
            return col
    return None


def _safe_ratio_delta(numer: float, denom: float) -> float:
    if not np.isfinite(numer) or not np.isfinite(denom) or abs(denom) <= 1e-9:
        return 0.0
    return float((numer - denom) / denom)


def compute_block_driver_weights(history_df: pd.DataFrame, target_date: Optional[str] = None) -> pd.DataFrame:
    required = {"date", "time_block", "total_drawal", "temperature", "humidity", "precipitation"}
    if history_df is None or history_df.empty:
        raise ValueError("history_df is empty")
    if not required.issubset(set(history_df.columns)):
        missing = sorted(list(required.difference(set(history_df.columns))))
        raise ValueError(f"history_df missing required columns: {missing}")

    hist = history_df.copy()
    hist["date"] = hist["date"].astype(str)
    if target_date:
        hist = hist[hist["date"] < str(target_date)]
    if hist.empty:
        hist = history_df.copy()
        hist["date"] = hist["date"].astype(str)

    hist["time_block"] = pd.to_numeric(hist["time_block"], errors="coerce")
    hist = hist[hist["time_block"].between(1, _BLOCK_COUNT)]
    if hist.empty:
        raise ValueError("No valid block rows in history_df")

    hist["total_drawal"] = pd.to_numeric(hist["total_drawal"], errors="coerce")
    hist["temperature"] = pd.to_numeric(hist["temperature"], errors="coerce")
    hist["humidity"] = pd.to_numeric(hist["humidity"], errors="coerce")
    hist["precipitation"] = pd.to_numeric(hist["precipitation"], errors="coerce")

    wind_col = _resolve_wind_column(hist)
    if wind_col is None:
        import logging
        logging.getLogger(__name__).warning(
            "No wind column found; using zero proxy – wind driver weights will be zero."
        )
        hist["wind_proxy"] = 0.0
        wind_col = "wind_proxy"
    hist[wind_col] = pd.to_numeric(hist[wind_col], errors="coerce")

    dt = pd.to_datetime(hist["date"], errors="coerce")
    hist["is_weekend"] = (dt.dt.weekday >= 5).astype(float)

    holiday_col = _resolve_holiday_column(hist)
    if holiday_col is not None:
        holiday_series = pd.to_numeric(hist[holiday_col], errors="coerce")
        hist["is_holiday_flag"] = (holiday_series.fillna(0.0) > 0.0).astype(float)
    elif "day_type" in hist.columns:
        day_type = hist["day_type"].astype(str).str.lower()
        hist["is_holiday_flag"] = day_type.str.contains("holiday").astype(float)
    else:
        hist["is_holiday_flag"] = 0.0

    global_medians = {
        "temperature": float(hist["temperature"].median(skipna=True)) if hist["temperature"].notna().any() else 0.0,
        "humidity": float(hist["humidity"].median(skipna=True)) if hist["humidity"].notna().any() else 0.0,
        "precipitation": float(hist["precipitation"].median(skipna=True)) if hist["precipitation"].notna().any() else 0.0,
        wind_col: float(hist[wind_col].median(skipna=True)) if hist[wind_col].notna().any() else 0.0,
    }

    grouped = {int(k): v for k, v in hist.groupby(hist["time_block"].astype(int))}
    rows: List[Dict[str, Any]] = []
    feature_cols = ["temperature", "humidity", "precipitation", wind_col]

    for block in range(1, _BLOCK_COUNT + 1):
        blk = grouped.get(block, pd.DataFrame(columns=hist.columns)).copy()
        blk["total_drawal"] = pd.to_numeric(blk.get("total_drawal"), errors="coerce")
        blk = blk[np.isfinite(blk["total_drawal"])]

        avg_load = float(blk["total_drawal"].mean()) if not blk.empty else 0.0
        denom = max(abs(avg_load), 1e-6)
        beta_temp = 0.0
        beta_hum = 0.0
        beta_rain = 0.0
        beta_wind = 0.0

        if len(blk) >= 8 and float(np.std(blk["total_drawal"].to_numpy(dtype=float))) > 1e-9:
            x_df = blk[feature_cols].copy()
            for col in feature_cols:
                x_df[col] = pd.to_numeric(x_df[col], errors="coerce")
                med = float(x_df[col].median(skipna=True)) if x_df[col].notna().any() else global_medians.get(col, 0.0)
                if not np.isfinite(med):
                    med = global_medians.get(col, 0.0)
                x_df[col] = x_df[col].fillna(med)

            x_mat = x_df.to_numpy(dtype=float)
            y_vec = blk["total_drawal"].to_numpy(dtype=float)
            if np.all(np.isfinite(x_mat)) and np.all(np.isfinite(y_vec)) and x_mat.shape[0] >= 8:
                try:
                    model = LinearRegression()
                    model.fit(x_mat, y_vec)
                    coefs = model.coef_.reshape(-1)
                    if coefs.size == 4:
                        beta_temp = float(coefs[0])
                        beta_hum = float(coefs[1])
                        beta_rain = float(coefs[2])
                        beta_wind = float(coefs[3])
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).debug(
                        "Block %d LinearRegression failed: %s", block, exc
                    )

        weekday_vals = blk.loc[blk["is_weekend"] < 0.5, "total_drawal"].to_numpy(dtype=float)
        weekend_vals = blk.loc[blk["is_weekend"] >= 0.5, "total_drawal"].to_numpy(dtype=float)
        weekday_avg = float(np.mean(weekday_vals)) if weekday_vals.size else np.nan
        weekend_avg = float(np.mean(weekend_vals)) if weekend_vals.size else np.nan
        if np.isfinite(weekday_avg) and abs(weekday_avg) > 1e-9 and np.isfinite(weekend_avg):
            daytype_weight = _safe_ratio_delta(weekend_avg, weekday_avg)
        else:
            daytype_weight = 0.0

        holiday_vals = blk.loc[blk["is_holiday_flag"] >= 0.5, "total_drawal"].to_numpy(dtype=float)
        normal_vals = blk.loc[blk["is_holiday_flag"] < 0.5, "total_drawal"].to_numpy(dtype=float)
        holiday_avg = float(np.mean(holiday_vals)) if holiday_vals.size else np.nan
        normal_avg = float(np.mean(normal_vals)) if normal_vals.size else np.nan
        if np.isfinite(holiday_avg) and np.isfinite(normal_avg) and abs(normal_avg) > 1e-9:
            holiday_weight = _safe_ratio_delta(holiday_avg, normal_avg)
        else:
            holiday_weight = 0.0

        rows.append({
            "block": block,
            "temp_weight": float(beta_temp / denom),
            "humidity_weight": float(beta_hum / denom),
            "rain_weight": float(beta_rain / denom),
            "wind_weight": float(beta_wind / denom),
            "daytype_weight": float(daytype_weight),
            "holiday_weight": float(holiday_weight),
            "avg_load": float(avg_load),
            "samples": int(len(blk)),
        })

    return pd.DataFrame(rows)


def compute_block_driver_weights_elastic_net(
    history_df: pd.DataFrame,
    target_date: Optional[str] = None,
    l1_ratio_grid: Optional[List[float]] = None,
    alpha_grid: Optional[np.ndarray] = None,
    min_samples: int = 14,
    max_iter: int = 10000,
    random_state: int = 42,
    add_temp_squared: bool = False,
    add_temp_humidity_interaction: bool = False,
    season_segment: bool = False,
    smooth_window: int = 0,
    apply_confidence_shrinkage: bool = True,
    shrink_power: float = 1.2,
) -> pd.DataFrame:
    required = {"date", "time_block", "total_drawal", "temperature", "humidity", "precipitation"}
    if history_df is None or history_df.empty:
        raise ValueError("history_df is empty")
    if not required.issubset(set(history_df.columns)):
        missing = sorted(list(required.difference(set(history_df.columns))))
        raise ValueError(f"history_df missing required columns: {missing}")

    hist = history_df.copy()
    hist["date"] = hist["date"].astype(str)
    if target_date:
        hist = hist[hist["date"] < str(target_date)]
    if hist.empty:
        hist = history_df.copy()
        hist["date"] = hist["date"].astype(str)

    hist["time_block"] = pd.to_numeric(hist["time_block"], errors="coerce")
    hist = hist[hist["time_block"].between(1, _BLOCK_COUNT)]
    if hist.empty:
        raise ValueError("No valid block rows in history_df")

    hist["total_drawal"] = pd.to_numeric(hist["total_drawal"], errors="coerce")
    hist["temperature"] = pd.to_numeric(hist["temperature"], errors="coerce")
    hist["humidity"] = pd.to_numeric(hist["humidity"], errors="coerce")
    hist["precipitation"] = pd.to_numeric(hist["precipitation"], errors="coerce")

    wind_col = _resolve_wind_column(hist)
    if wind_col is None:
        import logging
        logging.getLogger(__name__).warning(
            "No wind column found; using zero proxy – wind driver weights will be zero."
        )
        hist["wind_proxy"] = 0.0
        wind_col = "wind_proxy"
    hist[wind_col] = pd.to_numeric(hist[wind_col], errors="coerce")

    dt = pd.to_datetime(hist["date"], errors="coerce")
    hist["is_weekend"] = (dt.dt.weekday >= 5).astype(float)

    holiday_col = _resolve_holiday_column(hist)
    if holiday_col is not None:
        holiday_series = pd.to_numeric(hist[holiday_col], errors="coerce")
        hist["is_holiday_flag"] = (holiday_series.fillna(0.0) > 0.0).astype(float)
    elif "day_type" in hist.columns:
        day_type = hist["day_type"].astype(str).str.lower()
        hist["is_holiday_flag"] = day_type.str.contains("holiday").astype(float)
    else:
        hist["is_holiday_flag"] = 0.0

    if season_segment:
        season_col = pd.to_datetime(hist["date"], errors="coerce").dt.month.map(SEASON_BY_MONTH)
        hist["season_seg"] = season_col.fillna("unknown")
        if target_date:
            target_ts = pd.to_datetime(target_date, errors="coerce")
            if not pd.isna(target_ts):
                target_season = SEASON_BY_MONTH.get(int(target_ts.month), None)
                if target_season is not None:
                    seg_hist = hist[hist["season_seg"] == target_season].copy()
                    if len(seg_hist) >= max(96, int(min_samples * 8)):
                        hist = seg_hist

    global_medians = {
        "temperature": float(hist["temperature"].median(skipna=True)) if hist["temperature"].notna().any() else 0.0,
        "humidity": float(hist["humidity"].median(skipna=True)) if hist["humidity"].notna().any() else 0.0,
        "precipitation": float(hist["precipitation"].median(skipna=True)) if hist["precipitation"].notna().any() else 0.0,
        wind_col: float(hist[wind_col].median(skipna=True)) if hist[wind_col].notna().any() else 0.0,
    }

    ratio_grid = l1_ratio_grid if l1_ratio_grid else [0.15, 0.35, 0.5, 0.7, 0.9]
    alphas = np.asarray(alpha_grid, dtype=float) if alpha_grid is not None else np.logspace(-4, 0.8, 36)
    alphas = np.sort(np.unique(alphas[(alphas > 0) & np.isfinite(alphas)]))
    if alphas.size == 0:
        alphas = np.logspace(-4, 0.8, 36)

    grouped = {int(k): v for k, v in hist.groupby(hist["time_block"].astype(int))}
    rows: List[Dict[str, Any]] = []

    for block in range(1, _BLOCK_COUNT + 1):
        blk = grouped.get(block, pd.DataFrame(columns=hist.columns)).copy()
        blk["total_drawal"] = pd.to_numeric(blk.get("total_drawal"), errors="coerce")
        blk = blk[np.isfinite(blk["total_drawal"])]

        avg_load = float(blk["total_drawal"].mean()) if not blk.empty else 0.0
        denom = max(abs(avg_load), 1e-6)

        feature_df = pd.DataFrame({
            "temp": pd.to_numeric(blk.get("temperature"), errors="coerce"),
            "humidity": pd.to_numeric(blk.get("humidity"), errors="coerce"),
            "rain": pd.to_numeric(blk.get("precipitation"), errors="coerce"),
            "wind": pd.to_numeric(blk.get(wind_col), errors="coerce"),
            "daytype": pd.to_numeric(blk.get("is_weekend"), errors="coerce"),
            "holiday": pd.to_numeric(blk.get("is_holiday_flag"), errors="coerce"),
        })
        for col in ("temp", "humidity", "rain", "wind", "daytype", "holiday"):
            med = float(feature_df[col].median(skipna=True)) if feature_df[col].notna().any() else 0.0
            if col == "temp" and not np.isfinite(med):
                med = global_medians["temperature"]
            elif col == "humidity" and not np.isfinite(med):
                med = global_medians["humidity"]
            elif col == "rain" and not np.isfinite(med):
                med = global_medians["precipitation"]
            elif col == "wind" and not np.isfinite(med):
                med = global_medians[wind_col]
            if not np.isfinite(med):
                med = 0.0
            feature_df[col] = feature_df[col].fillna(med)

        if add_temp_squared:
            feature_df["temp_sq"] = np.square(feature_df["temp"].to_numpy(dtype=float))
        if add_temp_humidity_interaction:
            feature_df["temp_x_humidity"] = feature_df["temp"].to_numpy(dtype=float) * feature_df["humidity"].to_numpy(dtype=float)

        y_vec = blk["total_drawal"].to_numpy(dtype=float)

        coeffs = {
            "temp": 0.0,
            "humidity": 0.0,
            "rain": 0.0,
            "wind": 0.0,
            "daytype": 0.0,
            "holiday": 0.0,
        }
        model_r2 = 0.0
        alpha_opt = np.nan
        l1_ratio_opt = np.nan

        if len(feature_df) >= int(min_samples) and float(np.std(y_vec)) > 1e-9:
            x_mat = feature_df.to_numpy(dtype=float)
            if np.all(np.isfinite(x_mat)) and np.all(np.isfinite(y_vec)):
                cv_folds = int(np.clip(len(feature_df) // 6, 3, 5))
                if len(feature_df) <= cv_folds:
                    cv_folds = max(2, len(feature_df) - 1)
                if cv_folds >= 2:
                    try:
                        model = Pipeline(steps=[
                            ("scale", StandardScaler(with_mean=True, with_std=True)),
                            (
                                "enet",
                                ElasticNetCV(
                                    l1_ratio=ratio_grid,
                                    alphas=alphas,
                                    fit_intercept=True,
                                    cv=cv_folds,
                                    random_state=int(random_state),
                                    max_iter=int(max_iter),
                                    selection="cyclic",
                                ),
                            ),
                        ])
                        model.fit(x_mat, y_vec)

                        scaler = model.named_steps["scale"]
                        enet = model.named_steps["enet"]
                        alpha_opt = float(getattr(enet, "alpha_", np.nan))
                        l1_ratio_opt = float(getattr(enet, "l1_ratio_", np.nan))
                        scaled_coef = np.asarray(enet.coef_, dtype=float).reshape(-1)
                        scale = np.asarray(scaler.scale_, dtype=float).reshape(-1)
                        if scaled_coef.size == scale.size and scaled_coef.size == feature_df.shape[1]:
                            raw_coef = np.divide(
                                scaled_coef,
                                np.where(np.abs(scale) > 1e-12, scale, 1.0),
                                out=np.zeros_like(scaled_coef, dtype=float),
                                where=np.abs(scale) > 1e-12,
                            )
                            for i, col in enumerate(feature_df.columns.tolist()):
                                if col in coeffs:
                                    coeffs[col] = float(raw_coef[i])

                        y_hat = model.predict(x_mat)
                        ss_res = float(np.sum(np.square(y_vec - y_hat)))
                        ss_tot = float(np.sum(np.square(y_vec - np.mean(y_vec))))
                        if ss_tot > 1e-9:
                            model_r2 = float(1.0 - (ss_res / ss_tot))
                    except Exception as exc:
                        import logging
                        logging.getLogger(__name__).debug(
                            "Block %d ElasticNetCV failed: %s", block, exc
                        )

        sample_conf = float(np.clip(len(feature_df) / max(float(min_samples) * 3.0, 1.0), 0.0, 1.0))
        fit_conf = float(np.clip(model_r2, 0.0, 1.0))
        weight_conf = float(np.clip((0.6 * sample_conf) + (0.4 * fit_conf), 0.0, 1.0))
        shrink = float(weight_conf ** max(float(shrink_power), 0.0)) if apply_confidence_shrinkage else 1.0

        temp_w_raw = float(coeffs["temp"] / denom)
        hum_w_raw = float(coeffs["humidity"] / denom)
        rain_w_raw = float(coeffs["rain"] / denom)
        wind_w_raw = float(coeffs["wind"] / denom)
        daytype_w_raw = float(coeffs["daytype"] / denom)
        holiday_w_raw = float(coeffs["holiday"] / denom)

        rows.append({
            "block": block,
            "temp_weight_raw": temp_w_raw,
            "humidity_weight_raw": hum_w_raw,
            "rain_weight_raw": rain_w_raw,
            "wind_weight_raw": wind_w_raw,
            "daytype_weight_raw": daytype_w_raw,
            "holiday_weight_raw": holiday_w_raw,
            "temp_weight": float(temp_w_raw * shrink),
            "humidity_weight": float(hum_w_raw * shrink),
            "rain_weight": float(rain_w_raw * shrink),
            "wind_weight": float(wind_w_raw * shrink),
            "daytype_weight": float(daytype_w_raw * shrink),
            "holiday_weight": float(holiday_w_raw * shrink),
            "weight_confidence": weight_conf,
            "sample_confidence": sample_conf,
            "fit_confidence": fit_conf,
            "samples": int(len(feature_df)),
            "model_r2": float(model_r2),
            "alpha_opt": float(alpha_opt) if np.isfinite(alpha_opt) else np.nan,
            "l1_ratio_opt": float(l1_ratio_opt) if np.isfinite(l1_ratio_opt) else np.nan,
        })

    out = pd.DataFrame(rows)
    if int(smooth_window) and int(smooth_window) > 1:
        w = int(max(1, smooth_window))
        for col in ("temp_weight", "humidity_weight", "rain_weight", "wind_weight", "daytype_weight", "holiday_weight"):
            out[col] = out[col].rolling(window=w, center=True, min_periods=1).mean()
        if "weight_confidence" in out.columns:
            out["weight_confidence"] = out["weight_confidence"].rolling(window=w, center=True, min_periods=1).mean()

    return out


def run_weather_impact_elastic_net_engine(
    history_df: pd.DataFrame,
    target_date: str,
    base_load: Optional[np.ndarray] = None,
    momentum_lambda: float = 0.45,
    min_samples: int = 14,
    max_iter: int = 10000,
    random_state: int = 42,
    l1_ratio_grid: Optional[List[float]] = None,
    alpha_grid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Blockwise weather impact engine:
    1) Train Elastic Net with features temp/humidity/rain/lag_1_load/lag_96_load
    2) Convert weather coefficients to elasticity weights
    3) Compute weather deltas (today - yesterday)
    4) Weather impact = sum(weight * delta)
    5) DoD load delta = Load_t-1 - Load_t-2
    6) Momentum impact = lambda * DoD
    7) Simulated load = Base + Weather Impact + Momentum Impact
    """
    required = {"date", "time_block", "total_drawal", "temperature", "humidity", "precipitation"}
    optional_weather = ["apparent_temperature", "cloud_cover", "sunshine_duration", "direct_radiation", "wind_speed_10m"]
    if history_df is None or history_df.empty:
        raise ValueError("history_df is empty")
    if not required.issubset(set(history_df.columns)):
        missing = sorted(list(required.difference(set(history_df.columns))))
        raise ValueError(f"history_df missing required columns: {missing}")

    hist = history_df.copy()
    # Create missing optional weather columns as 0 so downstream code doesn't break
    for col in optional_weather:
        if col not in hist.columns:
            hist[col] = 0.0
    hist["date"] = hist["date"].astype(str)
    hist["time_block"] = pd.to_numeric(hist["time_block"], errors="coerce")
    hist["total_drawal"] = pd.to_numeric(hist["total_drawal"], errors="coerce")

    for col in ["temperature", "humidity", "precipitation"] + optional_weather:
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
    hist = hist[hist["time_block"].between(1, _BLOCK_COUNT)]
    hist = hist.dropna(subset=["date", "time_block", "total_drawal"])
    if hist.empty:
        raise ValueError("No valid rows in history_df after cleaning")

    hist["time_block"] = hist["time_block"].astype(int)
    hist = hist.sort_values(["date", "time_block"]).reset_index(drop=True)

    dates = sorted(hist["date"].dropna().astype(str).unique().tolist())
    if not dates:
        raise ValueError("No dates available for weather impact engine")
    resolved_date = str(target_date) if str(target_date) in dates else dates[-1]
    target_idx = dates.index(resolved_date)
    prev_date = dates[target_idx - 1] if target_idx >= 1 else None
    prev2_date = dates[target_idx - 2] if target_idx >= 2 else None

    # Sequence lags on full 15-min chronology.
    hist["lag_1_load"] = pd.to_numeric(hist["total_drawal"], errors="coerce").shift(1)
    hist["lag_96_load"] = pd.to_numeric(hist["total_drawal"], errors="coerce").shift(_BLOCK_COUNT)
    train = hist[hist["date"] < resolved_date].copy()

    ratio_grid = l1_ratio_grid if l1_ratio_grid else [0.15, 0.35, 0.5, 0.7, 0.9]
    alphas = np.asarray(alpha_grid, dtype=float) if alpha_grid is not None else np.logspace(-4, 0.8, 36)
    alphas = np.sort(np.unique(alphas[(alphas > 0) & np.isfinite(alphas)]))
    if alphas.size == 0:
        alphas = np.logspace(-4, 0.8, 36)

    global_medians = {
        "temperature": float(hist["temperature"].median(skipna=True)) if hist["temperature"].notna().any() else 0.0,
        "humidity": float(hist["humidity"].median(skipna=True)) if hist["humidity"].notna().any() else 0.0,
        "precipitation": float(hist["precipitation"].median(skipna=True)) if hist["precipitation"].notna().any() else 0.0,
        "apparent_temperature": float(hist["apparent_temperature"].median(skipna=True)) if hist["apparent_temperature"].notna().any() else 0.0,
        "cloud_cover": float(hist["cloud_cover"].median(skipna=True)) if hist["cloud_cover"].notna().any() else 0.0,
        "sunshine_duration": float(hist["sunshine_duration"].median(skipna=True)) if hist["sunshine_duration"].notna().any() else 0.0,
        "direct_radiation": float(hist["direct_radiation"].median(skipna=True)) if hist["direct_radiation"].notna().any() else 0.0,
        "wind_speed_10m": float(hist["wind_speed_10m"].median(skipna=True)) if hist["wind_speed_10m"].notna().any() else 0.0,
        "lag_1_load": float(hist["lag_1_load"].median(skipna=True)) if hist["lag_1_load"].notna().any() else 0.0,
        "lag_96_load": float(hist["lag_96_load"].median(skipna=True)) if hist["lag_96_load"].notna().any() else 0.0,
    }

    rows: List[Dict[str, Any]] = []
    grouped = {int(k): v for k, v in train.groupby(train["time_block"].astype(int))} if not train.empty else {}
    feature_cols = [
        "temperature", "humidity", "precipitation", 
        "apparent_temperature", "cloud_cover", "sunshine_duration", 
        "direct_radiation", "wind_speed_10m",
        "lag_1_load", "lag_96_load"
    ]

    for block in range(1, _BLOCK_COUNT + 1):
        blk = grouped.get(block, pd.DataFrame(columns=train.columns)).copy()
        blk["total_drawal"] = pd.to_numeric(blk.get("total_drawal"), errors="coerce")
        blk = blk[np.isfinite(blk["total_drawal"])]

        avg_load = float(blk["total_drawal"].mean()) if not blk.empty else 0.0
        denom = max(abs(avg_load), 1e-6)

        coeffs = {col: 0.0 for col in feature_cols}
        model_r2 = 0.0
        alpha_opt = np.nan
        l1_ratio_opt = np.nan

        if len(blk) >= int(min_samples) and float(np.std(blk["total_drawal"].to_numpy(dtype=float))) > 1e-9:
            x_df = blk[feature_cols].copy()
            for col in feature_cols:
                x_df[col] = pd.to_numeric(x_df[col], errors="coerce")
                med = float(x_df[col].median(skipna=True)) if x_df[col].notna().any() else global_medians.get(col, 0.0)
                if not np.isfinite(med):
                    med = global_medians.get(col, 0.0)
                x_df[col] = x_df[col].fillna(med)

            x_mat = x_df.to_numpy(dtype=float)
            y_vec = blk["total_drawal"].to_numpy(dtype=float)
            if np.all(np.isfinite(x_mat)) and np.all(np.isfinite(y_vec)):
                cv_folds = int(np.clip(len(x_df) // 6, 3, 5))
                if len(x_df) <= cv_folds:
                    cv_folds = max(2, len(x_df) - 1)
                if cv_folds >= 2:
                    try:
                        model = Pipeline(steps=[
                            ("scale", StandardScaler(with_mean=True, with_std=True)),
                            (
                                "enet",
                                ElasticNetCV(
                                    l1_ratio=ratio_grid,
                                    alphas=alphas,
                                    fit_intercept=True,
                                    cv=cv_folds,
                                    random_state=int(random_state),
                                    max_iter=int(max_iter),
                                    selection="cyclic",
                                ),
                            ),
                        ])
                        model.fit(x_mat, y_vec)

                        scaler = model.named_steps["scale"]
                        enet = model.named_steps["enet"]
                        alpha_opt = float(getattr(enet, "alpha_", np.nan))
                        l1_ratio_opt = float(getattr(enet, "l1_ratio_", np.nan))
                        scaled_coef = np.asarray(enet.coef_, dtype=float).reshape(-1)
                        scale = np.asarray(scaler.scale_, dtype=float).reshape(-1)
                        if scaled_coef.size == scale.size and scaled_coef.size == len(feature_cols):
                            raw_coef = np.divide(
                                scaled_coef,
                                np.where(np.abs(scale) > 1e-12, scale, 1.0),
                                out=np.zeros_like(scaled_coef, dtype=float),
                                where=np.abs(scale) > 1e-12,
                            )
                            for i, col in enumerate(feature_cols):
                                coeffs[col] = float(raw_coef[i])

                        y_hat = model.predict(x_mat)
                        ss_res = float(np.sum(np.square(y_vec - y_hat)))
                        ss_tot = float(np.sum(np.square(y_vec - np.mean(y_vec))))
                        if ss_tot > 1e-9:
                            model_r2 = float(1.0 - (ss_res / ss_tot))
                    except Exception:
                        pass

        sample_conf = float(np.clip(len(blk) / max(float(min_samples) * 3.0, 1.0), 0.0, 1.0))
        fit_conf = float(np.clip(model_r2, 0.0, 1.0))
        weight_conf = float(np.clip((0.6 * sample_conf) + (0.4 * fit_conf), 0.0, 1.0))

        res = {
            "block": block,
            "samples": int(len(blk)),
            "model_r2": float(model_r2),
            "alpha_opt": float(alpha_opt) if np.isfinite(alpha_opt) else np.nan,
            "l1_ratio_opt": float(l1_ratio_opt) if np.isfinite(l1_ratio_opt) else np.nan,
            "weight_confidence": float(weight_conf),
        }
        for col in feature_cols:
            res[f"{col}_weight"] = float(coeffs[col] / denom)
        rows.append(res)

    weight_df = pd.DataFrame(rows)
    if weight_df.empty:
        weight_df = pd.DataFrame({
            "block": np.arange(1, _BLOCK_COUNT + 1, dtype=int),
            "temp_weight": np.zeros(_BLOCK_COUNT, dtype=float),
            "humidity_weight": np.zeros(_BLOCK_COUNT, dtype=float),
            "rain_weight": np.zeros(_BLOCK_COUNT, dtype=float),
            "lag_1_weight": np.zeros(_BLOCK_COUNT, dtype=float),
            "lag_96_weight": np.zeros(_BLOCK_COUNT, dtype=float),
            "samples": np.zeros(_BLOCK_COUNT, dtype=int),
            "model_r2": np.zeros(_BLOCK_COUNT, dtype=float),
            "alpha_opt": np.full(_BLOCK_COUNT, np.nan, dtype=float),
            "l1_ratio_opt": np.full(_BLOCK_COUNT, np.nan, dtype=float),
            "weight_confidence": np.full(_BLOCK_COUNT, 0.0, dtype=float),
        })

    def _day_vector(date_value: Optional[str], col: str, fill: float = 0.0) -> np.ndarray:
        if not date_value:
            return np.full(_BLOCK_COUNT, float(fill), dtype=float)
        day = hist[hist["date"].astype(str) == str(date_value)][["time_block", col]].copy()
        if day.empty:
            return np.full(_BLOCK_COUNT, float(fill), dtype=float)
        day["time_block"] = pd.to_numeric(day["time_block"], errors="coerce")
        day[col] = pd.to_numeric(day[col], errors="coerce")
        day = day.dropna(subset=["time_block"]).sort_values("time_block")
        ser = day.set_index(day["time_block"].astype(int))[col].reindex(range(1, _BLOCK_COUNT + 1))
        if ser.notna().any():
            ser = ser.ffill().bfill().fillna(float(fill))
        else:
            ser = pd.Series(np.full(_BLOCK_COUNT, float(fill), dtype=float), index=range(1, _BLOCK_COUNT + 1))
        return ser.to_numpy(dtype=float)

    today_temp = _day_vector(resolved_date, "temperature", fill=global_medians["temperature"])
    yday_temp = _day_vector(prev_date, "temperature", fill=global_medians["temperature"])
    today_hum = _day_vector(resolved_date, "humidity", fill=global_medians["humidity"])
    yday_hum = _day_vector(prev_date, "humidity", fill=global_medians["humidity"])
    today_rain = _day_vector(resolved_date, "precipitation", fill=global_medians["precipitation"])
    yday_rain = _day_vector(prev_date, "precipitation", fill=global_medians["precipitation"])
    
    today_apparent = _day_vector(resolved_date, "apparent_temperature", fill=global_medians["apparent_temperature"])
    yday_apparent = _day_vector(prev_date, "apparent_temperature", fill=global_medians["apparent_temperature"])
    today_cloud = _day_vector(resolved_date, "cloud_cover", fill=global_medians["cloud_cover"])
    yday_cloud = _day_vector(prev_date, "cloud_cover", fill=global_medians["cloud_cover"])
    today_sun = _day_vector(resolved_date, "sunshine_duration", fill=global_medians["sunshine_duration"])
    yday_sun = _day_vector(prev_date, "sunshine_duration", fill=global_medians["sunshine_duration"])
    today_rad = _day_vector(resolved_date, "direct_radiation", fill=global_medians["direct_radiation"])
    yday_rad = _day_vector(prev_date, "direct_radiation", fill=global_medians["direct_radiation"])
    today_wind = _day_vector(resolved_date, "wind_speed_10m", fill=global_medians["wind_speed_10m"])
    yday_wind = _day_vector(prev_date, "wind_speed_10m", fill=global_medians["wind_speed_10m"])

    yday_load = _day_vector(prev_date, "total_drawal", fill=0.0)
    yday2_load = _day_vector(prev2_date, "total_drawal", fill=0.0)

    # Baseline anomaly guard: if yesterday's total load is an outlier vs its
    # same-weekday rolling window, fall back to the most-recent clean same-weekday
    # day to avoid corrupting both the baseline and the momentum term.
    if prev_date:
        try:
            prev_dt = pd.to_datetime(prev_date)
            prev_weekday = prev_dt.weekday()
            # Collect daily totals for same weekday in the training window (up to 8 weeks back)
            _same_wd_totals: list[float] = []
            for _d in reversed(dates[:target_idx - 1]):
                _dt = pd.to_datetime(_d)
                if _dt.weekday() == prev_weekday:
                    _vec = _day_vector(_d, "total_drawal", fill=0.0)
                    _tot = float(_vec.sum())
                    if _tot > 0:
                        _same_wd_totals.append(_tot)
                if len(_same_wd_totals) >= 8:
                    break
            if len(_same_wd_totals) >= 3:
                _sw_arr = np.asarray(_same_wd_totals, dtype=float)
                _sw_mean = float(_sw_arr.mean())
                _sw_std = float(_sw_arr.std())
                _yday_total = float(yday_load.sum())
                if _sw_std > 0 and abs(_yday_total - _sw_mean) > 2.0 * _sw_std:
                    # Yesterday is anomalous — find the most recent clean same-weekday fallback
                    logger.warning(
                        "Yesterday (%s) total load %.0f MW is outside 2σ band "
                        "[%.0f ± %.0f MW] — looking for clean same-weekday fallback.",
                        prev_date, _yday_total, _sw_mean, _sw_std,
                    )
                    for _d in reversed(dates[:target_idx - 1]):
                        _dt = pd.to_datetime(_d)
                        if _dt.weekday() == prev_weekday:
                            _vec = _day_vector(_d, "total_drawal", fill=0.0)
                            _tot = float(_vec.sum())
                            if _tot > 0 and abs(_tot - _sw_mean) <= 2.0 * _sw_std:
                                logger.warning("Using %s as clean baseline day instead of %s.", _d, prev_date)
                                yday_load = _vec
                                break
        except Exception as _e:
            logger.debug("Baseline anomaly guard skipped: %s", _e)

    temp_delta = today_temp - yday_temp
    hum_delta = today_hum - yday_hum
    rain_delta = today_rain - yday_rain
    apparent_delta = today_apparent - yday_apparent
    cloud_delta = today_cloud - yday_cloud
    sun_delta = today_sun - yday_sun
    rad_delta = today_rad - yday_rad
    wind_delta = today_wind - yday_wind
    
    dod_load_delta = yday_load - yday2_load if prev2_date else np.zeros(_BLOCK_COUNT, dtype=float)

    ordered = weight_df.sort_values("block")
    temp_w = _as_96_vector(ordered["temperature_weight"].to_numpy(dtype=float))
    hum_w = _as_96_vector(ordered["humidity_weight"].to_numpy(dtype=float))
    rain_w = _as_96_vector(ordered["precipitation_weight"].to_numpy(dtype=float))
    apparent_w = _as_96_vector(ordered["apparent_temperature_weight"].to_numpy(dtype=float))
    cloud_w = _as_96_vector(ordered["cloud_cover_weight"].to_numpy(dtype=float))
    sun_w = _as_96_vector(ordered["sunshine_duration_weight"].to_numpy(dtype=float))
    rad_w = _as_96_vector(ordered["direct_radiation_weight"].to_numpy(dtype=float))
    wind_w = _as_96_vector(ordered["wind_speed_10m_weight"].to_numpy(dtype=float))
    
    conf_w = _as_96_vector(ordered["weight_confidence"].to_numpy(dtype=float))

    base_vec = _as_96_vector(base_load if base_load is not None else yday_load)

    weather_impact_pct = (
        (temp_delta * temp_w) + 
        (hum_delta * hum_w) + 
        (rain_delta * rain_w) +
        (apparent_delta * apparent_w) +
        (cloud_delta * cloud_w) +
        (sun_delta * sun_w) +
        (rad_delta * rad_w) +
        (wind_delta * wind_w)
    )
    weather_impact_mw = base_vec * weather_impact_pct

    # Adaptive momentum lambda: on high-volatility days (large yesterday deviation)
    # reduce lambda so we don't amplify an anomalous baseline.
    _HIGH_VOLATILITY_MW = 150.0
    deviation = float(np.abs(dod_load_delta).mean())
    if deviation > _HIGH_VOLATILITY_MW:
        lam = float(np.clip(float(momentum_lambda) * 0.4, 0.15, 0.25))
        logger.debug(
            "High dod_load_delta deviation %.1f MW > %.1f threshold — reducing lambda to %.3f",
            deviation, _HIGH_VOLATILITY_MW, lam,
        )
    else:
        lam = float(np.clip(float(momentum_lambda), 0.3, 0.6))
    momentum_impact_mw = lam * dod_load_delta
    momentum_base = np.divide(
        momentum_impact_mw,
        np.maximum(np.abs(base_vec), 1e-6),
        out=np.zeros_like(momentum_impact_mw, dtype=float),
        where=np.abs(base_vec) > 1e-6,
    )
    simulated_load = base_vec + weather_impact_mw + momentum_impact_mw

    block_results_df = pd.DataFrame({
        "date": [resolved_date] * _BLOCK_COUNT,
        "block": np.arange(1, _BLOCK_COUNT + 1, dtype=int),
        "base_load": base_vec,
        "temp_delta": temp_delta,
        "humidity_delta": hum_delta,
        "rain_delta": rain_delta,
        "apparent_delta": apparent_delta,
        "cloud_delta": cloud_delta,
        "sun_delta": sun_delta,
        "rad_delta": rad_delta,
        "wind_delta": wind_delta,
        "temp_weight": temp_w,
        "humidity_weight": hum_w,
        "rain_weight": rain_w,
        "apparent_weight": apparent_w,
        "cloud_weight": cloud_w,
        "sun_weight": sun_w,
        "rad_weight": rad_w,
        "wind_weight": wind_w,
        "weather_impact_pct": weather_impact_pct,
        "weather_impact_mw": weather_impact_mw,
        "dod_load_delta_mw": dod_load_delta,
        "momentum_lambda": np.full(_BLOCK_COUNT, lam, dtype=float),
        "momentum_impact_mw": momentum_impact_mw,
        "simulated_load_mw": simulated_load,
        "weight_confidence": conf_w,
    })

    return {
        "target_date": resolved_date,
        "previous_date": prev_date,
        "previous_previous_date": prev2_date,
        "momentum_lambda": lam,
        "driver_weight_matrix": weight_df.sort_values("block").reset_index(drop=True),
        "weather_delta": {
            "temperature": temp_delta.tolist(),
            "humidity": hum_delta.tolist(),
            "rain": rain_delta.tolist(),
            "apparent": apparent_delta.tolist(),
            "cloud": cloud_delta.tolist(),
            "sun": sun_delta.tolist(),
            "radiation": rad_delta.tolist(),
            "wind": wind_delta.tolist(),
        },
        "component_base": {
            "temperature": (temp_delta * temp_w).tolist(),
            "humidity": (hum_delta * hum_w).tolist(),
            "precipitation": (rain_delta * rain_w).tolist(),
            "apparent": (apparent_delta * apparent_w).tolist(),
            "cloud": (cloud_delta * cloud_w).tolist(),
            "sun": (sun_delta * sun_w).tolist(),
            "radiation": (rad_delta * rad_w).tolist(),
            "wind": (wind_delta * wind_w).tolist(),
        },
        "weather_impact_pct": weather_impact_pct.tolist(),
        "weather_impact_mw": weather_impact_mw.tolist(),
        "dod_load_delta_mw": dod_load_delta.tolist(),
        "momentum_impact_mw": momentum_impact_mw.tolist(),
        "momentum_base": momentum_base.tolist(),
        "simulated_load_mw": simulated_load.tolist(),
        "weight_confidence": conf_w.tolist(),
        "block_results_df": block_results_df,
        "metadata": {
            "features": feature_cols,
            "train_dates": int(len(sorted(train["date"].astype(str).unique().tolist()))) if not train.empty else 0,
            "min_samples": int(min_samples),
        },
    }


def _coerce_weight_vector(weight_matrix: Any, column: str) -> np.ndarray:
    aliases = {
        "temperature_weight": ("temp_weight", "temp_weight_raw", "temp_weight_v2", "temp"),
        "humidity_weight": ("humidity_weight_raw", "hum_weight", "hum_weight_raw"),
        "precipitation_weight": ("rain_weight", "rain_weight_raw", "precip_weight", "precipitation"),
        "apparent_temperature_weight": ("apparent_weight", "apparent_weight_raw", "apparent_temp_weight"),
        "cloud_cover_weight": ("cloud_weight", "cloud_weight_raw"),
        "sunshine_duration_weight": ("sun_weight", "sun_weight_raw", "sunshine_weight"),
        "direct_radiation_weight": ("rad_weight", "rad_weight_raw", "radiation_weight"),
        "wind_speed_10m_weight": ("wind_weight", "wind_weight_raw", "wind_speed_weight", "wind_speed_10m", "wind"),
        "daytype_weight": ("day_type_weight", "weekend_weight", "daytype"),
        "holiday_weight": ("holiday",),
        "weight_confidence": ("fit_confidence", "sample_confidence", "confidence"),
    }

    if isinstance(weight_matrix, pd.DataFrame):
        use_col = column
        if use_col not in weight_matrix.columns:
            for alt in aliases.get(column, ()):
                if alt in weight_matrix.columns:
                    use_col = alt
                    break
        if use_col not in weight_matrix.columns:
            return np.zeros(_BLOCK_COUNT, dtype=float)
        if "block" in weight_matrix.columns:
            ordered = (
                weight_matrix[["block", use_col]]
                .copy()
                .assign(block=lambda x: pd.to_numeric(x["block"], errors="coerce"))
                .dropna(subset=["block"])
                .sort_values("block")
            )
            vec = ordered[use_col].to_numpy(dtype=float)
            return _as_96_vector(vec)
        return _as_96_vector(weight_matrix[use_col].to_numpy(dtype=float))
    if isinstance(weight_matrix, dict):
        if column in weight_matrix:
            return _as_96_vector(weight_matrix.get(column))
        for alt in aliases.get(column, ()):
            if alt in weight_matrix:
                return _as_96_vector(weight_matrix.get(alt))
        return np.zeros(_BLOCK_COUNT, dtype=float)
    return _as_96_vector(weight_matrix)


def run_block_driver_weight_delta_engine(
    target_date: str,
    baseline_load: np.ndarray,
    weight_matrix: Any,
    delta_inputs: Optional[Dict[str, Any]] = None,
    sliders: Optional[Dict[str, float]] = None,
    selection: Optional[Dict[str, Any]] = None,
    smooth_edges: bool = False,
    base_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    delta_cfg = delta_inputs or {}
    override_cfg = base_overrides or {}

    baseline_vec = _as_96_vector(baseline_load)

    temp_w = _coerce_weight_vector(weight_matrix, "temperature_weight")
    hum_w = _coerce_weight_vector(weight_matrix, "humidity_weight")
    rain_w = _coerce_weight_vector(weight_matrix, "precipitation_weight")
    apparent_w = _coerce_weight_vector(weight_matrix, "apparent_temperature_weight")
    cloud_w = _coerce_weight_vector(weight_matrix, "cloud_cover_weight")
    sun_w = _coerce_weight_vector(weight_matrix, "sunshine_duration_weight")
    rad_w = _coerce_weight_vector(weight_matrix, "direct_radiation_weight")
    wind_w = _coerce_weight_vector(weight_matrix, "wind_speed_10m_weight")
    
    daytype_w = _coerce_weight_vector(weight_matrix, "daytype_weight")
    holiday_w = _coerce_weight_vector(weight_matrix, "holiday_weight")
    weight_conf = _coerce_weight_vector(weight_matrix, "weight_confidence")

    temp_delta = _as_96_vector(delta_cfg.get("temp_delta", delta_cfg.get("temperature_delta", 0.0)))
    hum_delta = _as_96_vector(delta_cfg.get("humidity_delta", delta_cfg.get("hum_delta", 0.0)))
    rain_delta = _as_96_vector(delta_cfg.get("rain_delta", delta_cfg.get("precip_delta", 0.0)))
    apparent_delta = _as_96_vector(delta_cfg.get("apparent_delta", 0.0))
    cloud_delta = _as_96_vector(delta_cfg.get("cloud_delta", 0.0))
    sun_delta = _as_96_vector(delta_cfg.get("sun_delta", 0.0))
    rad_delta = _as_96_vector(delta_cfg.get("rad_delta", 0.0))
    wind_delta = _as_96_vector(delta_cfg.get("wind_delta", 0.0))
    
    daytype_flag = _as_96_vector(delta_cfg.get("daytype_flag", 0.0))
    holiday_flag = _as_96_vector(delta_cfg.get("holiday_flag", 0.0))

    temperature_base = temp_delta * temp_w
    humidity_base = hum_delta * hum_w
    precipitation_base = rain_delta * rain_w
    apparent_base = apparent_delta * apparent_w
    cloud_base = cloud_delta * cloud_w
    sun_base = sun_delta * sun_w
    radiation_base = rad_delta * rad_w
    wind_base = wind_delta * wind_w
    
    daytype_base = daytype_flag * daytype_w
    holiday_base = holiday_flag * holiday_w
    manual_base = _as_96_vector(delta_cfg.get("manual_base", 0.0))

    if override_cfg.get("temperature_base") is not None:
        temperature_base = _as_96_vector(override_cfg.get("temperature_base"))
    if override_cfg.get("humidity_base") is not None:
        humidity_base = _as_96_vector(override_cfg.get("humidity_base"))
    if override_cfg.get("precipitation_base") is not None:
        precipitation_base = _as_96_vector(override_cfg.get("precipitation_base"))
    if override_cfg.get("wind_base") is not None:
        wind_base = _as_96_vector(override_cfg.get("wind_base"))
    if override_cfg.get("daytype_base") is not None:
        daytype_base = _as_96_vector(override_cfg.get("daytype_base"))
    if override_cfg.get("holiday_base") is not None:
        holiday_base = _as_96_vector(override_cfg.get("holiday_base"))
    if override_cfg.get("manual_base") is not None:
        manual_base = _as_96_vector(override_cfg.get("manual_base"))

    weather_base_override = override_cfg.get("weather_base")
    if weather_base_override is not None:
        weather_override_vec = _as_96_vector(weather_base_override)
        has_component_override = any(
            override_cfg.get(key) is not None
            for key in ("temperature_base", "humidity_base", "precipitation_base", "wind_base")
        )
        if not has_component_override:
            temperature_base = weather_override_vec.copy()
            humidity_base = np.zeros(_BLOCK_COUNT, dtype=float)
            precipitation_base = np.zeros(_BLOCK_COUNT, dtype=float)
            wind_base = np.zeros(_BLOCK_COUNT, dtype=float)

    weather_base = temperature_base + humidity_base + precipitation_base + apparent_base + cloud_base + sun_base + radiation_base + wind_base

    slider_cfg = sliders or {}
    s_w = float(slider_cfg.get("weather", slider_cfg.get("S_w", 1.0)))
    s_d = float(slider_cfg.get("daytype", slider_cfg.get("S_d", 1.0)))
    s_h = float(slider_cfg.get("holiday", slider_cfg.get("S_h", 1.0)))
    s_m = float(slider_cfg.get("manual", slider_cfg.get("S_m", 1.0)))
    s_temp = float(slider_cfg.get("temperature", 1.0))
    s_hum = float(slider_cfg.get("humidity", 1.0))
    s_prec = float(slider_cfg.get("precipitation", 1.0))
    s_apparent = float(slider_cfg.get("apparent_temperature", 1.0))
    s_cloud = float(slider_cfg.get("cloud_cover", 1.0))
    s_sun = float(slider_cfg.get("sunshine_duration", 1.0))
    s_rad = float(slider_cfg.get("direct_radiation", 1.0))
    s_wind = float(slider_cfg.get("wind", 1.0))

    temperature_scaled = temperature_base * s_w * s_temp
    humidity_scaled = humidity_base * s_w * s_hum
    precipitation_scaled = precipitation_base * s_w * s_prec
    apparent_scaled = apparent_base * s_w * s_apparent
    cloud_scaled = cloud_base * s_w * s_cloud
    sun_scaled = sun_base * s_w * s_sun
    rad_scaled = radiation_base * s_w * s_rad
    wind_scaled = wind_base * s_w * s_wind
    
    weather_scaled = (
        temperature_scaled + humidity_scaled + precipitation_scaled + 
        apparent_scaled + cloud_scaled + sun_scaled + rad_scaled + wind_scaled
    )
    daytype_scaled = daytype_base * s_d
    holiday_scaled = holiday_base * s_h
    manual_scaled = manual_base * s_m

    final_impact = np.clip(
        weather_scaled + daytype_scaled + holiday_scaled + manual_scaled,
        _IMPACT_MIN,
        _IMPACT_MAX,
    )

    mask = _selection_mask(selection)
    if smooth_edges:
        mask = _smoothed_selection_mask(mask)
    final_load = baseline_vec * (1.0 + (final_impact * mask))

    matrix = pd.DataFrame({
        "date": [target_date] * _BLOCK_COUNT,
        "block": np.arange(1, _BLOCK_COUNT + 1, dtype=int),
        "temp_weight": temp_w,
        "humidity_weight": hum_w,
        "rain_weight": rain_w,
        "apparent_weight": apparent_w,
        "cloud_weight": cloud_w,
        "sun_weight": sun_w,
        "rad_weight": rad_w,
        "wind_weight": wind_w,
        "weight_confidence": weight_conf,
        "temp_delta": temp_delta,
        "humidity_delta": hum_delta,
        "rain_delta": rain_delta,
        "apparent_delta": apparent_delta,
        "cloud_delta": cloud_delta,
        "sun_delta": sun_delta,
        "rad_delta": rad_delta,
        "wind_delta": wind_delta,
        "weather_base": weather_base,
        "temperature_base": temperature_base,
        "humidity_base": humidity_base,
        "precipitation_base": precipitation_base,
        "apparent_base": apparent_base,
        "cloud_base": cloud_base,
        "sun_base": sun_base,
        "radiation_base": radiation_base,
        "wind_base": wind_base,
        "weather_scaled": weather_scaled,
        "temperature_scaled": temperature_scaled,
        "humidity_scaled": humidity_scaled,
        "precipitation_scaled": precipitation_scaled,
        "apparent_scaled": apparent_scaled,
        "cloud_scaled": cloud_scaled,
        "sun_scaled": sun_scaled,
        "rad_scaled": rad_scaled,
        "wind_scaled": wind_scaled,
        "final_impact": final_impact,
    })

    driver_weight_matrix = pd.DataFrame({
        "date": [target_date] * _BLOCK_COUNT,
        "block": np.arange(1, _BLOCK_COUNT + 1, dtype=int),
        "temp_weight": temp_w,
        "humidity_weight": hum_w,
        "rain_weight": rain_w,
        "apparent_weight": apparent_w,
        "cloud_weight": cloud_w,
        "sun_weight": sun_w,
        "rad_weight": rad_w,
        "wind_weight": wind_w,
        "weight_confidence": weight_conf,
    })

    return {
        "matrix_df": matrix,
        "driver_weight_matrix": driver_weight_matrix,
        "selection_mask": mask,
        "impact_vector": final_impact,
        "final_load": final_load,
    }


def _rank_block_contributors(contrib_map: Dict[str, float], baseline_mw: Optional[float] = None) -> List[Dict[str, Any]]:
    total_abs = sum(abs(v) for v in contrib_map.values()) or 1.0
    denom_mw = float(baseline_mw) if baseline_mw is not None and np.isfinite(float(baseline_mw)) else None
    ranked = sorted(contrib_map.items(), key=lambda kv: abs(kv[1]), reverse=True)
    rows = []
    for feature, val in ranked:
        load_pct = 0.0
        if denom_mw is not None and abs(denom_mw) > 1e-6:
            load_pct = float((float(val) / denom_mw) * 100.0)
        rows.append({
            "feature": feature,
            "contribution_mw": float(val),
            "contribution_pct": float((val / total_abs) * 100.0),
            "contribution_load_pct": load_pct,
            "direction": "up" if val >= 0 else "down",
        })
    return rows


def _recommended_action(primary_feature: str, risk_flag: str, net_impact_pct: float) -> str:
    if risk_flag == "high":
        if primary_feature == "weather":
            return "Increase reserve and monitor weather-driven slots."
        if primary_feature == "calendar":
            return "Review day-type/holiday assumptions before commitment."
        if primary_feature == "manual":
            return "Reduce manual override and apply staged correction."
        return "Trigger operator review and prepare corrective dispatch."
    if risk_flag == "medium":
        if primary_feature in ("weather", "calendar"):
            return "Monitor slot and pre-position balancing support."
        return "Track slot and apply guarded adjustment if deviation grows."
    if abs(net_impact_pct) >= 8:
        return "Keep slot under watch; no immediate intervention."
    return "No action required."


def run_short_term_pipeline(df: pd.DataFrame, target_date: str, actual_blocks: int = 40, config: Optional[Dict] = None) -> Dict:
    import time as _time
    _t0 = _time.monotonic()

    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    # Extract callback (not a real config param, remove before use)
    _progress_cb = cfg.pop("progress_callback", None)

    def _step(msg: str):
        elapsed = _time.monotonic() - _t0
        logger.info("[pipeline:%s] %s  (%.1fs elapsed)", target_date, msg, elapsed)
        if _progress_cb:
            try:
                _progress_cb({"type": "progress", "message": msg, "elapsed": round(elapsed, 1)})
            except Exception:
                pass

    _step("▶ START")
    cfg["similarity_weights"] = _normalize_similarity_weights(cfg.get("similarity_weights", {}))
    actual_blocks = int(np.clip(actual_blocks, 0, 96))  # cap at full day, no static config limit
    if actual_blocks < 2:
        actual_blocks = 0

    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("date column missing")
    df["date"] = df["date"].astype(str)
    # Ensure optional weather columns exist (may be absent in older datasets)
    for _oc in ["apparent_temperature", "cloud_cover", "sunshine_duration",
                 "direct_radiation", "wind_speed_10m", "cloud_cover_low"]:
        if _oc not in df.columns:
            df[_oc] = 0.0

    dates = sorted(df["date"].dropna().unique().tolist())
    if target_date not in dates:
        # Find latest robust date
        target_date = dates[-1]
        for d in reversed(dates):
            day_df = df[df["date"] == d]
            if day_df["time_block"].nunique() >= 96 and day_df["total_drawal"].mean() > cfg.get("min_valid_load_mw", 100):
                target_date = d
                break


    similarity_profile = {
        "weights": dict(cfg.get("similarity_weights", {})),
        "require_rain_match": bool(cfg.get("require_rain_match", True)),
        "source": "manual",
        "diagnostics": {},
    }
    if bool(cfg.get("auto_similarity_from_data", True)):
        _step("1/7 computing similarity profile...")
        similarity_profile = _compute_similarity_profile_from_data(df, target_date, cfg)
        cfg["similarity_weights"] = _normalize_similarity_weights(similarity_profile.get("weights", {}))
        if bool(cfg.get("auto_require_rain_match", True)):
            cfg["require_rain_match"] = bool(similarity_profile.get("require_rain_match", True))

    _step("2/7 selecting best baseline window...")
    best_window, best_mape = _best_baseline_window(
        df,
        target_date,
        cfg["baseline_window_candidates"],
        cfg["baseline_backtest_days"],
        weather_weight=float(cfg.get("baseline_weather_weight", 0.0)),
    )

    _step("3/7 finding similar days for baseline...")
    baseline_hist, similar_days = _similar_day_baseline(df, target_date, cfg)

    # ------------------------------------------------------------------
    # ML Baseline (Option 1 — hybrid upgrade)
    # Train an XGBoost/LightGBM/Ridge model on the full history and blend
    # its prediction with the similar-day baseline.  The delta engine is
    # unchanged — ML only improves the baseline it starts from.
    # ------------------------------------------------------------------
    _step("4/7 training ML baseline (XGBoost/LightGBM/Ridge)...")
    _ml_baseline_vec = np.zeros(_BLOCK_COUNT, dtype=float)
    _ml_baseline_meta: dict = {"model_name": "disabled"}
    if compute_ml_baseline is not None and blend_baselines is not None:
        try:
            _ml_baseline_vec, _ml_baseline_meta = compute_ml_baseline(df, target_date)
            if np.any(_ml_baseline_vec > 0):
                baseline_hist = blend_baselines(
                    ml_baseline=_ml_baseline_vec,
                    stat_baseline=baseline_hist,
                )
                logger.info(
                    "ML baseline blended (%s, blend_weight=%.2f, cached=%s).",
                    _ml_baseline_meta.get("model_name"),
                    _ml_baseline_meta.get("blend_weight", 0.45),
                    _ml_baseline_meta.get("is_cached", False),
                )
        except Exception as _ml_err:
            logger.warning("ML baseline failed, using similar-day baseline only: %s", _ml_err)

    target_df = _coerce_day_to_96_blocks(df[df["date"] == target_date], fill_load=False)
    if target_df.empty:
        raise ValueError("Target date not available")
    if int(target_df["time_block"].nunique()) != _BLOCK_COUNT:
        raise ValueError(f"Target date {target_date} does not contain {_BLOCK_COUNT} aligned blocks")

    season = _season(target_date)
    day_type = _day_type(target_date)

    # Compute holiday flags for target date (used by both behaviour and calendar)
    _hf_df_early = _compute_holiday_flags(
        df[df["date"] == str(target_date)], region=cfg.get("region", "punjab")
    )
    target_holiday_flags = {}
    if not _hf_df_early.empty:
        _hf_row_early = _hf_df_early.iloc[0]
        for _hfc in ["is_holiday", "bridge_day", "long_weekend", "holiday_before_weekend",
                      "holiday_after_weekend", "post_weekend_holiday", "mid_week_holiday",
                      "days_to_holiday", "days_since_holiday"]:
            target_holiday_flags[_hfc] = int(_hf_row_early.get(_hfc, 0))

    # Human behaviour adjustment based on region, season, day type
    hb_region = str(cfg.get("region", "punjab"))
    hb_weight = float(cfg.get("human_behaviour_weight", 1.0))

    # Calibrate behaviour from historical data if enabled
    if cfg.get("calibrate_behaviour_from_data", True):
        global _CALIBRATED_BEHAVIOUR
        _CALIBRATED_BEHAVIOUR = _calibrate_behaviour_profiles(df, hb_region, cfg)

    # Use extended day type if holiday flags available
    hb_day_type = _day_type_extended(target_date, target_holiday_flags) if target_holiday_flags else day_type
    human_behaviour_adj, hb_profile_label = _human_behaviour_adjustment(
        season, hb_region, hb_day_type, weight=hb_weight
    )

    history_dates = [d for d in dates if d < target_date]
    history_dates = history_dates[-cfg["weather_training_days"]:]
    train_df = df[df["date"].isin(history_dates)]
    if train_df.empty:
        train_df = df[df["date"] != target_date]

    _step("5/7 training weather model (ElasticNet/Ridge)...")
    weather_model, weather_training_frame, weather_effects = _train_weather_model(
        train_df,
        tune=bool(cfg.get("weather_tune", True)),
        tune_iters=int(cfg.get("weather_tune_iters", 12)),
    )
    actual = target_df["total_drawal"].to_numpy(dtype=float)
    load_anchor = _build_load_anchor(baseline_hist=baseline_hist, actual=actual, actual_blocks=actual_blocks)
    weather_pred, weather_coefs = _weather_baseline(
        model=weather_model,
        history_df=train_df,
        target_df=target_df,
        effects=weather_effects,
        load_anchor=load_anchor,
    )

    baseline_weather_df = df[df["date"].isin(similar_days["date"].tolist())] if not similar_days.empty else train_df
    weather_dev = _compute_weather_deviation(target_df, baseline_weather_df)

    coef_temp = float(weather_coefs.get("temperature", 0.0))
    coef_hum = float(weather_coefs.get("humidity", 0.0))
    coef_rain = float(weather_coefs.get("rain", 0.0))

    if not baseline_weather_df.empty:
        w_cols = [
            "temperature", "humidity", "precipitation",
            "apparent_temperature", "cloud_cover", "sunshine_duration",
            "direct_radiation", "wind_speed_10m"
        ]
        w_cols = [c for c in w_cols if c in baseline_weather_df.columns]
        base_block_weather = (
            baseline_weather_df.groupby("time_block")[w_cols]
            .mean()
            .reindex(range(1, 97))
            .ffill()
            .bfill()
        )
        tgt_w_cols = [c for c in w_cols if c in target_df.columns]
        target_block_weather = (
            target_df.set_index("time_block")[tgt_w_cols]
            .reindex(range(1, 97))
            .ffill()
            .bfill()
        )
        def _safe_delta(col):
            if col in target_block_weather.columns and col in base_block_weather.columns:
                return (target_block_weather[col] - base_block_weather[col]).to_numpy()
            return np.zeros(96)
        temp_delta = _safe_delta("temperature")
        hum_delta = _safe_delta("humidity")
        rain_delta = _safe_delta("precipitation")
        apparent_delta = _safe_delta("apparent_temperature")
        cloud_delta = _safe_delta("cloud_cover")
        sun_delta = _safe_delta("sunshine_duration")
        rad_delta = _safe_delta("direct_radiation")
        wind_delta = _safe_delta("wind_speed_10m")
    else:
        temp_delta = np.zeros(96)
        hum_delta = np.zeros(96)
        rain_delta = np.zeros(96)
        apparent_delta = np.zeros(96)
        cloud_delta = np.zeros(96)
        sun_delta = np.zeros(96)
        rad_delta = np.zeros(96)
        wind_delta = np.zeros(96)

    temp_profile = _learn_asymmetric_temp_profile(
        df=df,
        target_date=target_date,
        season=season,
        day_type=day_type,
        lookback_days=int(cfg.get("temp_asym_lookback_days", 180)),
        rolling_days=int(cfg.get("temp_asym_rolling_days", 7)),
        min_block_samples=int(cfg.get("temp_asym_min_samples", 8)),
        prior_coeff=coef_temp,
    )
    temp_up_coeff = np.asarray(temp_profile.get("block_increase_coeffs", np.full(96, coef_temp)), dtype=float)
    temp_down_coeff = np.asarray(temp_profile.get("block_reduction_coeffs", np.full(96, coef_temp)), dtype=float)
    if temp_up_coeff.size < 96:
        temp_up_coeff = np.pad(temp_up_coeff, (0, 96 - temp_up_coeff.size), mode="edge")
    if temp_down_coeff.size < 96:
        temp_down_coeff = np.pad(temp_down_coeff, (0, 96 - temp_down_coeff.size), mode="edge")
    # Saturation (User Fix #3)
    # Replace linear elasticity with capped response using tanh
    # impact = tanh(delta / 4) * 4 * coeff
    # This saturates beyond +/- 4 degrees delta.
    
    # Also apply Midday Amplification (User Fix #2) for blocks 45-60
    
    midday_mask = np.ones(96, dtype=float)
    midday_mask[44:60] = 1.25 # Blocks 45-60 (0-indexed 44-59)
    
    # Tanh saturation — widened to ±6°C for Indian summers (was ±4°C)
    sat_scale = 6.0
    saturated_delta = np.tanh(temp_delta / sat_scale) * sat_scale

    temp_coeff_vec = np.where(temp_delta >= 0.0, temp_up_coeff, temp_down_coeff)

    # ── Multiplicative weather delta (v3.0) ──────────────────────────
    # Weather impact scales with baseline load level: a 3°C rise adds more MW
    # when base load is 5000 MW vs 2000 MW. Convert MW/°C coefficients to
    # percentage-of-baseline before applying.
    reference_load = float(np.median(baseline_hist[baseline_hist > 100])) if np.any(baseline_hist > 100) else float(np.mean(baseline_hist) + 1e-6)
    pct_coeff_vec = temp_coeff_vec / max(reference_load, 1.0)
    temperature_impact_block = baseline_hist * pct_coeff_vec * saturated_delta * midday_mask

    weather_impact_block = (
        temperature_impact_block + 
        (hum_delta * float(weather_coefs.get("humidity", 0.0))) + 
        (rain_delta * float(weather_coefs.get("rain", 0.0))) +
        (apparent_delta * float(weather_coefs.get("apparent_temp", 0.0))) +
        (cloud_delta * float(weather_coefs.get("cloud", 0.0))) +
        (sun_delta * float(weather_coefs.get("sunshine", 0.0))) +
        (rad_delta * float(weather_coefs.get("radiation", 0.0))) +
        (wind_delta * float(weather_coefs.get("wind", 0.0)))
    )

    # ── Per-Period Weather Divergence Gate (v3.0) ──────────────────────
    # Compute gate per period (night/morning/afternoon/evening) instead of
    # a single daily average. Afternoon blocks with +3°C should get the delta
    # even if the daily mean gap is only 1.5°C.
    divergence_thresh = float(cfg.get("weather_divergence_threshold", 1.5))
    divergence_gate = np.ones(96, dtype=float)
    avg_temp_divergence = float(np.mean(np.abs(temp_delta)))
    if divergence_thresh > 0:
        periods = [(0, 24), (24, 48), (48, 72), (72, 96)]  # night/morning/afternoon/evening
        for p_start, p_end in periods:
            period_div = float(np.mean(np.abs(temp_delta[p_start:p_end])))
            gate_val = float(np.clip(
                (period_div - divergence_thresh) / max(divergence_thresh * 1.25, 0.1), 0.0, 1.0
            ))
            divergence_gate[p_start:p_end] = gate_val
        weather_impact_block = weather_impact_block * divergence_gate

    weather_baseline = baseline_hist + weather_impact_block

    # ══════════════════════════════════════════════════════════════════
    # HYBRID ADDITIVE FORECAST MODEL
    # ──────────────────────────────────────────────────────────────────
    # Forecast = Baseline + Weather_Residual + Calendar_Offset
    #          + Trend_Correction + Boundary_Correction
    #
    # Each component is additive MW — transparent, auditable, no
    # multiplicative compounding. The boundary correction anchors the
    # forecast to recent actual levels with exponential decay.
    # ══════════════════════════════════════════════════════════════════

    # Calendar factor (holiday/weekend/transition multiplier → converted to additive MW)
    calendar_factors = _calendar_factor(df, cfg.get("calendar_config"), holiday_flags=target_holiday_flags)
    calendar_factor_total = calendar_factors.get("total", np.ones(96))
    cal_daytype_factor = calendar_factors.get("day_type", np.ones(96))
    cal_holiday_factor = calendar_factors.get("holiday", np.ones(96))
    cal_transition_factor = calendar_factors.get("transition", np.ones(96))

    actual_full = actual.copy()
    actual_partial = actual[:actual_blocks]
    safe_baseline = np.maximum(baseline_hist, 1.0)

    # ── Component 1: Weather Residual (direct ML additive correction) ──
    # weather_impact_block is already in MW, gated by divergence
    weather_residual = weather_impact_block.copy()

    # ── Component 2: Calendar Offset (additive MW from factor) ─────────
    # Convert multiplicative calendar factor to additive MW offset
    calendar_offset = baseline_hist * (calendar_factor_total - 1.0)

    # ── Component 3: Piecewise Trend Correction (v3.0) ─────────────────
    # Fit separate slopes for short-term (14d) vs long-term (60d) residuals.
    # If short-term diverges >2× from long-term, trust short-term (regime change).
    # Clip is proportional to recent load (±5%) instead of fixed ±200 MW.
    similar_dates = similar_days["date"].tolist() if not similar_days.empty else []
    trend_correction = np.zeros(96, dtype=float)

    # Gather all recent daily residuals (up to 60 days)
    all_dates = sorted(df["date"].dropna().unique().tolist())
    target_idx = all_dates.index(target_date) if target_date in all_dates else len(all_dates) - 1
    lookback_60 = all_dates[max(0, target_idx - 60):target_idx]
    day_residuals_60 = []
    for d in lookback_60:
        day_vals = df[df["date"] == d].sort_values("time_block")["total_drawal"].to_numpy()
        if len(day_vals) == 96 and np.mean(day_vals) > 100:
            day_residuals_60.append(float(np.mean(day_vals - safe_baseline)))

    if len(day_residuals_60) >= 3:
        x_all = np.arange(len(day_residuals_60), dtype=float)
        slope_long = float(np.polyfit(x_all, day_residuals_60, 1)[0])

        # Short-term: last 14 entries (or all if fewer)
        short_n = min(14, len(day_residuals_60))
        x_short = np.arange(short_n, dtype=float)
        slope_short = float(np.polyfit(x_short, day_residuals_60[-short_n:], 1)[0])

        # Regime change detection: if short-term diverges >2× from long-term, favor it
        if abs(slope_long) > 1e-6 and abs(slope_short) > 2 * abs(slope_long):
            slope = slope_short
        else:
            slope = 0.7 * slope_short + 0.3 * slope_long

        # Proportional clip: ±5% of recent average load (not fixed ±200 MW)
        recent_avg_load = float(np.mean(baseline_hist[baseline_hist > 100])) if np.any(baseline_hist > 100) else 1000.0
        max_trend = 0.05 * recent_avg_load
        trend_mw = float(np.clip(slope * 1.0, -max_trend, max_trend))
        trend_correction = np.full(96, trend_mw, dtype=float)
    elif similar_dates and len(similar_dates) >= 2:
        # Fallback to similar-day residuals if not enough history
        day_residuals = []
        for d in similar_dates:
            day_vals = df[df["date"] == d].sort_values("time_block")["total_drawal"].to_numpy()
            if len(day_vals) == 96:
                day_residuals.append(np.mean(day_vals - safe_baseline))
        if len(day_residuals) >= 2:
            x = np.arange(len(day_residuals), dtype=float)
            slope = float(np.polyfit(x, day_residuals, 1)[0])
            recent_avg_load = float(np.mean(baseline_hist[baseline_hist > 100])) if np.any(baseline_hist > 100) else 1000.0
            max_trend = 0.05 * recent_avg_load
            trend_mw = float(np.clip(slope * 1.0, -max_trend, max_trend))
            trend_correction = np.full(96, trend_mw, dtype=float)

    # ── Component 4: Boundary Correction (anchor to actual level) ──────
    # Ratio-based correction over a wide window captures proportional level deviation.
    # Additive correction captures residual absolute offset. Both decay over 48 blocks (~12 hrs).
    boundary_window = min(12, actual_blocks)   # additive window (recent blocks)
    ratio_window = min(32, actual_blocks)       # ratio window (wider = more representative)
    boundary_bias_mw = 0.0
    level_ratio = 1.0
    if boundary_window > 0:
        recent_actual = actual_partial[-boundary_window:]
        recent_baseline = baseline_hist[actual_blocks - boundary_window:actual_blocks]
        boundary_bias_mw = float(np.median(recent_actual - recent_baseline))
    if ratio_window > 0:
        ratio_actual = actual_partial[-ratio_window:]
        ratio_baseline = baseline_hist[actual_blocks - ratio_window:actual_blocks]
        _valid = ratio_baseline > 100
        if _valid.sum() >= 2:
            level_ratio = float(np.clip(
                np.median(ratio_actual[_valid] / ratio_baseline[_valid]),
                0.75, 1.25,
            ))

    # Level-adjust baseline_hist only for the actual period so the Pattern Fit chart
    # shows baseline aligned with actual. Forecast period baseline is left unchanged
    # (the boundary_correction below handles the smooth level transition there).
    if level_ratio != 1.0 and actual_blocks > 0:
        for b in range(actual_blocks):
            baseline_hist[b] = baseline_hist[b] * level_ratio
        safe_baseline = np.maximum(baseline_hist, 1.0)

    # Additive boundary correction for forecast period — ratio already applied to actual
    # blocks above; don't re-apply ratio to forecast (causes overshot jump).
    boundary_correction = np.zeros(96, dtype=float)
    for b in range(actual_blocks, 96):
        distance = b - actual_blocks + 1
        decay = np.exp(-0.693 * distance / 48.0)
        boundary_correction[b] = boundary_bias_mw * decay

    # ── Similar-day residual shape (intraday pattern from historical days) ──
    sim_residual_shape = np.zeros(96, dtype=float)
    if similar_dates:
        resid_stack = []
        for d in similar_dates:
            day_vals = df[df["date"] == d].sort_values("time_block")["total_drawal"].to_numpy()
            if len(day_vals) == 96:
                resid_stack.append(day_vals - safe_baseline)
        if resid_stack:
            sim_residual_shape = np.median(np.vstack(resid_stack), axis=0)

    _step("6/7 applying India-specific adjustments...")
    # ── India-Specific Adjustments (agriculture, festivals, events) ───
    india_adj = np.zeros(96, dtype=float)
    india_meta = {}
    if compute_india_adjustments is not None:
        try:
            india_region = cfg.get("region", "punjab")
            # Extract per-block temperatures if available (enables CDH model)
            _block_temps = None
            _avg_temp = 25.0
            _avg_hum  = 50.0
            if target_df is not None and not target_df.empty:
                if "temperature" in target_df.columns:
                    _t = target_df.sort_values("time_block")["temperature"].to_numpy()
                    if len(_t) > 0:
                        _avg_temp = float(np.nanmean(_t))
                        _block_temps = _t if len(_t) == 96 else None
                if "humidity" in target_df.columns:
                    _h = target_df["humidity"].to_numpy()
                    if len(_h) > 0:
                        _avg_hum = float(np.nanmean(_h))
            india_result = compute_india_adjustments(
                state=india_region,
                target_date=str(target_date),
                baseline=baseline_hist,
                config=cfg,
                temperature=_avg_temp,
                humidity=_avg_hum,
                block_temps=_block_temps,
            )
            india_adj = np.asarray(india_result.get("total_adj_mw", np.zeros(96)), dtype=float)
            india_meta = india_result
        except Exception as _e:
            logger.warning(f"India intelligence failed: {_e}")

    _step("7/7 assembling final forecast...")
    # ── Assemble Final Forecast ────────────────────────────────────────
    forecast = np.zeros(96, dtype=float)
    for b in range(96):
        if b < actual_blocks:
            forecast[b] = actual[b]  # known blocks = exact actual
        else:
            distance = b - actual_blocks + 1
            # Similar-day shape influence grows as boundary correction decays
            shape_weight = 1.0 - np.exp(-0.693 * distance / 24.0)
            sim_shape_mw = sim_residual_shape[b] * shape_weight

            forecast[b] = (
                baseline_hist[b]
                + weather_residual[b]
                + calendar_offset[b]
                + trend_correction[b]
                + boundary_correction[b]
                + sim_shape_mw
                + india_adj[b]
            )
    forecast = np.maximum(forecast, 0)

    # ── Legacy variables for downstream KPI reporting ──────────────────
    boundary_ratio = float(np.median(actual_partial / safe_baseline[:actual_blocks]) if actual_blocks > 0 else 1.0)
    hybrid = baseline_hist + boundary_correction  # approximate hybrid for KPIs
    hybrid_pre_calendar = hybrid.copy()
    calendar_adjustment_block = calendar_offset
    bias_factor = boundary_ratio
    bias_applied = (baseline_hist + boundary_correction).copy()
    alpha = 1.0
    beta = 0.0
    trend_slope = float(trend_correction[48]) if len(trend_correction) > 48 else 0.0
    trend_center = 0.0
    trend_component = np.zeros(actual_blocks)
    trend_full = trend_correction.copy()
    rain_coeff = cfg["rain_coeffs"].get(season, 20.0)
    # Rain impact via log-saturation (rebuild plan Step 13)
    rain_impact_vec = np.full(96, rain_coeff, dtype=float)
    rain_impact_vec[69:] *= 1.8  # evening boost
    precip_raw = target_df["precipitation"].to_numpy(dtype=float) if "precipitation" in target_df.columns else np.zeros(96, dtype=float)
    precip_saturated = np.where(precip_raw > 0, np.log1p(precip_raw) * (5.0 / max(np.log1p(5.0), 1e-9)), 0.0)
    rain_impact = rain_impact_vec * precip_saturated
    pattern_adjustment = sim_residual_shape.copy()
    residual_correction = boundary_correction.copy()
    residuals = actual_partial - hybrid[:actual_blocks]
    residuals_clean = residuals.copy()

    # ── Stitch: exact actuals + smooth 8-block cosine taper into forecast ─────
    stitched = forecast.copy()
    stitched[:actual_blocks] = actual_partial
    taper_len = min(8, 96 - actual_blocks)
    if taper_len > 0 and actual_blocks > 0 and len(actual_partial) > 0:
        for t in range(taper_len):
            b = actual_blocks + t
            w = 0.5 * (1.0 - np.cos(np.pi * (t + 1) / (taper_len + 1)))  # cosine ease-in
            stitched[b] = actual_partial[-1] * (1.0 - w) + forecast[b] * w

    shrink_cfg = cfg.get("forecast_shrinkage", {}) if isinstance(cfg.get("forecast_shrinkage", {}), dict) else {}
    offpeak_weight = float(np.clip(float(shrink_cfg.get("offpeak_weight", 0.50)), 0.0, 1.0))
    peak_weight = float(np.clip(float(shrink_cfg.get("peak_weight", 0.25)), 0.0, 1.0))
    known_blocks_bonus = float(np.clip(float(shrink_cfg.get("known_blocks_bonus", 0.20)), 0.0, 0.5))
    peak_start_idx = int(np.clip(int(shrink_cfg.get("peak_start_block", 48)) - 1, 0, _BLOCK_COUNT - 1))
    peak_end_idx = int(np.clip(int(shrink_cfg.get("peak_end_block", 72)), peak_start_idx + 1, _BLOCK_COUNT))

    shrink_weights = np.full(_BLOCK_COUNT, offpeak_weight, dtype=float)
    shrink_weights[peak_start_idx:peak_end_idx] = peak_weight
    if actual_blocks > 0:
        progress = float(np.clip(actual_blocks / _BLOCK_COUNT, 0.0, 1.0))
        shrink_weights = np.clip(shrink_weights + (known_blocks_bonus * progress), 0.0, 1.0)
    shrink_weights[:actual_blocks] = 1.0

    stitched_raw = stitched.copy()
    stitched = (shrink_weights * stitched_raw) + ((1.0 - shrink_weights) * hybrid)
    stitched[:actual_blocks] = actual_partial
    stitched_pre_hybrid_ai = stitched.copy()

    _step("  → hybrid AI engine (ResNet-LSTM blend)...")
    hybrid_ai_result = None
    hybrid_ai_cfg = cfg.get("hybrid_ai", {}) if isinstance(cfg.get("hybrid_ai", {}), dict) else {}
    if bool(hybrid_ai_cfg.get("enabled", True)) and HybridForecastingEngine is not None and HybridForecastConfig is not None:
        try:
            hybrid_engine = HybridForecastingEngine(
                HybridForecastConfig(
                    lookback_days=int(hybrid_ai_cfg.get("lookback_days", 7)),
                    min_history_days=int(hybrid_ai_cfg.get("min_history_days", 21)),
                    training_days=int(hybrid_ai_cfg.get("training_days", 60)),
                    blend_weight=float(hybrid_ai_cfg.get("blend_weight", 0.35)),
                    min_blend_weight=float(hybrid_ai_cfg.get("min_blend_weight", 0.10)),
                    max_blend_weight=float(hybrid_ai_cfg.get("max_blend_weight", 0.45)),
                    random_state=int(hybrid_ai_cfg.get("random_state", 42)),
                    sequence_epochs=int(hybrid_ai_cfg.get("sequence_epochs", 60)),
                )
            )
            hybrid_ai_result = hybrid_engine.run(df=df, target_date=target_date, fallback_forecast=stitched_pre_hybrid_ai)
            if hybrid_ai_result is not None:
                stitched = np.asarray(hybrid_ai_result.forecast, dtype=float)
                stitched[:actual_blocks] = actual_partial
        except Exception as exc:
            logger.warning("Hybrid forecasting engine failed for %s: %s", target_date, exc)
            hybrid_ai_result = None

    baseline_vec = np.asarray(stitched, dtype=float)
    weather_base = np.divide(
        weather_impact_block,
        np.maximum(baseline_vec, 1e-6),
        out=np.zeros_like(weather_impact_block, dtype=float),
        where=np.abs(baseline_vec) > 1e-6,
    )
    temperature_base = np.divide(
        temperature_impact_block,
        np.maximum(baseline_vec, 1e-6),
        out=np.zeros_like(temp_delta, dtype=float),
        where=np.abs(baseline_vec) > 1e-6,
    )
    humidity_base = np.divide(
        hum_delta * coef_hum,
        np.maximum(baseline_vec, 1e-6),
        out=np.zeros_like(hum_delta, dtype=float),
        where=np.abs(baseline_vec) > 1e-6,
    )
    precipitation_base = np.divide(
        rain_delta * coef_rain,
        np.maximum(baseline_vec, 1e-6),
        out=np.zeros_like(rain_delta, dtype=float),
        where=np.abs(baseline_vec) > 1e-6,
    )
    apparent_base = np.divide(
        apparent_delta * float(weather_coefs.get("apparent_temp", 0.0)),
        np.maximum(baseline_vec, 1e-6),
        out=np.zeros_like(apparent_delta, dtype=float),
        where=np.abs(baseline_vec) > 1e-6,
    )
    cloud_base = np.divide(
        cloud_delta * float(weather_coefs.get("cloud", 0.0)),
        np.maximum(baseline_vec, 1e-6),
        out=np.zeros_like(cloud_delta, dtype=float),
        where=np.abs(baseline_vec) > 1e-6,
    )
    sun_base = np.divide(
        sun_delta * float(weather_coefs.get("sunshine", 0.0)),
        np.maximum(baseline_vec, 1e-6),
        out=np.zeros_like(sun_delta, dtype=float),
        where=np.abs(baseline_vec) > 1e-6,
    )
    radiation_base = np.divide(
        rad_delta * float(weather_coefs.get("radiation", 0.0)),
        np.maximum(baseline_vec, 1e-6),
        out=np.zeros_like(rad_delta, dtype=float),
        where=np.abs(baseline_vec) > 1e-6,
    )
    wind_base = np.divide(
        wind_delta * float(weather_coefs.get("wind", 0.0)),
        np.maximum(baseline_vec, 1e-6),
        out=np.zeros_like(wind_delta, dtype=float),
        where=np.abs(baseline_vec) > 1e-6,
    )
    daytype_base = np.asarray(cal_daytype_factor) - 1.0
    holiday_base = np.asarray(cal_holiday_factor) - 1.0
    manual_base = _as_96_vector(cfg.get("manual_base", np.zeros(96, dtype=float)))
    driver_layer = build_block_driver_matrix(
        target_date=target_date,
        baseline_load=baseline_vec,
        weather_base=weather_base,
        daytype_base=daytype_base,
        holiday_base=holiday_base,
        manual_base=manual_base,
        temperature_base=temperature_base,
        humidity_base=humidity_base,
        precipitation_base=precipitation_base,
        apparent_base=apparent_base,
        cloud_base=cloud_base,
        sun_base=sun_base,
        radiation_base=radiation_base,
        wind_base=wind_base,
        sliders=cfg.get("driver_sliders"),
        selection=cfg.get("selection"),
        smooth_edges=bool(cfg.get("selection_smoothing", False)),
    )
    block_driver_matrix = driver_layer["matrix_df"]
    selection_mask = np.asarray(driver_layer["selection_mask"], dtype=float)
    final_load = np.asarray(driver_layer["final_load"], dtype=float)

    weather_scaled = block_driver_matrix["weather_scaled"].to_numpy(dtype=float)
    temperature_scaled = block_driver_matrix["temperature_scaled"].to_numpy(dtype=float)
    humidity_scaled = block_driver_matrix["humidity_scaled"].to_numpy(dtype=float)
    precipitation_scaled = block_driver_matrix["precipitation_scaled"].to_numpy(dtype=float)
    apparent_scaled = block_driver_matrix["apparent_scaled"].to_numpy(dtype=float) if "apparent_scaled" in block_driver_matrix.columns else np.zeros(_BLOCK_COUNT, dtype=float)
    cloud_scaled = block_driver_matrix["cloud_scaled"].to_numpy(dtype=float) if "cloud_scaled" in block_driver_matrix.columns else np.zeros(_BLOCK_COUNT, dtype=float)
    sun_scaled = block_driver_matrix["sun_scaled"].to_numpy(dtype=float) if "sun_scaled" in block_driver_matrix.columns else np.zeros(_BLOCK_COUNT, dtype=float)
    rad_scaled = block_driver_matrix["rad_scaled"].to_numpy(dtype=float) if "rad_scaled" in block_driver_matrix.columns else np.zeros(_BLOCK_COUNT, dtype=float)
    wind_scaled = block_driver_matrix["wind_scaled"].to_numpy(dtype=float) if "wind_scaled" in block_driver_matrix.columns else np.zeros(_BLOCK_COUNT, dtype=float)
    daytype_scaled = block_driver_matrix["daytype_scaled"].to_numpy(dtype=float)
    holiday_scaled = block_driver_matrix["holiday_scaled"].to_numpy(dtype=float)
    manual_scaled = block_driver_matrix["manual_scaled"].to_numpy(dtype=float)
    clamped_impact = block_driver_matrix["final_impact"].to_numpy(dtype=float)

    weather_contrib_mw = baseline_vec * weather_scaled * selection_mask
    temperature_contrib_mw = baseline_vec * temperature_scaled * selection_mask
    humidity_contrib_mw = baseline_vec * humidity_scaled * selection_mask
    precipitation_contrib_mw = baseline_vec * precipitation_scaled * selection_mask
    apparent_contrib_mw = baseline_vec * apparent_scaled * selection_mask
    cloud_contrib_mw = baseline_vec * cloud_scaled * selection_mask
    sun_contrib_mw = baseline_vec * sun_scaled * selection_mask
    radiation_contrib_mw = baseline_vec * rad_scaled * selection_mask
    wind_contrib_mw = baseline_vec * wind_scaled * selection_mask
    daytype_contrib_mw = baseline_vec * daytype_scaled * selection_mask
    holiday_contrib_mw = baseline_vec * holiday_scaled * selection_mask
    manual_contrib_mw = baseline_vec * manual_scaled * selection_mask
    net_contribution_mw = final_load - baseline_vec
    applied_impact_pct = clamped_impact * selection_mask * 100.0

    residual_std = float(np.std(residuals)) if len(residuals) >= 2 else 0.0
    similar_residual_std = np.zeros(96, dtype=float)
    if similar_dates:
        similar_residuals = []
        for d in similar_dates:
            day_series = df[df["date"] == d].sort_values("time_block")["total_drawal"].to_numpy()
            if len(day_series) == 96:
                similar_residuals.append(day_series - baseline_hist)
        if similar_residuals:
            similar_residual_std = np.std(np.vstack(similar_residuals), axis=0)

    # Tail Decay (User Fix #5)
    # Decay momentum (trend) at night (Block > 84)
    trend_decay_mask = np.ones(96, dtype=float)
    for b in range(84, 96):
        dist = b - 84
        trend_decay_mask[b] = np.exp(-dist / 6.0)
        
    trend_weight = float(cfg.get("trend_weight", 0.05))
    trend_applied = trend_full * trend_weight * trend_decay_mask
    sigma_block = np.sqrt((similar_residual_std ** 2) + (residual_std ** 2))
    sigma_block = sigma_block + (np.abs(pattern_adjustment) * 0.15)
    sigma_floor = np.maximum(np.abs(net_contribution_mw) * 0.08, 1.0)
    sigma_block = np.maximum(sigma_block, sigma_floor)
    z_val = 1.2816
    p50 = stitched.copy()
    p10 = np.maximum(p50 - (z_val * sigma_block), 0.0)
    p90 = p50 + (z_val * sigma_block)
    if hybrid_ai_result is not None:
        p50 = np.asarray(hybrid_ai_result.forecast, dtype=float)
        p10 = np.asarray(hybrid_ai_result.p10, dtype=float)
        p90 = np.asarray(hybrid_ai_result.p90, dtype=float)
        p50[:actual_blocks] = actual_partial
        p10[:actual_blocks] = actual_partial
        p90[:actual_blocks] = actual_partial
    uncertainty_width_mw = p90 - p10
    uncertainty_width_pct = (uncertainty_width_mw / np.maximum(p50, 1e-6)) * 100.0
    forecast_confidence = np.clip(1.0 - (uncertainty_width_pct / 100.0), 0.05, 0.99)
    if hybrid_ai_result is not None:
        forecast_confidence = np.asarray(hybrid_ai_result.confidence, dtype=float)
        forecast_confidence[:actual_blocks] = 0.99

    weather_sens = np.abs(weather_scaled)
    calendar_sens = np.abs(daytype_scaled) + np.abs(holiday_scaled)
    operational_sens = np.abs((trend_applied + pattern_adjustment - rain_impact) / np.maximum(baseline_vec, 1e-6))
    manual_sens = np.abs(manual_scaled)
    sens_stack = np.vstack([weather_sens, calendar_sens, operational_sens, manual_sens])
    dominant_idx = np.argmax(sens_stack, axis=0)
    family_labels = np.array(["weather", "calendar", "operational", "manual"])
    dominant_family = family_labels[dominant_idx]
    sensitivity_score = np.sum(sens_stack, axis=0)

    block_contributors = []
    slot_sensitivity_profile = []
    forecast_uncertainty = []
    decision_signals = []
    top_contributor_slots = []
    for i in range(96):
        contrib_map = {
            "temperature": float(temperature_contrib_mw[i]),
            "humidity": float(humidity_contrib_mw[i]),
            "precipitation": float(precipitation_contrib_mw[i]),
            "wind": float(wind_contrib_mw[i]),
            "apparent_temperature": float(apparent_contrib_mw[i]),
            "cloud_cover": float(cloud_contrib_mw[i]),
            "sunshine_duration": float(sun_contrib_mw[i]),
            "direct_radiation": float(radiation_contrib_mw[i]),
            "daytype": float(daytype_contrib_mw[i]),
            "holiday": float(holiday_contrib_mw[i]),
            "manual": float(manual_contrib_mw[i]),
        }
        ranked = _rank_block_contributors(contrib_map, baseline_mw=float(baseline_vec[i]))
        abs_vals = np.asarray([abs(v) for v in contrib_map.values()], dtype=float)
        abs_sum = float(np.sum(abs_vals)) if np.sum(abs_vals) > 0 else 1.0
        dominance = float(np.max(abs_vals) / abs_sum)
        contributor_confidence = float(np.clip(0.35 + (0.40 * dominance) + (0.25 * forecast_confidence[i]), 0.05, 0.99))

        risk_score = (0.55 * abs(applied_impact_pct[i])) + (0.45 * uncertainty_width_pct[i])
        if risk_score >= 25:
            risk_flag = "high"
        elif risk_score >= 12:
            risk_flag = "medium"
        else:
            risk_flag = "low"

        primary_driver = ranked[0]["feature"] if ranked else "none"
        secondary_driver = ranked[1]["feature"] if len(ranked) > 1 else primary_driver
        expected_gain = abs(float(net_contribution_mw[i])) * (0.35 if risk_flag == "high" else 0.20 if risk_flag == "medium" else 0.08)

        block_contributors.append({
            "block": i + 1,
            "time": f"{(i*15)//60:02d}:{(i*15)%60:02d}",
            "baseline_mw": float(baseline_vec[i]),
            "forecast_mw": float(final_load[i]),
            "net_impact_pct": float(applied_impact_pct[i]),
            "net_contribution_mw": float(net_contribution_mw[i]),
            "forecast_confidence": float(forecast_confidence[i]),
            "contributor_confidence": contributor_confidence,
            "contributors_ranked": ranked,
        })

        slot_sensitivity_profile.append({
            "block": i + 1,
            "time": f"{(i*15)//60:02d}:{(i*15)%60:02d}",
            "weather_sensitivity": float(weather_sens[i]),
            "calendar_sensitivity": float(calendar_sens[i]),
            "operational_sensitivity": float(operational_sens[i]),
            "manual_sensitivity": float(manual_sens[i]),
            "dominant_family": str(dominant_family[i]),
            "sensitivity_score": float(sensitivity_score[i]),
        })

        forecast_uncertainty.append({
            "block": i + 1,
            "time": f"{(i*15)//60:02d}:{(i*15)%60:02d}",
            "p10_mw": float(p10[i]),
            "p50_mw": float(p50[i]),
            "p90_mw": float(p90[i]),
            "uncertainty_width_mw": float(uncertainty_width_mw[i]),
            "uncertainty_width_pct": float(uncertainty_width_pct[i]),
            "forecast_confidence": float(forecast_confidence[i]),
        })

        decision_signals.append({
            "block": i + 1,
            "time": f"{(i*15)//60:02d}:{(i*15)%60:02d}",
            "risk_flag": risk_flag,
            "risk_score": float(risk_score),
            "dominant_family": str(dominant_family[i]),
            "primary_driver": primary_driver,
            "secondary_driver": secondary_driver,
            "recommended_action": _recommended_action(primary_driver, risk_flag, float(applied_impact_pct[i])),
            "expected_gain_mw": float(expected_gain),
            "net_impact_pct": float(applied_impact_pct[i]),
            "uncertainty_pct": float(uncertainty_width_pct[i]),
            "confidence": float(min(forecast_confidence[i], contributor_confidence)),
        })

        top_contributor_slots.append({
            "block": i + 1,
            "time": f"{(i*15)//60:02d}:{(i*15)%60:02d}",
            "abs_net_contribution_mw": float(abs(net_contribution_mw[i])),
            "primary_driver": primary_driver,
            "risk_flag": risk_flag,
        })
    top_contributor_slots = sorted(top_contributor_slots, key=lambda x: x["abs_net_contribution_mw"], reverse=True)[:12]

    # Peak-level weather + calendar impact diagnostics
    hist_peak_idx = int(np.argmax(baseline_hist))
    weather_peak_idx = int(np.argmax(weather_baseline))
    hybrid_pre_cal_peak_idx = int(np.argmax(hybrid_pre_calendar))
    hybrid_peak_idx = int(np.argmax(hybrid))
    forecast_peak_idx = int(np.argmax(stitched))
    peak_i = forecast_peak_idx
    peak_impact = {
        "calendar_config": cfg.get("calendar_config"),
        "baseline_peak": {
            "time_block": hist_peak_idx + 1,
            "mw": float(np.max(baseline_hist)),
        },
        "weather_peak": {
            "time_block": weather_peak_idx + 1,
            "mw": float(np.max(weather_baseline)),
            "delta_vs_baseline_mw": float(np.max(weather_baseline) - np.max(baseline_hist)),
        },
        "hybrid_pre_calendar_peak": {
            "time_block": hybrid_pre_cal_peak_idx + 1,
            "mw": float(np.max(hybrid_pre_calendar)),
            "delta_vs_baseline_mw": float(np.max(hybrid_pre_calendar) - np.max(baseline_hist)),
        },
        "hybrid_peak": {
            "time_block": hybrid_peak_idx + 1,
            "mw": float(np.max(hybrid)),
            "calendar_effect_at_peak_mw": float(hybrid[hybrid_peak_idx] - hybrid_pre_calendar[hybrid_peak_idx]),
        },
        "forecast_peak": {
            "time_block": forecast_peak_idx + 1,
            "mw": float(np.max(stitched)),
            "weather_effect_mw": float(weather_baseline[peak_i] - baseline_hist[peak_i]),
            "calendar_effect_mw": float(calendar_adjustment_block[peak_i]),
            "net_weather_calendar_effect_mw": float(hybrid[peak_i] - baseline_hist[peak_i]),
        },
    }

    driver_table = []

    window_start = max(0, actual_blocks)
    window_len = max(1, 96 - window_start)
    temp_impact = float(np.sum(temperature_impact_block[window_start:])) / window_len
    hum_impact = float(np.sum((hum_delta * coef_hum)[window_start:])) / window_len
    rain_driver = float(np.sum((rain_delta * coef_rain)[window_start:])) / window_len
    rain_impact_total = float(np.sum(rain_impact[window_start:])) / window_len
    actual_len = max(1, int(actual_blocks))
    bias_mw = float(np.mean(residuals)) if len(residuals) else 0.0
    trend_amp = float(np.mean(np.abs(trend_component))) if len(trend_component) else 0.0
    trend_mw = trend_amp if trend_slope >= 0 else -trend_amp
    pattern_mw = float(np.sum(pattern_adjustment[window_start:])) / window_len
    residual_corr_mw = float(np.sum(residual_correction[window_start:])) / window_len

    driver_values = [
        ("Temperature impact", temp_impact),
        ("Humidity impact", hum_impact),
        ("Rain impact", -rain_impact_total),
        ("Bias MW", bias_mw),
        ("Trend MW", trend_mw),
        ("Pattern MW", pattern_mw),
        ("Residual correction", residual_corr_mw),
        ("Human behaviour", float(np.sum(human_behaviour_adj[window_start:])) / window_len),
    ]
    total_driver = sum(abs(v) for _, v in driver_values) or 1.0

    for label, val in driver_values:
        driver_table.append({
            "factor": label,
            "mw": round(val, 2),
            "pct": round(val / total_driver * 100, 2),
        })

    rows = []
    for i in range(96):
        rows.append({
            "date": target_date,
            "time_block": i + 1,
            "time": f"{(i*15)//60:02d}:{(i*15)%60:02d}",
            "historical_baseline": float(baseline_hist[i]),
            "weather_impact": float(weather_impact_block[i]),
            "weather_baseline": float(weather_baseline[i]),
            "hybrid_baseline": float(hybrid[i]),
            "calendar_adjustment": float(calendar_adjustment_block[i]),
            "bias_applied": float(bias_applied[i]),
            "trend_component": float(trend_applied[i]),
            "rain_adjustment": float(rain_impact[i]),
            "pattern_adjustment": float(pattern_adjustment[i]),
            "residual_correction": float(residual_correction[i]),
            "human_behaviour": float(human_behaviour_adj[i]),
            "forecast": float(stitched[i]),
            "actual": float(actual_full[i]) if (i < actual_blocks and i < len(actual_full)) else None,
            "is_actual": i < actual_blocks,
        })

    # Dip explanations (forecast window)
    forecast_vals = stitched
    window_start = max(0, actual_blocks)
    mean_forecast = float(np.mean(forecast_vals[window_start:])) if window_start < 96 else float(np.mean(forecast_vals))
    dip_threshold_mw = max(40.0, mean_forecast * 0.015)
    dip_threshold_pct = 1.5
    dip_explanations = []
    for i in range(window_start + 1, 96):
        prev_val = forecast_vals[i - 1]
        curr_val = forecast_vals[i]
        if prev_val <= 0:
            continue
        diff = curr_val - prev_val
        diff_pct = (diff / prev_val) * 100
        if diff > -dip_threshold_mw and diff_pct > -dip_threshold_pct:
            continue

        bias_component = bias_applied[i] - hybrid[i]
        trend_component_block = trend_full[i]
        rain_component = -rain_impact[i]
        weather_component = weather_pred[i] - baseline_hist[i]

        components = {
            "Bias MW": bias_component,
            "Trend MW": trend_component_block,
            "Rain impact": rain_component,
            "Weather shift": weather_component,
        }
        primary = min(components.items(), key=lambda kv: kv[1])[0]

        dip_explanations.append({
            "time_block": i + 1,
            "time": f"{(i*15)//60:02d}:{(i*15)%60:02d}",
            "drop_mw": round(float(diff), 2),
            "drop_pct": round(float(diff_pct), 2),
            "primary_driver": primary,
            "components": {k: round(float(v), 2) for k, v in components.items()},
        })

    dip_explanations = sorted(dip_explanations, key=lambda d: d["drop_mw"])[:6]

    response = {
        "date": target_date,
        "metadata": {
            "season": season,
            "day_type": day_type,
            "actual_blocks": actual_blocks,
            "rain_coeff": rain_coeff,
            "best_baseline_window": best_window,
            "best_baseline_mape": round(best_mape, 2) if best_mape is not None else None,
            "similar_days_count": int(len(similar_days)) if not similar_days.empty else 0,
            "region": hb_region,
            "human_behaviour_profile": hb_profile_label,
            "human_behaviour_weight": hb_weight,
            "behaviour_source": (
                "CalibratedCyclicData"
                if (_CALIBRATED_BEHAVIOUR and (INDIAN_STATE_REGIONS.get(hb_region.lower().strip(), hb_region.lower().strip()), season, hb_day_type) in _CALIBRATED_BEHAVIOUR)
                else "Hardcoded"
            ),
            "hybrid_ai_engine": hybrid_ai_result.metadata if hybrid_ai_result is not None else {"enabled": False},
            "hybrid_ai_explanation": hybrid_ai_result.explanation if hybrid_ai_result is not None else "Hybrid engine unavailable; fallback forecast retained.",
            "human_behaviour_learning": _CALIBRATED_BEHAVIOUR_META.get(
                (INDIAN_STATE_REGIONS.get(hb_region.lower().strip(), hb_region.lower().strip()), season, hb_day_type),
                {"source": "hardcoded_profiles"},
            ),
            "forecast_shrinkage": {
                "offpeak_weight": round(float(offpeak_weight), 3),
                "peak_weight": round(float(peak_weight), 3),
                "known_blocks_bonus": round(float(known_blocks_bonus), 3),
                "peak_blocks": [int(peak_start_idx + 1), int(peak_end_idx)],
            },
            "target_day_type_extended": hb_day_type if 'hb_day_type' in dir() else day_type,
            "holiday_flags": target_holiday_flags,
            "weather_divergence_gate": float(np.mean(divergence_gate)) if 'divergence_gate' in dir() else None,
            "avg_temp_divergence_c": float(avg_temp_divergence) if 'avg_temp_divergence' in dir() else None,
            "model_scope": "global",
            "feature_schema": {
                "cross_day_lags": ["lag_1", "lag_7"],
                "intraday_lags": ["lag_block_1", "lag_block_4"],
                "rolling_windows": ["rolling_4", "rolling_12"],
                "time_encoding": ["block_sin", "block_cos"],
            },
            "feature_count": int(len(_get_short_term_feature_columns(weather_training_frame))),
            "similarity_weights": cfg.get("similarity_weights"),
            "require_rain_match": bool(cfg.get("require_rain_match", True)),
            "similarity_profile_source": similarity_profile.get("source"),
            "similarity_profile_diagnostics": similarity_profile.get("diagnostics", {}),
            "temperature_delta_profile": {
                "source": temp_profile.get("source"),
                "prior_coeff": float(temp_profile.get("prior_coeff", coef_temp)),
                "global_increase_coeff": float(temp_profile.get("global_increase_coeff", coef_temp)),
                "global_reduction_coeff": float(temp_profile.get("global_reduction_coeff", coef_temp)),
                "global_increase_samples": int(temp_profile.get("global_increase_samples", 0)),
                "global_reduction_samples": int(temp_profile.get("global_reduction_samples", 0)),
                "diagnostics": temp_profile.get("diagnostics", {}),
            },
            "driver_sliders": cfg.get("driver_sliders"),
            "selection": cfg.get("selection"),
            "selection_smoothing": bool(cfg.get("selection_smoothing", False)),
            "decision_mode": "block_exogenous_attribution_v1",
            "similar_days": similar_days[["date", "similarity_score", "temp_diff", "hum_diff", "rain_match"]].to_dict("records") if not similar_days.empty else [],
            "weather_coefs": {k: round(float(v), 4) for k, v in weather_coefs.items()} if weather_coefs else {},
            "drivers": driver_table,
        },
        "weights": {
            "alpha": round(alpha, 3),
            "beta": round(beta, 3),
            "weather_deviation": round(weather_dev, 3),
        },
        "calendar_factor": calendar_factor_total.tolist() if calendar_factor_total is not None else None,
        "series": {
            "blocks": list(range(1, 97)),
            "historical_baseline": baseline_hist.tolist(),
            "ml_baseline": _ml_baseline_vec.tolist(),
            "ml_baseline_meta": _ml_baseline_meta,
            "weather_feature_deltas": {
                "temperature": temp_delta.tolist(),
                "humidity": hum_delta.tolist(),
                "precipitation": rain_delta.tolist(),
                "apparent_temperature": apparent_delta.tolist(),
                "cloud_cover": cloud_delta.tolist(),
                "sunshine_duration": sun_delta.tolist(),
                "direct_radiation": rad_delta.tolist(),
                "wind_speed_10m": wind_delta.tolist(),
            },
            "temperature_effective_coeff": temp_coeff_vec.tolist(),
            "temperature_increase_coeff": temp_up_coeff.tolist(),
            "temperature_reduction_coeff": temp_down_coeff.tolist(),
            "weather_impact": weather_impact_block.tolist(),
            "weather_baseline": weather_baseline.tolist(),
            "hybrid_baseline": hybrid.tolist(),
            "calendar_adjustment": calendar_adjustment_block.tolist(),
            "cal_daytype_factor": cal_daytype_factor.tolist(),
            "cal_holiday_factor": cal_holiday_factor.tolist(),
            "cal_transition_factor": cal_transition_factor.tolist(),
            "bias_applied": bias_applied.tolist(),
            "trend": trend_full.tolist(),
            "trend_component": (trend_full * trend_weight).tolist(),
            "rain_adjustment": rain_impact.tolist(),
            "pattern_adjustment": pattern_adjustment.tolist(),
            "residual_correction": residual_correction.tolist(),
            "human_behaviour": human_behaviour_adj.tolist(),
            "temperature_contrib_mw": temperature_contrib_mw.tolist(),
            "humidity_contrib_mw": humidity_contrib_mw.tolist(),
            "precipitation_contrib_mw": precipitation_contrib_mw.tolist(),
            "wind_contrib_mw": wind_contrib_mw.tolist(),
            "apparent_contrib_mw": apparent_contrib_mw.tolist(),
            "cloud_contrib_mw": cloud_contrib_mw.tolist(),
            "sunshine_contrib_mw": sun_contrib_mw.tolist(),
            "radiation_contrib_mw": radiation_contrib_mw.tolist(),
            "weather_feature_impacts_pct": {
                "temperature": (temperature_scaled * selection_mask * 100.0).tolist(),
                "humidity": (humidity_scaled * selection_mask * 100.0).tolist(),
                "precipitation": (precipitation_scaled * selection_mask * 100.0).tolist(),
                "wind": (wind_scaled * selection_mask * 100.0).tolist(),
                "apparent_temperature": (apparent_scaled * selection_mask * 100.0).tolist(),
                "cloud_cover": (cloud_scaled * selection_mask * 100.0).tolist(),
                "sunshine_duration": (sun_scaled * selection_mask * 100.0).tolist(),
                "direct_radiation": (rad_scaled * selection_mask * 100.0).tolist(),
                "weather_total": (weather_scaled * selection_mask * 100.0).tolist(),
            },
            "forecast_raw_before_shrinkage": stitched_raw.tolist(),
            "forecast_after_shrinkage_before_hybrid_ai": stitched_pre_hybrid_ai.tolist(),
            "hybrid_ai_forecast": (
                np.asarray(hybrid_ai_result.forecast, dtype=float).tolist()
                if hybrid_ai_result is not None
                else stitched_pre_hybrid_ai.tolist()
            ),
            "forecast": stitched.tolist(),
            "final_load": final_load.tolist(),
            "selection_mask": selection_mask.tolist(),
            "actual": [float(actual_full[i]) if i < actual_blocks else None for i in range(len(actual_full))],
        },
        "explanation": hybrid_ai_result.explanation if hybrid_ai_result is not None else "Forecast generated by statistical fallback pipeline.",
        "bias_factor": round(bias_factor, 4),
        "trend_curve": trend_full.tolist(),
        "rain_adjustment": rain_impact.tolist(),
        "driver_contributions": driver_table,
        "similar_days": similar_days[["date", "similarity_score", "temp_diff", "hum_diff", "rain_match"]].to_dict("records") if not similar_days.empty else [],
        "peak_impact": peak_impact,
        "forecast_df": rows,
        "dip_explanations": dip_explanations,
        "block_driver_matrix": block_driver_matrix.to_dict("records"),
        "block_contributors": block_contributors,
        "slot_sensitivity_profile": slot_sensitivity_profile,
        "forecast_uncertainty": forecast_uncertainty,
        "decision_signals": decision_signals,
        "india_intelligence": india_meta,
        "top_contributor_slots": top_contributor_slots,
    }

    _step("✓ DONE — pipeline complete")
    return response


# ── Weather DB fetch helper ───────────────────────────────────────────────────
def _fetch_t2_weather_from_db(t2_date: str, region: str) -> Optional[pd.DataFrame]:
    """
    Fetch block-level weather for t2_date from the Django pipeline API
    (weather_mean table in MySQL).  Returns a 96-row DataFrame with columns
    matching the pipeline schema, or None if unavailable.
    """
    import requests as _req

    pipeline_base = os.environ.get("PIPELINE_API_BASE_URL", "http://localhost:8001").rstrip("/")
    api_token     = os.environ.get("PIPELINE_API_TOKEN", os.environ.get("API_SECRET_KEY", "flagbearer"))
    state         = region.upper().replace(" ", "_").replace("-", "_")

    try:
        resp = _req.get(
            f"{pipeline_base}/weather/mean",
            params={"state": state, "date": t2_date, "limit": 100},
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=10,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            return None

        wdf = pd.DataFrame(rows)
        wdf["date"]       = pd.to_datetime(wdf["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        wdf["time_block"] = pd.to_numeric(wdf.get("time_block", wdf.get("block", pd.Series(dtype=float))),
                                          errors="coerce").astype("Int64")
        wdf = wdf[wdf["time_block"].between(1, 96)].sort_values("time_block").reset_index(drop=True)
        if len(wdf) < 10:
            return None
        return wdf
    except Exception as _e:
        logging.getLogger(__name__).warning("T+2 weather DB fetch failed for %s %s: %s", region, t2_date, _e)
        return None


def _inject_weather_into_rows(target_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Overwrite weather columns in target_df with values from weather_df,
    matched by time_block.  Any column present in weather_df but not in
    target_df is added.  time_block must be an integer key in both frames.
    """
    WEATHER_COLS = [
        "temperature", "humidity", "precipitation",
        "apparent_temperature", "cloud_cover", "cloud_cover_low",
        "sunshine_duration", "direct_radiation", "wind_speed_10m",
    ]

    result = target_df.copy()
    result["time_block"] = pd.to_numeric(result["time_block"], errors="coerce").astype("Int64")
    wdf = weather_df.copy()
    wdf["time_block"] = pd.to_numeric(wdf["time_block"], errors="coerce").astype("Int64")

    available = [c for c in WEATHER_COLS if c in wdf.columns]
    if not available:
        return result

    # Reindex weather to cover all 96 blocks, forward-fill gaps
    wdf_full = wdf.set_index("time_block")[available].reindex(range(1, 97)).ffill().bfill()

    for col in available:
        vals = wdf_full[col].values
        if col not in result.columns:
            result[col] = 0.0
        # Align by time_block index
        tb = result["time_block"].to_numpy(dtype=int)
        result[col] = [float(vals[b - 1]) if 1 <= b <= 96 else float(vals[0]) for b in tb]

    return result


# ── T+2 (Day-After-Tomorrow) Forecast Pipeline ────────────────────────────────
def run_t2_pipeline(
    df: pd.DataFrame,
    t1_date: str,
    config: Optional[Dict] = None,
    region: Optional[str] = None,
) -> Dict:
    """
    Run the short-term pipeline for T+2 (the day after t1_date).

    Strategy:
    - Forecast T+1 first with actual_blocks=0.
    - Replace the T+1 day in the history panel with the forecasted T+1 curve so
      the T+2 run sees a sequential prior day rather than hidden actuals or a
      separate baseline lookup.
    - Forecast T+2 with actual_blocks=0.
    - Apply a short seam taper so T+1 block 96 and T+2 block 1 remain continuous.
    """
    try:
        t1_dt = pd.Timestamp(t1_date)
    except Exception:
        raise ValueError(f"Invalid t1_date: {t1_date!r}")

    t2_dt = t1_dt + pd.Timedelta(days=1)
    t2_date = t2_dt.strftime("%Y-%m-%d")

    cfg = dict(config or {})
    seam_cfg = cfg.get("t2_seam_bridge", {}) if isinstance(cfg.get("t2_seam_bridge", {}), dict) else {}
    seam_window = int(np.clip(int(seam_cfg.get("window_blocks", 8)), 1, _BLOCK_COUNT))

    t1_rows = df[df["date"].astype(str) == t1_date].copy()
    if t1_rows.empty:
        raise ValueError(f"T+1 date {t1_date} not found in dataframe; cannot forecast sequential T+2.")

    # ── Step 1: Forecast T+1 with zero actuals (pure forward forecast) ───────
    t1_result = run_short_term_pipeline(
        df=df,
        target_date=t1_date,
        actual_blocks=0,
        config=cfg,
    )
    t1_forecast = _normalize_block_vector(
        ((t1_result.get("series") or {}).get("forecast") if isinstance(t1_result, dict) else None),
        length=_BLOCK_COUNT,
        fill_value=0.0,
    )

    # ── Step 2: Build T+1 synthetic row using forecasted load as history ─────
    #   Weather columns are kept from the real T+1 day (they already exist in df).
    t1_synthetic = _build_synthetic_forecast_day(
        template_df=t1_rows,
        target_date=t1_date,
        forecast=t1_forecast,
        zero_actuals=False,
    )

    # ── Step 3: Build T+2 rows — use DB weather if available ─────────────────
    dates_in_df = sorted(df["date"].dropna().astype(str).unique().tolist()) if "date" in df.columns else []

    # Fetch real T+2 weather from MySQL via Django API
    t2_weather_df = None
    if region:
        t2_weather_df = _fetch_t2_weather_from_db(t2_date, region)
        if t2_weather_df is not None:
            logging.getLogger(__name__).info(
                "[T+2] Fetched %d weather blocks for %s %s from DB",
                len(t2_weather_df), region, t2_date,
            )
        else:
            logging.getLogger(__name__).warning(
                "[T+2] No DB weather for %s %s — using T+1 weather as proxy", region, t2_date,
            )

    if t2_date in dates_in_df:
        # T+2 already in df — use it but override weather with fresh DB data
        t2_rows = df[df["date"].astype(str) == t2_date].copy()
        if t2_weather_df is not None:
            t2_rows = _inject_weather_into_rows(t2_rows, t2_weather_df)
    else:
        # T+2 not in df — synthesise from T+1 template, then inject DB weather
        t2_rows = _build_synthetic_forecast_day(
            template_df=t1_rows,
            target_date=t2_date,
            forecast=np.zeros(_BLOCK_COUNT, dtype=float),
            zero_actuals=True,
        )
        if t2_weather_df is not None:
            t2_rows = _inject_weather_into_rows(t2_rows, t2_weather_df)

    # ── Step 4: Build extended history — replace T+1 actual with T+1 forecast -
    #   History up to (not including) T+1, then T+1-as-forecast, then T+2.
    history_without_t1 = df[df["date"].astype(str) != t1_date].copy()
    history_without_t1_t2 = history_without_t1[history_without_t1["date"].astype(str) != t2_date].copy()
    df_extended = pd.concat([history_without_t1_t2, t1_synthetic, t2_rows], ignore_index=True)

    result = run_short_term_pipeline(
        df=df_extended,
        target_date=t2_date,
        actual_blocks=0,
        config=cfg,
    )

    result_series = result.get("series", {}) if isinstance(result, dict) else {}
    t2_forecast = _normalize_block_vector(result_series.get("forecast"), length=_BLOCK_COUNT, fill_value=0.0)
    t2_adjusted, seam_gap_before, seam_gap_after = _apply_seam_continuity(
        forecast=t2_forecast,
        previous_terminal_mw=float(t1_forecast[-1]),
        window=seam_window,
    )

    if isinstance(result_series, dict):
        result_series["forecast"] = t2_adjusted.tolist()
        if "final_load" in result_series:
            result_series["final_load"] = t2_adjusted.tolist()
        if "hybrid_ai_forecast" in result_series:
            result_series["hybrid_ai_forecast"] = t2_adjusted.tolist()

    if isinstance(result.get("forecast_df"), list):
        for idx, row in enumerate(result["forecast_df"][:_BLOCK_COUNT]):
            if isinstance(row, dict):
                row["forecast"] = float(t2_adjusted[idx])

    metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
    if isinstance(metadata, dict):
        metadata["sequential_forecast"] = {
            "enabled": True,
            "t1_seed_date": str(t1_date),
            "t2_target_date": str(t2_date),
            "seam_window_blocks": int(seam_window),
            "seam_gap_before_mw": round(float(seam_gap_before), 3),
            "seam_gap_after_mw": round(float(seam_gap_after), 3),
        }

    result["horizon"] = "t2"
    result["t1_date"] = t1_date
    result["t2_date"] = t2_date
    result["t1_forecast"] = t1_forecast.tolist()
    return result
