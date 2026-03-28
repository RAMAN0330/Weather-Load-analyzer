from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .data import BLOCK_COUNT, build_daily_load_panel
from .features import build_sequence_dataset, build_target_features
from .models import GaussianProcessResidualRefiner, MixtureOfExpertsForecaster, SequenceResNetLSTMForecaster


@dataclass
class HybridForecastConfig:
    lookback_days: int = 7
    min_history_days: int = 21
    training_days: int = 60
    blend_weight: float = 0.35
    min_blend_weight: float = 0.10
    max_blend_weight: float = 0.45
    random_state: int = 42
    sequence_epochs: int = 60


@dataclass
class HybridForecastResult:
    forecast: np.ndarray
    p10: np.ndarray
    p90: np.ndarray
    confidence: np.ndarray
    metadata: Dict[str, Any]
    explanation: str


class HybridForecastingEngine:
    def __init__(self, config: Optional[HybridForecastConfig] = None) -> None:
        self.config = config or HybridForecastConfig()

    def _resolve_blend_weight(self, validation_mape: Optional[float]) -> float:
        if validation_mape is None or not np.isfinite(validation_mape):
            return float(self.config.min_blend_weight)
        if validation_mape <= 3.0:
            return float(self.config.max_blend_weight)
        if validation_mape <= 4.5:
            return float(min(self.config.max_blend_weight, self.config.blend_weight + 0.05))
        if validation_mape <= 6.0:
            return float(self.config.blend_weight)
        return float(self.config.min_blend_weight)

    def run(
        self,
        df: pd.DataFrame,
        target_date: str,
        fallback_forecast: Optional[np.ndarray] = None,
    ) -> Optional[HybridForecastResult]:
        panel = build_daily_load_panel(df)
        if target_date not in panel.daily_matrix.index:
            return None

        target_idx = panel.daily_matrix.index.get_loc(target_date)
        if int(target_idx) < max(self.config.lookback_days, self.config.min_history_days):
            return None

        start_idx = max(0, int(target_idx) - int(self.config.training_days))
        train_matrix = panel.daily_matrix.iloc[start_idx:target_idx + 1].copy()
        train_context = panel.context.iloc[start_idx:target_idx + 1].reset_index(drop=True).copy()
        train_panel = build_daily_load_panel(
            pd.DataFrame(
                {
                    "date": np.repeat(train_matrix.index.to_numpy(), BLOCK_COUNT),
                    "time_block": np.tile(np.arange(1, BLOCK_COUNT + 1), len(train_matrix)),
                    "total_drawal": train_matrix.to_numpy(dtype=float).reshape(-1),
                }
            )
        )
        train_panel.context = train_context

        dataset = build_sequence_dataset(train_panel, lookback_days=self.config.lookback_days)
        if len(dataset.dates) < 8:
            return None

        # Use all samples before the target day for training; the last row corresponds to the target date.
        train_mask = np.asarray([d < str(target_date) for d in dataset.dates], dtype=bool)
        if int(np.sum(train_mask)) < 6:
            return None

        X_train = dataset.X_seq[train_mask]
        y_curve_train = dataset.y_curve[train_mask]
        y_scale_train = dataset.y_scale[train_mask]
        gating_train = dataset.gating_frame.loc[train_mask].reset_index(drop=True)
        recent_actuals_train = dataset.recent_actuals[train_mask]

        seq_model = SequenceResNetLSTMForecaster(
            lookback_days=self.config.lookback_days,
            random_state=self.config.random_state,
            epochs=self.config.sequence_epochs,
        )
        seq_model.fit(X_train, y_curve_train, y_scale_train)

        base_curve_preds = []
        base_scale_preds = []
        base_daily_forecasts = []
        for seq in X_train:
            curve, scale = seq_model.predict(seq)
            base_curve_preds.append(curve)
            base_scale_preds.append(scale)
            base_daily_forecasts.append(curve * scale)
        base_curve_preds = np.asarray(base_curve_preds, dtype=float)
        base_scale_preds = np.asarray(base_scale_preds, dtype=float)
        base_daily_forecasts = np.asarray(base_daily_forecasts, dtype=float)

        moe_model = MixtureOfExpertsForecaster()
        moe_model.fit(gating_train, base_curve_preds, base_scale_preds, y_curve_train, y_scale_train)

        moe_daily_forecasts = []
        for idx in range(len(X_train)):
            moe_pred = moe_model.predict(gating_train.iloc[idx].to_numpy(dtype=float), base_curve_preds[idx], base_scale_preds[idx])
            moe_daily_forecasts.append(moe_pred.curve * moe_pred.scale)
        moe_daily_forecasts = np.asarray(moe_daily_forecasts, dtype=float)

        gpr_model = GaussianProcessResidualRefiner()
        gpr_model.fit(gating_train, moe_daily_forecasts, y_curve_train * y_scale_train[:, None])

        target_features = build_target_features(train_panel, target_date, lookback_days=self.config.lookback_days)
        seq_curve, seq_scale = seq_model.predict(target_features["X_seq"])
        moe_result = moe_model.predict(target_features["gating_features"], seq_curve, seq_scale)
        base_forecast = moe_result.curve * moe_result.scale
        residual_mean, residual_std = gpr_model.predict(target_features["context"], base_forecast)
        hybrid_forecast = np.maximum(base_forecast + residual_mean, 0.0)

        blend_weight = self._resolve_blend_weight(seq_model.validation_mape_)
        if fallback_forecast is not None:
            fallback = np.asarray(fallback_forecast, dtype=float).reshape(-1)
            if fallback.size >= BLOCK_COUNT:
                fallback = fallback[:BLOCK_COUNT]
                hybrid_forecast = (blend_weight * hybrid_forecast) + ((1.0 - blend_weight) * fallback)
        sigma = np.maximum(residual_std, np.maximum(np.std(recent_actuals_train[-min(len(recent_actuals_train), 7):], axis=0), 1.0) * 0.08)
        p10 = np.maximum(hybrid_forecast - (1.2816 * sigma), 0.0)
        p90 = hybrid_forecast + (1.2816 * sigma)
        confidence = np.clip(1.0 - ((p90 - p10) / np.maximum(hybrid_forecast, 1.0)), 0.05, 0.99)

        metadata = {
            "engine": "resnet_lstm_moe_gpr",
            "sequence_backend": seq_model.backend_name_,
            "lookback_days": int(self.config.lookback_days),
            "training_days": int(np.sum(train_mask)),
            "sequence_validation_mape": round(float(seq_model.validation_mape_ or 0.0), 3),
            "blend_weight": round(float(blend_weight), 3),
            "gating_weights": {k: round(float(v), 4) for k, v in moe_result.gating_weights.items()},
            "expert_adjustments": {k: round(float(v), 4) for k, v in moe_result.expert_adjustments.items()},
            "gpr_training_points": int(gpr_model.training_points_),
        }
        explanation = (
            f"Hybrid engine used {metadata['training_days']} training days; "
            f"MoE weights weekday={metadata['gating_weights'].get('weekday', 0):.2f}, "
            f"weekend={metadata['gating_weights'].get('weekend', 0):.2f}, "
            f"high_volatility={metadata['gating_weights'].get('high_volatility', 0):.2f}; "
            f"GPR residual refinement applied with blend {metadata['blend_weight']:.2f}."
        )

        return HybridForecastResult(
            forecast=np.asarray(hybrid_forecast, dtype=float),
            p10=np.asarray(p10, dtype=float),
            p90=np.asarray(p90, dtype=float),
            confidence=np.asarray(confidence, dtype=float),
            metadata=metadata,
            explanation=explanation,
        )
