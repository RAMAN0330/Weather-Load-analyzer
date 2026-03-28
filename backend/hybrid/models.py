from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    _TORCH_AVAILABLE = False

from .data import BLOCK_COUNT


def _softmax(vec: np.ndarray) -> np.ndarray:
    x = np.asarray(vec, dtype=float).reshape(-1)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / max(float(np.sum(ex)), 1e-9)


class SequenceResNetLSTMForecaster:
    def __init__(
        self,
        lookback_days: int,
        random_state: int = 42,
        epochs: int = 60,
        lr: float = 1e-3,
    ) -> None:
        self.lookback_days = int(lookback_days)
        self.random_state = int(random_state)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self._curve_model = None
        self._scale_model = None
        self._torch_model = None
        self._using_torch = False
        self.validation_mape_: Optional[float] = None
        self.training_samples_: int = 0
        self.backend_name_: str = "ridge_fallback"

    def _build_torch_model(self) -> "nn.Module":
        class ResidualBlock(nn.Module):
            def __init__(self, channels: int) -> None:
                super().__init__()
                self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
                self.act = nn.ReLU()

            def forward(self, x):
                residual = x
                out = self.act(self.conv1(x))
                out = self.conv2(out)
                out = self.act(out + residual)
                return out

        class Model(nn.Module):
            def __init__(self, in_channels: int) -> None:
                super().__init__()
                self.input_proj = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
                self.res1 = ResidualBlock(32)
                self.res2 = ResidualBlock(32)
                self.lstm = nn.LSTM(input_size=32, hidden_size=48, num_layers=1, batch_first=True)
                self.curve_head = nn.Linear(48, 1)
                self.scale_head = nn.Sequential(
                    nn.Linear(48, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                feat = torch.relu(self.input_proj(x))
                feat = self.res2(self.res1(feat))
                seq = feat.transpose(1, 2)
                lstm_out, _ = self.lstm(seq)
                curve = self.curve_head(lstm_out).squeeze(-1)
                pooled = lstm_out.mean(dim=1)
                scale = self.scale_head(pooled).squeeze(-1)
                return curve, scale

        return Model(self.lookback_days)

    def fit(self, X_seq: np.ndarray, y_curve: np.ndarray, y_scale: np.ndarray) -> None:
        X = np.asarray(X_seq, dtype=float)
        yc = np.asarray(y_curve, dtype=float)
        ys = np.asarray(y_scale, dtype=float).reshape(-1)
        self.training_samples_ = int(X.shape[0])

        flat = X.reshape(X.shape[0], -1)
        ridge_curve_model = Ridge(alpha=1.0, random_state=self.random_state)
        ridge_curve_model.fit(flat, yc)
        ridge_scale_model = LinearRegression()
        ridge_scale_model.fit(flat, np.log(np.maximum(ys, 1.0)))
        ridge_curve = np.asarray(ridge_curve_model.predict(flat), dtype=float)
        ridge_scale = np.exp(ridge_scale_model.predict(flat))

        best_backend = "ridge_fallback"
        best_curve_model = ridge_curve_model
        best_scale_model = ridge_scale_model
        best_torch_model = None
        best_using_torch = False
        best_pred_curve = ridge_curve
        best_pred_scale = ridge_scale

        daily_true = np.maximum(yc, 0.0) * ys[:, None]
        ridge_daily_pred = np.maximum(ridge_curve, 0.0) * ridge_scale[:, None]
        best_mape = float(np.mean(np.abs(ridge_daily_pred - daily_true) / np.maximum(daily_true, 1.0)) * 100.0)

        if _TORCH_AVAILABLE and X.shape[0] >= 24:
            torch.manual_seed(self.random_state)
            model = self._build_torch_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()

            x_t = torch.tensor(X, dtype=torch.float32)
            y_curve_t = torch.tensor(yc, dtype=torch.float32)
            y_scale_t = torch.tensor(np.log(np.maximum(ys, 1.0)), dtype=torch.float32)
            model.train()
            for _ in range(self.epochs):
                optimizer.zero_grad(set_to_none=True)
                pred_curve, pred_scale = model(x_t)
                loss = loss_fn(pred_curve, y_curve_t) + (0.15 * loss_fn(pred_scale, y_scale_t))
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pred_curve_t, pred_scale_t = model(x_t)
            torch_curve = np.asarray(pred_curve_t.numpy(), dtype=float)
            torch_scale = np.exp(pred_scale_t.numpy())
            torch_daily_pred = np.maximum(torch_curve, 0.0) * torch_scale[:, None]
            torch_mape = float(np.mean(np.abs(torch_daily_pred - daily_true) / np.maximum(daily_true, 1.0)) * 100.0)
            if torch_mape + 0.1 < best_mape:
                best_backend = "resnet_lstm"
                best_torch_model = model
                best_using_torch = True
                best_curve_model = None
                best_scale_model = None
                best_pred_curve = torch_curve
                best_pred_scale = torch_scale
                best_mape = torch_mape

        self._curve_model = best_curve_model
        self._scale_model = best_scale_model
        self._torch_model = best_torch_model
        self._using_torch = best_using_torch
        self.backend_name_ = best_backend
        self.validation_mape_ = best_mape

    def predict(self, X_seq: np.ndarray) -> Tuple[np.ndarray, float]:
        X = np.asarray(X_seq, dtype=float).reshape(1, self.lookback_days, BLOCK_COUNT)
        if self._using_torch and self._torch_model is not None:
            with torch.no_grad():
                curve_t, scale_t = self._torch_model(torch.tensor(X, dtype=torch.float32))
            curve = curve_t.numpy().reshape(BLOCK_COUNT)
            scale = float(np.exp(scale_t.numpy().reshape(-1)[0]))
        else:
            flat = X.reshape(1, -1)
            curve = np.asarray(self._curve_model.predict(flat), dtype=float).reshape(BLOCK_COUNT)
            scale = float(np.exp(self._scale_model.predict(flat).reshape(-1)[0]))

        curve = np.clip(curve, 0.6, 1.5)
        curve = curve / max(float(np.mean(curve)), 1e-6)
        return curve, max(scale, 1.0)


@dataclass
class MoEForecastResult:
    curve: np.ndarray
    scale: float
    gating_weights: Dict[str, float]
    expert_adjustments: Dict[str, float]


class MixtureOfExpertsForecaster:
    expert_names = ("weekday", "weekend", "high_volatility")

    def __init__(self) -> None:
        self.curve_residuals_: Dict[str, np.ndarray] = {}
        self.scale_residuals_: Dict[str, float] = {}
        self.global_curve_residual_ = np.zeros(BLOCK_COUNT, dtype=float)
        self.global_scale_residual_ = 0.0

    def fit(
        self,
        gating_frame,
        base_curve_preds: np.ndarray,
        base_scale_preds: np.ndarray,
        y_curve: np.ndarray,
        y_scale: np.ndarray,
    ) -> None:
        gf = gating_frame.reset_index(drop=True)
        curve_resid = np.asarray(y_curve, dtype=float) - np.asarray(base_curve_preds, dtype=float)
        scale_resid = np.asarray(y_scale, dtype=float) - np.asarray(base_scale_preds, dtype=float)
        self.global_curve_residual_ = np.mean(curve_resid, axis=0)
        self.global_scale_residual_ = float(np.mean(scale_resid))

        masks = {
            "weekday": (gf["is_weekend"].to_numpy(dtype=float) < 0.5),
            "weekend": (gf["is_weekend"].to_numpy(dtype=float) >= 0.5),
            "high_volatility": (gf["is_high_volatility"].to_numpy(dtype=float) >= 0.5),
        }
        for name, mask in masks.items():
            if int(np.sum(mask)) >= 3:
                self.curve_residuals_[name] = np.mean(curve_resid[mask], axis=0)
                self.scale_residuals_[name] = float(np.mean(scale_resid[mask]))
            else:
                self.curve_residuals_[name] = self.global_curve_residual_.copy()
                self.scale_residuals_[name] = float(self.global_scale_residual_)

    def _gating_weights(self, features: np.ndarray) -> np.ndarray:
        feat = np.asarray(features, dtype=float).reshape(-1)
        is_weekend = feat[0] if feat.size > 0 else 0.0
        is_high_vol = feat[1] if feat.size > 1 else 0.0
        vol_z = feat[2] if feat.size > 2 else 0.0

        scores = np.array(
            [
                1.5 * (1.0 - is_weekend) - (0.4 * max(vol_z, 0.0)),
                1.5 * is_weekend,
                1.2 * is_high_vol + (0.8 * max(vol_z, 0.0)),
            ],
            dtype=float,
        )
        return _softmax(scores)

    def predict(self, features: np.ndarray, base_curve: np.ndarray, base_scale: float) -> MoEForecastResult:
        weights = self._gating_weights(features)
        residual_curve = np.zeros(BLOCK_COUNT, dtype=float)
        residual_scale = 0.0
        for idx, name in enumerate(self.expert_names):
            residual_curve += float(weights[idx]) * self.curve_residuals_.get(name, self.global_curve_residual_)
            residual_scale += float(weights[idx]) * float(self.scale_residuals_.get(name, self.global_scale_residual_))

        adjusted_curve = np.asarray(base_curve, dtype=float) + residual_curve
        adjusted_curve = np.clip(adjusted_curve, 0.6, 1.5)
        adjusted_curve = adjusted_curve / max(float(np.mean(adjusted_curve)), 1e-6)
        adjusted_scale = max(float(base_scale + residual_scale), 1.0)
        return MoEForecastResult(
            curve=adjusted_curve,
            scale=adjusted_scale,
            gating_weights={name: float(weights[idx]) for idx, name in enumerate(self.expert_names)},
            expert_adjustments={name: float(np.mean(self.curve_residuals_.get(name, self.global_curve_residual_))) for name in self.expert_names},
        )


class GaussianProcessResidualRefiner:
    def __init__(self) -> None:
        self.model_: Optional[GaussianProcessRegressor] = None
        self.training_points_: int = 0

    def fit(
        self,
        gating_frame,
        base_forecasts: np.ndarray,
        actual_curves: np.ndarray,
    ) -> None:
        gf = gating_frame.reset_index(drop=True)
        X_rows = []
        y_rows = []
        max_days = min(len(base_forecasts), 14)
        if max_days <= 0:
            return

        for row_idx in range(len(base_forecasts) - max_days, len(base_forecasts)):
            base_curve = np.asarray(base_forecasts[row_idx], dtype=float)
            actual_curve = np.asarray(actual_curves[row_idx], dtype=float)
            ctx = gf.iloc[row_idx]
            for block in range(BLOCK_COUNT):
                angle = (2.0 * np.pi * block) / BLOCK_COUNT
                X_rows.append(
                    [
                        np.sin(angle),
                        np.cos(angle),
                        float(ctx["is_weekend"]),
                        float(ctx["is_high_volatility"]),
                        float(ctx["volatility_zscore"]),
                        float(base_curve[block] / max(np.mean(base_curve), 1.0)),
                    ]
                )
                y_rows.append(float(actual_curve[block] - base_curve[block]))

        X = np.asarray(X_rows, dtype=float)
        y = np.asarray(y_rows, dtype=float)
        if len(X) < BLOCK_COUNT:
            return

        kernel = (1.0 * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds="fixed")) + WhiteKernel(noise_level=1.0, noise_level_bounds="fixed")
        self.model_ = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True, random_state=42)
        self.model_.fit(X, y)
        self.training_points_ = int(len(X))

    def predict(self, context: Dict[str, float], base_curve: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.model_ is None:
            return np.zeros(BLOCK_COUNT, dtype=float), np.full(BLOCK_COUNT, np.std(base_curve) * 0.08, dtype=float)

        rows = []
        base = np.asarray(base_curve, dtype=float)
        base_mean = max(float(np.mean(base)), 1.0)
        for block in range(BLOCK_COUNT):
            angle = (2.0 * np.pi * block) / BLOCK_COUNT
            rows.append(
                [
                    np.sin(angle),
                    np.cos(angle),
                    float(context.get("is_weekend", 0.0)),
                    float(context.get("is_high_volatility", 0.0)),
                    float(context.get("volatility_zscore", 0.0)),
                    float(base[block] / base_mean),
                ]
            )

        pred, std = self.model_.predict(np.asarray(rows, dtype=float), return_std=True)
        return np.asarray(pred, dtype=float), np.asarray(std, dtype=float)
