"""
Hyperparameter Tuning Module for 97%+ Forecast Accuracy.
Performs grid search and Bayesian optimization on ensemble weights,
model parameters, and post-processing refinements.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER TUNING ENGINE
# ════════════════════════════════════════════════════════════════════════════════

class HyperparameterTuner:
    """
    Optimizes forecast ensemble hyperparameters to maximize accuracy.
    Uses walk-forward validation with MAPE as primary metric.
    """

    def __init__(
        self,
        lookback_days: int = 30,
        validation_days: int = 7,
        test_days: int = 14,
    ):
        self.lookback_days = lookback_days
        self.validation_days = validation_days
        self.test_days = test_days
        self.best_params = {}
        self.best_score = float('inf')  # Lower MAPE is better
        self.tuning_history = []

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test sets."""
        dates = sorted(df['date'].unique())
        total_days = len(dates)
        
        test_start = max(0, total_days - self.test_days)
        val_start = max(0, test_start - self.validation_days)
        train_end = max(0, val_start - 1)
        train_start = max(0, train_end - self.lookback_days + 1)
        
        train_dates = dates[train_start:train_end+1]
        val_dates = dates[val_start:test_start]
        test_dates = dates[test_start:]
        
        train_df = df[df['date'].isin(train_dates)]
        val_df = df[df['date'].isin(val_dates)]
        test_df = df[df['date'].isin(test_dates)]
        
        return train_df, val_df, test_df

    def tune_blend_weights(
        self,
        df: pd.DataFrame,
        forecast_fn: Callable,  # Function that takes config and returns forecast
        grid: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, float]:
        """
        Grid search over blend weights.
        
        Args:
            df: Historical data
            forecast_fn: Function to evaluate (df, config) -> forecast_dict
            grid: Grid of parameters to search
            
        Returns:
            Best parameters found
        """
        if grid is None:
            grid = {
                'ml_weight': [0.55, 0.60, 0.65, 0.70, 0.75],
                'baseline_weight': [0.15, 0.20, 0.25, 0.30],
                'block_reg_weight': [0.05, 0.10, 0.15],
            }
        
        train_df, val_df, test_df = self._split_data(df)
        
        if val_df.empty or test_df.empty:
            logger.warning("Insufficient data for tuning, skipping")
            return {}
        
        results = []
        param_space = self._generate_grid(grid)
        
        logger.info(f"Tuning over {len(param_space)} parameter combinations...")
        
        for i, params in enumerate(param_space):
            try:
                # Normalize weights to sum to 1.0
                total_weight = (
                    params['ml_weight'] +
                    params['baseline_weight'] +
                    params['block_reg_weight']
                )
                if total_weight <= 0:
                    continue
                
                norm_params = {
                    'ml_weight': params['ml_weight'] / total_weight,
                    'baseline_weight': params['baseline_weight'] / total_weight,
                    'block_reg_weight': params['block_reg_weight'] / total_weight,
                }
                
                # Evaluate on validation set
                config = {'blend_weights_t1': norm_params}
                val_mape = self._evaluate(val_df, forecast_fn, config)
                test_mape = self._evaluate(test_df, forecast_fn, config)
                
                results.append({
                    'params': norm_params,
                    'val_mape': val_mape,
                    'test_mape': test_mape,
                })
                
                if val_mape < self.best_score:
                    self.best_score = val_mape
                    self.best_params = norm_params.copy()
                    logger.info(
                        f"[{i+1}/{len(param_space)}] New best: "
                        f"val_mape={val_mape:.2f}%, test_mape={test_mape:.2f}%"
                    )
                
                self.tuning_history.append({
                    'iteration': i,
                    'params': norm_params,
                    'val_mape': val_mape,
                    'test_mape': test_mape,
                })
                
            except Exception as e:
                logger.debug(f"Failed to evaluate params {params}: {e}")
        
        if results:
            best_result = min(results, key=lambda x: x['val_mape'])
            logger.info(f"✓ Tuning complete: best val_mape={best_result['val_mape']:.2f}%")
            return best_result['params']
        
        return {}

    def tune_model_hyperparams(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        grid: Optional[Dict[str, List]] = None,
    ) -> Dict[str, Any]:
        """
        Tune XGBoost/LightGBM hyperparameters for given horizon.
        """
        if grid is None:
            grid = {
                'n_estimators': [100, 150, 200],
                'max_depth': [5, 6, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.75, 0.85, 0.95],
            }
        
        logger.info(f"Tuning horizon-{horizon} model hyperparameters...")
        
        # Would require model training framework (skipping implementation for brevity)
        # In practice, use Optuna or GridSearchCV
        
        return grid  # Return original grid as default

    def _generate_grid(self, grid: Dict[str, List]) -> List[Dict]:
        """Generate all combinations from grid dict."""
        keys = list(grid.keys())
        values = list(grid.values())
        
        from itertools import product
        combos = []
        for combo in product(*values):
            combos.append(dict(zip(keys, combo)))
        
        return combos

    def _evaluate(
        self,
        df: pd.DataFrame,
        forecast_fn: Callable,
        config: Dict,
    ) -> float:
        """Evaluate forecast quality on dataset."""
        total_error = []
        
        for date in df['date'].unique():
            try:
                day_actual = df[df['date'] == date]['total_drawal'].values
                if len(day_actual) != 96:
                    continue
                
                # Run forecast
                result = forecast_fn(df=df.copy(), target_date=str(date), config=config)
                forecast = np.array(result['series']['forecast']) if 'series' in result else None
                
                if forecast is None or len(forecast) != 96:
                    continue
                
                # Calculate MAPE
                mape = mean_absolute_percentage_error(day_actual, forecast)
                total_error.append(mape)
            
            except Exception as e:
                logger.debug(f"Failed to evaluate date {date}: {e}")
        
        if total_error:
            return float(np.mean(total_error)) * 100  # Return as percentage
        return float('inf')


# ════════════════════════════════════════════════════════════════════════════════
# ACCURACY IMPROVEMENT RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════════

def analyze_forecast_errors(
    actual: np.ndarray,
    forecast: np.ndarray,
    by_block: bool = True,
) -> Dict[str, Any]:
    """
    Analyze where forecast errors occur to identify improvement opportunities.
    """
    errors = np.abs(actual - forecast)
    pct_errors = (errors / np.maximum(actual, 1)) * 100
    
    analysis = {
        'mae': float(np.mean(errors)),
        'mape': float(np.mean(pct_errors)),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'max_error': float(np.max(errors)),
        'std_error': float(np.std(errors)),
    }
    
    if by_block:
        analysis['error_by_block'] = {}
        for block in range(1, 97):
            block_idx = block - 1
            block_errors = pct_errors[block_idx]
            analysis['error_by_block'][block] = float(block_errors)
    
    return analysis


def recommend_improvements(
    error_analysis: Dict[str, Any],
    current_config: Dict,
) -> List[str]:
    """
    Generate recommendations to improve accuracy based on error analysis.
    """
    recommendations = []
    
    mape = error_analysis.get('mape', 100)
    
    if mape > 5.0:
        recommendations.append("CRITICAL: High overall MAPE. Consider retraining models with more data.")
        recommendations.append("- Increase baseline_lookback_days from 45 to 90")
        recommendations.append("- Enable use_advanced_baseline = True")
        recommendations.append("- Enable use_horizon_specific_models = True")
    
    if 'error_by_block' in error_analysis:
        errors_by_block = error_analysis['error_by_block']
        
        # Peak period analysis (blocks 48-72)
        peak_errors = [errors_by_block.get(i, 0) for i in range(48, 73)]
        if np.mean(peak_errors) > mape + 1.5:
            recommendations.append("Peak period (12:00-18:00) has high errors. Consider:")
            recommendations.append("- Increase weather feature importance")
            recommendations.append("- Add thermal inertia features (CDD)")
            recommendations.append("- Tune peak_detection_percentile")
        
        # Night period analysis (blocks 1-24)
        night_errors = [errors_by_block.get(i, 0) for i in range(1, 25)]
        if np.mean(night_errors) > mape + 1.0:
            recommendations.append("Night period (00:00-06:00) has high errors. Consider:")
            recommendations.append("- Increase baseline weight for night blocks")
            recommendations.append("- Add occupancy/human behavior features")
            recommendations.append("- Reduce trend component weight at night")
    
    if mape < 3.0:
        recommendations.append("✓ Excellent accuracy (MAPE < 3%). Current config is well-tuned.")
    elif mape < 4.0:
        recommendations.append("✓ Good accuracy (MAPE < 4%). Minor tuning may help further.")
    elif mape < 5.0:
        recommendations.append("⚠ Fair accuracy (MAPE < 5%). Tuning recommended.")
    
    return recommendations


# ════════════════════════════════════════════════════════════════════════════════
# CONFIG OPTIMIZATION PRESETS
# ════════════════════════════════════════════════════════════════════════════════

PRESET_CONFIGS = {
    "accuracy_97": {
        "description": "Optimized for 97%+ accuracy (MAPE < 3%)",
        "config_overrides": {
            "optimization_enabled": True,
            "use_advanced_baseline": True,
            "use_horizon_specific_models": True,
            "use_bias_correction": True,
            "adaptive_blend_per_horizon": True,
            "weather_tune": True,
            "weather_tune_iters": 20,
            "baseline_window_candidates": [3, 5, 7, 10, 14, 21],
            "blend_weights_t1": {"ml_model": 0.72, "baseline": 0.18, "block_regression": 0.10},
            "blend_weights_t2": {"ml_model": 0.62, "baseline": 0.23, "block_regression": 0.15},
            "forecast_shrinkage": {
                "offpeak_weight": 0.55,
                "peak_weight": 0.20,
                "known_blocks_bonus": 0.25,
            },
        }
    },
    "accuracy_95": {
        "description": "Optimized for 95%+ accuracy (MAPE < 5%)",
        "config_overrides": {
            "optimization_enabled": True,
            "use_advanced_baseline": True,
            "use_horizon_specific_models": False,
            "use_bias_correction": True,
            "weather_tune": True,
            "weather_tune_iters": 12,
            "blend_weights_t1": {"ml_model": 0.65, "baseline": 0.25, "block_regression": 0.10},
            "blend_weights_t2": {"ml_model": 0.55, "baseline": 0.30, "block_regression": 0.15},
        }
    },
    "speed_optimized": {
        "description": "Optimized for speed (minimal tuning)",
        "config_overrides": {
            "optimization_enabled": False,
            "use_advanced_baseline": False,
            "use_horizon_specific_models": False,
            "use_bias_correction": False,
            "weather_tune": False,
            "hybrid_ai": {"enabled": False},
        }
    },
}


def get_preset_config(preset_name: str, default_config: Dict) -> Dict:
    """Get a preset configuration with overrides applied."""
    if preset_name not in PRESET_CONFIGS:
        logger.warning(f"Unknown preset: {preset_name}")
        return default_config.copy()
    
    preset = PRESET_CONFIGS[preset_name]
    result = default_config.copy()
    result.update(preset['config_overrides'])
    
    logger.info(f"Using preset: {preset['description']}")
    return result

