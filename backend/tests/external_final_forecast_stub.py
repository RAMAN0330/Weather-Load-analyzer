import numpy as np


def final_forecast(df, target_date: str, actual_blocks: int = 0, config=None):
    """
    Minimal external final forecast module used by unit tests.
    Returns a deterministic 96-block forecast vector.
    """
    _ = df
    _ = target_date
    _ = actual_blocks
    _ = config
    return np.linspace(2000.0, 2095.0, 96)

