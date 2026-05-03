import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal external feature module used by unit tests.
    The pipeline loads this file via SHORT_TERM_EXTERNAL_FEATURES_PATH.
    """
    out = df.copy()
    temp = pd.to_numeric(out.get("temperature", 0.0), errors="coerce").fillna(0.0)
    hum = pd.to_numeric(out.get("humidity", 0.0), errors="coerce").fillna(0.0)
    out["ext_temp2"] = 2.0 * temp
    out["ext_heat_index_proxy"] = temp * (1.0 + (hum / 100.0))
    return out

