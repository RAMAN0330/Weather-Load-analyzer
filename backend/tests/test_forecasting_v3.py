"""Tests for Forecast Engine v3 (backend/forecasting) — synthetic data only, no network."""
from __future__ import annotations

import os

os.environ.setdefault("FASTAPI_AUTH_DISABLED", "true")
os.environ.setdefault("FORECAST_CACHE_TTL_S", "0")

import numpy as np
import pandas as pd
import pytest

from backend.forecasting import BLOCKS_PER_DAY, FEATURE_VERSION, MODEL_VERSION
from backend.forecasting import models as fmodels
from backend.forecasting.baselines import recent_day, seasonal_naive
from backend.forecasting.data import DataSourceError, merge_sldc_load, normalise_load_rows, normalise_weather_rows
from backend.forecasting.features import ALL_GROUPS, build_design, build_load_matrix, calendar_frame, feature_columns
from backend.forecasting.metrics import compute_metrics, pinball_loss, wape
from backend.forecasting.physics import cooling_degree, dew_point_magnus, rolling_degree_hours, wet_bulb_stull
from backend.forecasting.quality import validate_forecast_blocks, validate_load, validate_weather
from backend.forecasting.service import ForecastService, InsufficientHistory
from backend.forecasting.spatial import (
    HARYANA_DISTRICT_WEIGHT_PCT,
    build_state_weather_features,
    weighted_mean,
)

DISTRICTS = ["GURUGRAM", "FARIDABAD", "HISAR", "ROHTAK", "SIRSA"]
START = pd.Timestamp("2025-04-01")
N_DAYS = 120


# --------------------------------------------------------------------------- synthetic data
def make_synthetic(n_days: int = N_DAYS, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temperature-driven synthetic Haryana-like load + 5-district weather."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, periods=n_days, freq="D")
    hour = (np.arange(BLOCKS_PER_DAY) + 0.5) / 4.0
    doy = days.dayofyear.to_numpy()
    seasonal = 31.0 + 5.0 * np.sin(2 * np.pi * (doy - 80) / 365.0)
    anom = np.zeros(n_days)
    for i in range(1, n_days):  # AR(1) synoptic anomaly (test data generation only)
        anom[i] = 0.6 * anom[i - 1] + rng.normal(0, 2.8)
    diurnal = 7.0 * np.sin(2 * np.pi * (hour - 9.0) / 24.0)
    offsets = np.array([0.6, 0.9, 1.8, 0.0, 1.2])
    temp = (
        seasonal[:, None, None]
        + anom[:, None, None]
        + diurnal[None, :, None]
        + offsets[None, None, :]
        + rng.normal(0, 0.6, (n_days, BLOCKS_PER_DAY, len(DISTRICTS)))
    )
    hum = np.clip(65 - 1.6 * (temp - 30) + rng.normal(0, 4, temp.shape), 8, 97)

    w = np.array([HARYANA_DISTRICT_WEIGHT_PCT[d] for d in DISTRICTS])
    tw = (temp * w).sum(axis=2) / w.sum()
    cd = np.maximum(tw - 26.0, 0.0)
    shape = 6200 + 900 * np.sin(2 * np.pi * (hour - 6) / 24) + 700 * np.exp(-0.5 * ((hour - 20.5) / 1.6) ** 2)
    week = np.where(days.dayofweek.to_numpy() >= 5, 0.93, 1.0)
    load = shape[None, :] * week[:, None] + 170.0 * cd + rng.normal(0, 35, (n_days, BLOCKS_PER_DAY))

    blk = np.arange(1, BLOCKS_PER_DAY + 1)
    load_df = pd.DataFrame(
        {"date": np.repeat(days, BLOCKS_PER_DAY), "block": np.tile(blk, n_days), "load": load.ravel()}
    )
    n = n_days * BLOCKS_PER_DAY * len(DISTRICTS)
    wx = pd.DataFrame(
        {
            "date": np.repeat(days, BLOCKS_PER_DAY * len(DISTRICTS)),
            "block": np.tile(np.repeat(blk, len(DISTRICTS)), n_days),
            "district": np.tile(DISTRICTS, n_days * BLOCKS_PER_DAY),
            "temperature": temp.ravel(),
            "humidity": hum.ravel(),
            "precipitation": np.zeros(n),
            "cloud_cover": np.full(n, 20.0),
            "wind_speed_10m": np.full(n, 3.0),
            "direct_radiation": np.clip(600 * np.sin(np.pi * (np.tile(np.repeat(hour, len(DISTRICTS)), n_days) - 6) / 12), 0, None),
        }
    )
    return load_df, wx


LOAD_DF, WX_DF = make_synthetic()


def synthetic_loader(load_df: pd.DataFrame = LOAD_DF, wx_df: pd.DataFrame = WX_DF):
    def _loader(region: str, from_date: str, to_date: str):
        f, t = pd.Timestamp(from_date), pd.Timestamp(to_date)
        lmask = (load_df["date"] >= f) & (load_df["date"] <= t)
        wmask = (wx_df["date"] >= f) & (wx_df["date"] <= t)
        return load_df.loc[lmask].reset_index(drop=True), wx_df.loc[wmask].reset_index(drop=True)

    return _loader


@pytest.fixture(scope="module")
def service(tmp_path_factory) -> ForecastService:
    return ForecastService(loader=synthetic_loader(), records_dir=tmp_path_factory.mktemp("records"))


# --------------------------------------------------------------------------- physics
def test_stull_reference_value():
    tw, valid = wet_bulb_stull(20.0, 50.0)
    assert valid
    assert abs(float(tw) - 13.7) <= 0.1


def test_stull_validity_mask_and_nan():
    tw, valid = wet_bulb_stull(np.array([20.0, 20.0, 60.0, np.nan]), np.array([2.0, 50.0, 50.0, 50.0]))
    assert valid.tolist() == [False, True, False, False]
    assert np.isnan(tw[[0, 2, 3]]).all() and np.isfinite(tw[1])


def test_dew_point_reference_value():
    assert abs(float(dew_point_magnus(25.0, 60.0)) - 16.7) <= 0.1
    assert np.isnan(dew_point_magnus(np.nan, 60.0))


def test_cooling_degree_and_rolling_cdh():
    cd = cooling_degree(np.array([20.0, 26.0, 30.0, np.nan]), base_c=24.0)
    assert cd[0] == 0.0 and cd[1] == 2.0 and cd[2] == 6.0 and np.isnan(cd[3])
    x = np.full(8, 4.0)  # 4 °C over every block → 4 blocks * 4 * 0.25 h = 4 °C·h
    r = rolling_degree_hours(x, 4)
    assert np.isnan(r[:3]).all() and np.allclose(r[3:], 4.0)
    x2 = x.copy()
    x2[5] = np.nan  # 3/4 present → rescaled, not zero-filled
    assert np.isclose(rolling_degree_hours(x2, 4)[6], 4.0)
    x2[4:7] = np.nan
    assert np.isnan(rolling_degree_hours(x2, 4)[6])


# --------------------------------------------------------------------------- spatial
def test_weighted_mean_renormalises_over_missing():
    m = np.array([[30.0, 40.0], [np.nan, 40.0], [np.nan, np.nan]])
    w = np.array([3.0, 1.0])
    mean, wp = weighted_mean(m, w)
    assert np.isclose(mean[0], 32.5) and np.isclose(mean[1], 40.0) and np.isnan(mean[2])
    assert wp.tolist() == [4.0, 1.0, 0.0]


def test_state_features_coverage_and_nan_not_zero():
    day = pd.Timestamp("2025-05-01")
    rows = []
    for b in range(1, BLOCKS_PER_DAY + 1):
        rows.append({"date": day, "block": b, "district": "Gurugram", "temperature": 40.0, "humidity": 30.0, "precipitation": np.nan})
        rows.append({"date": day, "block": b, "district": "HISAR", "temperature": 30.0 if b != 10 else np.nan, "humidity": 50.0, "precipitation": np.nan})
    wx = build_state_weather_features(pd.DataFrame(rows), "haryana")
    g, h = HARYANA_DISTRICT_WEIGHT_PCT["GURUGRAM"], HARYANA_DISTRICT_WEIGHT_PCT["HISAR"]
    total = sum(HARYANA_DISTRICT_WEIGHT_PCT.values())
    r1, r10 = wx.loc[(day, 1)], wx.loc[(day, 10)]
    assert np.isclose(r1["temp_wmean"], (40 * g + 30 * h) / (g + h))
    assert np.isclose(r1["temp_mean"], 35.0)
    assert np.isclose(r10["temp_wmean"], 40.0)  # HISAR missing → renormalised
    assert np.isclose(r1["wx_weight_coverage"], (g + h) / total)
    assert np.isclose(r10["wx_weight_coverage"], g / total)
    assert r1["wx_district_count"] == 2 and r10["wx_district_count"] == 1
    assert np.isclose(r1["hot_share"], g / (g + h))
    assert np.isnan(wx["precip_wmean"]).all()  # missing precip stays NaN
    # physics per district before aggregation: f(mean) != mean(f)
    tw_g, _ = wet_bulb_stull(40.0, 30.0)
    tw_h, _ = wet_bulb_stull(30.0, 50.0)
    assert np.isclose(r1["wb_wmean"], (tw_g * g + tw_h * h) / (g + h))
    assert np.isclose(r1["night_min_temp"], wx.loc[day, "temp_wmean"].iloc[:24].min())


def test_hot_day_streak():
    days = pd.date_range("2025-06-01", periods=4)
    temps = [39.0, 40.0, 30.0, 41.0]
    rows = [
        {"date": d, "block": b, "district": "ROHTAK", "temperature": t, "humidity": 40.0}
        for d, t in zip(days, temps)
        for b in range(1, BLOCKS_PER_DAY + 1)
    ]
    wx = build_state_weather_features(pd.DataFrame(rows), "haryana")
    streak = wx["hot_day_streak"].groupby(level="date").first().tolist()
    assert streak == [1.0, 2.0, 0.0, 1.0]


# --------------------------------------------------------------------------- data normalisation
def test_data_normalisation_and_sldc_priority():
    sldc = normalise_load_rows([
        {"date": "2025-01-01", "time_block": 1, "total_drawal": 100.0},
        {"date": "2025-01-01", "time_block": 1, "total_drawal": 110.0},
        {"date": "2025-01-01", "time_block": 3, "total_drawal": 300.0},
    ])
    aux = normalise_load_rows([
        {"date": "2025-01-01", "block": b, "load_mw": 999.0} for b in (1, 2, 3, 4)
    ])
    m = merge_sldc_load(sldc, aux).set_index("block")["load"]
    assert m[1] == 105.0 and m[2] == 999.0 and m[3] == 300.0 and m[4] == 999.0
    assert sldc.attrs["source_duplicate_blocks"] == 1
    wx = normalise_weather_rows([
        {"date": "2025-01-01", "time_block": 5, "location": "gurgaon", "temperature_2m": 20.0, "relative_humidity_2m": 40}
    ])
    assert wx.loc[0, "district"] == "GURUGRAM" and wx.loc[0, "humidity"] == 40
    assert np.isnan(wx.loc[0, "precipitation"])  # never zero-filled


# --------------------------------------------------------------------------- quality
def test_validate_load_contract():
    days = pd.date_range("2025-01-01", periods=3)
    df = pd.DataFrame({"date": np.repeat(days, 96), "block": np.tile(np.arange(1, 97), 3), "load": 5000.0})
    df = df[~((df["date"] == days[2]))]  # whole day missing
    df = pd.concat([df, df.iloc[[0]]])  # one duplicate
    df.loc[df.index[5], "load"] = -1.0
    sec, status, warns = validate_load(df, days[0], days[-1])
    assert sec["expected_blocks"] == 288 and sec["present_blocks"] == 192
    assert sec["duplicate_blocks"] == 1 and sec["nonpositive_blocks"] == 1
    assert sec["missing_dates"] == ["2025-01-03"]
    assert status == "degraded" and warns


def test_validate_weather_missing_districts():
    sec, status, warns = validate_weather(WX_DF, "2025-05-01", "2025-05-01", "haryana")
    assert sec["districts_expected"] == 21 and sec["districts_seen"] == 5
    assert "FATEHABAD" in sec["missing_districts"]
    assert status == "degraded" and any("districts missing" in w for w in warns)


def test_validate_forecast_blocks_gate():
    ok = [{"p10": 90.0, "p50": 100.0, "p90": 110.0} for _ in range(96)]
    assert validate_forecast_blocks(ok)[0] == "ok"
    assert validate_forecast_blocks(ok[:95])[0] == "failed"
    bad = [dict(b) for b in ok]
    bad[3] = {"p10": 120.0, "p50": 100.0, "p90": 110.0}
    assert validate_forecast_blocks(bad)[0] == "failed"
    bad[3] = {"p10": None, "p50": 100.0, "p90": 110.0}
    assert validate_forecast_blocks(bad)[0] == "failed"
    ramp = [dict(b) for b in ok]
    ramp[50] = {"p10": 1990.0, "p50": 2000.0, "p90": 2010.0}
    status, warns = validate_forecast_blocks(ramp, ramp_threshold_mw=500)
    assert status == "ok" and warns


# --------------------------------------------------------------------------- metrics / baselines / features
def test_metrics_basic():
    y = np.full((2, 96), 100.0)
    p = np.full((2, 96), 110.0)
    assert np.isclose(wape(y, p), 10.0)
    m = compute_metrics(y, p, p - 20, p + 20)
    assert np.isclose(m["wape"], 10.0) and np.isclose(m["accuracy_wape"], 90.0)
    assert np.isclose(m["bias_mw"], 10.0) and np.isclose(m["daily_peak_error_mw"], 10.0)
    assert m["interval_coverage_pct"] == 100.0
    assert np.isclose(pinball_loss(np.array([1.0]), np.array([[0.0, 1.0, 2.0]])), (0.1 + 0 + 0.1) / 3)


def test_recent_day_matches_day_type():
    days = pd.date_range("2025-03-03", periods=21)  # starts on a Monday
    L = np.arange(len(days), dtype=float)[:, None] * np.ones((1, 96))
    off = calendar_frame(days)["is_weekend"].to_numpy().astype(bool)  # weekends only
    tgt = np.array([14, 19])  # Monday 2025-03-17, Saturday 2025-03-22
    rd = recent_day(L, tgt, 1, off)
    assert np.allclose(rd[0], np.mean([11, 10, 9]))  # Fri, Thu, Wed before origin Sunday
    assert np.allclose(rd[1], np.mean([13, 12, 6]))  # previous Sun, Sat, Sun
    assert np.allclose(seasonal_naive(L, tgt)[0], 7)


def test_design_is_origin_safe_and_groups_selectable():
    days = pd.date_range(START, periods=N_DAYS)
    lm = build_load_matrix(LOAD_DF, days)
    wx = build_state_weather_features(WX_DF, "haryana", days=days)
    d = build_design(lm, wx, 2, target_days=days[40:41])
    L = lm.values
    assert np.allclose(d["lag_h_load"], L[38]) and np.allclose(d["lag_7d_load"], L[33])
    assert np.allclose(d["roll7_mean_load"], L[32:39].mean(axis=0))
    base_only = build_design(lm, wx, 1, target_days=days[40:41], groups=["base"])
    assert not any(c.startswith(("temp_", "wb_", "cdh")) for c in base_only.columns)
    assert set(feature_columns(ALL_GROUPS)) <= set(d.columns)


# --------------------------------------------------------------------------- service
def test_forecast_contract_shape(service):
    res = service.forecast("haryana", "2025-07-20", 1, "lgbm_residual")
    assert res["forecast_id"].startswith("hry-20250720-h1-")
    assert res["origin_date"] == "2025-07-19" and res["model_version"] == MODEL_VERSION
    assert res["feature_version"] == FEATURE_VERSION and res["created_at"].endswith("Z")
    blocks = res["blocks"]
    assert len(blocks) == 96 and [b["block"] for b in blocks] == list(range(1, 97))
    assert blocks[0]["time"] == "00:00" and blocks[95]["time"] == "23:45"
    q = np.array([[b["p10"], b["p50"], b["p90"]] for b in blocks])
    assert np.isfinite(q).all() and (q[:, 0] <= q[:, 1]).all() and (q[:, 1] <= q[:, 2]).all()
    assert all(b["actual"] is not None for b in blocks)  # past target → actuals filled
    assert np.isclose(res["summary"]["energy_mwh"], q[:, 1].sum() * 0.25, atol=1.0)
    assert set(res["drivers"]) == {
        "temp_weighted_max", "temp_p90_max", "wet_bulb_weighted_max", "wet_bulb_p90_max",
        "cdh_24h_end", "night_min_temp", "hot_share_peak", "precip_total_mm",
    }
    qual = res["quality"]
    assert set(qual) == {"status", "weather_coverage_pct", "load_share_coverage_pct", "history_days", "missing_load_blocks", "warnings"}
    assert res["status"] == qual["status"] == "degraded"  # only 5 of 21 districts
    assert 0 < qual["load_share_coverage_pct"] < 100
    assert res["feature_importance"] and {"feature", "gain_pct"} <= set(res["feature_importance"][0])
    rec = service._records_dir / f"{res['forecast_id']}.json"
    assert rec.exists()
    before = rec.read_text()
    service.forecast("haryana", "2025-07-20", 1, "lgbm_residual")  # cached model, same id
    assert rec.read_text() == before  # immutable record


def test_future_target_has_null_actuals(service):
    res = service.forecast("haryana", "2025-07-30", 2, "recent_day")  # last data day = 2025-07-29
    assert res["origin_date"] == "2025-07-28"
    assert all(b["actual"] is None for b in res["blocks"])
    assert len(res["blocks"]) == 96


def test_insufficient_history_raises():
    svc = ForecastService(loader=synthetic_loader(), records_dir=False)
    with pytest.raises(InsufficientHistory):
        svc.forecast("haryana", "2025-04-20", 1, "seasonal_naive")


def test_leakage_future_load_does_not_change_forecast():
    target, h = pd.Timestamp("2025-07-10"), 1
    origin = target - pd.Timedelta(days=h)
    tampered = LOAD_DF.copy()
    after = tampered["date"] > origin
    tampered.loc[after, "load"] = tampered.loc[after, "load"] * 1.7 + 500
    tampered = tampered.drop(tampered.index[after.to_numpy() & (tampered["block"].to_numpy() % 3 == 0)])
    a = ForecastService(loader=synthetic_loader(), records_dir=False)
    b = ForecastService(loader=synthetic_loader(tampered), records_dir=False)
    cols = a.feature_cols
    pa, pb = a.prepare("haryana", target, h), b.prepare("haryana", target, h)
    pd.testing.assert_frame_equal(pa["target_rows"][cols], pb["target_rows"][cols])
    pd.testing.assert_frame_equal(a._train_rows(pa["design"], origin), b._train_rows(pb["design"], origin))
    fa = a.forecast("haryana", target, h, "lgbm_residual")
    fb = b.forecast("haryana", target, h, "lgbm_residual")
    assert [x["p50"] for x in fa["blocks"]] == [x["p50"] for x in fb["blocks"]]
    assert fa["forecast_id"] == fb["forecast_id"]
    # backtest path (full matrix, structural shifting) yields the same features
    days = pd.date_range(START, periods=N_DAYS)
    full = build_design(build_load_matrix(LOAD_DF, days), build_state_weather_features(WX_DF, "haryana", days=days), h, target_days=[target])
    np.testing.assert_allclose(full[cols].to_numpy(), pa["target_rows"][cols].to_numpy(), equal_nan=True)


def test_backtest_lgbm_residual_beats_seasonal_naive(service, capsys):
    res = service.backtest("haryana", "2025-07-01", "2025-07-28", [1], ["seasonal_naive", "recent_day", "lgbm_residual"])
    assert res["weather_note"] and res["date_from"] == "2025-07-01" and res["date_to"] == "2025-07-28"
    by = {r["model"]: r for r in res["results"]}
    for r in by.values():
        assert r["n_days"] == 28 and r["horizon"] == 1
        assert {"wape", "accuracy_wape", "mae", "rmse", "bias_mw", "peak_wape", "daily_peak_error_mw",
                "p90_abs_error_mw", "p95_abs_error_mw", "pinball_loss", "interval_coverage_pct"} == set(r["metrics"])
    w = {k: v["metrics"]["wape"] for k, v in by.items()}
    with capsys.disabled():
        print(f"\n[v3 synthetic backtest WAPE %] {w}")
    assert w["lgbm_residual"] < w["seasonal_naive"]
    assert len(res["daily"]) == 3 * 28 and {"model", "horizon", "date", "wape", "mae", "peak_error_mw"} == set(res["daily"][0])


def test_backtest_caps_target_dates(service):
    res = service.backtest("haryana", "2025-05-01", "2025-07-29", [2], ["seasonal_naive"])
    assert res["results"][0]["n_days"] <= 62
    assert res["date_to"] == (pd.Timestamp("2025-05-01") + pd.Timedelta(days=61)).strftime("%Y-%m-%d")


def test_lightgbm_fallback_to_sklearn(monkeypatch):
    monkeypatch.setattr(fmodels, "_import_lightgbm", lambda: None)
    svc = ForecastService(loader=synthetic_loader(), records_dir=False)
    res = svc.forecast("haryana", "2025-06-15", 1, "lgbm_direct")
    assert any("HistGradientBoosting" in w for w in res["quality"]["warnings"])
    q = np.array([[b["p10"], b["p50"], b["p90"]] for b in res["blocks"]])
    assert (np.diff(q, axis=1) >= 0).all()


# --------------------------------------------------------------------------- router
@pytest.fixture(scope="module")
def client(service):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend import forecast_v3_router as r

    app = FastAPI()
    app.include_router(r.router)
    r.set_service(service)
    yield TestClient(app)
    r.set_service(None)


def test_router_models(client):
    body = client.get("/api/v3/models").json()
    assert body["model_version"] == MODEL_VERSION and body["feature_version"] == FEATURE_VERSION
    keys = {m["key"] for m in body["models"]}
    assert keys == {"lgbm_residual", "lgbm_direct", "recent_day", "seasonal_naive"}
    assert [m["key"] for m in body["models"] if m["default"]] == ["lgbm_residual"]


def test_router_forecast_and_backtest(client):
    r = client.post("/api/v3/forecast", json={"region": "haryana", "target_date": "2025-07-20", "horizon": 1, "model": "lgbm_residual"})
    assert r.status_code == 200, r.text
    assert len(r.json()["blocks"]) == 96
    r = client.post("/api/v3/backtest", json={"region": "haryana", "date_from": "2025-07-20", "date_to": "2025-07-24", "horizons": [1, 2], "models": ["seasonal_naive", "recent_day"]})
    assert r.status_code == 200, r.text
    assert len(r.json()["results"]) == 4


def test_router_data_quality(client):
    r = client.get("/api/v3/data-quality", params={"region": "haryana", "days": 30, "to": "2025-07-29"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["from"] == "2025-06-30" and body["to"] == "2025-07-29"
    assert body["load"]["expected_blocks"] == 2880 and body["load"]["present_blocks"] == 2880
    assert body["weather"]["districts_expected"] == 21 and body["status"] in {"ok", "degraded", "failed"}


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/api/v3/forecast", {"target_date": "2025-07-20", "horizon": 3}),
        ("/api/v3/forecast", {"target_date": "20-07-2025"}),
        ("/api/v3/forecast", {"target_date": "2025-07-20", "model": "prophet"}),
        ("/api/v3/backtest", {"date_from": "2025-07-20", "date_to": "2025-07-10"}),
        ("/api/v3/forecast", {"target_date": "2025-04-10"}),  # insufficient history
    ],
)
def test_router_422_with_string_detail(client, path, payload):
    r = client.post(path, json=payload)
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str) and r.json()["detail"]


def test_router_503_on_data_source_failure():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend import forecast_v3_router as r

    def broken(region, f, t):
        raise DataSourceError("pipeline API down")

    app = FastAPI()
    app.include_router(r.router)
    prev = r._service
    r.set_service(ForecastService(loader=broken, records_dir=False))
    try:
        resp = TestClient(app).post("/api/v3/forecast", json={"target_date": "2025-07-20"})
        assert resp.status_code == 503 and "pipeline API down" in resp.json()["detail"]
    finally:
        r.set_service(prev)
