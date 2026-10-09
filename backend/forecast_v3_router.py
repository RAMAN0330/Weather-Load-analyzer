"""FastAPI router for Forecast Engine v3 (contract: docs/forecast-v3-api.md).

Routes (all behind the global Bearer auth middleware):
  POST /api/v3/forecast      day-ahead quantile forecast
  POST /api/v3/backtest      rolling-origin backtest
  GET  /api/v3/data-quality  input data-quality report
  GET  /api/v3/models        available models

Error bodies are always ``{"detail": "<human readable>"}``: request validation
and insufficient history → 422, upstream data failures → 503.

Tests (or alternative deployments) can inject a service with
``set_service(ForecastService(loader=...))``.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from datetime import date
from typing import Any, List, Literal, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

try:
    from .forecasting.data import DataSourceError
    from .forecasting.models import MODEL_SPECS
    from .forecasting.service import MAX_BACKTEST_DAYS, ForecastService, InsufficientHistory
except ImportError:  # script mode (backend/ on sys.path)
    from forecasting.data import DataSourceError
    from forecasting.models import MODEL_SPECS
    from forecasting.service import MAX_BACKTEST_DAYS, ForecastService, InsufficientHistory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3", tags=["forecast-v3"])

ModelKey = Literal["lgbm_residual", "lgbm_direct", "recent_day", "seasonal_naive"]
assert set(ModelKey.__args__) == set(MODEL_SPECS)  # keep in sync with models.py

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_REGION_RE = re.compile(r"^[A-Za-z][A-Za-z _-]{1,40}$")


def _check_date(v: str, name: str) -> str:
    if not isinstance(v, str) or not _DATE_RE.match(v):
        raise ValueError(f"{name} must be a date in YYYY-MM-DD format")
    try:
        date.fromisoformat(v)
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid calendar date") from exc
    return v


def _check_region(v: str) -> str:
    if not isinstance(v, str) or not _REGION_RE.match(v.strip()):
        raise ValueError("region must be a region name such as 'haryana'")
    return v.strip().lower()


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    region: str = "haryana"
    target_date: str
    horizon: Literal[1, 2] = 1
    model: ModelKey = "lgbm_residual"

    @field_validator("region")
    @classmethod
    def _v_region(cls, v: str) -> str:
        return _check_region(v)

    @field_validator("target_date")
    @classmethod
    def _v_target(cls, v: str) -> str:
        return _check_date(v, "target_date")


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    region: str = "haryana"
    date_from: str
    date_to: str
    horizons: List[Literal[1, 2]] = Field(default_factory=lambda: [1, 2], min_length=1)
    models: List[ModelKey] = Field(
        default_factory=lambda: ["seasonal_naive", "recent_day", "lgbm_residual"], min_length=1
    )

    @field_validator("region")
    @classmethod
    def _v_region(cls, v: str) -> str:
        return _check_region(v)

    @field_validator("date_from", "date_to")
    @classmethod
    def _v_dates(cls, v: str, info) -> str:
        return _check_date(v, info.field_name)

    @model_validator(mode="after")
    def _order(self) -> "BacktestRequest":
        if date.fromisoformat(self.date_to) < date.fromisoformat(self.date_from):
            raise ValueError("date_to must be on or after date_from")
        # de-duplicate while preserving order
        self.horizons = list(dict.fromkeys(self.horizons))
        self.models = list(dict.fromkeys(self.models))
        return self


# ---------------------------------------------------------------- service DI
_service: Optional[ForecastService] = None
_service_lock = threading.Lock()


def set_service(service: Optional[ForecastService]) -> None:
    """Inject the ForecastService used by the routes (``None`` resets to default)."""
    global _service
    with _service_lock:
        _service = service


def get_service() -> ForecastService:
    global _service
    with _service_lock:
        if _service is None:
            _service = ForecastService()
        return _service


# ------------------------------------------------------------------- helpers
def _validation_detail(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "__root__")
        msg = str(err.get("msg", "invalid value")).removeprefix("Value error, ")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or "Invalid request"


def _parse(model_cls: type[BaseModel], payload: Any) -> Any:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object")
    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_validation_detail(exc)) from exc


async def _run(fn, *args, **kwargs) -> Any:
    """Run blocking work off the event loop and map domain errors to HTTP."""
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except InsufficientHistory as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DataSourceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected
        logger.exception("forecast v3 failure")
        raise HTTPException(status_code=500, detail=f"Forecast engine error: {type(exc).__name__}") from exc


# -------------------------------------------------------------------- routes
@router.post("/forecast")
async def post_forecast(payload: Any = Body(...)) -> dict:
    req: ForecastRequest = _parse(ForecastRequest, payload)
    return await _run(get_service().forecast, req.region, req.target_date, req.horizon, req.model)


@router.post("/backtest")
async def post_backtest(payload: Any = Body(...)) -> dict:
    req: BacktestRequest = _parse(BacktestRequest, payload)
    return await _run(get_service().backtest, req.region, req.date_from, req.date_to, req.horizons, req.models)


@router.get("/data-quality")
async def get_data_quality(
    region: str = Query("haryana"),
    days: str = Query("30"),
    to: Optional[str] = Query(None, description="Optional last date (YYYY-MM-DD); default yesterday IST"),
) -> dict:
    try:
        region_n = _check_region(region)
        if to is not None:
            _check_date(to, "to")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not days.isdigit() or not 1 <= int(days) <= 366:
        raise HTTPException(status_code=422, detail="days must be an integer between 1 and 366")
    return await _run(get_service().data_quality, region_n, int(days), to)


@router.get("/models")
async def get_models() -> dict:
    return ForecastService.models()


__all__ = ["router", "set_service", "get_service", "ForecastRequest", "BacktestRequest", "MAX_BACKTEST_DAYS"]
