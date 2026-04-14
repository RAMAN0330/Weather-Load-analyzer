"""
pipeline/views.py

Django REST Framework views that replicate every endpoint in api.py.
Token authentication mirrors the FastAPI bearer-token pattern.

All read endpoints accept:
    ?state=HARYANA&from_date=2025-01-01&to_date=2025-01-31&limit=200
"""

from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import AuthenticationFailed, ValidationError
from rest_framework import status

from .db_utils import (
    TABLE_MAP, fetch_rows, fetch_states, fetch_table_counts,
    fetch_count, fetch_date_range, upsert_rows,
)


# ── Auth helper ───────────────────────────────────────────────────────────────

def _check_token(request):
    """Raise AuthenticationFailed if the Bearer token is wrong/missing."""
    secret = settings.API_SECRET_KEY
    if not secret:
        raise AuthenticationFailed("API_SECRET_KEY not configured on server.")
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != secret:
        raise AuthenticationFailed("Invalid or missing Bearer token.")


def _require_param(request, name: str) -> str:
    val = request.query_params.get(name)
    if not val:
        raise ValidationError({name: "This query parameter is required."})
    return val


# ── Health ────────────────────────────────────────────────────────────────────

class HealthView(APIView):
    def get(self, request):
        return Response({
            "status": "ok",
            "db": settings.DATABASES["default"]["NAME"],
            "docs": "Use /states, /tables, /weather/mean, /load, /forecast, /sldc …",
        })


# ── Utility endpoints ─────────────────────────────────────────────────────────

class StatesView(APIView):
    def get(self, request):
        _check_token(request)
        return Response(fetch_states())


class TablesView(APIView):
    def get(self, request):
        _check_token(request)
        return Response(fetch_table_counts())


class CountView(APIView):
    def get(self, request):
        _check_token(request)
        state = _require_param(request, "state")
        table = request.query_params.get("table")
        if table and table not in TABLE_MAP:
            raise ValidationError({"table": f"Unknown table '{table}'. Valid: {list(TABLE_MAP)}"})
        return Response(fetch_count(state, table or None))


class DateRangeView(APIView):
    def get(self, request):
        _check_token(request)
        state = _require_param(request, "state")
        table = request.query_params.get("table", "load")
        if table not in TABLE_MAP:
            raise ValidationError({"table": f"Unknown table '{table}'."})
        return Response(fetch_date_range(state, table))


# ── Read endpoints ────────────────────────────────────────────────────────────

def _read_params(request):
    """Extract common date-filter params from query string."""
    return {
        "state":     _require_param(request, "state"),
        "date":      request.query_params.get("date"),
        "from_date": request.query_params.get("from_date"),
        "to_date":   request.query_params.get("to_date"),
        "limit":     min(int(request.query_params.get("limit", 200)), 200_000),
    }


class WeatherMeanView(APIView):
    def get(self, request):
        _check_token(request)
        p = _read_params(request)
        rows = fetch_rows("weather_mean", **p)
        return Response(rows)


class WeatherLocView(APIView):
    def get(self, request):
        _check_token(request)
        p = _read_params(request)
        location = request.query_params.get("location")
        extra_where = "location = %s" if location else ""
        extra_params = [location] if location else []
        rows = fetch_rows(
            "weather_loc",
            state=p["state"],
            date=p["date"],
            from_date=p["from_date"],
            to_date=p["to_date"],
            limit=p["limit"],
            extra_where=extra_where,
            extra_params=extra_params,
            order_by="date, time_block, location",
        )
        return Response(rows)


class LoadView(APIView):
    def get(self, request):
        _check_token(request)
        p = _read_params(request)
        rows = fetch_rows("load", **p)
        return Response(rows)


class ForecastView(APIView):
    def get(self, request):
        _check_token(request)
        p = _read_params(request)
        rows = fetch_rows("forecast", order_by="date, block", **p)
        return Response(rows)


class SldcView(APIView):
    def get(self, request):
        _check_token(request)
        p = _read_params(request)
        rows = fetch_rows("sldc", **p)
        return Response(rows)


class SldcForecastView(APIView):
    def get(self, request):
        _check_token(request)
        p = _read_params(request)
        rows = fetch_rows("sldc_forecast", **p)
        return Response(rows)


# ── Write endpoints ───────────────────────────────────────────────────────────

class UpsertView(APIView):
    """
    POST /upsert
    Body: {"state": "HARYANA", "table": "load", "rows": [{...}, ...]}
    """
    def post(self, request):
        _check_token(request)
        data = request.data
        table = (data.get("table") or "").lower()
        state = data.get("state") or ""
        rows = data.get("rows") or []

        if table not in TABLE_MAP:
            raise ValidationError({"table": f"Unknown table '{table}'. Valid: {list(TABLE_MAP)}"})
        if not state:
            raise ValidationError({"state": "Required."})

        result = upsert_rows(table, state, list(rows))
        return Response({"status": "ok", **result})


class BulkUpsertView(APIView):
    """
    POST /bulk-upsert
    Body: {"state": "HARYANA", "data": {"load": [{...}], "weather_mean": [{...}]}}
    """
    def post(self, request):
        _check_token(request)
        data = request.data
        state = data.get("state") or ""
        tables_data = data.get("data") or {}

        if not state:
            raise ValidationError({"state": "Required."})

        results = {}
        for table_key, rows in tables_data.items():
            key = table_key.lower()
            if key not in TABLE_MAP:
                results[table_key] = {"status": "error", "detail": f"Unknown table '{key}'."}
                continue
            try:
                res = upsert_rows(key, state, list(rows))
                results[table_key] = {"status": "ok", **res}
            except Exception as exc:
                results[table_key] = {"status": "error", "detail": str(exc)}

        return Response(results)


