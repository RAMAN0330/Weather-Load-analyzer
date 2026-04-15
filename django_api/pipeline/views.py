"""
pipeline/views.py

Django REST Framework views that replicate every endpoint in api.py.
Token authentication mirrors the FastAPI bearer-token pattern.

All read endpoints accept:
    ?state=HARYANA&from_date=2025-01-01&to_date=2025-01-31&limit=200
"""

import math
from datetime import date as date_type, datetime, timedelta
from decimal import Decimal

import requests
from django.conf import settings
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import AuthenticationFailed, ValidationError
from rest_framework import status

from .db_utils import (
    TABLE_MAP, fetch_rows, fetch_states, fetch_table_counts,
    fetch_count, fetch_date_range, upsert_rows, raw_query,
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


# ── FastAPI compatibility gateway for unported compute-only tools ────────────

class FastApiProxyView(APIView):
    """Route legacy simulator/compute endpoints through Django without moving ML training."""

    def dispatch(self, request, *args, **kwargs):
        prefix = (kwargs.get("prefix") or "").strip("/")
        path = (kwargs.get("path") or "").strip("/")
        if prefix:
            path = f"{prefix}/{path}" if path else prefix
        url = f"{settings.FASTAPI_BASE_URL}/api/{path.lstrip('/')}"
        headers = {}
        content_type = request.headers.get("Content-Type")
        if content_type:
            headers["Content-Type"] = content_type
        auth = request.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        try:
            upstream = requests.request(
                method=request.method,
                url=url,
                params=request.GET,
                data=request.body if request.method not in ("GET", "HEAD") else None,
                headers=headers,
                timeout=180,
            )
        except requests.RequestException as exc:
            return Response({"detail": f"FastAPI compatibility route failed: {exc}"}, status=502)

        response = HttpResponse(
            upstream.content,
            status=upstream.status_code,
            content_type=upstream.headers.get("Content-Type", "application/json"),
        )
        return response


# ── Django app API (fast, no model training) ─────────────────────────────────

LOAD_ALIASES = ("total_drawal", "load", "load_mw", "actual", "actual_load", "mw", "demand")
FORECAST_ALIASES = ("forecasted", "forecast", "forecast_mw", "predicted")
TEMP_ALIASES = ("temperature", "temperature_2m", "temp")
HUMIDITY_ALIASES = ("humidity", "relative_humidity_2m", "relative_humidity")
PRECIP_ALIASES = ("precipitation", "rain", "rainfall")


def _clean_json(value):
    if isinstance(value, dict):
        return {k: _clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [_clean_json(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _date_str(value):
    if value is None:
        return None
    if hasattr(value, "date") and not isinstance(value, date_type):
        value = value.date()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)[:10]


def _state_from_request(request, default="HARYANA"):
    data = getattr(request, "data", {}) or {}
    raw = data.get("state") or data.get("region") or request.query_params.get("state") or default
    return str(raw).strip().upper().replace("-", "_").replace(" ", "_")


def _first_value(row, aliases):
    for key in aliases:
        if key in row:
            val = row.get(key)
            try:
                f = float(val)
                return f if math.isfinite(f) else None
            except (TypeError, ValueError):
                return None
    return None


def _block_no(row):
    val = row.get("time_block", row.get("block"))
    try:
        n = int(float(val))
        return n if 1 <= n <= 96 else None
    except (TypeError, ValueError):
        return None


def _vector_from_rows(rows, aliases=LOAD_ALIASES):
    sums = {i: 0.0 for i in range(1, 97)}
    counts = {i: 0 for i in range(1, 97)}
    for row in rows:
        block = _block_no(row)
        val = _first_value(row, aliases)
        if block is None or val is None or val <= 1:
            continue
        sums[block] += val
        counts[block] += 1
    return [round(sums[i] / counts[i], 6) if counts[i] else None for i in range(1, 97)]


def _date_offset(date_str, days):
    return (datetime.fromisoformat(str(date_str)[:10]) + timedelta(days=days)).date().isoformat()


def _load_dates(state):
    rows = raw_query(
        """
        SELECT date, COUNT(*) AS rows_count
        FROM `load`
        WHERE state = %s
        GROUP BY date
        ORDER BY date
        """,
        (state,),
    )
    return [_date_str(r["date"]) for r in rows if r.get("date") is not None]


def _latest_date(state):
    dates = _load_dates(state)
    return dates[-1] if dates else None


def _from_date_for_days(state, table_key, days):
    date_range = fetch_date_range(state, table_key)
    latest = _date_str(date_range.get("max_date")) if date_range else None
    if not latest:
        return None
    return _date_offset(latest, -max(int(days or 60), 1) + 1)


def _load_day_vector(state, day):
    rows = fetch_rows("load", state=state, date=day, limit=100_000)
    return _vector_from_rows(rows, LOAD_ALIASES)


def _pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 4:
        return None
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    x_mean = sum(x_vals) / len(x_vals)
    y_mean = sum(y_vals) / len(y_vals)
    num = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    den_x = sum((x - x_mean) ** 2 for x in x_vals)
    den_y = sum((y - y_mean) ** 2 for y in y_vals)
    if den_x <= 0 or den_y <= 0:
        return None
    return num / math.sqrt(den_x * den_y)


class V2ConfigView(APIView):
    def get(self, request):
        state = _state_from_request(request)
        dates = _load_dates(state)
        latest = dates[-1] if dates else None
        available = [str(s).lower() for s in fetch_states()]
        return Response(_clean_json({
            "dates": dates,
            "all_dates": dates,
            "latest_date": latest,
            "partial_latest_date": latest,
            "default_date": latest,
            "best_baseline_window": 7,
            "best_mape": None,
            "baseline_window_mapes": [],
            "available_regions": available,
            "default_region": str(state).lower(),
            "source": "django",
        }))


class V2SettingsView(APIView):
    def get(self, request):
        return Response({
            "alert_thresholds": {"warning_pct": 5, "critical_pct": 10},
            "baseline_defaults": {"window_days": 7, "weather_similarity": True},
            "tariffs": {
                "energy_charge_per_kwh": 6.5,
                "demand_charge_per_kw": 350,
                "deviation_penalty_per_kwh": 8.0,
            },
            "source": "django",
        })


class V2ReloadView(APIView):
    def post(self, request):
        state = _state_from_request(request)
        dates = _load_dates(state)
        counts = fetch_count(state, "load")
        return Response(_clean_json({
            "status": "ok",
            "rows": counts.get("load", 0),
            "latest_date": dates[-1] if dates else None,
            "partial_latest_date": dates[-1] if dates else None,
            "source": "django",
        }))


class V2LoadBenchmarksView(APIView):
    def post(self, request):
        state = _state_from_request(request)
        date = (request.data or {}).get("date") or _latest_date(state)
        if not date:
            raise ValidationError({"date": "No load dates available."})
        t1 = _date_offset(date, -1)
        t7 = _date_offset(date, -7)
        t365 = _date_offset(date, -365)
        return Response(_clean_json({
            "today": _load_day_vector(state, date),
            "t1": _load_day_vector(state, t1),
            "t7": _load_day_vector(state, t7),
            "t365": _load_day_vector(state, t365),
            "dates": {"today": date, "t1": t1, "t7": t7, "t365": t365},
            "source": "django",
        }))


class V2LoadSeriesView(APIView):
    def post(self, request):
        state = _state_from_request(request)
        dates = (request.data or {}).get("dates") or []
        return Response(_clean_json({
            "series": {str(d): _load_day_vector(state, str(d)) for d in dates},
            "source": "django",
        }))


class V2LoadChangeView(APIView):
    def post(self, request):
        state = _state_from_request(request)
        payload = request.data or {}
        dates = payload.get("dates") or []
        if not dates or len(dates) <= 1:
            dates = _load_dates(state)
        if len(dates) > 1:
            daily = []
            totals = {}
            for d in dates:
                vals = [v for v in _load_day_vector(state, str(d)) if v is not None]
                totals[str(d)] = sum(vals)
            sorted_dates = sorted(str(d) for d in dates)
            for prev, curr in zip(sorted_dates, sorted_dates[1:]):
                if totals.get(prev, 0) > 0:
                    daily.append({
                        "date": curr,
                        "value": round(((totals.get(curr, 0) - totals[prev]) / totals[prev]) * 100, 2),
                    })
            return Response({"type": "daily", "data": daily, "source": "django"})

        date1 = payload.get("date1") or (dates[0] if dates else None)
        date2 = payload.get("date2")
        v1 = _load_day_vector(state, date1) if date1 else [None] * 96
        v2 = _load_day_vector(state, date2) if date2 else [None] * 96
        change = []
        for a, b in zip(v1, v2):
            change.append(round(((a or 0) - (b or 0)) / max(b or 0, 1e-6) * 100, 2) if a is not None else None)
        return Response({"type": "blocks", "blocks": list(range(1, 97)), "change_pct": change, "source": "django"})


class V2AnalysisView(APIView):
    def post(self, request):
        state = _state_from_request(request)
        requested = (request.data or {}).get("date") or _latest_date(state)
        if not requested:
            raise ValidationError({"date": "No load dates available."})
        dates = _load_dates(state)
        if requested in dates:
            end_idx = dates.index(requested) + 1
            window_dates = dates[max(0, end_idx - 7):end_idx]
        else:
            window_dates = dates[-7:]
        if not window_dates:
            return Response({"date": requested, "correlations": {}, "attribution": [], "source": "django"})

        load_rows = fetch_rows("load", state, from_date=window_dates[0], to_date=window_dates[-1], limit=200_000)
        weather_rows = fetch_rows("weather_mean", state, from_date=window_dates[0], to_date=window_dates[-1], limit=200_000)
        load_map = {
            (_date_str(r.get("date")), _block_no(r)): _first_value(r, LOAD_ALIASES)
            for r in load_rows
        }
        weather_map = {
            (_date_str(r.get("date")), _block_no(r)): r
            for r in weather_rows
        }
        correlations = {}
        for label, aliases in {
            "temperature": TEMP_ALIASES,
            "humidity": HUMIDITY_ALIASES,
            "precipitation": PRECIP_ALIASES,
        }.items():
            x_vals, y_vals = [], []
            for key, load_val in load_map.items():
                wx = weather_map.get(key)
                wx_val = _first_value(wx or {}, aliases)
                if load_val is not None and wx_val is not None:
                    x_vals.append(load_val)
                    y_vals.append(wx_val)
            corr = _pearson(x_vals, y_vals)
            if corr is not None:
                correlations[label] = round(corr, 3)
        return Response(_clean_json({
            "date": requested,
            "correlations": correlations,
            "attribution": [],
            "source": "django",
        }))


# ── /api/pipeline compatibility endpoints for frontend DB tools ──────────────

class PipelineStatesCompatView(APIView):
    def get(self, request):
        return Response(fetch_states())


class PipelineWeatherCompatView(APIView):
    table_key = "weather_mean"

    def get(self, request, state):
        days = int(request.query_params.get("days", 60))
        from_date = _from_date_for_days(state.upper(), self.table_key, days)
        rows = fetch_rows(self.table_key, state.upper(), from_date=from_date, limit=200_000)
        return Response(_clean_json(rows))


class PipelineWeatherLocCompatView(PipelineWeatherCompatView):
    table_key = "weather_loc"


class PipelineLoadCompatView(APIView):
    def get(self, request, state):
        days = int(request.query_params.get("days", 60))
        state = state.upper()
        from_date = _from_date_for_days(state, "load", days)
        rows = fetch_rows("load", state, from_date=from_date, limit=200_000)
        out = []
        for row in rows:
            out.append({
                "date": _date_str(row.get("date")),
                "time_block": _block_no(row),
                "load": _first_value(row, LOAD_ALIASES),
            })
        return Response(_clean_json(out))


class PipelineForecastCompatView(APIView):
    def get(self, request, state):
        days = int(request.query_params.get("days", 60))
        state = state.upper()
        from_date = _from_date_for_days(state, "forecast", days)
        rows = fetch_rows("forecast", state, from_date=from_date, limit=200_000, order_by="date, block")
        out = []
        for row in rows:
            out.append({
                "date": _date_str(row.get("date")),
                "time_block": _block_no(row),
                "forecasted": _first_value(row, FORECAST_ALIASES),
            })
        return Response(_clean_json(out))


class PipelineSimilarityCompatView(APIView):
    def get(self, request, state):
        state = state.upper()
        target_date = request.query_params.get("target_date")
        top_k = int(request.query_params.get("top_k", 5))
        if not target_date:
            raise ValidationError({"target_date": "Required."})

        rows = fetch_rows("weather_mean", state, limit=200_000)
        profiles = {}
        for row in rows:
            d = _date_str(row.get("date"))
            block = _block_no(row)
            if not d or block is None:
                continue
            profiles.setdefault(d, {})[block] = {
                "temp": _first_value(row, TEMP_ALIASES),
                "humidity": _first_value(row, HUMIDITY_ALIASES),
                "precip": _first_value(row, PRECIP_ALIASES),
            }
        if target_date not in profiles:
            return Response({"target_date": target_date, "similar": [], "profiles": {}})

        def _flat(profile):
            vec = []
            for block in range(1, 97):
                slot = profile.get(block, {})
                vec.extend([slot.get("temp") or 0.0, slot.get("humidity") or 0.0, slot.get("precip") or 0.0])
            return vec

        target = _flat(profiles[target_date])
        similar = []
        for d, profile in profiles.items():
            if d == target_date:
                continue
            vec = _flat(profile)
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, vec)))
            similar.append({"date": d, "distance": round(dist, 4)})
        similar.sort(key=lambda x: x["distance"])
        similar = similar[:top_k]

        selected = [target_date] + [r["date"] for r in similar]
        profile_out = {}
        for d in selected:
            profile_out[d] = {
                "temp": [(profiles[d].get(i, {}).get("temp")) for i in range(1, 97)],
                "humidity": [(profiles[d].get(i, {}).get("humidity")) for i in range(1, 97)],
                "precip": [(profiles[d].get(i, {}).get("precip")) for i in range(1, 97)],
            }

        return Response(_clean_json({
            "target_date": target_date,
            "similar": similar,
            "profiles": profile_out,
            "source": "django",
        }))
