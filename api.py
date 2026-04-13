"""
api.py – FastAPI server with READ + WRITE endpoints for rishu_db.

Run standalone:
    uvicorn api:app --host 0.0.0.0 --port 8001 --reload

Interactive docs: http://localhost:8001/docs

All endpoints require:  Authorization: Bearer <API_SECRET_KEY>
"""

import math
from typing import Optional, List, Dict, Any
import pymysql
from fastapi import FastAPI, Depends, HTTPException, Query, Body, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD,
    API_SECRET_KEY,
)

app = FastAPI(
    title="Load Forecast API",
    description="Read & write energy pipeline data",
    version="1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer = HTTPBearer()

TABLE_MAP = {
    "weather_mean": "weather_mean",
    "weather_loc":  "weather_loc",
    "load":         "`load`",
    "forecast":     "forecast",
    "sldc":         "sldc",
    "sldc_forecast": "sldc_forecast",
}

PK_COLS = {
    "weather_mean": ["state", "date", "time_block"],
    "weather_loc":  ["state", "date", "time_block", "location"],
    "load":         ["state", "date", "time_block"],
    "forecast":     ["state", "date", "block"],
    "sldc":         ["state", "date", "time_block"],
    "sldc_forecast": ["state", "date", "time_block"],
}


def _conn():
    return pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT,
        user=MYSQL_USER, password=MYSQL_PASSWORD,
        database=MYSQL_DB, charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def _query(sql: str, params=()):
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        c.close()


def _execute(sql: str, params_list: list):
    """Execute a write query with executemany. Returns affected row count."""
    c = _conn()
    try:
        with c.cursor() as cur:
            cur.executemany(sql, params_list)
            n = cur.rowcount
        c.commit()
        return n
    finally:
        c.close()


def _clean(v):
    """Replace NaN / inf / NaT / empty-string with None."""
    if v is None:
        return None
    if isinstance(v, str) and v.lower() in ("nat", "nan", "none", "null", ""):
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v

def require_token(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    if not API_SECRET_KEY:
        raise HTTPException(500, "API_SECRET_KEY not configured on server.")
    if creds.credentials != API_SECRET_KEY:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token.")
    return creds.credentials


# ─────────────────────────────────────────────────────────────
# Shared WHERE builder
# ─────────────────────────────────────────────────────────────

def _where(state, date=None, from_date=None, to_date=None):
    cond, p = ["state=%s"], [state.upper()]
    if date:
        cond.append("date=%s"); p.append(date)
    else:
        if from_date: cond.append("date>=%s"); p.append(from_date)
        if to_date:   cond.append("date<=%s"); p.append(to_date)
    return " AND ".join(cond), p


# ═════════════════════════════════════════════════════════════
# READ  endpoints
# ═════════════════════════════════════════════════════════════

@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "db": MYSQL_DB, "docs": "/docs"}


@app.get("/states", tags=["read"])
def get_states(_=Depends(require_token)):
    """List distinct states across all tables."""
    states = set()
    for table in TABLE_MAP.values():
        try:
            rows = _query(f"SELECT DISTINCT state FROM {table}")
            states.update(r["state"] for r in rows)
        except Exception:
            pass
    return sorted(states)


@app.get("/tables", tags=["read"])
def get_tables(_=Depends(require_token)):
    """List all tables and their row counts."""
    rows = _query("SHOW TABLES")
    key = list(rows[0].keys())[0] if rows else "Tables"
    result = {}
    for r in rows:
        tbl = r[key]
        cnt = _query(f"SELECT COUNT(*) as c FROM `{tbl}`")
        result[tbl] = cnt[0]["c"]
    return result


@app.get("/weather/mean", tags=["read"])
def get_weather_mean(
    state: str,
    date: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = Query(200, le=200000),
    _=Depends(require_token),
):
    w, p = _where(state, date, from_date, to_date)
    return _query(
        f"SELECT * FROM weather_mean WHERE {w} ORDER BY date, time_block LIMIT %s",
        tuple(p + [limit]),
    )


@app.get("/weather/loc", tags=["read"])
def get_weather_loc(
    state: str,
    date: Optional[str] = None,
    location: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = Query(200, le=200000),
    _=Depends(require_token),
):
    w, p = _where(state, date, from_date, to_date)
    if location:
        w += " AND location=%s"; p.append(location)
    return _query(
        f"SELECT * FROM weather_loc WHERE {w} ORDER BY date, time_block, location LIMIT %s",
        tuple(p + [limit]),
    )


@app.get("/load", tags=["read"])
def get_load(
    state: str,
    date: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = Query(200, le=200000),
    _=Depends(require_token),
):
    w, p = _where(state, date, from_date, to_date)
    return _query(
        f"SELECT * FROM `load` WHERE {w} ORDER BY date, time_block LIMIT %s",
        tuple(p + [limit]),
    )


@app.get("/forecast", tags=["read"])
def get_forecast(
    state: str,
    date: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = Query(200, le=200000),
    _=Depends(require_token),
):
    w, p = _where(state, date, from_date, to_date)
    return _query(
        f"SELECT * FROM forecast WHERE {w} ORDER BY date, block LIMIT %s",
        tuple(p + [limit]),
    )


@app.get("/sldc", tags=["read"])
def get_sldc(
    state: str,
    date: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = Query(200, le=200000),
    _=Depends(require_token),
):
    w, p = _where(state, date, from_date, to_date)
    return _query(
        f"SELECT * FROM sldc WHERE {w} ORDER BY date, time_block LIMIT %s",
        tuple(p + [limit]),
    )


@app.get("/sldc/forecast", tags=["read"])
def get_sldc_forecast(
    state: str,
    date: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = Query(200, le=200000),
    _=Depends(require_token),
):
    w, p = _where(state, date, from_date, to_date)
    return _query(
        f"SELECT * FROM sldc_forecast WHERE {w} ORDER BY date, time_block LIMIT %s",
        tuple(p + [limit]),
    )


# ═════════════════════════════════════════════════════════════
# WRITE  endpoints
# ═════════════════════════════════════════════════════════════

class UpsertPayload(BaseModel):
    """
    POST body for writing data.

    Example:
    {
        "state": "RAJASTHAN",
        "table": "load",
        "rows": [
            {"date": "2025-04-01", "time_block": 1, "total_drawal": 5200.0},
            {"date": "2025-04-01", "time_block": 2, "total_drawal": 5150.0}
        ]
    }
    """
    state: str
    table: str
    rows: List[Dict[str, Any]]


@app.post("/upsert", tags=["write"])
def upsert_data(payload: UpsertPayload, _=Depends(require_token)):
    """
    UPSERT rows into any table. Automatically adds the `state` column.
    Uses ON DUPLICATE KEY UPDATE for conflict resolution.
    """
    suffix = payload.table.lower()
    if suffix not in TABLE_MAP:
        raise HTTPException(400, f"Unknown table '{suffix}'. Valid: {list(TABLE_MAP)}")

    if not payload.rows:
        return {"status": "ok", "rows_affected": 0}

    table = TABLE_MAP[suffix]
    state = payload.state.strip().upper()

    # Inject state and collect all column names
    for row in payload.rows:
        row["state"] = state

    all_cols = list(payload.rows[0].keys())
    # Remove 'id' if present
    all_cols = [c for c in all_cols if c.lower() != "id"]

    pk = set(PK_COLS.get(suffix, []))
    non_pk = [c for c in all_cols if c not in pk]

    col_sql    = ", ".join(f"`{c}`" for c in all_cols)
    val_sql    = ", ".join(["%s"] * len(all_cols))
    update_sql = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in (non_pk or all_cols[:1]))

    sql = (f"INSERT INTO {table} ({col_sql}) VALUES ({val_sql}) "
           f"ON DUPLICATE KEY UPDATE {update_sql}")

    params_list = [
        tuple(_clean(row.get(c)) for c in all_cols)
        for row in payload.rows
    ]

    n = _execute(sql, params_list)
    return {"status": "ok", "rows_affected": n, "table": suffix, "state": state}


class BulkUpsertPayload(BaseModel):
    """
    POST body for writing to multiple tables at once.

    Example:
    {
        "state": "HARYANA",
        "data": {
            "weather_mean": [ {row1}, {row2}, ... ],
            "load": [ {row1}, {row2}, ... ]
        }
    }
    """
    state: str
    data: Dict[str, List[Dict[str, Any]]]


@app.post("/bulk-upsert", tags=["write"])
def bulk_upsert(payload: BulkUpsertPayload, _=Depends(require_token)):
    """UPSERT rows into multiple tables in one call."""
    results = {}
    for table_suffix, rows in payload.data.items():
        inner = UpsertPayload(state=payload.state, table=table_suffix, rows=rows)
        try:
            res = upsert_data(inner, _)
            results[table_suffix] = res
        except HTTPException as e:
            results[table_suffix] = {"status": "error", "detail": e.detail}
    return results


@app.delete("/delete", tags=["write"])
def delete_data(
    state: str,
    table: str,
    date: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    _=Depends(require_token),
):
    """Delete rows for a state (optionally filtered by date range)."""
    suffix = table.lower()
    if suffix not in TABLE_MAP:
        raise HTTPException(400, f"Unknown table '{suffix}'. Valid: {list(TABLE_MAP)}")

    tbl = TABLE_MAP[suffix]
    w, p = _where(state, date, from_date, to_date)

    c = _conn()
    try:
        with c.cursor() as cur:
            cur.execute(f"DELETE FROM {tbl} WHERE {w}", tuple(p))
            n = cur.rowcount
        c.commit()
    finally:
        c.close()

    return {"status": "ok", "rows_deleted": n, "table": suffix, "state": state.upper()}


# ═════════════════════════════════════════════════════════════
# UTILITY endpoints
# ═════════════════════════════════════════════════════════════

@app.get("/count", tags=["utility"])
def row_count(
    state: str,
    table: Optional[str] = None,
    _=Depends(require_token),
):
    """Get row counts for a state, optionally for a specific table."""
    state = state.upper()
    if table:
        suffix = table.lower()
        if suffix not in TABLE_MAP:
            raise HTTPException(400, f"Unknown table '{suffix}'.")
        tbl = TABLE_MAP[suffix]
        rows = _query(f"SELECT COUNT(*) as c FROM {tbl} WHERE state=%s", (state,))
        return {suffix: rows[0]["c"]}

    counts = {}
    for suffix, tbl in TABLE_MAP.items():
        try:
            rows = _query(f"SELECT COUNT(*) as c FROM {tbl} WHERE state=%s", (state,))
            counts[suffix] = rows[0]["c"]
        except Exception:
            counts[suffix] = 0
    return counts


@app.get("/date-range", tags=["utility"])
def date_range(
    state: str,
    table: str = "load",
    _=Depends(require_token),
):
    """Get min/max dates for a state in a given table."""
    suffix = table.lower()
    if suffix not in TABLE_MAP:
        raise HTTPException(400, f"Unknown table '{suffix}'.")
    tbl = TABLE_MAP[suffix]
    rows = _query(
        f"SELECT MIN(date) as min_date, MAX(date) as max_date FROM {tbl} WHERE state=%s",
        (state.upper(),),
    )
    return rows[0] if rows else {"min_date": None, "max_date": None}
