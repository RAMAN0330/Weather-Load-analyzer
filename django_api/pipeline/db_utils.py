"""
pipeline/db_utils.py

Low-level helpers that use Django's database connection for raw SQL.
Used for UPSERT (ON DUPLICATE KEY UPDATE) and DELETE operations that
involve composite primary keys Django ORM can't express natively.
"""

import math

from django.db import connection

# ── Table registry ────────────────────────────────────────────────────────────

TABLE_MAP = {
    "weather_mean":  "weather_mean",
    "weather_loc":   "weather_loc",
    "load":          "`load`",        # reserved word
    "forecast":      "forecast",
    "sldc":          "sldc",
    "sldc_forecast": "sldc_forecast",
}

PK_COLS = {
    "weather_mean":  ["state", "date", "time_block"],
    "weather_loc":   ["state", "date", "time_block", "location"],
    "load":          ["state", "date", "time_block"],
    "forecast":      ["state", "date", "block"],
    "sldc":          ["state", "date", "time_block"],
    "sldc_forecast": ["state", "date", "time_block"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(v):
    """Replace NaN / inf / NaT / empty-string sentinel values with None."""
    if v is None:
        return None
    if isinstance(v, str) and v.lower() in ("nat", "nan", "none", "null", ""):
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _where_clause(state: str, date=None, from_date=None, to_date=None):
    """Build a WHERE clause for state + optional date filter."""
    cond = ["state = %s"]
    params = [state.upper()]
    if date:
        cond.append("date = %s")
        params.append(date)
    else:
        if from_date:
            cond.append("date >= %s")
            params.append(from_date)
        if to_date:
            cond.append("date <= %s")
            params.append(to_date)
    return " AND ".join(cond), params


def raw_query(sql: str, params=()):
    """Execute a SELECT and return a list of dicts."""
    with connection.cursor() as cur:
        cur.execute(sql, params)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def raw_execute(sql: str, params_list: list) -> int:
    """Execute a write statement (executemany). Returns rowcount."""
    with connection.cursor() as cur:
        cur.executemany(sql, params_list)
        return cur.rowcount


def raw_execute_one(sql: str, params: tuple) -> int:
    """Execute a single write statement. Returns rowcount."""
    with connection.cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount


# ── Read helpers ──────────────────────────────────────────────────────────────

def fetch_rows(table_key: str, state: str, date=None, from_date=None,
               to_date=None, limit: int = 200, extra_where: str = "",
               extra_params: list = None, order_by: str = "date, time_block"):
    """Generic SELECT * for any table with state/date filters."""
    tbl = TABLE_MAP[table_key]
    w, params = _where_clause(state, date, from_date, to_date)
    if extra_where:
        w += f" AND {extra_where}"
        params += (extra_params or [])
    params.append(limit)
    return raw_query(
        f"SELECT * FROM {tbl} WHERE {w} ORDER BY {order_by} LIMIT %s",
        tuple(params),
    )


def fetch_states():
    """Return sorted list of distinct states across all tables."""
    states = set()
    for tbl in TABLE_MAP.values():
        try:
            rows = raw_query(f"SELECT DISTINCT state FROM {tbl}")
            states.update(r["state"] for r in rows)
        except Exception:
            pass
    return sorted(states)


def fetch_table_counts():
    """Return {table_name: row_count} for all known tables."""
    rows = raw_query("SHOW TABLES")
    key = list(rows[0].keys())[0] if rows else "Tables"
    result = {}
    for r in rows:
        tbl = r[key]
        cnt = raw_query(f"SELECT COUNT(*) AS c FROM `{tbl}`")
        result[tbl] = cnt[0]["c"]
    return result


def fetch_count(state: str, table_key: str = None):
    """Row counts for a state, across one table or all tables."""
    state = state.upper()
    if table_key:
        tbl = TABLE_MAP[table_key]
        rows = raw_query(f"SELECT COUNT(*) AS c FROM {tbl} WHERE state = %s", (state,))
        return {table_key: rows[0]["c"]}
    counts = {}
    for key, tbl in TABLE_MAP.items():
        try:
            rows = raw_query(f"SELECT COUNT(*) AS c FROM {tbl} WHERE state = %s", (state,))
            counts[key] = rows[0]["c"]
        except Exception:
            counts[key] = 0
    return counts


def fetch_date_range(state: str, table_key: str = "load"):
    """Min/max dates for a state in a given table."""
    tbl = TABLE_MAP[table_key]
    rows = raw_query(
        f"SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM {tbl} WHERE state = %s",
        (state.upper(),),
    )
    return rows[0] if rows else {"min_date": None, "max_date": None}


# ── Write helpers ─────────────────────────────────────────────────────────────

def upsert_rows(table_key: str, state: str, rows: list) -> dict:
    """
    UPSERT rows into the given table using ON DUPLICATE KEY UPDATE.
    Returns {"rows_affected": n, "table": table_key, "state": state}.
    """
    if not rows:
        return {"rows_affected": 0, "table": table_key, "state": state}

    tbl = TABLE_MAP[table_key]
    state = state.strip().upper()

    for row in rows:
        row["state"] = state

    all_cols = [c for c in rows[0].keys() if c.lower() != "id"]
    pk = set(PK_COLS.get(table_key, []))
    non_pk = [c for c in all_cols if c not in pk]

    col_sql = ", ".join(f"`{c}`" for c in all_cols)
    val_sql = ", ".join(["%s"] * len(all_cols))
    update_sql = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in (non_pk or all_cols[:1]))

    sql = (
        f"INSERT INTO {tbl} ({col_sql}) VALUES ({val_sql}) "
        f"ON DUPLICATE KEY UPDATE {update_sql}"
    )
    params_list = [
        tuple(_clean(row.get(c)) for c in all_cols)
        for row in rows
    ]

    n = raw_execute(sql, params_list)
    return {"rows_affected": n, "table": table_key, "state": state}


def delete_rows(table_key: str, state: str, date=None, from_date=None, to_date=None) -> int:
    """DELETE rows for state + optional date range. Returns rows deleted."""
    tbl = TABLE_MAP[table_key]
    w, params = _where_clause(state, date, from_date, to_date)
    n = raw_execute_one(f"DELETE FROM {tbl} WHERE {w}", tuple(params))
    return n
