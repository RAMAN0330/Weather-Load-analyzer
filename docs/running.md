# Running Forecast Studio

Stack: **PostgreSQL** (dummy data) → **FastAPI** (all APIs, auth, forecasting) → **React** (nginx).

```bash
docker compose up --build        # http://localhost  — log in as demo / demo12345
```

On first start the `postgres` container creates the schema (`db/init/01_schema.sql`)
and the one-shot `seed` service fills it with dummy data from `mockdata/` (5 states,
21 Haryana districts, 2023-01-01 → today; weather and forecasts to today + 2), then
creates the demo login. Seeding takes a few minutes and is skipped on later starts.

| Task | Command |
|---|---|
| Re-seed from scratch | `docker compose run --rm seed python -m db.seed --force` |
| Seed fewer states / days | set `SEED_STATES=HARYANA` and/or `SEED_FROM=2024-06-01` in `.env` |
| Change demo login | `DEMO_USERNAME`, `DEMO_EMAIL`, `DEMO_PASSWORD` in `.env` |
| Database password | `POSTGRES_PASSWORD` in `.env` (default `forecast`, local only) |
| Inspect data | Postgres is published on `127.0.0.1:5432` (db `forecast_studio`) |

Local development without Docker: start Postgres, set
`DATABASE_URL=postgresql+psycopg2://forecast:forecast@localhost:5432/forecast_studio`,
run `python -m db.seed`, `uvicorn backend.main:app --port 8000`, and `npm run dev`
in `frontend/` (Vite proxies `/api`, `/ml-api` and `/auth` to port 8000).

**Daily cycle.** FastAPI runs a background job (every `DAILY_JOB_INTERVAL_S`,
default 900 s) that (1) tops up the dummy data feed — newly settled load blocks
for today, weather and external forecasts to today + 2 — so the database keeps
moving forward even if it was seeded days ago (`DUMMY_DATA_FEED=false` turns this
off), and (2) issues today's **T+1 and T+2** forecasts, cached for the day.
Forecasting has no date picker: only tomorrow and the day after can be
forecast; past dates are for backtests. The app is Haryana-only.

**Caching.** PostgreSQL reads are cached in-process for `DB_CACHE_TTL_S`
(default 300 s, cleared whenever new data arrives); daily forecasts are cached
per operating day; the v3 engine also caches trained models and input frames.
`GET /api/v3/status` shows cache hits/misses and the daily job state.

The data is synthetic: realistic in shape (seasonal peaks, daily cycle, weekends,
weather response) but not real grid measurements. The seeded snapshot ends at the
seed date; re-seed to move "today" forward.
