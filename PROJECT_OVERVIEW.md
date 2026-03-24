# GridIntel (RD) — Project Overview

GridIntel is a local-first **grid load forecasting + analytics dashboard** built as:

- **Backend:** FastAPI (Python) that reads `final_data.csv`, runs short‑term forecasting, and serves JSON APIs.
- **Frontend:** React + Vite that renders operator views (Load Analysis, Weather Analysis, Optimizer, Simulator, Analysis, Forecast).

If you’re new to the UI workflow, start with `PAGE_GUIDE.md`.

---

## Repo Layout

- `backend/`
  - `backend/main.py`: FastAPI app + API surface (`/api/*`, `/api/v2/*`)
  - `backend/short_term_pipeline.py`: core short‑term pipeline (96×15‑min blocks)
  - `backend/engine.py`, `backend/gridintel_engine.py`: helper engines / analytics
  - `backend/tests/`: backend unit tests (unittest)
- `frontend/`
  - `frontend/src/App.jsx`: main UI shell, charts, API calls
  - `frontend/src/features/forecast/ForecastPage.tsx`: Forecast tab layout
  - `frontend/src/features/simulator/*`: Simulator store + UI
- `final_data.csv`: primary dataset used by the backend
- `exports/`: generated charts/reports/scenario storage
- Docs:
  - `README.md`: quick start
  - `PAGE_GUIDE.md`: UI workflow ladder
  - `Backend.md`: KPI definitions (ops + forecasting)
  - `frontend.md`: UI/UX design notes

---

## Quick Start (Dev)

### Prerequisites

- Python 3.9+ (3.10/3.11 recommended)
- Node.js 18+ (works with 16+, but use 18+ if possible)

### 1) Backend

From repo root:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r backend/requirements.txt

# Optional (enables extra models / faster trees if installed)
pip install lightgbm xgboost

uvicorn backend.main:app --reload --port 8000
```

API docs:

- Swagger: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

### 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

By default, Vite runs on port **3000** (`frontend/vite.config.js`), so open:

- `http://localhost:3000`

---

## Configuration

### Frontend API base URL

The frontend uses `VITE_API_BASE_URL` and automatically appends `/api`.

- Default in dev: `/api` (works with Vite proxy)
- Default in prod build: `http://localhost:8000/api`

Where it’s set:

- Root `.env`: `VITE_API_BASE_URL=http://localhost:8000`
- `frontend/.env`: `VITE_API_BASE_URL=http://localhost:8000`

To use the Vite proxy instead, set:

```env
VITE_API_BASE_URL=/api
```

### Dataset path

Backend reads `final_data.csv` from repo root:

- `backend/main.py` uses `DATA_PATH = <repo>/final_data.csv`

If you replace `final_data.csv`, either restart the backend or call:

- `POST /api/v2/reload`

---

## UI Tabs (What They Do)

High-level workflow is documented in `PAGE_GUIDE.md`. In short:

- **Load Analysis:** structural load view + comparisons
- **Weather Analysis:** weather driver influence and diagnostics
- **Optimizer:** baseline window / pattern fit / error diagnostics
- **Simulator:** what‑if editing and scenario comparison
- **Analysis:** deeper KPI + driver attribution matrix
- **Forecast:** live/short‑term stream view + CSV download

---

## Core APIs (Used by the Frontend)

### v2 endpoints (primary)

- `GET /api/v2/config`: available dates/regions, defaults
- `GET /api/v2/settings`: thresholds and UI settings defaults
- `POST /api/v2/dayahead`: day‑ahead forecast payload
- `POST /api/v2/live`: short‑term “Live Ops” forecast (supports partial actual window)
- `POST /api/v2/analysis`: diagnostics / KPI analysis for a date
- `POST /api/v2/load_benchmarks`: today vs T‑1/T‑7/T‑365 series
- `POST /api/v2/load_series`: multi‑day series fetch
- `POST /api/v2/load_change`: load change series
- `POST /api/v2/reload`: reload `final_data.csv` into memory

### Simulator endpoints

- `POST /api/simulator/blocks`: block grid + impacts for simulator
- `GET /api/simulator/scenarios`: list saved scenarios
- `POST /api/simulator/scenarios`: create/update scenario
- `DELETE /api/simulator/scenarios/{scenario_id}`: delete scenario

### Legacy endpoints (older UI / utilities)

`backend/main.py` also exposes `/api/*` and `/analytics/*` endpoints (kept for backwards compatibility and internal tooling).

---

## Forecast CSV Download

The Forecast tab exposes a **Download** action that exports the backend’s `forecast_df` as a CSV (96 rows, one per 15‑minute block).

File naming pattern:

- `shortterm_<date>.csv`

---

## Data Expectations (Minimum Columns)

`final_data.csv` is expected to contain (at least):

- `date` (string or date-like)
- `time_block` (1–96)
- `total_drawal` (MW)

Weather/driver columns are used when present (examples):

- `temperature`, `humidity`, `precipitation`
- `apparent_temperature`, `cloud_cover`, `sunshine_duration`, `direct_radiation`, `wind_speed_10m`

To sanity-check the latest dates:

```bash
python backend/check_data.py
```

---

## Tests

Backend unit tests:

```bash
python -m unittest discover -s backend/tests
```

Frontend build (sanity):

```bash
cd frontend
npm run build
```

---

## Troubleshooting

### Frontend loads but API calls fail

- Ensure backend is running at `http://localhost:8000`
- If using Vite proxy, set `VITE_API_BASE_URL=/api` and restart `npm run dev`

### “No data available for requested date”

- Confirm `final_data.csv` has that `date` and complete `time_block` rows
- Call `POST /api/v2/reload` after replacing the dataset

### Optional model dependencies missing

The pipeline can run with just sklearn, but some model branches are enabled only when installed:

```bash
pip install lightgbm xgboost torch
```

---

## Where to Look Next

- Pipeline logic: `backend/short_term_pipeline.py`
- API orchestration: `backend/main.py`
- UI entry: `frontend/src/App.jsx`
- Forecast view layout: `frontend/src/features/forecast/ForecastPage.tsx`

