# GridIntel: Energy Forecasting & Analytics Platform

This project consists of a **FastAPI Backend** and a **React Frontend**.

For a more complete guide (architecture, APIs, troubleshooting), see `PROJECT_OVERVIEW.md`.

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.8+**
- **Node.js 16+** & **npm**

### 2. Backend Setup
The backend handles data processing and serves API endpoints.

#### Option A: Using Pip / venv (Recommended)
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r backend/requirements.txt
```

#### ▶️ Run Backend Server
Navigate to the project root:
```bash
cd "c:\Users\RamanSharma\OneDrive - GNA-Energy\Desktop\RD"
uvicorn backend.main:app --reload
```
*Server will start at: `http://localhost:8000`*
*Swagger Docs: `http://localhost:8000/docs`*

---

### 3. Frontend Setup
The frontend is a React application located in the `frontend` folder.

#### Install Dependencies (First Time Only)
```bash
cd frontend
npm install
```

#### ▶️ Run Frontend App
```bash
cd frontend
npm run dev
```
*App will typically run at: `http://localhost:3000` (check terminal output)*
*If prompt asks to open in browser, say 'y'.*

---

## 📂 Project Structure

- **`/backend`**: FastAPI application (`main.py`), API routes.
- **`/frontend`**: React application (`App.jsx`, components).
- **`process_data.py`**: Data cleaning and feature engineering scripts.

## ✨ Key Features
- **Dashboard**: Overview of Rolling Stats, Load Duration, and Anomalies.
- **Analytics Suite**: Pattern, Anomaly, Weather Impact, and Baseline Optimizer views.
- **Reporting & Export**: Generate reports and export charts from the UI.
