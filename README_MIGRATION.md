# Grid Intel: React + FastAPI Migration Guide

You have successfully migrated the codebase to a Client-Server architecture.

## 1. Architecture Overview
-   **Backend (`/backend`)**: Python FastAPI service handling data processing and analytics.
-   **Frontend (`/frontend`)**: React + Vite application with the new "Cyber-Glass" UI.

## 2. Setup Instructions

### Step A: Start the Backend
Open a terminal and run:
```bash
# Install dependencies (already installed by agent, but just in case)
pip install -r backend/requirements.txt

# Start the API Server
python -m uvicorn backend.main:app --reload
```
You should see: `Uvicorn running on http://127.0.0.1:8000`

### Step B: Start the Frontend
You will need **Node.js** installed for this step.
Open a **new terminal**:
```bash
cd frontend

# Install JavaScript dependencies
npm install

# Start the Dev Server
npm run dev
```
You should see: `Local: http://localhost:5173/`

## 3. Usage
Open `http://localhost:5173/` in your browser.
The frontend will automatically connect to the backend running on port 8000.
