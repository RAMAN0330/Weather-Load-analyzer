import os
import subprocess
import time

import requests


def test_baseline_sync():
    base_url = "http://localhost:8000"
    
    payload = {
        "feature": "total_drawal",
        "start_date": "2025-10-01",
        "end_date": "2025-10-15"
    }
    
    print("Starting FastAPI server...")
    server_proc = subprocess.Popen(
        ["python", "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
        env={**os.environ, "PYTHONPATH": "."},
        cwd="."
    )
    
    time.sleep(5)
    
    try:
        # 1. Get Forecast
        print("Requesting Forecast...")
        resp = requests.post(f"{base_url}/analytics/predict/forecast", json=payload)
        data = resp.json()
        
        if resp.status_code == 200:
            used_window = data.get("kpis", {}).get("used_window")
            print(f"SUCCESS: Forecast returned 200 OK. Used Window: {used_window}")
            
            # 2. Get Baseline Optimizer Result
            print("Requesting Baseline Optimizer...")
            resp_opt = requests.post(f"{base_url}/analytics/baseline", json=payload)
            opt_data = resp_opt.json()
            best_window = opt_data.get("kpis", {}).get("best_window")
            print(f"Baseline Optimizer Best Window: {best_window}")
            
            if used_window == best_window:
                print("VERIFICATION SUCCESS: Forecast used the best window from baseline optimizer!")
            else:
                print(f"VERIFICATION FAILURE: used_window ({used_window}) != best_window ({best_window})")
        else:
            print(f"FAILURE: Forecast returned {resp.status_code}")
            
    finally:
        server_proc.kill()

if __name__ == "__main__":
    test_baseline_sync()
