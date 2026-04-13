"""
Start the FastAPI server optimized for Windows (ProactorEventLoop + httptools).
Usage:
  python backend/run.py           # production
  uvicorn backend.main:app --reload --port 8000   # dev with auto-reload

On Linux/macOS you can also install uvloop (`pip install uvloop`) and it will
be picked up automatically by uvicorn when loop="uvloop" is set.
"""
import sys
import os
import asyncio

# On Windows, Python 3.8+ defaults to ProactorEventLoop which supports subprocesses
# and overlapped I/O well. No change needed — just ensure it's set explicitly.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,           # True only in dev
        http="httptools",       # faster HTTP parser than h11
        workers=1,              # single worker — ML state is in-process
        log_level="info",
        access_log=False,       # disable per-request logs for throughput
        timeout_keep_alive=30,
    )
