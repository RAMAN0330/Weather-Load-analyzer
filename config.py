"""
config.py – centralised configuration for rishu_pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

# MySQL
MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_DB   = os.environ.get("MYSQL_DB", "rishu_db")
MYSQL_USER = os.environ.get("MYSQL_USER", "admin")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")

# API
API_SECRET_KEY = os.environ.get("API_SECRET_KEY", "")
FASTAPI_HOST   = os.environ.get("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT   = int(os.environ.get("FASTAPI_PORT", "8001"))

# ngrok
NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN", "")
