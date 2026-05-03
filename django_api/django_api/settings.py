"""
django_api/settings.py

Django settings for the pipeline API project.
Reads MySQL credentials from the root .env (same as config.py).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level above django_api/)
BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent  # RD/
load_dotenv(ROOT_DIR / ".env")

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-secret-key-change-in-production")

DEBUG = os.environ.get("DJANGO_DEBUG", "False").lower() == "true"

ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")


INSTALLED_APPS = [
    "rest_framework",
    "corsheaders",
    "pipeline",
    "accounts",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "django_api.urls"

WSGI_APPLICATION = "django_api.wsgi.application"

# ── MySQL database ────────────────────────────────────────────────────────────
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": os.environ.get("MYSQL_DB", "rishu_db"),
        "USER": os.environ.get("MYSQL_USER", "admin"),
        "PASSWORD": os.environ.get("MYSQL_PASSWORD", '123gnaenergy'),
        "HOST": os.environ.get("MYSQL_HOST", "insights-db.cwrczdscnsl8.ap-south-1.rds.amazonaws.com"),
        "PORT": os.environ.get("MYSQL_PORT", "3306"),
        "OPTIONS": {
            "charset": "utf8mb4",
            "init_command": "SET sql_mode='STRICT_TRANS_TABLES'",
        },
    }
}

# ── REST Framework ────────────────────────────────────────────────────────────
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": [],
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "UNAUTHENTICATED_USER": None,
}

# ── CORS ─────────────────────────────────────────────────────────────────────
_CORS_ORIGINS = os.environ.get("CORS_ALLOWED_ORIGINS", "")
if _CORS_ORIGINS:
    CORS_ALLOWED_ORIGINS = [o.strip() for o in _CORS_ORIGINS.split(",") if o.strip()]
    CORS_ALLOW_ALL_ORIGINS = False
else:
    CORS_ALLOW_ALL_ORIGINS = True


# ── API auth token ────────────────────────────────────────────────────────────
API_SECRET_KEY = os.environ.get("API_SECRET_KEY", "")
FASTAPI_BASE_URL = os.environ.get("FASTAPI_BASE_URL", "http://localhost:8000").rstrip("/")

# ── Misc ──────────────────────────────────────────────────────────────────────
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
USE_TZ = False

# pipeline and built-in Django tables skip migrations (managed=False tables).
# accounts uses real migrations to create vp_* tables in MySQL.
MIGRATION_MODULES = {
    "pipeline": None,
    # "accounts" is intentionally NOT listed here — it uses normal migrations
}
