"""
run_django.py  –  Start the Django API server.

Usage:
    python django_api/run_django.py [--port 8001] [--host 0.0.0.0]

This runs Django's development server (good for local use).
For production use gunicorn:
    gunicorn django_api.wsgi:application --bind 0.0.0.0:8001 --workers 2
"""

import os
import sys
import argparse

# Make sure Django can find its settings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_api.settings")

parser = argparse.ArgumentParser(description="Start Django pipeline API")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", default="8001")
args = parser.parse_args()

import django
django.setup()

from django.core.management import call_command
call_command("runserver", f"{args.host}:{args.port}", "--noreload")
