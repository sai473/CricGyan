"""
Vercel Python serverless entry — wraps the FastAPI app with Mangum.

Requires lite model exports under models/saved/ (see README). Installs from requirements.txt (numpy + lightgbm only).
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from mangum import Mangum

from backend.main import app

handler = Mangum(app, lifespan="auto")
