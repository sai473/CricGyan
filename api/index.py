"""
Vercel Python serverless entry — wraps the FastAPI app with Mangum.

Set in Vercel: Python 3.11, installCommand: pip install -r requirements.txt
Note: LightGBM + model pickles may exceed free-tier limits; use Pro or deploy API on Railway.
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
