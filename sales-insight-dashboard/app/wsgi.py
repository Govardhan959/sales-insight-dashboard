"""
wsgi.py

Purpose:
    Production entry point. Runs the Flask app through Waitress
    instead of Flask's built-in development server.
"""

from waitress import serve
from app import app

if __name__ == "__main__":
    print("Starting production server on http://127.0.0.1:8000")
    serve(app, host="0.0.0.0", port=8000)