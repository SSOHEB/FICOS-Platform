"""
FICOS — Top-Level FastAPI Server Runner
Starts the production backend API server on http://localhost:8000.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import uvicorn
from src.config.settings import get_settings
from src.application.api import app

settings = get_settings()

if __name__ == "__main__":
    print("=================================================================")
    print("  FICOS MARITIME INTELLIGENCE & CHARTERING DECISION API SERVER   ")
    print(f"  Listening on http://{settings.host}:{settings.port} (Swagger: /docs)")
    print(f"  Environment: {settings.environment} | Version: {settings.model_version}")
    print("=================================================================")
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=(settings.environment.lower() == "development"))
