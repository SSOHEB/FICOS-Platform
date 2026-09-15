"""
FICOS — Application Root Entrypoint
====================================
Standard ASGI application entrypoint supporting:
  - CLI execution: `uvicorn main:app --host 0.0.0.0 --port 8000`
  - Direct runner: `python main.py`
  - Container orchestration / Cloud PaaS runners
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config.settings import get_settings
from src.application.api import app

settings = get_settings()

if __name__ == "__main__":
    import uvicorn
    print(f">> Starting FICOS FastAPI Server on {settings.host}:{settings.port} [{settings.environment}]...")
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=(settings.environment.lower() == "development")
    )
