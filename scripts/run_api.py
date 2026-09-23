"""
FICOS — Application API Server Launcher
=======================================
Starts the FastAPI backend application on port 8000.
"""
import sys
from pathlib import Path
import uvicorn

# Ensure repository root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.settings import get_settings

def main():
    settings = get_settings()
    uvicorn.run(
        "src.application.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )

if __name__ == "__main__":
    main()
