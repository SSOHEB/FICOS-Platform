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
from src.application.api import app

if __name__ == "__main__":
    print("=================================================================")
    print("  FICOS MARITIME INTELLIGENCE & CHARTERING DECISION API SERVER   ")
    print("  Listening on http://localhost:8000 (Swagger docs at /docs)     ")
    print("=================================================================")
    uvicorn.run("src.application.api:app", host="0.0.0.0", port=8000, reload=True)
