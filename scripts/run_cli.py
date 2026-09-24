"""
FICOS — Interactive Command-Line Interface Launcher
===================================================
Launches the FICOS operational CLI for freight procurement recommendations.
"""
import sys
from pathlib import Path

# Ensure repository root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.api.__main__ import main

if __name__ == "__main__":
    main()
