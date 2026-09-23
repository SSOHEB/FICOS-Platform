"""
FICOS — Canonical Configuration Interface
=========================================
Re-exports canonical single-source-of-truth configuration from configs/canonical_config.py.
"""
from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.canonical_config import *
