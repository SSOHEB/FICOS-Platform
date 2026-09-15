"""
FICOS — Model Registry
Reads, writes, and queries the registry manifest (registry/manifest.json).

Key distinction:
  - 'promoted'  → can be used for inference
  - 'secondary' → available but not primary
  - 'fallback'  → regime-dependent; use FLEXIBLE_INDEX instead
  - 'excluded'  → failed validation; must never be used for inference
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from src.domain.schemas import ModelRegistryEntry, ModelStatus


def _registry_root() -> Path:
    """Return the registry/ directory relative to the project root."""
    return Path(__file__).resolve().parent.parent.parent / "registry"


def _manifest_path() -> Path:
    return _registry_root() / "manifest.json"


class ModelRegistry:
    """
    Lightweight filesystem + JSON registry.
    NOT MLflow. Intentionally simple.
    """

    def __init__(self, manifest_path: Optional[str] = None):
        self._path = Path(manifest_path) if manifest_path else _manifest_path()
        self._manifest: Dict[str, Any] = self._load()

    # ──────────────────────────────────────────────────────────
    # LOAD / SAVE
    # ──────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        if not self._path.exists():
            raise FileNotFoundError(
                f"Registry manifest not found at {self._path}. "
                f"Create registry/manifest.json before running inference."
            )
        with open(self._path, "r", encoding="utf-8") as f:
            return json.load(f)

    def reload(self) -> None:
        """Re-read the manifest from disk (e.g., after promotion)."""
        self._manifest = self._load()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2)

    # ──────────────────────────────────────────────────────────
    # QUERY
    # ──────────────────────────────────────────────────────────

    def all_entries(self) -> List[Dict[str, Any]]:
        return self._manifest.get("models", [])

    def get(
        self,
        asset: str,
        horizon_days: int,
        status_filter: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Return the registry entry for a given asset/horizon.
        Optionally filter by status.
        Returns None if no matching entry exists.
        """
        for entry in self.all_entries():
            if (
                entry.get("asset", "").lower() == asset.lower()
                and entry.get("horizon_days") == horizon_days
            ):
                if status_filter is None or entry.get("status") == status_filter:
                    return entry
        return None

    def get_promoted(self, asset: str, horizon_days: int) -> Optional[Dict[str, Any]]:
        """Return the promoted entry only. Returns None if asset/horizon is not promoted."""
        return self.get(asset, horizon_days, status_filter="promoted")

    def is_promoted(self, asset: str, horizon_days: int) -> bool:
        return self.get_promoted(asset, horizon_days) is not None

    def is_excluded(self, asset: str, horizon_days: int) -> bool:
        entry = self.get(asset, horizon_days)
        return entry is not None and entry.get("status") == "excluded"

    def list_promoted_pairs(self) -> List[Dict[str, Any]]:
        return [e for e in self.all_entries() if e.get("status") == "promoted"]

    def list_promoted_models(self) -> List[Dict[str, Any]]:
        """Alias for list_promoted_pairs."""
        return self.list_promoted_pairs()


    def get_p10_p90(self, asset: str, horizon_days: int) -> Dict[str, float]:
        """Return P10/P90 uncertainty bounds from the registry."""
        entry = self.get_promoted(asset, horizon_days)
        if entry:
            return {
                "p10": entry.get("p10_bound", -1500.0),
                "p90": entry.get("p90_bound", 1500.0),
                "tau": entry.get("optimal_tau", 0.01),
                "historical_precision": entry.get("historical_precision", 50.0),
            }
        # Fallback wide bounds for unregistered/excluded pairs
        return {"p10": -1500.0, "p90": 1500.0, "tau": 0.01, "historical_precision": 50.0}

    # ──────────────────────────────────────────────────────────
    # WRITE
    # ──────────────────────────────────────────────────────────

    def add_or_update(self, entry: Dict[str, Any]) -> None:
        """
        Add or update a registry entry.
        Matching is by (asset, horizon_days).
        """
        models = self._manifest.setdefault("models", [])
        for i, e in enumerate(models):
            if (
                e.get("asset") == entry["asset"]
                and e.get("horizon_days") == entry["horizon_days"]
            ):
                models[i] = entry
                self._save()
                return
        models.append(entry)
        self._save()

    def promote(self, asset: str, horizon_days: int) -> None:
        """
        Promote a candidate model to 'promoted' status.
        Must only be called after successful validation.
        """
        import datetime
        entry = self.get(asset, horizon_days)
        if entry is None:
            raise ValueError(f"No registry entry for {asset} {horizon_days}d.")
        if entry.get("status") == "excluded":
            raise ValueError(
                f"Cannot promote {asset} {horizon_days}d — it is EXCLUDED by validation."
            )
        entry["status"] = "promoted"
        entry["promoted_at"] = datetime.datetime.utcnow().isoformat()
        self.add_or_update(entry)
        print(f"[Registry] Promoted: {asset} {horizon_days}d → 'promoted'")

    def describe(self) -> None:
        """Print a summary of all registry entries."""
        print("\n=== FICOS Model Registry ===")
        for e in self.all_entries():
            asset = e.get("asset", "?")
            h = e.get("horizon_days", "?")
            mtype = e.get("model_type", "?")
            status = e.get("status", "?")
            print(f"  {asset:12s} {h:3}d  {mtype:30s}  [{status}]")
        print()


# ──────────────────────────────────────────────────────────────
# Convenience singleton loader
# ──────────────────────────────────────────────────────────────

_registry_instance: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Return a cached registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance


if __name__ == "__main__":
    reg = ModelRegistry()
    reg.describe()
