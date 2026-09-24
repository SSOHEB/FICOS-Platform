"""FICOS API Application package."""
from .api import app
from .recommendation_service import RecommendationService

__all__ = ["app", "RecommendationService"]
