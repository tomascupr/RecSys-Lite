"""Recommendation models for RecSys-Lite."""

import os

from recsys_lite.models.als import ALSModel
from recsys_lite.models.base import BaseRecommender
from recsys_lite.models.bpr import BPRModel
from recsys_lite.models.gru4rec import GRU4Rec
from recsys_lite.models.item2vec import Item2VecModel

# Use mock LightFM implementation in CI environment where building real LightFM fails
if os.environ.get("CI") == "true":
    # In CI environment, use mock implementation
    from recsys_lite.models.lightfm_mock import LightFMModel
else:
    # In normal environments, use real implementation
    try:
        from recsys_lite.models.lightfm_model import LightFMModel
    except ImportError:
        # Fallback to mock if lightfm isn't available
        from recsys_lite.models.lightfm_mock import LightFMModel

__all__ = [
    "BaseRecommender",
    "ALSModel",
    "BPRModel",
    "Item2VecModel",
    "LightFMModel",
    "GRU4Rec",
]