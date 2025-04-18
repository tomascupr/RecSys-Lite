"""Recommendation models for RecSys-Lite."""

import os

from recsys_lite.models.als import ALSModel
from recsys_lite.models.base import (
    BaseRecommender,
    FactorizationModelMixin,
    ModelPersistenceMixin,
    ModelRegistry,
)
from recsys_lite.models.bpr import BPRModel
from recsys_lite.models.ease import EASEModel
from recsys_lite.models.item2vec import Item2VecModel

# Use mock implementations in CI environment where building real dependencies fails
if os.environ.get("CI") == "true":
    # In CI environment, use mock implementations
    from recsys_lite.models.gru4rec_mock import GRU4Rec
    from recsys_lite.models.lightfm_mock import LightFMModel
    # Mock implementations for text embedding and hybrid
    from recsys_lite.models.text_embedding_mock import TextEmbeddingModel
    from recsys_lite.models.hybrid_mock import HybridModel
else:
    # In normal environments, use real implementations
    try:
        from recsys_lite.models.lightfm_model import LightFMModel
    except ImportError:
        # Fallback to mock if lightfm isn't available
        from recsys_lite.models.lightfm_mock import LightFMModel

    try:
        from recsys_lite.models.gru4rec import GRU4Rec
    except ImportError:
        # Fallback to mock if torch isn't available
        from recsys_lite.models.gru4rec_mock import GRU4Rec
        
    try:
        from recsys_lite.models.text_embedding import TextEmbeddingModel
    except ImportError:
        # Fallback if sentence-transformers isn't available
        from recsys_lite.models.text_embedding_mock import TextEmbeddingModel
    
    try:
        from recsys_lite.models.hybrid import HybridModel
    except ImportError:
        # Fallback for hybrid model
        from recsys_lite.models.hybrid_mock import HybridModel

# Register models with the registry
ModelRegistry.register("als", ALSModel)
ModelRegistry.register("bpr", BPRModel)
ModelRegistry.register("item2vec", Item2VecModel)
ModelRegistry.register("ease", EASEModel)
ModelRegistry.register("lightfm", LightFMModel)
ModelRegistry.register("gru4rec", GRU4Rec)
ModelRegistry.register("text_embedding", TextEmbeddingModel)
ModelRegistry.register("hybrid", HybridModel)

__all__ = [
    "BaseRecommender",
    "FactorizationModelMixin",
    "ModelPersistenceMixin",
    "ModelRegistry",
    "ALSModel",
    "BPRModel",
    "Item2VecModel",
    "EASEModel",
    "LightFMModel",
    "GRU4Rec",
    "TextEmbeddingModel",
    "HybridModel",
]
