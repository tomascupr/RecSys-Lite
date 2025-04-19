"""Recommendation models for RecSys-Lite."""

import os

# We'll use this approach to simplify imports and avoid mypy warnings
from typing import Any, Optional, Type

from recsys_lite.models.als import ALSModel
from recsys_lite.models.base import (
    BaseRecommender,
    FactorizationModelMixin,
    ModelPersistenceMixin,
    ModelRegistry,
)
from recsys_lite.models.bpr import BPRModel
from recsys_lite.models.ease import EASEModel

# Import all mock implementations first as fallbacks
from recsys_lite.models.gru4rec_mock import GRU4Rec as GRU4RecMock
from recsys_lite.models.hybrid_mock import HybridModel as HybridModelMock
from recsys_lite.models.item2vec import Item2VecModel
from recsys_lite.models.lightfm_mock import LightFMModel as LightFMModelMock
from recsys_lite.models.text_embedding_mock import TextEmbeddingModel as TextEmbeddingModelMock


# Create an import registry to manage both real and mock implementations
class _ModelImporter:
    def __init__(self):
        self.registry = {}

    def register(self, model_type: str, model_class: Type[BaseRecommender]) -> None:
        self.registry[model_type] = model_class

    def get(self, model_type: str) -> Optional[Type[BaseRecommender]]:
        return self.registry.get(model_type, None)


# Global importer
_importer = _ModelImporter()

# For mypy, we'll just use Any as the type for these variables
GRU4Rec: Any
HybridModel: Any
LightFMModel: Any
TextEmbeddingModel: Any

# Set up mock implementations as defaults
GRU4Rec = GRU4RecMock
HybridModel = HybridModelMock
LightFMModel = LightFMModelMock
TextEmbeddingModel = TextEmbeddingModelMock

# In normal environments, try real implementations
if os.environ.get("CI") != "true":
    try:
        from recsys_lite.models.lightfm_model import LightFMModel as _LightFMModel

        LightFMModel = _LightFMModel
    except ImportError:
        pass

    try:
        from recsys_lite.models.gru4rec import GRU4Rec as _GRU4Rec

        GRU4Rec = _GRU4Rec
    except ImportError:
        pass

    try:
        from recsys_lite.models.text_embedding import TextEmbeddingModel as _TextEmbeddingModel

        TextEmbeddingModel = _TextEmbeddingModel
    except ImportError:
        pass

    try:
        from recsys_lite.models.hybrid import HybridModel as _HybridModel

        HybridModel = _HybridModel
    except ImportError:
        pass

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
