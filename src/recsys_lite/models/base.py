"""Base model class for RecSys-Lite."""

import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp


class ModelPersistenceMixin:
    """Mixin for model persistence operations."""
    
    def save_model(self, path: str) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_state = self._get_model_state()
        model_filename = f"{self._get_model_type()}_model.pkl"
        with open(os.path.join(path, model_filename), "wb") as f:
            pickle.dump(model_state, f)
    
    def load_model(self, path: str) -> None:
        """Load model from disk.
        
        Args:
            path: Path to load model from
        """
        model_filename = f"{self._get_model_type()}_model.pkl"
        with open(os.path.join(path, model_filename), "rb") as f:
            model_state = pickle.load(f)
        self._set_model_state(model_state)
    
    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for serialization.
        
        Returns:
            Dictionary with model state
        """
        pass
        
    @abstractmethod
    def _set_model_state(self, model_state: Dict[str, Any]) -> None:
        """Set model state from deserialized data.
        
        Args:
            model_state: Dictionary with model state
        """
        pass
        
    @abstractmethod
    def _get_model_type(self) -> str:
        """Get model type identifier.
        
        Returns:
            Model type string
        """
        pass


class FactorizationModelMixin:
    """Mixin for models with user and item factors."""
    
    user_factors: Optional[np.ndarray] = None
    item_factors: Optional[np.ndarray] = None
    
    def get_item_factors(self) -> np.ndarray:
        """Get item factors matrix.
        
        Returns:
            Item factors matrix
        """
        if self.item_factors is None:
            return np.array([])
        return self.item_factors
        
    def get_user_factors(self) -> np.ndarray:
        """Get user factors matrix.
        
        Returns:
            User factors matrix
        """
        if self.user_factors is None:
            return np.array([])
        return self.user_factors


class BaseRecommender(ABC, ModelPersistenceMixin):
    """Abstract base class for recommendation models."""
    
    model_type: str = ""  # Override in subclasses
    
    def _get_model_type(self) -> str:
        """Get model type.
        
        Returns:
            Model type string
        """
        return self.model_type

    @abstractmethod
    def fit(self, user_item_matrix: sp.csr_matrix, **kwargs: Any) -> None:
        """Fit the model on user-item interaction data.

        Args:
            user_item_matrix: Sparse user-item interaction matrix
            **kwargs: Additional model-specific parameters
        """
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: Union[int, str],
        user_items: sp.csr_matrix,
        n_items: int = 10,
        **kwargs: Any,
    ) -> Tuple[NDArray[np.int_], NDArray[np.float32]]:
        """Generate recommendations for a user.

        Args:
            user_id: User ID
            user_items: Sparse user-item interaction matrix
            n_items: Number of recommendations to return
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (item_ids, scores)
        """
        pass


class ModelRegistry:
    """Registry for recommendation models."""
    
    _registry: Dict[str, Type[BaseRecommender]] = {}
    
    @classmethod
    def register(cls, model_type: str, model_class: Type[BaseRecommender]) -> None:
        """Register a model class.
        
        Args:
            model_type: Model type string
            model_class: Model class
        """
        cls._registry[model_type.lower()] = model_class
    
    @classmethod
    def get_model_class(cls, model_type: str) -> Type[BaseRecommender]:
        """Get model class by type.
        
        Args:
            model_type: Model type string
            
        Returns:
            Model class
            
        Raises:
            ValueError: If model type is not registered
        """
        model_class = cls._registry.get(model_type.lower())
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")
        return model_class
        
    @classmethod
    def create_model(cls, model_type: str, **kwargs: Any) -> BaseRecommender:
        """Create model instance by type.
        
        Args:
            model_type: Model type string
            **kwargs: Model parameters
            
        Returns:
            Model instance
        """
        model_class = cls.get_model_class(model_type)
        return model_class(**kwargs)
        
    @classmethod
    def load_model(cls, model_type: str, path: str) -> BaseRecommender:
        """Load model from disk.
        
        Args:
            model_type: Model type string
            path: Path to load model from
            
        Returns:
            Loaded model instance
        """
        model = cls.create_model(model_type)
        model.load_model(path)
        return model
