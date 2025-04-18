"""Base model class for RecSys-Lite."""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp


class BaseRecommender(ABC):
    """Abstract base class for recommendation models."""

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

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from
        """
        pass
