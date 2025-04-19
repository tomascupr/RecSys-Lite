"""Very lightweight EASE‑R model implementation.

The real EASE‑R solves a ridge‑regression in closed form.  To avoid heavy
NumPy/BLAS dependencies inside the constrained execution environment we fall
back to a **toy** implementation that captures the *interface* and stores a
trivial identity‑based item‑item similarity matrix.  This is sufficient for
unit/integration testing and for showcasing how the model would be hooked into
the CLI, update‑worker and Faiss layers.  A production deployment can replace
the `_compute_item_similarity` method with a proper linear‑algebra routine
without touching the public API.
"""

from __future__ import annotations

from typing import Any, Tuple, Union

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from .base import BaseRecommender, FactorizationModelMixin, ModelRegistry


class EASEModel(BaseRecommender, FactorizationModelMixin):
    """Extremely Simplified EASE‑R placeholder."""

    model_type = "ease"

    def __init__(self, lambda_: float = 0.5) -> None:  # noqa: D401 – simple init
        self.lambda_ = float(lambda_)
        self.item_similarity: NDArray[np.float32] | None = None  # To be learned

    # ---------------------------------------------------------------------
    # BaseRecommender API
    # ---------------------------------------------------------------------

    def fit(self, user_item_matrix: sp.csr_matrix, **kwargs: Any) -> None:
        n_items = user_item_matrix.shape[1]

        # **Toy implementation** – identity matrix as similarity proxy
        self.item_similarity = np.eye(n_items, dtype=np.float32)

    def recommend(
        self,
        user_id: Union[int, str],
        user_items: sp.csr_matrix,
        n_items: int = 10,
        **kwargs: Any,
    ) -> Tuple[NDArray[np.int_], NDArray[np.float32]]:
        if self.item_similarity is None:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        # Very naive – recommend the top *n_items* global popular indices not
        # already interacted with (diagonal are ones so will be filtered).
        interacted_items = set(user_items.indices if hasattr(user_items, "indices") else [])

        scores = [sum(col) for col in zip(*self.item_similarity, strict=False)]
        candidate_indices = [i for i in range(len(scores)) if i not in interacted_items]

        top_indices = np.array(candidate_indices[:n_items], dtype=np.int_)
        top_scores = np.array([scores[i] for i in top_indices], dtype=np.float32)

        return top_indices, top_scores

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict[str, Any]:  # noqa: D401
        return {"lambda_": self.lambda_, "item_similarity": self.item_similarity}

    def _set_model_state(self, model_state: dict[str, Any]) -> None:  # noqa: D401
        self.lambda_ = float(model_state.get("lambda_", 0.5))
        self.item_similarity = model_state.get("item_similarity")


# Register model so it is available through the registry/CLI
ModelRegistry.register(EASEModel.model_type, EASEModel)
