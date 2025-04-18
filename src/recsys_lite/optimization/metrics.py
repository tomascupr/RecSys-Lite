"""Evaluation metrics for recommendation models."""

from typing import List

import numpy as np


def hr_at_k(true_items: List[int], recommended_items: List[int], k: int = 20) -> float:
    """Calculate Hit Rate at K.

    Args:
        true_items: List of ground truth item IDs
        recommended_items: List of recommended item IDs
        k: Number of recommendations to consider

    Returns:
        Hit Rate at K
    """
    # Truncate recommended items to top k
    recommended_items = recommended_items[:k]

    # Check if any of the true items are in the recommendations
    hits = len(set(true_items) & set(recommended_items))

    # Return 1 if there's at least one hit, 0 otherwise
    return 1.0 if hits > 0 else 0.0


def ndcg_at_k(
    true_items: List[int], recommended_items: List[int], k: int = 20
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K.

    Args:
        true_items: List of ground truth item IDs
        recommended_items: List of recommended item IDs
        k: Number of recommendations to consider

    Returns:
        NDCG at K
    """
    # Truncate recommended items to top k
    recommended_items = recommended_items[:k]

    # Calculate relevance (1 if in true items, 0 otherwise)
    relevance = np.array(
        [1.0 if item in true_items else 0.0 for item in recommended_items]
    )

    # If no relevant items, return 0
    if sum(relevance) == 0:
        return 0.0

    # Calculate DCG
    discounts = np.log2(
        np.arange(2, len(relevance) + 2)
    )  # [log2(2), log2(3), ..., log2(k+1)]
    dcg = np.sum(relevance / discounts)

    # Calculate ideal DCG (all relevant items at the top)
    ideal_relevance = np.zeros_like(relevance)
    ideal_relevance[: min(len(true_items), k)] = 1.0
    idcg = np.sum(ideal_relevance / discounts)

    # Return NDCG
    return float(dcg / idcg)
