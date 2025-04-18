"""State management for RecSys-Lite API."""

import time
from typing import Any, Dict, Optional

import scipy.sparse as sp

from recsys_lite.models.base import BaseRecommender


class APIState:
    """Container for API shared state."""
    
    def __init__(self):
        """Initialize API state."""
        # Model and mapping data
        self.user_mapping: Dict[str, int] = {}
        self.item_mapping: Dict[str, int] = {}
        self.reverse_user_mapping: Dict[int, str] = {}
        self.reverse_item_mapping: Dict[int, str] = {}
        self.item_data: Dict[str, Dict[str, Any]] = {}
        self.faiss_index = None
        self.model: Optional[BaseRecommender] = None
        self.model_type: Optional[str] = None
        self.user_item_matrix: Optional[sp.csr_matrix] = None
        
        # Performance metrics
        self.request_count: int = 0
        self.recommendation_count: int = 0
        self.error_count: int = 0
        self.start_time: float = time.time()
    
    def increase_request_count(self) -> None:
        """Increment request counter."""
        self.request_count += 1
    
    def increase_recommendation_count(self, count: int = 1) -> None:
        """Increment recommendation counter.
        
        Args:
            count: Number of recommendations to add
        """
        self.recommendation_count += count
    
    def increase_error_count(self) -> None:
        """Increment error counter."""
        self.error_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics.
        
        Returns:
            Dictionary of metrics
        """
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": round(uptime, 2),
            "request_count": self.request_count,
            "recommendation_count": self.recommendation_count,
            "error_count": self.error_count,
            "recommendations_per_second": round(self.recommendation_count / max(uptime, 1), 2),
            "model_type": self.model_type,
            "model_info": {
                "users": len(self.user_mapping) if self.user_mapping else 0,
                "items": len(self.item_mapping) if self.item_mapping else 0,
            },
        }