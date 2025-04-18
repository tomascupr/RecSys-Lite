"""Worker for incremental model updates."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import faiss
import numpy as np
import scipy.sparse as sp


class UpdateWorker:
    """Worker for incremental model updates."""
    
    def __init__(
        self,
        db_path: Path,
        model: Any,
        faiss_index: faiss.Index,
        item_id_map: Dict[int, str],
        batch_size: int = 1000,
        interval: int = 60,
    ) -> None:
        """Initialize update worker.
        
        Args:
            db_path: Path to DuckDB database
            model: Recommendation model with partial_fit_users method
            faiss_index: Faiss index for similar item lookup
            item_id_map: Mapping from index to item ID
            batch_size: Maximum number of events to process per batch
            interval: Update interval in seconds
        """
        self.db_path = db_path
        self.model = model
        self.faiss_index = faiss_index
        self.item_id_map = item_id_map
        self.batch_size = batch_size
        self.interval = interval
        self.last_timestamp = 0
    
    def run(self) -> None:
        """Run the update worker loop."""
        while True:
            try:
                # Get new events
                user_item_matrix, user_ids, new_items = self._get_new_events()
                
                if user_ids.size > 0:
                    # Update user factors
                    self._update_user_factors(user_item_matrix, user_ids)
                
                if new_items:
                    # Update item factors and index
                    self._update_item_vectors(new_items)
                
                # Sleep until next update
                time.sleep(self.interval)
            
            except Exception as e:
                print(f"Error in update worker: {e}")
                time.sleep(self.interval)
    
    def _get_new_events(self) -> tuple:
        """Get new events from the database.
        
        Returns:
            Tuple of user-item matrix, user IDs, and new items
        """
        conn = duckdb.connect(str(self.db_path))
        
        # Get events since last timestamp
        events_df = conn.execute(
            f"""
            SELECT user_id, item_id, qty
            FROM events
            WHERE ts > {self.last_timestamp}
            ORDER BY ts
            LIMIT {self.batch_size}
            """
        ).fetchdf()
        
        # Update last timestamp
        if not events_df.empty:
            self.last_timestamp = conn.execute(
                f"""
                SELECT MAX(ts)
                FROM events
                WHERE ts > {self.last_timestamp}
                LIMIT {self.batch_size}
                """
            ).fetchone()[0]
        
        conn.close()
        
        if events_df.empty:
            return sp.csr_matrix((0, 0)), np.array([]), []
        
        # TODO: Convert events to user-item matrix
        # This will be implemented when the model interface is finalized
        
        return sp.csr_matrix((0, 0)), np.array([]), []
    
    def _update_user_factors(
        self, user_item_matrix: sp.csr_matrix, user_ids: np.ndarray
    ) -> None:
        """Update user factors for new events.
        
        Args:
            user_item_matrix: Sparse user-item interaction matrix
            user_ids: IDs of users to update
        """
        self.model.partial_fit_users(user_item_matrix, user_ids)
    
    def _update_item_vectors(self, new_items: List[str]) -> None:
        """Update item vectors and Faiss index.
        
        Args:
            new_items: List of new item IDs
        """
        # Get item vectors for new items
        item_vectors = self.model.get_item_vectors_matrix(new_items)
        
        # Add to Faiss index
        if item_vectors.shape[0] > 0:
            self.faiss_index.add(item_vectors)
            
            # Update item ID map
            start_idx = len(self.item_id_map)
            for i, item_id in enumerate(new_items):
                self.item_id_map[start_idx + i] = item_id