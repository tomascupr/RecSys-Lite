"""Faiss index builder for fast similarity search."""

import os
import pickle
from typing import List, Optional, Tuple, Union

import faiss
import numpy as np
from numpy.typing import NDArray


class FaissIndexBuilder:
    """Builder for Faiss similarity search indexes."""

    def __init__(
        self,
        vectors: NDArray[np.float32],
        ids: Optional[List[Union[int, str]]] = None,
        index_type: str = "IVF_Flat",
        metric: str = "inner_product",
        nlist: int = 100,
        nprobe: int = 10,
    ) -> None:
        """Initialize Faiss index builder.

        Args:
            vectors: Item vectors for indexing (n_items x dim)
            ids: List of item IDs (optional)
            index_type: Type of index ('Flat', 'IVF_Flat', 'IVF_HNSW')
            metric: Distance metric ('inner_product', 'l2', 'cosine')
            nlist: Number of clusters for IVF indexes
            nprobe: Number of clusters to probe during search
        """
        self.vectors = vectors.astype(np.float32)  # Convert to float32 for Faiss
        self.dim = vectors.shape[1]
        self.ids = ids or list(range(vectors.shape[0]))
        self.index_type = index_type
        self.metric = metric
        self.nlist = min(nlist, vectors.shape[0] // 10)  # Avoid too many lists
        self.nprobe = nprobe

        # ID mapping
        self.id_to_index = {id_: idx for idx, id_ in enumerate(self.ids)}
        self.index_to_id = {idx: id_ for idx, id_ in enumerate(self.ids)}

        # Create index
        self.index = self._create_index()

    def _create_index(self) -> faiss.Index:
        """Create Faiss index based on configuration.

        Returns:
            Faiss index
        """
        # Set distance metric
        if self.metric == "inner_product":
            metric_param = faiss.METRIC_INNER_PRODUCT
        elif self.metric == "cosine":
            metric_param = faiss.METRIC_INNER_PRODUCT
            # Normalize vectors for cosine similarity using inner product
            faiss.normalize_L2(self.vectors)
        else:  # L2 distance
            metric_param = faiss.METRIC_L2

        # Create index based on type
        if self.index_type == "Flat":
            index = (
                faiss.IndexFlatIP(self.dim)
                if self.metric in ["inner_product", "cosine"]
                else faiss.IndexFlatL2(self.dim)
            )
        elif self.index_type == "IVF_Flat":
            quantizer = faiss.IndexFlatL2(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, metric_param)
            # Train index
            index.train(self.vectors)
        elif self.index_type == "IVF_HNSW":
            quantizer = faiss.IndexHNSWFlat(self.dim, 32)  # 32 neighbors
            index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, metric_param)
            # Train index
            index.train(self.vectors)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Set number of clusters to probe (for IVF indexes)
        if self.index_type.startswith("IVF"):
            index.nprobe = self.nprobe

        # Add vectors to index
        index.add(self.vectors)

        return index

    def search(self, query: NDArray[np.float32], k: int = 10) -> Tuple[NDArray[np.float32], NDArray[np.object_]]:
        """Search for similar items.

        Args:
            query: Query vector(s)
            k: Number of results to return

        Returns:
            Tuple of (distances, item IDs)
        """
        # Make sure query is 2D
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Convert to float32
        query = query.astype(np.float32)

        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query)

        # Search
        distances, indices = self.index.search(query, k)

        # Map indices to item IDs
        item_ids = np.array([[self.index_to_id[int(idx)] for idx in row] for row in indices])

        return distances, item_ids

    def add_items(self, vectors: NDArray[np.float32], ids: Optional[List[Union[int, str]]] = None) -> None:
        """Add new items to the index.

        Args:
            vectors: Item vectors to add
            ids: Item IDs
        """
        # Convert to float32
        vectors = vectors.astype(np.float32)

        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(vectors)

        # Add to index
        self.index.add(vectors)

        # Update ID mappings
        start_idx = len(self.ids)
        new_ids = ids or list(range(start_idx, start_idx + vectors.shape[0]))

        for i, id_ in enumerate(new_ids):
            idx = start_idx + i
            self.id_to_index[id_] = idx
            self.index_to_id[idx] = id_
            self.ids.append(id_)

    def save(self, path: str) -> None:
        """Save index and metadata to disk.

        Args:
            path: Directory path to save index and metadata
        """
        os.makedirs(path, exist_ok=True)

        # Save Faiss index
        index_path = os.path.join(path, "index.faiss")
        faiss.write_index(self.index, index_path)

        # Save metadata
        metadata = {
            "dim": self.dim,
            "ids": self.ids,
            "id_to_index": self.id_to_index,
            "index_to_id": self.index_to_id,
            "index_type": self.index_type,
            "metric": self.metric,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
        }
        metadata_path = os.path.join(path, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

    @classmethod
    def load(cls, path: str) -> "FaissIndexBuilder":
        """Load index and metadata from disk.

        Args:
            path: Directory path with saved index and metadata

        Returns:
            FaissIndexBuilder instance
        """
        # Load metadata
        metadata_path = os.path.join(path, "metadata.pkl")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # Load Faiss index
        index_path = os.path.join(path, "index.faiss")
        index = faiss.read_index(index_path)

        # Create instance
        instance = cls.__new__(cls)

        # Set attributes
        instance.dim = metadata["dim"]
        instance.ids = metadata["ids"]
        instance.id_to_index = metadata["id_to_index"]
        instance.index_to_id = metadata["index_to_id"]
        instance.index_type = metadata["index_type"]
        instance.metric = metadata["metric"]
        instance.nlist = metadata["nlist"]
        instance.nprobe = metadata["nprobe"]
        instance.index = index

        # Dummy vectors since we loaded the index directly
        instance.vectors = np.zeros((len(instance.ids), instance.dim), dtype=np.float32)

        return instance
