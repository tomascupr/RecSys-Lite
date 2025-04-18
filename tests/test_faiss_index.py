"""Tests for Faiss index builder."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from recsys_lite.indexing import FaissIndexBuilder


@pytest.fixture
def sample_vectors():
    """Create sample vectors for testing."""
    # Create random vectors
    rng = np.random.RandomState(42)
    n_items = 100
    dim = 10

    vectors = rng.random((n_items, dim)).astype(np.float32)
    ids = [f"I_{i}" for i in range(n_items)]

    return vectors, ids


def test_faiss_index_creation(sample_vectors):
    """Test Faiss index creation."""
    vectors, ids = sample_vectors

    # Create index
    index_builder = FaissIndexBuilder(
        vectors=vectors,
        ids=ids,
        index_type="Flat",
        metric="inner_product",
    )

    # Check index properties
    assert index_builder.dim == vectors.shape[1]
    assert len(index_builder.ids) == len(ids)
    assert index_builder.index is not None

    # Test search
    query = np.random.random(vectors.shape[1]).astype(np.float32)
    distances, item_ids = index_builder.search(query, k=5)

    # Check search results
    assert distances.shape == (1, 5)
    assert item_ids.shape == (1, 5)

    # Check that all returned items are valid IDs
    for item_id in item_ids[0]:
        assert item_id in ids


def test_faiss_index_add_items(sample_vectors):
    """Test adding items to Faiss index."""
    vectors, ids = sample_vectors

    # Use first 80 vectors for initial index
    init_vectors = vectors[:80]
    init_ids = ids[:80]

    # Create index
    index_builder = FaissIndexBuilder(
        vectors=init_vectors,
        ids=init_ids,
        index_type="Flat",
        metric="inner_product",
    )

    # Check initial state
    assert len(index_builder.ids) == 80

    # Add remaining vectors
    new_vectors = vectors[80:]
    new_ids = ids[80:]
    index_builder.add_items(new_vectors, new_ids)

    # Check updated state
    assert len(index_builder.ids) == 100

    # Test search after adding items
    query = np.random.random(vectors.shape[1]).astype(np.float32)
    distances, item_ids = index_builder.search(query, k=10)

    # Check that some new items are in the results
    item_id_set = set(item_ids[0])
    new_id_set = set(new_ids)
    assert len(item_id_set.intersection(new_id_set)) > 0


def test_faiss_index_save_load(sample_vectors):
    """Test saving and loading Faiss index."""
    vectors, ids = sample_vectors

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create index
        index_builder = FaissIndexBuilder(
            vectors=vectors,
            ids=ids,
            index_type="Flat",
            metric="inner_product",
        )

        # Save index
        save_path = Path(temp_dir) / "index"
        index_builder.save(save_path)

        # Check that files were created
        assert (save_path / "index.faiss").exists()
        assert (save_path / "metadata.pkl").exists()

        # Load index
        loaded_index = FaissIndexBuilder.load(save_path)

        # Check loaded index properties
        assert loaded_index.dim == vectors.shape[1]
        assert len(loaded_index.ids) == len(ids)
        assert loaded_index.index is not None

        # Test search with loaded index
        query = np.random.random(vectors.shape[1]).astype(np.float32)
        original_distances, original_item_ids = index_builder.search(query, k=5)
        loaded_distances, loaded_item_ids = loaded_index.search(query, k=5)

        # Check that results are similar
        # Since we're using the same index structure and data,
        # the results should be identical
        np.testing.assert_array_equal(original_distances, loaded_distances)
        np.testing.assert_array_equal(original_item_ids, loaded_item_ids)
