"""Tests for the VectorService class."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from recsys_lite.api.errors import VectorRetrievalError
from recsys_lite.api.services import VectorService, EntityType
from recsys_lite.models.base import BaseRecommender


class TestVectorService:
    """Test suite for VectorService."""

    def test_get_vector_with_vector_provider(self):
        """Test get_vector with a model that implements the VectorProvider protocol."""
        # Create a mock model
        mock_model = MagicMock(spec=BaseRecommender)
        mock_model.get_user_vectors.return_value = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        mock_model.get_item_vectors.return_value = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
        
        # Create vector service
        vector_service = VectorService()
        
        # Test user vector retrieval
        user_vector = vector_service.get_vector(
            model=mock_model,
            entity_type=EntityType.USER,
            entity_idx=1
        )
        
        # Verify the result
        assert isinstance(user_vector, np.ndarray)
        assert user_vector.shape == (1, 3)
        assert np.array_equal(user_vector, np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        mock_model.get_user_vectors.assert_called_once_with([1])
        
        # Test item vector retrieval
        item_vector = vector_service.get_vector(
            model=mock_model,
            entity_type=EntityType.ITEM,
            entity_idx=2,
            entity_id="item_2"
        )
        
        # Verify the result
        assert isinstance(item_vector, np.ndarray)
        assert item_vector.shape == (1, 3)
        assert np.array_equal(item_vector, np.array([[4.0, 5.0, 6.0]], dtype=np.float32))
        mock_model.get_item_vectors.assert_called_once_with([2])
    
    def test_get_vector_with_factors(self):
        """Test get_vector with a model that implements the get_*_factors methods."""
        # Create a mock model
        mock_model = MagicMock(spec=BaseRecommender)
        # No get_*_vectors methods
        del mock_model.get_user_vectors
        del mock_model.get_item_vectors
        
        # Set up factors
        user_factors = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        item_factors = np.array([[0.0, 0.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        mock_model.get_user_factors.return_value = user_factors
        mock_model.get_item_factors.return_value = item_factors
        
        # Create vector service
        vector_service = VectorService()
        
        # Test user vector retrieval
        user_vector = vector_service.get_vector(
            model=mock_model,
            entity_type=EntityType.USER,
            entity_idx=1
        )
        
        # Verify the result
        assert isinstance(user_vector, np.ndarray)
        assert user_vector.shape == (1, 2)
        assert np.array_equal(user_vector, np.array([[1.0, 2.0]], dtype=np.float32))
        mock_model.get_user_factors.assert_called_once()
        
        # Test item vector retrieval
        item_vector = vector_service.get_vector(
            model=mock_model,
            entity_type=EntityType.ITEM,
            entity_idx=1
        )
        
        # Verify the result
        assert isinstance(item_vector, np.ndarray)
        assert item_vector.shape == (1, 2)
        assert np.array_equal(item_vector, np.array([[5.0, 6.0]], dtype=np.float32))
        mock_model.get_item_factors.assert_called_once()
    
    def test_get_vector_fallback_to_random(self):
        """Test get_vector fallback to random vectors."""
        # Create a mock model with no vector methods
        mock_model = MagicMock(spec=BaseRecommender)
        del mock_model.get_user_vectors
        del mock_model.get_item_vectors
        mock_model.get_user_factors.return_value = None
        mock_model.get_item_factors.return_value = None
        mock_model.factors = 5  # Set vector size
        
        # Create vector service
        vector_service = VectorService()
        
        # Test with explicit vector size
        with patch('numpy.random.random') as mock_random:
            mock_random.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            
            vector = vector_service.get_vector(
                model=mock_model,
                entity_type=EntityType.USER,
                entity_idx=1,
                vector_size=3
            )
            
            assert vector.shape == (1, 3)
            mock_random.assert_called_once_with(3)
        
        # Test with model.factors
        with patch('numpy.random.random') as mock_random:
            mock_random.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
            
            vector = vector_service.get_vector(
                model=mock_model,
                entity_type=EntityType.ITEM,
                entity_idx=1
            )
            
            assert vector.shape == (1, 5)
            mock_random.assert_called_once_with(5)
    
    def test_get_vector_error_handling(self):
        """Test error handling in get_vector."""
        # Create a mock model that raises an exception
        mock_model = MagicMock(spec=BaseRecommender)
        mock_model.get_user_vectors.side_effect = ValueError("Test error")
        mock_model.get_item_vectors.side_effect = ValueError("Test error")
        mock_model.get_user_factors.side_effect = ValueError("Test error")
        mock_model.get_item_factors.side_effect = ValueError("Test error")
        
        # Also make the random vector generation fail
        with patch('numpy.random.random', side_effect=Exception("Random error")):
            # Create vector service
            vector_service = VectorService()
            
            # Test that VectorRetrievalError is raised
            with pytest.raises(VectorRetrievalError) as excinfo:
                vector_service.get_vector(
                    model=mock_model,
                    entity_type=EntityType.USER,
                    entity_idx=1,
                    entity_id="user_1"
                )
            
            # Verify the error details
            assert "user" in str(excinfo.value)
            assert "user_1" in str(excinfo.value)
            
            # Test with item
            with pytest.raises(VectorRetrievalError) as excinfo:
                vector_service.get_vector(
                    model=mock_model,
                    entity_type=EntityType.ITEM,
                    entity_idx=2,
                    entity_id="item_2"
                )
            
            # Verify the error details
            assert "item" in str(excinfo.value)
            assert "item_2" in str(excinfo.value)
    
    def test_get_user_vector_and_get_item_vector(self):
        """Test the convenience methods get_user_vector and get_item_vector."""
        # Create a mock vector service with a mocked get_vector method
        vector_service = VectorService()
        vector_service.get_vector = MagicMock(return_value=np.array([[1.0, 2.0]], dtype=np.float32))
        
        # Test get_user_vector
        mock_model = MagicMock(spec=BaseRecommender)
        user_vector = vector_service.get_user_vector(mock_model, 1, 2)
        
        # Verify the call
        vector_service.get_vector.assert_called_once_with(
            model=mock_model,
            entity_type=EntityType.USER,
            entity_idx=1,
            vector_size=2
        )
        
        # Reset mock
        vector_service.get_vector.reset_mock()
        
        # Test get_item_vector
        item_vector = vector_service.get_item_vector(mock_model, 2, "item_2", 3)
        
        # Verify the call
        vector_service.get_vector.assert_called_once_with(
            model=mock_model,
            entity_type=EntityType.ITEM,
            entity_idx=2,
            entity_id="item_2",
            vector_size=3
        )