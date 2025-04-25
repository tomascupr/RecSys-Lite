"""Router for recommendation endpoints."""

from typing import List

from fastapi import APIRouter, Depends, Query

from recsys_lite.api.dependencies import get_api_state, get_recommendation_service
from recsys_lite.api.errors import ItemNotFoundError, ModelNotInitializedError, UserNotFoundError, VectorRetrievalError
from recsys_lite.api.models import Recommendation, RecommendationResponse
from recsys_lite.api.services import RecommendationService
from recsys_lite.api.state import APIState
from recsys_lite.utils.logging import get_logger, log_exception, LogLevel

logger = get_logger("api.routers.recommendations")

router = APIRouter()


@router.get("/recommend", response_model=RecommendationResponse)
async def recommend(
    user_id: str = Query(..., description="User ID to get recommendations for"),
    k: int = Query(10, description="Number of recommendations to return"),
    use_faiss: bool = Query(True, description="Whether to use Faiss index or direct model"),
    
    # Pagination parameters
    page: int = Query(1, description="Page number (1-based)", ge=1),
    page_size: int = Query(10, description="Number of items per page", ge=1, le=100),
    
    # Filtering parameters
    categories: List[str] = Query(None, description="Filter by categories"),
    brands: List[str] = Query(None, description="Filter by brands"),
    min_price: float = Query(None, description="Minimum price"),
    max_price: float = Query(None, description="Maximum price"),
    exclude_items: List[str] = Query(None, description="Item IDs to exclude"),
    include_items: List[str] = Query(None, description="Limit to these item IDs"),
    
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    state: APIState = Depends(get_api_state),
) -> RecommendationResponse:
    """Get recommendations for a user with pagination and filtering.

    Args:
        user_id: User ID to get recommendations for
        k: Number of recommendations to return
        use_faiss: Whether to use Faiss index (faster) or direct model predictions
        page: Page number (1-based)
        page_size: Number of items per page (max 100)
        categories: Filter by categories
        brands: Filter by brands
        min_price: Minimum price filter
        max_price: Maximum price filter
        exclude_items: Item IDs to exclude from recommendations
        include_items: Limit recommendations to these item IDs
        recommendation_service: Recommendation service from dependency injection
        state: API state for metrics

    Returns:
        Recommendation response with pagination and filter information

    Raises:
        UserNotFoundError: If user ID is not found
        ModelNotInitializedError: If recommendation system is not initialized
        VectorRetrievalError: If vector retrieval fails
    """
    try:
        # Get more recommendations than requested to allow for filtering
        buffer_factor = 3  # Request 3x more to account for filtering
        use_buffer = any([categories, brands, min_price, max_price, exclude_items, include_items])
        buffer_k = k * buffer_factor if use_buffer else k
        
        # Cap at a reasonable maximum
        buffer_k = min(buffer_k, 1000)
        
        logger.debug(
            f"Generating recommendations for user {user_id}",
            extra={
                "user_id": user_id,
                "k": k,
                "buffer_k": buffer_k,
                "use_faiss": use_faiss,
                "page": page,
                "page_size": page_size
            }
        )
        
        # Get recommendations
        item_ids, scores, item_metadata = recommendation_service.recommend_for_user(
            user_id=user_id, k=buffer_k, use_faiss=use_faiss
        )

        # Apply filtering if any filter parameters were provided
        filter_info = None
        if any([categories, brands, min_price is not None, max_price is not None, exclude_items, include_items]):
            logger.debug(
                f"Applying filters to recommendations for user {user_id}",
                extra={
                    "user_id": user_id,
                    "categories": categories,
                    "brands": brands,
                    "min_price": min_price,
                    "max_price": max_price,
                    "exclude_items_count": len(exclude_items) if exclude_items else 0,
                    "include_items_count": len(include_items) if include_items else 0
                }
            )
            
            item_ids, scores, item_metadata, filter_info = recommendation_service.filter_recommendations(
                item_ids=item_ids,
                scores=scores,
                item_metadata=item_metadata,
                categories=categories,
                brands=brands,
                min_price=min_price,
                max_price=max_price,
                exclude_items=exclude_items,
                include_items=include_items
            )

        # Apply pagination
        pagination_info = None
        if page > 1 or len(item_ids) > page_size:
            logger.debug(
                f"Applying pagination to recommendations for user {user_id}",
                extra={
                    "user_id": user_id,
                    "page": page,
                    "page_size": page_size,
                    "total_items": len(item_ids)
                }
            )
            
            item_ids, scores, item_metadata, pagination_info = recommendation_service.paginate_results(
                item_ids=item_ids,
                scores=scores,
                item_metadata=item_metadata,
                page=page,
                page_size=page_size
            )

        # Create recommendation objects
        recommendations = [
            Recommendation(
                item_id=item_id,
                score=float(score),
                title=metadata.get("title"),
                category=metadata.get("category"),
                brand=metadata.get("brand"),
                price=metadata.get("price"),
                img_url=metadata.get("img_url"),
            )
            for item_id, score, metadata in zip(item_ids, scores, item_metadata, strict=False)
        ]

        # Update metrics
        state.increase_recommendation_count(len(recommendations))

        logger.info(
            f"Generated {len(recommendations)} recommendations for user {user_id}",
            extra={
                "user_id": user_id,
                "recommendation_count": len(recommendations),
                "use_faiss": use_faiss,
                "filtered": filter_info is not None,
                "paginated": pagination_info is not None
            }
        )

        # Return response
        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            pagination=pagination_info,
            filter_info=filter_info,
        )
    except (UserNotFoundError, ModelNotInitializedError, VectorRetrievalError):
        # Known errors - re-raise to be handled by exception handlers
        raise
    except Exception as e:
        # Unexpected errors
        log_exception(
            logger,
            "Error generating recommendations",
            e,
            level=LogLevel.ERROR,
            extra={
                "user_id": user_id,
                "k": k,
                "use_faiss": use_faiss,
                "page": page,
                "page_size": page_size
            }
        )
        state.increase_error_count()
        raise


@router.get("/similar-items", response_model=RecommendationResponse)
async def similar_items(
    item_id: str = Query(..., description="Item ID to find similar items for"),
    k: int = Query(10, description="Number of similar items to return"),
    
    # Pagination parameters
    page: int = Query(1, description="Page number (1-based)", ge=1),
    page_size: int = Query(10, description="Number of items per page", ge=1, le=100),
    
    # Filtering parameters
    categories: List[str] = Query(None, description="Filter by categories"),
    brands: List[str] = Query(None, description="Filter by brands"),
    min_price: float = Query(None, description="Minimum price"),
    max_price: float = Query(None, description="Maximum price"),
    exclude_items: List[str] = Query(None, description="Item IDs to exclude"),
    
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
    state: APIState = Depends(get_api_state),
) -> RecommendationResponse:
    """Get similar items with pagination and filtering.

    Args:
        item_id: Item ID to find similar items for
        k: Number of similar items to return
        page: Page number (1-based)
        page_size: Number of items per page (max 100)
        categories: Filter by categories
        brands: Filter by brands
        min_price: Minimum price filter
        max_price: Maximum price filter
        exclude_items: Item IDs to exclude from similar items
        recommendation_service: Recommendation service from dependency injection
        state: API state for metrics

    Returns:
        Recommendation response with pagination and filter information

    Raises:
        ItemNotFoundError: If item ID is not found
        ModelNotInitializedError: If recommendation system is not initialized
        VectorRetrievalError: If vector retrieval fails
    """
    try:
        # Get more recommendations than requested to allow for filtering
        buffer_factor = 3  # Request 3x more to account for filtering
        use_buffer = any([categories, brands, min_price, max_price, exclude_items])
        buffer_k = k * buffer_factor if use_buffer else k
        
        # Cap at a reasonable maximum
        buffer_k = min(buffer_k, 1000)
        
        logger.debug(
            f"Finding similar items for item {item_id}",
            extra={
                "item_id": item_id,
                "k": k,
                "buffer_k": buffer_k,
                "page": page,
                "page_size": page_size
            }
        )
        
        # Get similar items
        item_ids, scores, item_metadata = recommendation_service.find_similar_items(item_id=item_id, k=buffer_k)

        # Always exclude the original item from similar items
        if exclude_items is None:
            exclude_items = [item_id]
        elif item_id not in exclude_items:
            exclude_items.append(item_id)

        # Apply filtering if any filter parameters were provided
        filter_info = None
        if any([categories, brands, min_price is not None, max_price is not None, exclude_items]):
            logger.debug(
                f"Applying filters to similar items for item {item_id}",
                extra={
                    "item_id": item_id,
                    "categories": categories,
                    "brands": brands,
                    "min_price": min_price,
                    "max_price": max_price,
                    "exclude_items_count": len(exclude_items) if exclude_items else 0
                }
            )
            
            item_ids, scores, item_metadata, filter_info = recommendation_service.filter_recommendations(
                item_ids=item_ids,
                scores=scores,
                item_metadata=item_metadata,
                categories=categories,
                brands=brands,
                min_price=min_price,
                max_price=max_price,
                exclude_items=exclude_items,
                include_items=None
            )

        # Apply pagination
        pagination_info = None
        if page > 1 or len(item_ids) > page_size:
            logger.debug(
                f"Applying pagination to similar items for item {item_id}",
                extra={
                    "item_id": item_id,
                    "page": page,
                    "page_size": page_size,
                    "total_items": len(item_ids)
                }
            )
            
            item_ids, scores, item_metadata, pagination_info = recommendation_service.paginate_results(
                item_ids=item_ids,
                scores=scores,
                item_metadata=item_metadata,
                page=page,
                page_size=page_size
            )

        # Create recommendation objects
        recommendations = [
            Recommendation(
                item_id=similar_item_id,
                score=float(score),
                title=metadata.get("title"),
                category=metadata.get("category"),
                brand=metadata.get("brand"),
                price=metadata.get("price"),
                img_url=metadata.get("img_url"),
            )
            for similar_item_id, score, metadata in zip(item_ids, scores, item_metadata, strict=False)
        ]

        # Update metrics
        state.increase_recommendation_count(len(recommendations))

        logger.info(
            f"Found {len(recommendations)} similar items for item {item_id}",
            extra={
                "item_id": item_id,
                "similar_items_count": len(recommendations),
                "filtered": filter_info is not None,
                "paginated": pagination_info is not None
            }
        )

        # Return response
        return RecommendationResponse(
            user_id=item_id,  # Use item_id as user_id for similar items
            recommendations=recommendations,
            pagination=pagination_info,
            filter_info=filter_info,
        )
    except (ItemNotFoundError, ModelNotInitializedError, VectorRetrievalError):
        # Known errors - re-raise to be handled by exception handlers
        raise
    except Exception as e:
        # Unexpected errors
        log_exception(
            logger,
            "Error finding similar items",
            e,
            level=LogLevel.ERROR,
            extra={
                "item_id": item_id,
                "k": k,
                "page": page,
                "page_size": page_size
            }
        )
        state.increase_error_count()
        raise
