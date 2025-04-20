# Changelog

All notable changes to the RecSys-Lite project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Pagination support for recommendation and similar-items endpoints
- Comprehensive filtering capabilities for recommendation endpoints:
  - Category-based filtering
  - Brand-based filtering
  - Price range filtering (min/max price)
  - Item exclusion and inclusion lists
- Enhanced response format with pagination and filter information
- Automatic buffer sizing when filtering to maintain result count

### Changed
- Recommendation endpoints now use a more consistent response format
- Similar-items endpoint now returns same format as recommendations endpoint
- Improved error handling for pagination and filter parameters

## [0.2.0] - 2025-04-19

### Added
- Text embedding model using all-MiniLM-L6-v2 for content-based recommendations
- Hybrid model for combining multiple recommendation approaches
- ONNX runtime acceleration for improved inference performance
- Dynamic model weighting based on user interaction patterns
- Field weighting for improved item text representation
- `train_hybrid` command for creating hybrid models
- LLM dependencies group (`recsys-lite[llm]`)
- Cold-start user handling strategies
- Weighted average of item embeddings for user profile generation

### Changed
- Updated CLI interface to support new model types
- Updated ModelType enum to include TEXT_EMBEDDING and HYBRID options
- Improved model persistence for maintaining vector representations
- Enhanced recommendation algorithm to consider item content
- Optimized hyperparameter spaces for text embedding models

### Fixed
- Properly handle empty user interaction histories 
- Type annotation improvements for static analysis

## [0.1.0] - Initial Release

### Added
- Core recommendation system functionality
- ALS, BPR, Item2Vec, LightFM, GRU4Rec, and EASE models
- FastAPI-based recommendation service
- DuckDB data storage
- CLI tools for data ingestion and model training
- React recommendation widget
- FAISS indexing for fast similarity search
- Hyperparameter optimization with Optuna
- Incremental model updating
- GDPR compliance tools