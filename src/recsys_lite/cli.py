"""Command-line interface for RecSys-Lite."""

import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Type, cast

import duckdb
import numpy as np
import scipy.sparse as sp
import typer
from scipy.sparse import csr_matrix

from recsys_lite.ingest import ingest_data
from recsys_lite.models import BaseRecommender, ALSModel, BPRModel, GRU4Rec, Item2VecModel, LightFMModel
from recsys_lite.optimization import OptunaOptimizer
from recsys_lite.indexing import FaissIndexBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("recsys-lite")

# Create the CLI app
app = typer.Typer(help="RecSys-Lite: Lightweight recommendation system")


class ModelType(str, Enum):
    """Available model types."""
    ALS = "als"
    BPR = "bpr"
    ITEM2VEC = "item2vec"
    LIGHTFM = "lightfm"
    GRU4REC = "gru4rec"


class MetricType(str, Enum):
    """Available evaluation metrics."""
    HR_10 = "hr@10"
    HR_20 = "hr@20"
    NDCG_10 = "ndcg@10"
    NDCG_20 = "ndcg@20"


@app.command()
def ingest(
    events: Path = typer.Argument(..., help="Path to events parquet file"),
    items: Path = typer.Argument(..., help="Path to items CSV file"),
    db: Path = typer.Option("recsys.db", help="Path to DuckDB database"),
) -> None:
    """Ingest data into DuckDB database."""
    ingest_data(events, items, db)
    typer.echo(f"Data ingested successfully into {db}")


@app.command()
def train(
    model_type: ModelType = typer.Argument(..., help="Model type to train"),
    db: Path = typer.Option("recsys.db", help="Path to DuckDB database"),
    output: Path = typer.Option("model_artifacts", help="Output directory for model artifacts"),
    test_size: float = typer.Option(0.2, help="Fraction of data to use for testing"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    params_file: Optional[Path] = typer.Option(None, help="JSON file with model parameters"),
) -> None:
    """Train a recommendation model."""
    # Load parameters from file if provided
    if params_file:
        with open(params_file, "r") as f:
            params = json.load(f)
    else:
        # Default parameters for each model type
        if model_type == ModelType.ALS:
            params = {"factors": 128, "regularization": 0.01, "alpha": 1.0, "iterations": 15}
        elif model_type == ModelType.BPR:
            params = {"factors": 100, "learning_rate": 0.01, "regularization": 0.01, "iterations": 100}
        elif model_type == ModelType.ITEM2VEC:
            params = {"vector_size": 100, "window": 5, "min_count": 1, "sg": 1, "epochs": 5}
        elif model_type == ModelType.LIGHTFM:
            params = {"no_components": 100, "learning_rate": 0.05, "loss": "warp", "epochs": 50}
        elif model_type == ModelType.GRU4REC:
            params = {"hidden_size": 100, "n_layers": 1, "dropout": 0.1, "batch_size": 32, "learning_rate": 0.001, "n_epochs": 10}
    
    typer.echo(f"Training {model_type.value} model with parameters: {params}")
    
    # Connect to database
    conn = duckdb.connect(str(db))
    
    # Load data
    events_df = conn.execute("SELECT user_id, item_id, qty FROM events").fetchdf()
    
    # Create user and item ID mappings
    unique_users = events_df["user_id"].unique()
    unique_items = events_df["item_id"].unique()
    
    user_to_idx = {user: i for i, user in enumerate(unique_users)}
    item_to_idx = {item: i for i, item in enumerate(unique_items)}
    
    # Create train/test split
    np.random.seed(seed)
    train_mask = np.random.rand(len(events_df)) >= test_size
    
    train_df = events_df[train_mask]
    test_df = events_df[~train_mask]
    
    # Create user-item matrices
    train_matrix = _create_interaction_matrix(train_df, user_to_idx, item_to_idx)
    test_matrix = _create_interaction_matrix(test_df, user_to_idx, item_to_idx)
    
    # Train model based on type
    model: BaseRecommender
    
    if model_type == ModelType.ALS:
        model = cast(BaseRecommender, ALSModel(**params))
        model.fit(train_matrix)
    elif model_type == ModelType.BPR:
        model = cast(BaseRecommender, BPRModel(**params))
        model.fit(train_matrix)
    elif model_type == ModelType.ITEM2VEC:
        # Create session data for item2vec
        sessions = []
        for user in unique_users:
            user_items = events_df[events_df["user_id"] == user]["item_id"].tolist()
            if len(user_items) > 1:
                sessions.append(user_items)
        
        model = cast(BaseRecommender, Item2VecModel(**params))
        # Item2Vec needs special handling for fit
        if isinstance(model, Item2VecModel):
            model.fit(train_matrix, user_sessions=sessions)
    elif model_type == ModelType.LIGHTFM:
        model = cast(BaseRecommender, LightFMModel(**params))
        model.fit(train_matrix)
    elif model_type == ModelType.GRU4REC:
        # Create session data for GRU4Rec
        sessions = []
        for user in unique_users:
            user_items = events_df[events_df["user_id"] == user]["item_id"].tolist()
            if len(user_items) > 1:
                # Convert item IDs to indices
                user_items_idx = [item_to_idx[item] for item in user_items]
                sessions.append(user_items_idx)
        
        model = cast(BaseRecommender, GRU4Rec(n_items=len(unique_items), **params))
        # GRU4Rec needs special handling for fit
        if isinstance(model, GRU4Rec):
            model.fit(train_matrix, sessions=sessions)
    
    # Save model and mappings
    output_dir = output / model_type.value
    os.makedirs(output_dir, exist_ok=True)
    
    # Save mappings
    with open(output_dir / "user_mapping.json", "w") as f:
        json.dump(user_to_idx, f)
    
    with open(output_dir / "item_mapping.json", "w") as f:
        json.dump(item_to_idx, f)
    
    # Create Faiss index for item vectors
    item_vectors: Optional[np.ndarray] = None
    
    if model_type in [ModelType.ALS, ModelType.BPR, ModelType.LIGHTFM]:
        # These models use get_item_factors()
        if hasattr(model, 'get_item_factors'):
            item_vectors = model.get_item_factors()
    elif model_type == ModelType.ITEM2VEC:
        # Item2Vec uses get_item_vectors_matrix()
        if isinstance(model, Item2VecModel):
            item_vector_keys = list(item_to_idx.keys())
            item_vectors = model.get_item_vectors_matrix(item_vector_keys)
            
    # If we have item vectors, create the index
    if item_vectors is not None:
        index_builder = FaissIndexBuilder(
            vectors=item_vectors,
            ids=list(item_to_idx.keys()),
            index_type="IVF_Flat",
        )
        index_builder.save(str(output_dir / "faiss_index"))
    
    typer.echo(f"Model trained and saved to {output_dir}")


@app.command()
def optimize(
    model_type: ModelType = typer.Argument(..., help="Model type to optimize"),
    db: Path = typer.Option("recsys.db", help="Path to DuckDB database"),
    output: Path = typer.Option("model_artifacts", help="Output directory for model artifacts"),
    metric: MetricType = typer.Option(MetricType.NDCG_20, help="Evaluation metric"),
    trials: int = typer.Option(20, help="Number of optimization trials"),
    test_size: float = typer.Option(0.2, help="Fraction of data to use for testing"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """Optimize hyperparameters for a recommendation model."""
    typer.echo(f"Optimizing {model_type.value} model using {metric.value} with {trials} trials")
    
    # Connect to database
    conn = duckdb.connect(str(db))
    
    # Load data
    events_df = conn.execute("SELECT user_id, item_id, qty FROM events").fetchdf()
    
    # Create user and item ID mappings
    unique_users = events_df["user_id"].unique()
    unique_items = events_df["item_id"].unique()
    
    user_to_idx = {user: i for i, user in enumerate(unique_users)}
    item_to_idx = {item: i for i, item in enumerate(unique_items)}
    
    # Create train/test split
    np.random.seed(seed)
    train_mask = np.random.rand(len(events_df)) >= test_size
    
    train_df = events_df[train_mask]
    test_df = events_df[~train_mask]
    
    # Create user-item matrices
    train_matrix = _create_interaction_matrix(train_df, user_to_idx, item_to_idx)
    test_matrix = _create_interaction_matrix(test_df, user_to_idx, item_to_idx)
    
    # Define parameter space based on model type
    # We need to use Any here to satisfy mypy
    model_class: Any
    
    if model_type == ModelType.ALS:
        model_class = ALSModel
        param_space = {
            "factors": {"type": "int", "low": 50, "high": 200, "step": 10},
            "regularization": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
            "alpha": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
            "iterations": {"type": "int", "low": 5, "high": 30},
        }
    elif model_type == ModelType.BPR:
        model_class = BPRModel
        param_space = {
            "factors": {"type": "int", "low": 50, "high": 200, "step": 10},
            "learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
            "regularization": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
            "iterations": {"type": "int", "low": 50, "high": 200, "step": 10},
        }
    elif model_type == ModelType.LIGHTFM:
        model_class = LightFMModel
        param_space = {
            "no_components": {"type": "int", "low": 50, "high": 200, "step": 10},
            "learning_rate": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
            "item_alpha": {"type": "float", "low": 0.0, "high": 0.1},
            "user_alpha": {"type": "float", "low": 0.0, "high": 0.1},
            "loss": {"type": "categorical", "choices": ["warp", "bpr", "logistic"]},
            "epochs": {"type": "int", "low": 20, "high": 100, "step": 10},
        }
    elif model_type == ModelType.ITEM2VEC:
        # Create session data for item2vec
        sessions = []
        for user in unique_users:
            user_items = events_df[events_df["user_id"] == user]["item_id"].tolist()
            if len(user_items) > 1:
                sessions.append(user_items)
        
        # Can't use the standard optimization for Item2Vec due to different input format
        typer.echo("Item2Vec optimization is not implemented. Using default parameters.")
        params = {"vector_size": 100, "window": 5, "min_count": 1, "sg": 1, "epochs": 5}
        model = cast(BaseRecommender, Item2VecModel(**params))
        # Item2Vec needs special handling for fit
        if isinstance(model, Item2VecModel):
            model.fit(train_matrix, user_sessions=sessions)
        
        # Save model
        output_dir = output / model_type.value
        os.makedirs(output_dir, exist_ok=True)
        
        # Save mappings
        with open(output_dir / "user_mapping.json", "w") as f:
            json.dump(user_to_idx, f)
        
        with open(output_dir / "item_mapping.json", "w") as f:
            json.dump(item_to_idx, f)
        
        # Create Faiss index
        item_vectors = model.get_item_vectors_matrix(list(item_to_idx.keys()))
        index_builder = FaissIndexBuilder(
            vectors=item_vectors,
            ids=list(item_to_idx.keys()),
            index_type="IVF_Flat",
        )
        index_builder.save(str(output_dir / "faiss_index"))
        
        typer.echo(f"Item2Vec model trained and saved to {output_dir}")
        return
    elif model_type == ModelType.GRU4REC:
        # Create session data for GRU4Rec
        sessions = []
        for user in unique_users:
            user_items = events_df[events_df["user_id"] == user]["item_id"].tolist()
            if len(user_items) > 1:
                # Convert item IDs to indices
                user_items_idx = [item_to_idx[item] for item in user_items]
                sessions.append(user_items_idx)
        
        # Can't use the standard optimization for GRU4Rec due to different input format
        typer.echo("GRU4Rec optimization is not implemented. Using default parameters.")
        # Make sure all numeric parameters are the right type for mypy
        gru_params: Dict[str, Any] = {
            "hidden_size": 100, 
            "n_layers": 1, 
            "dropout": 0.1, 
            "batch_size": 32, 
            "learning_rate": 0.001, 
            "n_epochs": 10,
            "n_items": len(unique_items),
            "use_cuda": False,
        }
        model = cast(BaseRecommender, GRU4Rec(**gru_params))
        # GRU4Rec needs special handling for fit
        if isinstance(model, GRU4Rec):
            model.fit(train_matrix, sessions=sessions)
        
        # Save model
        output_dir = output / model_type.value
        os.makedirs(output_dir, exist_ok=True)
        
        # Save mappings
        with open(output_dir / "user_mapping.json", "w") as f:
            json.dump(user_to_idx, f)
        
        with open(output_dir / "item_mapping.json", "w") as f:
            json.dump(item_to_idx, f)
        
        # Save model
        model.save_model(str(output_dir / "model.pt"))
        
        typer.echo(f"GRU4Rec model trained and saved to {output_dir}")
        return
    
    # Create optimizer
    optimizer = OptunaOptimizer(
        model_class=model_class,
        metric=metric.value,
        direction="maximize",
        n_trials=trials,
        study_name=f"{model_type.value}_{metric.value}",
        seed=seed,
    )
    
    # Run optimization
    best_params = optimizer.optimize(
        train_data=train_matrix,
        valid_data=test_matrix,
        param_space=param_space,
        user_mapping=user_to_idx,
        item_mapping=item_to_idx,
    )
    
    # Train best model
    output_dir = output / model_type.value
    os.makedirs(output_dir, exist_ok=True)
    
    best_model = optimizer.get_best_model(train_matrix)
    
    # Save mappings
    with open(output_dir / "user_mapping.json", "w") as f:
        json.dump(user_to_idx, f)
    
    with open(output_dir / "item_mapping.json", "w") as f:
        json.dump(item_to_idx, f)
    
    # Save best parameters
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best_params, f)
    
    # Create Faiss index
    item_vectors = best_model.get_item_factors()
    index_builder = FaissIndexBuilder(
        vectors=item_vectors,
        ids=list(item_to_idx.keys()),
        index_type="IVF_Flat",
    )
    index_builder.save(str(output_dir / "faiss_index"))
    
    typer.echo(f"Best parameters: {best_params}")
    typer.echo(f"Best score ({metric.value}): {optimizer.best_value}")
    typer.echo(f"Model trained and saved to {output_dir}")


@app.command()
def serve(
    model_dir: Path = typer.Option("model_artifacts/als", help="Model directory"),
    host: str = typer.Option("0.0.0.0", help="Host to bind server"),
    port: int = typer.Option(8000, help="Port to bind server"),
) -> None:
    """Start the FastAPI server."""
    import uvicorn
    from recsys_lite.api.main import create_app
    
    app = create_app(model_dir=model_dir)
    typer.echo(f"Starting server with model from {model_dir}")
    uvicorn.run(app, host=host, port=port)


@app.command()
def worker(
    model_dir: Path = typer.Option("model_artifacts/als", help="Model directory"),
    db: Path = typer.Option("recsys.db", help="Path to DuckDB database"),
    interval: int = typer.Option(60, help="Update interval in seconds"),
) -> None:
    """Start the update worker."""
    from recsys_lite.update.worker import UpdateWorker
    
    typer.echo(f"Starting update worker with model from {model_dir}")
    
    # Load model ID mappings
    with open(model_dir / "user_mapping.json", "r") as f:
        user_mapping = json.load(f)
    
    with open(model_dir / "item_mapping.json", "r") as f:
        item_mapping = json.load(f)
    
    # Determine model type from directory name
    model_type = os.path.basename(model_dir)
    
    # Load model
    model: BaseRecommender
    if model_type == "als":
        from recsys_lite.models import ALSModel
        model = cast(BaseRecommender, ALSModel())
        # TODO: Load model state
    elif model_type == "bpr":
        from recsys_lite.models import BPRModel
        model = cast(BaseRecommender, BPRModel())
        # TODO: Load model state
    else:
        typer.echo(f"Unsupported model type for incremental updates: {model_type}")
        return
    
    # Load Faiss index
    index_builder = FaissIndexBuilder.load(str(model_dir / "faiss_index"))
    
    # Create update worker
    # Convert index_to_id to the expected type Dict[int, str]
    item_id_map = {int(k): str(v) for k, v in index_builder.index_to_id.items()}
    
    worker = UpdateWorker(
        db_path=db,
        model=model,
        faiss_index=index_builder.index,
        item_id_map=item_id_map,
        interval=interval,
    )
    
    # Run worker
    worker.run()


@app.command()
def gdpr(
    action: str = typer.Argument(..., help="GDPR action (delete-user, export-user)"),
    user_id: str = typer.Argument(..., help="User ID"),
    db: Path = typer.Option("recsys.db", help="Path to DuckDB database"),
    output: Optional[Path] = typer.Option(None, help="Output path for export"),
) -> None:
    """GDPR compliance operations."""
    if action == "delete-user":
        typer.echo(f"Deleting data for user {user_id}")
        conn = duckdb.connect(str(db))
        conn.execute(f"DELETE FROM events WHERE user_id = '{user_id}'")
        typer.echo(f"Deleted user data. You should retrain models.")
    elif action == "export-user":
        typer.echo(f"Exporting data for user {user_id}")
        conn = duckdb.connect(str(db))
        
        # Get user events
        events_df = conn.execute(f"SELECT * FROM events WHERE user_id = '{user_id}'").fetchdf()
        
        # Get item details for items the user interacted with
        item_ids = events_df["item_id"].tolist()
        if item_ids:
            quoted_items = [f"'{item}'" for item in item_ids]
            items_list = ", ".join(quoted_items)
            items_sql = f"SELECT * FROM items WHERE item_id IN ({items_list})"
            items_df = conn.execute(items_sql).fetchdf()
        else:
            items_df = None
        
        # Create export data
        export_data = {
            "user_id": user_id,
            "events": events_df.to_dict(orient="records") if not events_df.empty else [],
            "items": items_df.to_dict(orient="records") if items_df is not None and not items_df.empty else [],
        }
        
        # Output to file or stdout
        if output:
            os.makedirs(os.path.dirname(output), exist_ok=True)
            with open(output, "w") as f:
                json.dump(export_data, f, indent=2)
            typer.echo(f"User data exported to {output}")
        else:
            typer.echo(json.dumps(export_data, indent=2))
    else:
        typer.echo(f"Unknown GDPR action: {action}")


def _create_interaction_matrix(
    df: Any,
    user_mapping: Dict[str, int],
    item_mapping: Dict[str, int],
) -> csr_matrix:
    """Create user-item interaction matrix.
    
    Args:
        df: DataFrame with user-item interactions
        user_mapping: Mapping from user IDs to indices
        item_mapping: Mapping from item IDs to indices
    
    Returns:
        Sparse user-item interaction matrix
    """
    # Get matrix dimensions
    n_users = len(user_mapping)
    n_items = len(item_mapping)
    
    # Create sparse matrix
    row_indices = [user_mapping[user] for user in df["user_id"]]
    col_indices = [item_mapping[item] for item in df["item_id"]]
    data = df["qty"].astype(float).values  # Use quantities as interaction values
    
    matrix = csr_matrix((data, (row_indices, col_indices)), shape=(n_users, n_items))
    
    return matrix


if __name__ == "__main__":
    app()