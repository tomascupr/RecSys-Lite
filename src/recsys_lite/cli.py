"""Command-line interface for RecSys-Lite."""

import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import typer
from scipy.sparse import csr_matrix

from recsys_lite.indexing import FaissIndexBuilder

# We import lazily for stream to avoid optional dependency issues.
# Import message queue ingest
from recsys_lite.ingest import ingest_data, queue_ingest, stream_events
from recsys_lite.models import (
    ALSModel,
    BPRModel,
    GRU4Rec,
    HybridModel,
    Item2VecModel,
    LightFMModel,
    TextEmbeddingModel,
)
from recsys_lite.optimization import OptunaOptimizer

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
    EASE = "ease"
    TEXT_EMBEDDING = "text_embedding"
    HYBRID = "hybrid"


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


# ---------------------------------------------------------------------------
# Streaming ingest commands
# ---------------------------------------------------------------------------


@app.command(name="stream-ingest")
def stream_ingest(
    events_dir: Path = typer.Argument(..., help="Directory containing incremental parquet files"),
    db: Path = typer.Option("recsys.db", help="DuckDB database to append to"),
    poll_interval: int = typer.Option(5, help="Polling interval in seconds"),
) -> None:
    """Run a simple *file based* streaming ingest loop.

    The command watches *events_dir* for new ``*.parquet`` files and appends
    their contents to the ``events`` table in the specified DuckDB database.
    """

    typer.echo(f"Starting streaming ingest – watching '{events_dir}' every {poll_interval}s …")

    try:
        stream_events(events_dir, db, poll_interval=poll_interval)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


class QueueType(str, Enum):
    """Available message queue types."""

    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"


@app.command(name="queue-ingest")
def queue_ingest_command(
    queue_type: QueueType = typer.Argument(
        QueueType.RABBITMQ, help="Type of message queue (rabbitmq or kafka)"
    ),
    db: Path = typer.Option("recsys.db", help="DuckDB database to append to"),
    batch_size: int = typer.Option(100, help="Number of messages to process in a batch"),
    poll_interval: int = typer.Option(5, help="Polling interval in seconds if no messages"),
    # RabbitMQ specific options
    rabbitmq_host: str = typer.Option("localhost", help="RabbitMQ host"),
    rabbitmq_port: int = typer.Option(5672, help="RabbitMQ port"),
    rabbitmq_queue: str = typer.Option("events", help="RabbitMQ queue name"),
    rabbitmq_username: str = typer.Option("guest", help="RabbitMQ username"),
    rabbitmq_password: str = typer.Option("guest", help="RabbitMQ password"),
    rabbitmq_vhost: str = typer.Option("/", help="RabbitMQ virtual host"),
    # Kafka specific options
    kafka_servers: str = typer.Option("localhost:9092", help="Kafka bootstrap servers"),
    kafka_topic: str = typer.Option("events", help="Kafka topic"),
    kafka_group: str = typer.Option("recsys-lite", help="Kafka consumer group"),
) -> None:
    """Run a *message queue* based streaming ingest process.

    The command connects to a message queue (RabbitMQ or Kafka) and consumes
    event messages, appending them to the DuckDB database.

    For RabbitMQ, messages should be JSON objects with at least 'user_id' and 'item_id' fields.
    Optional fields include 'qty' (defaults to 1) and 'timestamp' (defaults to current time).

    This requires the optional message queue dependencies:
    pip install recsys-lite[mq]

    Or to install the specific dependencies:
    - For RabbitMQ: pip install pika
    - For Kafka: pip install kafka-python
    """
    # Construct queue configuration based on the selected queue type
    queue_config: Dict[str, Any] = {}

    if queue_type == QueueType.RABBITMQ:
        try:
            # Check for pika availability without importing
            import importlib.util

            if importlib.util.find_spec("pika") is None:
                raise ImportError("pika package not found")
        except ImportError as err:
            typer.echo(
                "Error: RabbitMQ support requires the pika package.\n"
                "Install it with: pip install recsys-lite[mq]",
                err=True,
            )
            raise typer.Exit(code=1) from err

        queue_config = {
            "host": rabbitmq_host,
            "port": rabbitmq_port,
            "queue": rabbitmq_queue,
            "username": rabbitmq_username,
            "password": rabbitmq_password,
            "virtual_host": rabbitmq_vhost,
        }
    elif queue_type == QueueType.KAFKA:
        try:
            # Check for kafka availability without importing
            import importlib.util

            if importlib.util.find_spec("kafka") is None:
                raise ImportError("kafka-python package not found")
        except ImportError as err:
            typer.echo(
                "Error: Kafka support requires the kafka-python package.\n"
                "Install it with: pip install recsys-lite[mq]",
                err=True,
            )
            raise typer.Exit(code=1) from err

        queue_config = {
            "bootstrap_servers": kafka_servers,
            "topic": kafka_topic,
            "group_id": kafka_group,
        }

    typer.echo(
        f"Starting {queue_type.value} queue-based ingest – "
        f"batch size: {batch_size}, poll interval: {poll_interval}s"
    )

    try:
        queue_ingest(
            queue_config=queue_config,
            db_path=db,
            queue_type=queue_type.value,
            batch_size=batch_size,
            poll_interval=poll_interval,
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# GDPR commands
# ---------------------------------------------------------------------------


@app.command()
def gdpr(ctx: typer.Context) -> None:
    """GDPR compliance tools for data export and user deletion."""
    # Called without subcommand
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


# Create a GDPR subgroup
gdpr_app = typer.Typer()
app.add_typer(gdpr_app, name="gdpr")


@gdpr_app.command(name="export-user")
def gdpr_export_user(
    user_id: str = typer.Argument(..., help="User ID to export data for"),
    db: Path = typer.Option("recsys.db", help="DuckDB database path"),
    output: Path = typer.Option(None, help="Output JSON file path (defaults to user_id.json)"),
) -> None:
    """Export all data for a specific user (GDPR compliance)."""
    if output is None:
        output = Path(f"{user_id}.json")

    # Connect to database
    conn = duckdb.connect(str(db))

    # Query user events
    events_df = conn.execute("SELECT * FROM events WHERE user_id = ?", [user_id]).fetchdf()

    # Get item metadata for interacted items
    if len(events_df) > 0:
        item_ids = events_df["item_id"].tolist()
        placeholders = ", ".join(["?"] * len(item_ids))
        item_df = conn.execute(
            f"SELECT * FROM items WHERE item_id IN ({placeholders})", item_ids
        ).fetchdf()
    else:
        item_df = conn.execute("SELECT * FROM items WHERE 1=0").fetchdf()

    # Prepare export data
    export_data = {
        "user_id": user_id,
        "export_timestamp": int(os.time()),
        "events": events_df.to_dict(orient="records"),
        "items": item_df.to_dict(orient="records"),
    }

    # Write to file
    with open(output, "w") as f:
        json.dump(export_data, f, indent=2)

    typer.echo(f"User data exported to {output}")


@gdpr_app.command(name="delete-user")
def gdpr_delete_user(
    user_id: str = typer.Argument(..., help="User ID to delete data for"),
    db: Path = typer.Option("recsys.db", help="DuckDB database path"),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm deletion without prompting"),
) -> None:
    """Delete all data for a specific user (GDPR compliance)."""
    # Connect to database
    conn = duckdb.connect(str(db))

    # Count user events
    event_count = conn.execute(
        "SELECT COUNT(*) FROM events WHERE user_id = ?", [user_id]
    ).fetchone()[0]

    if event_count == 0:
        typer.echo(f"No data found for user {user_id}")
        return

    # Confirm deletion
    if not confirm:
        typer.confirm(
            f"This will delete {event_count} events for user {user_id}. Continue?",
            abort=True,
        )

    # Delete user data
    conn.execute("DELETE FROM events WHERE user_id = ?", [user_id])

    # Add to deleted users table (create if not exists)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS deleted_users (
            user_id VARCHAR,
            deletion_timestamp BIGINT
        )
    """
    )

    conn.execute("INSERT INTO deleted_users VALUES (?, ?)", [user_id, int(os.time())])

    typer.echo(f"Deleted {event_count} events for user {user_id}")
    typer.echo("Note: User vectors will be removed during the next model update")


# ---------------------------------------------------------------------------
# Model commands
# ---------------------------------------------------------------------------


@app.command()
def train(
    model_type: ModelType = typer.Argument(..., help="Type of model to train"),
    db: Path = typer.Option("recsys.db", help="DuckDB database path"),
    output: Path = typer.Option(None, help="Output directory"),
    test_size: float = typer.Option(0.2, help="Test set size for evaluation"),
    seed: int = typer.Option(42, help="Random seed"),
    params_file: Optional[Path] = typer.Option(None, help="JSON file with model parameters"),
) -> None:
    """Train a recommendation model."""
    # Set default output directory if not specified
    if output is None:
        output = Path(f"model_artifacts/{model_type.value}")

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    # Load data from DuckDB
    logger.info(f"Loading data from {db}")
    conn = duckdb.connect(str(db))

    # Get user-item interaction matrix
    user_item_df = conn.execute(
        """
        SELECT user_id, item_id, CAST(SUM(qty) AS FLOAT) as interaction
        FROM events
        GROUP BY user_id, item_id
    """
    ).fetchdf()

    # Get item metadata
    item_df = conn.execute("SELECT * FROM items").fetchdf()

    # Create mappings
    logger.info("Creating user and item mappings")
    unique_users = user_item_df["user_id"].unique()
    unique_items = user_item_df["item_id"].unique()

    user_mapping = {user: idx for idx, user in enumerate(unique_users)}
    item_mapping = {item: idx for idx, item in enumerate(unique_items)}
    # Create reverse mapping for items
    reverse_item_mapping = {idx: item for item, idx in item_mapping.items()}

    # Create item metadata dictionary
    item_data = {}
    for _, row in item_df.iterrows():
        item_id = row["item_id"]
        item_data[item_id] = row.to_dict()

    # Create user-item matrix
    if model_type == ModelType.TEXT_EMBEDDING:
        # For text embedding, we don't need to create a matrix
        user_item_matrix = None
    else:
        # Create sparse matrix for collaborative filtering models
        rows = []
        cols = []
        data = []

        for _, row in user_item_df.iterrows():
            user_idx = user_mapping[row["user_id"]]
            item_idx = item_mapping[row["item_id"]]
            interaction = row["interaction"]

            rows.append(user_idx)
            cols.append(item_idx)
            data.append(interaction)

        user_item_matrix = csr_matrix(
            (data, (rows, cols)), shape=(len(user_mapping), len(item_mapping))
        )

    # Load parameters from file if specified
    params = {}
    if params_file and params_file.exists():
        with open(params_file, "r") as f:
            params = json.load(f)

    # Initialize model based on type with parameters
    logger.info(f"Initializing {model_type.value} model")

    if model_type == ModelType.ALS:
        model = ALSModel(
            factors=params.get("factors", 128),
            regularization=params.get("regularization", 0.01),
            alpha=params.get("alpha", 1.0),
            iterations=params.get("iterations", 15),
        )
    elif model_type == ModelType.BPR:
        model = BPRModel(
            factors=params.get("factors", 100),
            learning_rate=params.get("learning_rate", 0.05),
            regularization=params.get("regularization", 0.01),
            iterations=params.get("iterations", 100),
        )
    elif model_type == ModelType.ITEM2VEC:
        model = Item2VecModel(
            vector_size=params.get("vector_size", 100),
            window=params.get("window", 5),
            min_count=params.get("min_count", 5),
            sg=params.get("sg", 1),
            epochs=params.get("epochs", 5),
        )
    elif model_type == ModelType.LIGHTFM:
        model = LightFMModel(
            no_components=params.get("no_components", 64),
            learning_rate=params.get("learning_rate", 0.05),
            loss=params.get("loss", "warp"),
            epochs=params.get("epochs", 50),
        )
    elif model_type == ModelType.GRU4REC:
        model = GRU4Rec(
            hidden_size=params.get("hidden_size", 100),
            n_layers=params.get("n_layers", 1),
            dropout=params.get("dropout", 0.1),
            batch_size=params.get("batch_size", 64),
            learning_rate=params.get("learning_rate", 0.001),
            n_epochs=params.get("n_epochs", 10),
        )
    elif model_type == ModelType.EASE:
        model = LightFMModel(
            lambda_=params.get("lambda_", 0.5),
        )
    elif model_type == ModelType.TEXT_EMBEDDING:
        model = TextEmbeddingModel(
            model_name=params.get("model_name", "all-MiniLM-L6-v2"),
            item_text_fields=params.get(
                "item_text_fields", ["title", "category", "brand", "description"]
            ),
            field_weights=params.get(
                "field_weights",
                {
                    "title": 2.0,
                    "category": 1.0,
                    "brand": 1.0,
                    "description": 3.0,
                },
            ),
            normalize_vectors=params.get("normalize_vectors", True),
            batch_size=params.get("batch_size", 64),
            max_length=params.get("max_length", 512),
        )
    elif model_type == ModelType.HYBRID:
        # For hybrid, we don't support direct training
        typer.echo("Error: Hybrid models should be created using the train-hybrid command")
        raise typer.Exit(code=1)
    else:
        typer.echo(f"Unknown model type: {model_type}")
        raise typer.Exit(code=1)

    # Train model
    logger.info(f"Training {model_type.value} model")

    if model_type == ModelType.TEXT_EMBEDDING:
        # For text embedding, we need item data
        model.fit(
            user_item_matrix=user_item_matrix,
            item_data=item_data,
            output_dir=output,
        )
    else:
        # For collaborative filtering models
        model.fit(user_item_matrix)

    # Save model, mappings, and Faiss index
    logger.info(f"Saving model and artifacts to {output}")
    model.save_model(str(output))

    # Save mappings
    with open(output / "user_mapping.json", "w") as f:
        json.dump(user_mapping, f)

    with open(output / "item_mapping.json", "w") as f:
        json.dump(item_mapping, f)

    # Create Faiss index for similarity search
    logger.info("Building Faiss index")
    index_builder = FaissIndexBuilder()
    index_builder.build_index(
        model=model,
        item_mapping=item_mapping,
        reverse_item_mapping=reverse_item_mapping,
        output_dir=output / "faiss_index",
    )

    logger.info(f"{model_type.value} model training complete")


@app.command()
def train_hybrid(
    models_dir: List[Path] = typer.Argument(..., help="List of model directories to combine"),
    output: Path = typer.Option(None, help="Output directory"),
    weights: Optional[List[float]] = typer.Option(None, help="Model weights (comma-separated)"),
    dynamic: bool = typer.Option(True, help="Use dynamic weighting based on user history"),
    cold_start_threshold: int = typer.Option(5, help="Threshold for cold-start users"),
    cold_start_strategy: str = typer.Option(
        "content_boost", help="Strategy for cold-start users (content_boost|content_only|equal)"
    ),
):
    """Create a hybrid model combining multiple recommenders."""
    # Set default output directory if not specified
    if output is None:
        output = Path("model_artifacts/hybrid")

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    # Load component models
    logger.info(f"Loading {len(models_dir)} component models")
    models = []
    model_types = []

    for model_dir in models_dir:
        # Determine model type from directory
        model_files = list(model_dir.glob("*_model.pkl"))
        if not model_files:
            logger.warning(f"No model found in {model_dir}")
            continue

        model_file = model_files[0]
        model_type = model_file.stem.split("_")[0]
        model_types.append(model_type)

        # Load model
        try:
            from recsys_lite.models import ModelRegistry

            model = ModelRegistry.load_model(model_type, str(model_dir))
            models.append(model)
            logger.info(f"Loaded {model_type} model from {model_dir}")
        except Exception as e:
            logger.error(f"Error loading model from {model_dir}: {e}")

    if not models:
        logger.error("No models could be loaded")
        raise typer.Exit(code=1)

    # Parse weights if provided
    weight_values = None
    if weights:
        weight_values = [float(w) for w in weights]

        # Validate weights
        if len(weight_values) != len(models):
            logger.warning(
                f"Number of weights ({len(weight_values)}) doesn't match "
                f"number of models ({len(models)}). Using equal weights."
            )
            weight_values = None

    # Initialize hybrid model
    logger.info(f"Creating hybrid model with {len(models)} components")
    hybrid_model = HybridModel(
        models=models,
        weights=weight_values,
        dynamic_weighting=dynamic,
        cold_start_threshold=cold_start_threshold,
        cold_start_strategy=cold_start_strategy,
    )

    # Save hybrid model
    logger.info(f"Saving hybrid model to {output}")
    hybrid_model.save_model(str(output))

    # Copy mappings from first model
    first_model_dir = models_dir[0]
    for mapping_file in ["user_mapping.json", "item_mapping.json"]:
        src_path = first_model_dir / mapping_file
        if src_path.exists():
            import shutil

            shutil.copy(src_path, output / mapping_file)

    # Create Faiss index
    logger.info("Building Faiss index for hybrid model")

    # Load mappings
    with open(output / "item_mapping.json", "r") as f:
        item_mapping = json.load(f)

    reverse_item_mapping = {int(idx): item for item, idx in item_mapping.items()}

    index_builder = FaissIndexBuilder()
    index_builder.build_index(
        model=hybrid_model,
        item_mapping=item_mapping,
        reverse_item_mapping=reverse_item_mapping,
        output_dir=output / "faiss_index",
    )

    logger.info(f"Hybrid model ({'+'.join(model_types)}) creation complete")


@app.command()
def optimize(
    model_type: ModelType = typer.Argument(..., help="Type of model to optimize"),
    db: Path = typer.Option("recsys.db", help="DuckDB database path"),
    output: Path = typer.Option(None, help="Output directory"),
    metric: MetricType = typer.Option(MetricType.NDCG_20, help="Evaluation metric"),
    trials: int = typer.Option(20, help="Number of optimization trials"),
    test_size: float = typer.Option(0.2, help="Test set size for evaluation"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Optimize hyperparameters for a recommendation model."""
    # Set default output directory if not specified
    if output is None:
        output = Path(f"model_artifacts/{model_type.value}")

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    # Load data from DuckDB
    logger.info(f"Loading data from {db}")
    conn = duckdb.connect(str(db))

    # Get user-item interaction matrix
    user_item_df = conn.execute(
        """
        SELECT user_id, item_id, CAST(SUM(qty) AS FLOAT) as interaction
        FROM events
        GROUP BY user_id, item_id
    """
    ).fetchdf()

    # Create mappings
    logger.info("Creating user and item mappings")
    unique_users = user_item_df["user_id"].unique()
    unique_items = user_item_df["item_id"].unique()

    user_mapping = {user: idx for idx, user in enumerate(unique_users)}
    item_mapping = {item: idx for idx, item in enumerate(unique_items)}

    # Create user-item matrix
    rows = []
    cols = []
    data = []

    for _, row in user_item_df.iterrows():
        user_idx = user_mapping[row["user_id"]]
        item_idx = item_mapping[row["item_id"]]
        interaction = row["interaction"]

        rows.append(user_idx)
        cols.append(item_idx)
        data.append(interaction)

    user_item_matrix = csr_matrix(
        (data, (rows, cols)), shape=(len(user_mapping), len(item_mapping))
    )

    # Define parameter spaces for each model type
    if model_type == ModelType.ALS:
        param_space = {
            "factors": [32, 64, 128, 256],
            "regularization": [0.001, 0.01, 0.1, 1.0],
            "alpha": [0.1, 1.0, 10.0, 40.0],
            "iterations": [10, 15, 20],
        }
    elif model_type == ModelType.BPR:
        param_space = {
            "factors": [32, 64, 100, 128, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "regularization": [0.001, 0.01, 0.1],
            "iterations": [50, 100, 200],
        }
    elif model_type == ModelType.ITEM2VEC:
        param_space = {
            "vector_size": [32, 64, 100, 128, 200],
            "window": [3, 5, 10],
            "min_count": [1, 3, 5],
            "sg": [0, 1],
            "epochs": [3, 5, 10],
        }
    elif model_type == ModelType.LIGHTFM:
        param_space = {
            "no_components": [32, 64, 128],
            "learning_rate": [0.01, 0.05, 0.1],
            "loss": ["warp", "bpr", "logistic"],
            "epochs": [10, 30, 50],
        }
    elif model_type == ModelType.GRU4REC:
        param_space = {
            "hidden_size": [50, 100, 200],
            "n_layers": [1, 2],
            "dropout": [0.0, 0.1, 0.2, 0.3],
            "batch_size": [32, 64, 128],
            "learning_rate": [0.0001, 0.001, 0.01],
            "n_epochs": [5, 10, 15],
        }
    elif model_type == ModelType.EASE:
        param_space = {
            "lambda_": [0.1, 0.5, 1.0, 5.0, 10.0],
        }
    elif model_type == ModelType.TEXT_EMBEDDING:
        param_space = {
            "model_name": [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "paraphrase-multilingual-MiniLM-L12-v2",
            ],
            "batch_size": [32, 64, 128],
            "field_weights": [
                # Title focused
                {"title": 3.0, "category": 1.0, "brand": 1.0, "description": 2.0},
                # Description focused
                {"title": 2.0, "category": 1.0, "brand": 1.0, "description": 3.0},
                # Balanced
                {"title": 2.0, "category": 1.5, "brand": 1.5, "description": 2.0},
            ],
        }
    elif model_type == ModelType.HYBRID:
        typer.echo("Optimization not supported for hybrid models")
        raise typer.Exit(code=1)
    else:
        typer.echo(f"Unknown model type: {model_type}")
        raise typer.Exit(code=1)

    # Initialize optimizer
    optimizer = OptunaOptimizer(
        model_type=model_type.value,
        param_space=param_space,
        metric=metric.value,
        n_trials=trials,
        test_size=test_size,
        random_state=seed,
    )

    # Run optimization
    logger.info(f"Running optimization with {trials} trials")
    best_params = optimizer.optimize(user_item_matrix)

    # Save best parameters
    params_file = output / "best_params.json"
    with open(params_file, "w") as f:
        json.dump(best_params, f, indent=2)

    logger.info(f"Best parameters saved to {params_file}")

    # Train model with best parameters
    logger.info("Training model with best parameters")

    # Use the train command to train with best parameters
    # We use a temporary file to pass the parameters
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp:
        json.dump(best_params, temp)
        temp_path = temp.name

    # Train model with best parameters
    train(
        model_type=model_type,
        db=db,
        output=output,
        test_size=test_size,
        seed=seed,
        params_file=Path(temp_path),
    )

    # Clean up temporary file
    os.unlink(temp_path)

    logger.info("Optimization complete")


@app.command()
def serve(
    model_dir: Path = typer.Option("model_artifacts/als", help="Model directory"),
    host: str = typer.Option("0.0.0.0", help="Host to listen on"),
    port: int = typer.Option(8000, help="Port to listen on"),
    workers: int = typer.Option(4, help="Number of worker processes"),
    log_level: str = typer.Option("info", help="Log level (debug, info, warning, error)"),
) -> None:
    """Start the recommendation API server."""
    import uvicorn

    from recsys_lite.api.main import create_app

    # Check if model directory exists
    if not model_dir.exists():
        typer.echo(f"Model directory {model_dir} does not exist")
        raise typer.Exit(code=1)

    # Create app with model path
    app = create_app(model_dir=str(model_dir))

    # Start server
    typer.echo(f"Starting API server at http://{host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
    )


@app.command()
def worker(
    model_dir: Path = typer.Option("model_artifacts/als", help="Model directory"),
    db: Path = typer.Option("recsys.db", help="DuckDB database path"),
    interval: int = typer.Option(60, help="Update interval in seconds"),
    incremental_dir: Optional[Path] = typer.Option(
        None, help="Directory to watch for incremental data"
    ),
) -> None:
    """Start the update worker for incremental model updates."""
    from recsys_lite.update.worker import UpdateWorker

    # Check if model directory exists
    if not model_dir.exists():
        typer.echo(f"Model directory {model_dir} does not exist")
        raise typer.Exit(code=1)

    # Check if database exists
    if not db.exists():
        typer.echo(f"Database {db} does not exist")
        raise typer.Exit(code=1)

    # Create worker
    worker = UpdateWorker(
        model_dir=str(model_dir),
        db_path=str(db),
        interval=interval,
        incremental_dir=str(incremental_dir) if incremental_dir else None,
    )

    # Start worker
    typer.echo(f"Starting update worker with {interval}s interval")
    worker.run()


if __name__ == "__main__":
    app()
