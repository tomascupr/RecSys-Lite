"""CLI entrypoint for RecSys-Lite: command registration and helpers."""
import logging
import typer
from pathlib import Path
from typing import Tuple, Dict
from scipy.sparse import csr_matrix

import duckdb
from recsys_lite.indexing import FaissIndexBuilder
from recsys_lite.optimization.optimizer import OptunaOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("recsys-lite")

# Create the CLI app
app = typer.Typer(help="RecSys-Lite: Lightweight recommendation system")

def get_interactions_matrix(db_path: Path) -> Tuple[csr_matrix, Dict[str, int], Dict[str, int]]:
    """Get user-item interaction matrix from DuckDB."""
    conn = duckdb.connect(str(db_path))
    df = conn.execute(
        """
        SELECT user_id, item_id, CAST(SUM(qty) AS FLOAT) as interaction
        FROM events
        GROUP BY user_id, item_id
        """
    ).fetchdf()
    users = conn.execute("SELECT DISTINCT user_id FROM events").fetchdf()["user_id"]
    items = conn.execute("SELECT DISTINCT item_id FROM events").fetchdf()["item_id"]
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {it: j for j, it in enumerate(items)}
    rows, cols, data = [], [], []
    for _, row in df.iterrows():
        rows.append(user_map[row["user_id"]])
        cols.append(item_map[row["item_id"]])
        data.append(row["interaction"])
    matrix = csr_matrix((data, (rows, cols)), shape=(len(user_map), len(item_map)))
    conn.close()
    return matrix, user_map, item_map

# Import command modules (they register with the global `app`)
from recsys_lite.cli.ingest import ingest, stream_ingest, queue_ingest_command  # noqa: F401
from recsys_lite.cli.gdpr import export_user, delete_user  # noqa: F401
from recsys_lite.cli.types import ModelType, MetricType, QueueType  # noqa: F401
from recsys_lite.cli.model import train, train_hybrid  # noqa: F401
from recsys_lite.cli.optimize import optimize  # noqa: F401
from recsys_lite.cli.serve import serve, worker  # noqa: F401

__all__ = [
    "app",
    "get_interactions_matrix",
    "ingest",
    "stream_ingest",
    "queue_ingest_command",
    "export_user",
    "delete_user",
    "ModelType",
    "MetricType",
    "QueueType",
    "train",
    "train_hybrid",
    "optimize",
    "serve",
    "worker",
]
from pathlib import Path
from typing import Tuple, Dict
from scipy.sparse import csr_matrix
import duckdb

def get_interactions_matrix(db_path: Path) -> Tuple[csr_matrix, Dict[str, int], Dict[str, int]]:
    """Get user-item interaction matrix."""
    # Use package-level duckdb (patchable in tests)
    conn = duckdb.connect(str(db_path))
    # Aggregate interactions
    df = conn.execute(
        """
        SELECT user_id, item_id, CAST(SUM(qty) AS FLOAT) as interaction
        FROM events
        GROUP BY user_id, item_id
        """
    ).fetchdf()
    # Unique users and items
    users = conn.execute("SELECT DISTINCT user_id FROM events").fetchdf()["user_id"]
    items = conn.execute("SELECT DISTINCT item_id FROM events").fetchdf()["item_id"]
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {it: j for j, it in enumerate(items)}
    # Build sparse matrix
    rows, cols, data = [], [], []
    for _, row in df.iterrows():
        rows.append(user_map[row["user_id"]])
        cols.append(item_map[row["item_id"]])
        data.append(row["interaction"])
    matrix = csr_matrix((data, (rows, cols)), shape=(len(user_map), len(item_map)))
    conn.close()
    return matrix, user_map, item_map