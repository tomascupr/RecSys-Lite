"""Serving and update worker commands for RecSys-Lite CLI."""
from pathlib import Path
import typer

from recsys_lite.cli import app


@app.command()
def serve(
    model_dir: Path = typer.Option("model_artifacts/als", help="Model directory"),
    host: str = typer.Option("0.0.0.0", help="Host to listen on"),
    port: int = typer.Option(8000, help="Port to listen on"),
    workers: int = typer.Option(4, help="Number of worker processes"),
    log_level: str = typer.Option("info", help="Log level"),
) -> None:
    """Start the recommendation API server."""
    import uvicorn
    from recsys_lite.api.main import create_app

    if not model_dir.exists():
        typer.echo(f"Model directory {model_dir} does not exist")
        raise typer.Exit(code=1)

    app_instance = create_app(model_dir=str(model_dir))
    typer.echo(f"Starting API server at http://{host}:{port}")
    uvicorn.run(app_instance, host=host, port=port, workers=workers, log_level=log_level)


@app.command()
def worker(
    model_dir: Path = typer.Option("model_artifacts/als", help="Model directory"),
    db: Path = typer.Option("recsys.db", help="DuckDB database path"),
    interval: int = typer.Option(60, help="Update interval in seconds"),
    incremental_dir: Path = typer.Option(None, help="Directory to watch for incremental data"),
) -> None:
    """Start the update worker for incremental model updates."""
    from recsys_lite.api.loaders import load_model, load_faiss_index
    from recsys_lite.update.worker import UpdateWorker

    if not model_dir.exists():
        typer.echo(f"Model directory {model_dir} does not exist")
        raise typer.Exit(code=1)
    if not db.exists():
        typer.echo(f"Database {db} does not exist")
        raise typer.Exit(code=1)

    model, model_type = load_model(Path(model_dir))
    faiss_index = load_faiss_index(Path(model_dir))
    import json
    item_map = json.loads((model_dir / "item_mapping.json").read_text())
    reverse_map = {int(v): k for k, v in item_map.items()}

    worker_instance = UpdateWorker(
        db_path=db,
        model=model,
        faiss_index=faiss_index,
        item_id_map=reverse_map,
        interval=interval,
        incremental_dir=incremental_dir,
    )
    typer.echo(f"Starting update worker with {interval}s interval")
    worker_instance.run()