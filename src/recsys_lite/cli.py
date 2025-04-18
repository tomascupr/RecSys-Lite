"""Command-line interface for RecSys-Lite."""

from pathlib import Path
from typing import Optional

import typer

from recsys_lite.ingest import ingest_data

app = typer.Typer(help="RecSys-Lite: Lightweight recommendation system")


@app.command()
def ingest(
    events: Path = typer.Argument(..., help="Path to events parquet file"),
    items: Path = typer.Argument(..., help="Path to items CSV file"),
    db: Path = typer.Option("recsys.db", help="Path to DuckDB database"),
) -> None:
    """Ingest data into DuckDB database."""
    ingest_data(events, items, db)
    typer.echo(f"Data ingested successfully into {db}")


if __name__ == "__main__":
    app()