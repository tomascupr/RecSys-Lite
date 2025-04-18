"""Data ingestion functionality for RecSys-Lite."""

from pathlib import Path

import duckdb


def ingest_data(events_path: Path, items_path: Path, db_path: Path) -> None:
    """Ingest data into DuckDB database.

    Args:
        events_path: Path to events parquet file
        items_path: Path to items CSV file
        db_path: Path to DuckDB database
    """
    conn = duckdb.connect(str(db_path))
    
    # Create events table
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS events AS
        SELECT * FROM read_parquet('{events_path}')
        """
    )
    
    # Create items table
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS items AS
        SELECT * FROM read_csv('{items_path}')
        """
    )
    
    conn.close()