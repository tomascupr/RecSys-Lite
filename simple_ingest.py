"""Simple script to ingest data into DuckDB."""

import duckdb
from pathlib import Path

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
    
    # Show table info
    print("Events table:")
    result = conn.execute("SELECT * FROM events LIMIT 5").fetchall()
    for row in result:
        print(row)
    
    print("\nItems table:")
    result = conn.execute("SELECT * FROM items LIMIT 5").fetchall()
    for row in result:
        print(row)
    
    # Get table counts
    event_count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    item_count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    
    print(f"\nIngested {event_count} events and {item_count} items")
    
    conn.close()

if __name__ == "__main__":
    events_path = Path("data/sample_data/events.parquet")
    items_path = Path("data/sample_data/items.csv")
    db_path = Path("data/recsys.db")
    
    print(f"Ingesting data from {events_path} and {items_path} into {db_path}")
    ingest_data(events_path, items_path, db_path)