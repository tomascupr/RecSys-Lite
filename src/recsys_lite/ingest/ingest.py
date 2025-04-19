"""Data ingestion functionality for RecSys-Lite."""

from pathlib import Path
import time

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


# ---------------------------------------------------------------------------
# Streaming ingest helpers
# ---------------------------------------------------------------------------


def stream_events(
    events_dir: Path,
    db_path: Path,
    poll_interval: int = 5,
) -> None:
    """Continuously ingest parquet files dropped into a directory.

    The function performs a very simple *file‑based* streaming ingestion.  Any
    ``*.parquet`` file that appears in *events_dir* is **appended** to the
    ``events`` table inside the DuckDB database.  Already processed files are
    tracked in‑memory for the lifetime of the process, so the same file will
    not be imported twice.

    The implementation purposefully avoids external dependencies (e.g.
    ``watchdog``) – it just polls the directory every *poll_interval* seconds
    which is usually sufficient for low‑volume, near‑real‑time pipelines.

    Args:
        events_dir: Directory to watch for new parquet files.
        db_path:   Path to the DuckDB database containing an ``events`` table.
        poll_interval: Number of seconds to wait between directory scans.
    """

    events_dir = events_dir.expanduser().resolve()
    processed: set[str] = set()

    if not events_dir.exists():
        raise FileNotFoundError(f"Events directory '{events_dir}' does not exist")

    print(
        f"[stream-ingest] Watching '{events_dir}' for parquet files. "
        "Press Ctrl+C to stop."
    )

    try:
        while True:
            # Discover parquet files that have not been processed yet
            for parquet_file in sorted(events_dir.glob("*.parquet")):
                if parquet_file.name in processed:
                    continue

                try:
                    _append_parquet_to_events(parquet_file, db_path)
                    processed.add(parquet_file.name)
                    print(f"[stream-ingest] Ingested {parquet_file.name}")
                except Exception as exc:  # pragma: no cover – guard rail only
                    # We do not want the outer loop to die because of one bad file
                    print(
                        f"[stream-ingest] Failed to ingest {parquet_file.name}: {exc}"
                    )

            # Sleep before the next scan
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n[stream-ingest] Stopped – goodbye!")


def _append_parquet_to_events(parquet_file: Path, db_path: Path) -> None:
    """Helper that appends the content of *parquet_file* into ``events`` table."""

    conn = duckdb.connect(str(db_path))

    # Ensure the events table exists – if not, create it on‑the‑fly.
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS events AS
        SELECT * FROM read_parquet('{parquet_file}') WHERE 0=1
        """
    )

    # Append the actual data
    conn.execute(
        f"INSERT INTO events SELECT * FROM read_parquet('{parquet_file}')"
    )

    conn.close()
