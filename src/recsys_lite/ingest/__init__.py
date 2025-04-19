"""Data ingestion module for RecSys-Lite."""

from .ingest import ingest_data, queue_ingest, stream_events

__all__ = ["ingest_data", "stream_events", "queue_ingest"]
