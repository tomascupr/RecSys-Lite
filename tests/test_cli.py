"""Light‑weight integration tests for the Typer CLI.

The goal is *not* to exercise the full training pipeline – that would require
DuckDB, FAISS and the heavy recommendation models – but to make sure that the
public commands are *callable* and that the high‑level side‑effects (creating
an output directory, returning a mapping, …) behave as advertised.

We therefore patch only the **external** dependencies (database connection,
models, FAISS index builder).  Everything else runs through the real
implementation so that the tests still give us a meaningful smoke‑signal in
case the CLI contract breaks in the future.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

# ---------------------------------------------------------------------------
#  Production objects we want to test.
# ---------------------------------------------------------------------------

from recsys_lite.cli import app as typer_app  # The real Typer instance
import recsys_lite.cli as cli_mod


# ---------------------------------------------------------------------------
#  Helpers / stubs
# ---------------------------------------------------------------------------


class _DummyFetcher:
    """Mimic the object DuckDB returns from ``execute``.

    We expose only the ``fetchdf`` method required by the CLI.
    """

    def __init__(self, payload: Any) -> None:  # payload is usually a DataFrame
        self._payload = payload

    def fetchdf(self) -> Any:  # noqa: D401 – simple stub
        return self._payload


class _DummyConn:
    """Very small subset of the DuckDB connection interface."""

    def __init__(self, events_df: pd.DataFrame):
        self._events_df = events_df

    # The CLI issues three different queries – we only need to distinguish the
    # *intent* of each.
    def execute(self, query: str) -> _DummyFetcher:  # noqa: D401 – stub
        if "SELECT DISTINCT user_id" in query:
            return _DummyFetcher({"user_id": self._events_df["user_id"].unique()})
        if "SELECT DISTINCT item_id" in query:
            return _DummyFetcher({"item_id": self._events_df["item_id"].unique()})
        # The full table (user_id, item_id, qty)
        return _DummyFetcher(self._events_df)

    def close(self) -> None:  # noqa: D401 – nothing to do
        return None


class _DummyALSModel:  # noqa: D101 – test stub
    def __init__(self, **_kwargs):
        # Accept arbitrary keyword arguments so that the CLI can pass its
        # parameter dictionary unmodified.
        pass

    def fit(self, _matrix):  # noqa: D401 – we ignore the data content
        return None

    def get_item_factors(self):  # small 1×1 matrix – just enough for the CLI
        return [[0.0]]


class _DummyFaissBuilder:  # noqa: D101 – test stub
    def __init__(self, *args, **kwargs):  # noqa: D401 – signature irrelevant
        pass

    def save(self, _path: str) -> None:  # noqa: D401 – no‑op
        return None


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cli_runner() -> CliRunner:  # noqa: D401 – Pytest fixture
    return CliRunner()


@pytest.fixture()
def patched_environment(monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: D401
    """Patch heavy dependencies inside *cli_mod* in‑place."""

    # 1) DuckDB connection – supply a minimal in‑memory dataset
    events_df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "item_id": ["i1", "i2", "i3"],
            "qty": [1, 2, 3],
        }
    )
    monkeypatch.setattr(
        cli_mod,  # patch attribute directly on the imported module
        "duckdb",
        SimpleNamespace(connect=lambda _path: _DummyConn(events_df)),
    )

    # 2) Replace ALSModel with a lightweight stub
    monkeypatch.setattr(cli_mod, "ALSModel", _DummyALSModel)

    # 3) Avoid importing FAISS
    monkeypatch.setattr(cli_mod, "FaissIndexBuilder", _DummyFaissBuilder)


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------


def test_train_command_runs(tmp_path: Path, cli_runner: CliRunner, patched_environment):
    """`recsys‑lite train` exits with *0* and writes the mapping files."""

    output_dir = tmp_path / "artifacts"

    result = cli_runner.invoke(
        typer_app,
        [
            "train",
            "als",
            "--db",
            "dummy.db",  # the path is irrelevant – we patch duckdb.connect
            "--output",
            str(output_dir),
            "--test-size",
            "0.5",  # keep matrices tiny
        ],
    )

    assert result.exit_code == 0, result.stdout

    # The CLI should create one sub‑directory per model type
    model_root = output_dir / "als"
    assert model_root.is_dir()

    # And write the user / item mapping JSON files
    assert (model_root / "user_mapping.json").exists()
    assert (model_root / "item_mapping.json").exists()


def test_get_interactions_matrix(monkeypatch: pytest.MonkeyPatch):
    """Smoke‑test the public helper with the same dummy DuckDB stub."""

    events_df = pd.DataFrame(
        {
            "user_id": ["A", "A", "B"],
            "item_id": ["X", "Y", "Z"],
            "qty": [1, 1, 1],
        }
    )

    monkeypatch.setattr(
        cli_mod,
        "duckdb",
        SimpleNamespace(connect=lambda _p: _DummyConn(events_df)),
    )

    from recsys_lite.cli import get_interactions_matrix

    matrix, user_map, item_map = get_interactions_matrix(Path("dummy.db"))

    # Basic sanity checks
    assert matrix.shape == (2, 3)  # 2 users × 3 items
    assert user_map == {"A": 0, "B": 1}
    assert item_map == {"X": 0, "Y": 1, "Z": 2}
