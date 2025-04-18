"""Tests for the RecSys-Lite CLI module."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from recsys_lite.cli import (
    ModelType,
    MetricType,
    app,
    get_interactions_matrix,
    optimize_hyperparameters,
)


@pytest.fixture
def cli_runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_data_dir(tmpdir):
    """Create a temporary directory for data."""
    # Create dummy data files
    events_file = tmpdir.join("events.parquet")
    items_file = tmpdir.join("items.csv")
    db_file = tmpdir.join("test.db")
    model_dir = tmpdir.mkdir("model_artifacts")
    
    return {
        "tmpdir": tmpdir,
        "events_file": events_file,
        "items_file": items_file,
        "db_file": db_file,
        "model_dir": model_dir,
    }


def test_ingest_command(cli_runner, temp_data_dir, monkeypatch):
    """Test the ingest command."""
    # Mock the ingest_data function
    mock_ingest = MagicMock()
    monkeypatch.setattr("recsys_lite.cli.ingest_data", mock_ingest)
    
    # Run the command
    result = cli_runner.invoke(
        app, 
        [
            "ingest", 
            str(temp_data_dir["events_file"]), 
            str(temp_data_dir["items_file"]), 
            "--db", str(temp_data_dir["db_file"])
        ]
    )
    
    # Check the result
    assert result.exit_code == 0
    assert "Data ingested successfully" in result.stdout
    
    # Verify the mock was called correctly
    mock_ingest.assert_called_once_with(
        temp_data_dir["events_file"], 
        temp_data_dir["items_file"], 
        temp_data_dir["db_file"]
    )


@patch("recsys_lite.cli.ALSModel")
@patch("recsys_lite.cli.get_interactions_matrix")
def test_train_als_command(mock_get_matrix, mock_als_model, cli_runner, temp_data_dir, monkeypatch):
    """Test the train command with ALS model."""
    # Mock the necessary functions and objects
    mock_model_instance = MagicMock()
    mock_als_model.return_value = mock_model_instance
    
    # Mock user-item matrix
    mock_matrix = MagicMock()
    mock_get_matrix.return_value = (mock_matrix, {}, {})
    
    # Mock os.makedirs
    monkeypatch.setattr("os.makedirs", MagicMock())
    
    # Create a parameters file
    params_file = temp_data_dir["tmpdir"].join("als_params.json")
    params = {
        "factors": 64,
        "regularization": 0.02,
        "alpha": 1.5,
        "iterations": 10,
    }
    params_file.write(json.dumps(params))
    
    # Run the command
    result = cli_runner.invoke(
        app, 
        [
            "train", 
            "als", 
            "--db", str(temp_data_dir["db_file"]),
            "--output", str(temp_data_dir["model_dir"]),
            "--params-file", str(params_file)
        ]
    )
    
    # Check the result
    assert result.exit_code == 0
    assert "Training model: als" in result.stdout
    
    # Verify the model was called with the parameters
    mock_als_model.assert_called_once_with(
        factors=params["factors"],
        regularization=params["regularization"],
        alpha=params["alpha"],
        iterations=params["iterations"]
    )
    
    # Verify fit was called
    mock_model_instance.fit.assert_called_once()


@patch("recsys_lite.cli.duckdb.connect")
def test_get_interactions_matrix(mock_connect, temp_data_dir):
    """Test the get_interactions_matrix function."""
    # Mock the database connection and results
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    
    # Mock the fetchdf results
    mock_conn.execute.return_value.fetchdf.return_value = MagicMock(
        to_numpy=MagicMock(return_value=[(1, "item1"), (2, "item2"), (1, "item3")])
    )
    
    # Set up user and item lists for the mock
    users = ["user1", "user2"]
    items = ["item1", "item2", "item3"]
    
    # Mock the execute calls to return users and items
    def mock_execute(query):
        if "SELECT DISTINCT user_id" in query:
            df_mock = MagicMock()
            df_mock.fetchdf.return_value = {"user_id": users}
            return df_mock
        elif "SELECT DISTINCT item_id" in query:
            df_mock = MagicMock()
            df_mock.fetchdf.return_value = {"item_id": items}
            return df_mock
        elif "SELECT user_id, item_id, qty" in query:
            df_mock = MagicMock()
            # Return a dataframe with user_id, item_id, qty columns
            df_mock.fetchdf.return_value = {
                "user_id": ["user1", "user1", "user2"],
                "item_id": ["item1", "item2", "item3"],
                "qty": [1, 2, 3]
            }
            return df_mock
        return MagicMock()
    
    mock_conn.execute.side_effect = mock_execute
    
    # Call the function
    matrix, user_mapping, item_mapping = get_interactions_matrix(temp_data_dir["db_file"])
    
    # Check the results
    assert user_mapping == {"user1": 0, "user2": 1}
    assert item_mapping == {"item1": 0, "item2": 1, "item3": 2}
    
    # Verify connect was called with the correct path
    mock_connect.assert_called_once_with(str(temp_data_dir["db_file"]))


@patch("recsys_lite.cli.OptunaOptimizer")
@patch("recsys_lite.cli.get_interactions_matrix")
def test_optimize_hyperparameters(mock_get_matrix, mock_optimizer, temp_data_dir):
    """Test the optimize_hyperparameters function."""
    # Mock the necessary functions and objects
    mock_optimizer_instance = MagicMock()
    mock_optimizer.return_value = mock_optimizer_instance
    
    # Mock optimize return value
    mock_optimizer_instance.optimize.return_value = {
        "factors": 64,
        "regularization": 0.02,
    }
    
    # Mock get_best_model return value
    mock_model = MagicMock()
    mock_optimizer_instance.get_best_model.return_value = mock_model
    
    # Mock user-item matrix
    mock_matrix = MagicMock()
    mock_user_mapping = {"user1": 0, "user2": 1}
    mock_item_mapping = {"item1": 0, "item2": 1}
    mock_get_matrix.return_value = (mock_matrix, mock_user_mapping, mock_item_mapping)
    
    # Call the function
    result = optimize_hyperparameters(
        model_type=ModelType.ALS,
        db_path=temp_data_dir["db_file"],
        output_dir=temp_data_dir["model_dir"],
        metric=MetricType.NDCG_10,
        n_trials=10,
        test_size=0.2,
        seed=42
    )
    
    # Check the results
    assert result == {
        "factors": 64,
        "regularization": 0.02,
    }
    
    # Verify the optimizer was called correctly
    mock_optimizer.assert_called_once()
    mock_optimizer_instance.optimize.assert_called_once()
    mock_optimizer_instance.get_best_model.assert_called_once()