"""Simplified test script for API."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Add the project's src directory to the Python path
project_dir = Path(__file__).parent.parent
src_dir = project_dir / "src"
sys.path.insert(0, str(src_dir))

# Import the API we want to test
try:
    from recsys_lite.api.main import app

    print("Imported app directly")
except ImportError:
    from recsys_lite.api.main import create_app

    app = create_app()
    print("Created app from create_app function")

# Create a test client
client = TestClient(app)

print("Testing API endpoints...")

# Test the health endpoint
print("\nTesting /health endpoint")
response = client.get("/health")
print(f"Status code: {response.status_code}")
print(f"Response: {response.json()}")

# Test the metrics endpoint
print("\nTesting /metrics endpoint")
response = client.get("/metrics")
print(f"Status code: {response.status_code}")
print(f"Response: {response.json()}")

print("\nAPI test completed!")
