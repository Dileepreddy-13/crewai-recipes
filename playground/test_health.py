"""Unit tests for Issue #170: Playground liveness /health endpoint."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

playground_dir = Path(__file__).parent.resolve()
if str(playground_dir) not in sys.path:
    sys.path.insert(0, str(playground_dir))

from main import app  # noqa: E402

client = TestClient(app)


def test_health_endpoint() -> None:
    """Verify /health returns 200 with status ok and non-empty discovered recipes."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["recipes"], int)
    assert data["recipes"] > 0
    assert isinstance(data["recipe_names"], list)
    assert len(data["recipe_names"]) == data["recipes"]
    assert data["recipe_names"] == sorted(data["recipe_names"])
