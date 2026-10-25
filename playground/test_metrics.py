"""Unit tests for Issue #173: Latency and token usage metrics in playground."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

playground_dir = Path(__file__).parent.resolve()
if str(playground_dir) not in sys.path:
    sys.path.insert(0, str(playground_dir))

from main import app, extract_usage_metrics, response_cache  # noqa: E402

client = TestClient(app)


def test_extract_usage_metrics_dict():
    """Verify extract_usage_metrics parses a raw dict."""
    crew = MagicMock()
    crew.usage_metrics = {
        "total_tokens": 8214,
        "prompt_tokens": 6102,
        "completion_tokens": 2112,
        "successful_requests": 3,
    }
    metrics = extract_usage_metrics(crew)
    assert metrics == {
        "total_tokens": 8214,
        "prompt_tokens": 6102,
        "completion_tokens": 2112,
        "successful_requests": 3,
    }


def test_extract_usage_metrics_fallback_to_none():
    """Verify unpopulated or mock usage_metrics returns None rather than crashing."""
    crew = MagicMock()
    # If not set, crew.usage_metrics is a MagicMock, which must safely return None
    assert extract_usage_metrics(crew) is None

    crew.usage_metrics = None
    assert extract_usage_metrics(crew) is None

    crew.usage_metrics = {}
    assert extract_usage_metrics(crew) is None


def test_run_endpoint_schema_with_usage():
    """Verify POST /run response schema includes latency_seconds and usage dict."""
    response_cache.clear()
    mock_crew = MagicMock()
    mock_crew.kickoff.return_value = "Result with usage"
    mock_crew.usage_metrics = {
        "total_tokens": 8214,
        "prompt_tokens": 6102,
        "completion_tokens": 2112,
        "successful_requests": 3,
    }

    def _exec(mod):
        mod.build_crew = MagicMock(return_value=mock_crew)

    with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
        with patch(
            "importlib.machinery.SourceFileLoader.exec_module", side_effect=_exec
        ):
            payload = {
                "recipe": "faq-bot",
                "inputs": {"question": "Pricing", "customer_name": "Alice"},
                "bypass_cache": True,
            }
            res = client.post("/run", json=payload)
            assert res.status_code == 200
            data = res.json()

            # Schema checks
            assert "output" in data
            assert data["output"] == "Result with usage"
            assert "cached" in data
            assert data["cached"] is False
            assert "latency_seconds" in data
            assert isinstance(data["latency_seconds"], (int, float))
            assert data["latency_seconds"] >= 0.0

            assert "usage" in data
            assert isinstance(data["usage"], dict)
            assert data["usage"]["total_tokens"] == 8214
            assert data["usage"]["prompt_tokens"] == 6102
            assert data["usage"]["completion_tokens"] == 2112
            assert data["usage"]["successful_requests"] == 3


def test_run_endpoint_schema_when_usage_unpopulated():
    """Verify POST /run falls back to usage: null when metrics are missing."""
    response_cache.clear()
    mock_crew = MagicMock()
    mock_crew.kickoff.return_value = "Result without usage"
    mock_crew.usage_metrics = None

    def _exec(mod):
        mod.build_crew = MagicMock(return_value=mock_crew)

    with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
        with patch(
            "importlib.machinery.SourceFileLoader.exec_module", side_effect=_exec
        ):
            payload = {
                "recipe": "faq-bot",
                "inputs": {"question": "Pricing", "customer_name": "Bob"},
                "bypass_cache": True,
            }
            res = client.post("/run", json=payload)
            assert res.status_code == 200
            data = res.json()

            assert "latency_seconds" in data
            assert isinstance(data["latency_seconds"], (int, float))
            assert "usage" in data
            assert data["usage"] is None


def test_cached_run_includes_metrics_schema():
    """Verify cached response preserves latency_seconds and usage fields."""
    response_cache.clear()
    response_cache.set(
        "faq-bot", {"question": "Pricing", "customer_name": "Charlie"}, "Cached Text"
    )

    with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}):
        payload = {
            "recipe": "faq-bot",
            "inputs": {"question": "Pricing", "customer_name": "Charlie"},
        }
        res = client.post("/run", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["cached"] is True
        assert data["output"] == "Cached Text"
        assert "latency_seconds" in data
        assert "usage" in data
        assert data["usage"] is None


def test_frontend_renders_footer_elements():
    """Verify index.html contains the result footer and formatting logic."""
    index_html = (playground_dir / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="result-footer"' in index_html
    assert 'class="result-footer"' in index_html
    assert "renderResultStats" in index_html
    assert "clearResultStats" in index_html
    assert "LLM call" in index_html
    assert "tokens" in index_html
    assert "prompt:" in index_html
    assert "completion:" in index_html
