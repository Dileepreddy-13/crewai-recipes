"""Offline tests for LLM timeout handling in the playground (#111)."""

import os
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

playground_dir = Path(__file__).parent.resolve()
if str(playground_dir) not in sys.path:
    sys.path.insert(0, str(playground_dir))

from main import app, execute_recipe_stream, is_timeout_error, timeout_message  # noqa: E402

client = TestClient(app)


class APITimeoutError(Exception):
    """Stand-in for openai.APITimeoutError, matched by class name."""


def _slow_crew(exc: Exception) -> MagicMock:
    crew = MagicMock()
    crew.kickoff.side_effect = exc
    return crew


def test_is_timeout_error_walks_the_cause_chain() -> None:
    try:
        try:
            raise APITimeoutError("Request timed out.")
        except APITimeoutError as inner:
            raise RuntimeError("litellm wrapper") from inner
    except RuntimeError as outer:
        assert is_timeout_error(outer)
    assert not is_timeout_error(ValueError("bad input"))


def test_timeout_message_says_what_to_do() -> None:
    with patch.dict(os.environ, {}, clear=True):
        assert timeout_message() == (
            "LLM call timed out after 120s, try again, or raise LLM_TIMEOUT"
        )
    with patch.dict(os.environ, {"LLM_TIMEOUT": "45"}, clear=True):
        assert "after 45s" in timeout_message()


def test_run_returns_504_on_timeout() -> None:
    crew = _slow_crew(APITimeoutError("Request timed out."))
    module = MagicMock()
    module.build_crew = MagicMock(return_value=crew)
    with patch.dict(os.environ, {"LLM_API_KEY": "k", "LLM_TIMEOUT": "120"}):
        with (
            patch("importlib.util.spec_from_file_location"),
            patch("importlib.util.module_from_spec", return_value=module),
        ):
            response = client.post(
                "/run",
                json={
                    "recipe": "faq-bot",
                    "inputs": {"question": "Hi", "customer_name": "T"},
                    "no_cache": True,
                },
            )
    assert response.status_code == 504
    assert "raise LLM_TIMEOUT" in response.json()["detail"]


def test_stream_pushes_timeout_error_event() -> None:
    crew = _slow_crew(APITimeoutError("Request timed out."))
    module = MagicMock()
    module.build_crew = MagicMock(return_value=crew)
    events = []
    with patch.dict(os.environ, {"LLM_API_KEY": "k", "LLM_TIMEOUT": "120"}):
        with (
            patch("importlib.util.spec_from_file_location"),
            patch("importlib.util.module_from_spec", return_value=module),
        ):
            execute_recipe_stream(
                "faq-bot",
                {"question": "Hi", "customer_name": "T"},
                events.append,
                threading.Event(),
            )
    errors = [e for e in events if e and e.get("type") == "error"]
    assert errors, events
    assert "timed out after 120s" in errors[-1]["error"]
    assert "raise LLM_TIMEOUT" in errors[-1]["error"]
