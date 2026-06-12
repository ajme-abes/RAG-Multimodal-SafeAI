"""
Tests for retry_with_backoff (utils.py)
All tests are pure unit tests — no real API calls are made.
APIError is faked via a real subclass so isinstance() checks inside utils.py pass.
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../app")))

from google.genai.errors import APIError
from utils import retry_with_backoff


class FakeAPIError(APIError):
    """
    A real subclass of APIError so isinstance(err, APIError) returns True.
    APIError.__init__ requires a specific signature we bypass here.
    """
    def __init__(self, code: int):
        # Skip the parent __init__ to avoid needing a full Response object
        self.code = code
        self.message = f"Fake API error {code}"

    def __str__(self):
        return self.message


def _make_api_error(code: int) -> FakeAPIError:
    """Return a real APIError subclass instance with the given HTTP status code."""
    return FakeAPIError(code)


class TestRetryWithBackoff:

    # ── Success path ──────────────────────────────────────────────────────────

    def test_returns_value_on_first_success(self):
        func = MagicMock(return_value="ok")
        result = retry_with_backoff(func, max_retries=3, initial_delay=0)
        assert result == "ok"
        func.assert_called_once()

    # ── Retry on retriable codes ───────────────────────────────────────────────

    @patch("utils.time.sleep")
    def test_retries_on_429_then_succeeds(self, mock_sleep):
        err = _make_api_error(429)
        func = MagicMock(side_effect=[err, err, "recovered"])
        result = retry_with_backoff(func, max_retries=5, initial_delay=0, backoff_factor=2)
        assert result == "recovered"
        assert func.call_count == 3

    @patch("utils.time.sleep")
    def test_retries_on_503_then_succeeds(self, mock_sleep):
        err = _make_api_error(503)
        func = MagicMock(side_effect=[err, "ok"])
        result = retry_with_backoff(func, max_retries=3, initial_delay=0)
        assert result == "ok"
        assert func.call_count == 2

    # ── Exhausted retries ─────────────────────────────────────────────────────

    @patch("utils.time.sleep")
    def test_raises_after_max_retries_exhausted(self, mock_sleep):
        err = _make_api_error(429)
        func = MagicMock(side_effect=err)
        with pytest.raises(FakeAPIError):
            retry_with_backoff(func, max_retries=3, initial_delay=0)
        assert func.call_count == 3

    # ── Non-retriable codes raise immediately ─────────────────────────────────

    @patch("utils.time.sleep")
    def test_raises_immediately_on_non_retriable_code(self, mock_sleep):
        err = _make_api_error(400)  # Bad request — not retriable
        func = MagicMock(side_effect=err)
        with pytest.raises(FakeAPIError):
            retry_with_backoff(func, max_retries=5, initial_delay=0)
        # Should not retry — called exactly once
        assert func.call_count == 1

    @patch("utils.time.sleep")
    def test_raises_immediately_on_401(self, mock_sleep):
        err = _make_api_error(401)
        func = MagicMock(side_effect=err)
        with pytest.raises(FakeAPIError):
            retry_with_backoff(func, max_retries=5, initial_delay=0)
        assert func.call_count == 1

    # ── Non-API exceptions propagate immediately ──────────────────────────────

    def test_non_api_exception_propagates_immediately(self):
        func = MagicMock(side_effect=ValueError("bad input"))
        with pytest.raises(ValueError, match="bad input"):
            retry_with_backoff(func, max_retries=5, initial_delay=0)
        func.assert_called_once()

    def test_runtime_error_propagates_immediately(self):
        func = MagicMock(side_effect=RuntimeError("crash"))
        with pytest.raises(RuntimeError):
            retry_with_backoff(func, max_retries=5, initial_delay=0)
        func.assert_called_once()

    # ── Sleep is actually called between retries ───────────────────────────────

    @patch("utils.time.sleep")
    def test_sleep_is_called_between_retries(self, mock_sleep):
        err = _make_api_error(429)
        func = MagicMock(side_effect=[err, err, "ok"])
        retry_with_backoff(func, max_retries=5, initial_delay=1, backoff_factor=2)
        # Should sleep twice (after attempt 1 and attempt 2)
        assert mock_sleep.call_count == 2

    # ── Backoff delay grows with each retry ───────────────────────────────────

    @patch("utils.random.uniform", return_value=0.0)   # Remove jitter noise
    @patch("utils.time.sleep")
    def test_delay_doubles_with_backoff_factor(self, mock_sleep, mock_random):
        err = _make_api_error(429)
        func = MagicMock(side_effect=[err, err, err, "ok"])
        retry_with_backoff(func, max_retries=5, initial_delay=1, backoff_factor=2)
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        # Delays should be 1, 2, 4 (doubling each time, jitter zeroed out)
        assert sleep_calls[0] == pytest.approx(1.0, abs=0.01)
        assert sleep_calls[1] == pytest.approx(2.0, abs=0.01)
        assert sleep_calls[2] == pytest.approx(4.0, abs=0.01)
