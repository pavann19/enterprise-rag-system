from unittest.mock import patch

import pytest

from service.rate_limiter import RateLimiter


def test_rejects_non_positive_max_requests():
    with pytest.raises(ValueError):
        RateLimiter(max_requests=0, window_seconds=60)


def test_rejects_non_positive_window():
    with pytest.raises(ValueError):
        RateLimiter(max_requests=5, window_seconds=0)


def test_allows_requests_under_the_limit():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True


def test_rejects_requests_over_the_limit():
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    limiter.allow("client-a")
    limiter.allow("client-a")
    assert limiter.allow("client-a") is False


def test_rejected_request_is_not_recorded():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    limiter.allow("client-a")
    assert limiter.allow("client-a") is False
    # if the rejected call had been recorded, a third call would still be
    # rejected even after the limit resets — verified via the window test below
    assert len(limiter._requests["client-a"]) == 1


def test_different_keys_have_independent_limits():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-b") is True  # unaffected by client-a's usage
    assert limiter.allow("client-a") is False
    assert limiter.allow("client-b") is False


def test_old_requests_age_out_of_the_window():
    limiter = RateLimiter(max_requests=1, window_seconds=60)

    with patch("time.monotonic", return_value=1000.0):
        assert limiter.allow("client-a") is True
        assert limiter.allow("client-a") is False

    with patch("time.monotonic", return_value=1000.0 + 61):
        assert limiter.allow("client-a") is True  # window has passed


def test_retry_after_seconds_zero_when_no_history():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    assert limiter.retry_after_seconds("never-seen") == 0.0


def test_retry_after_seconds_counts_down_within_window():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    with patch("time.monotonic", return_value=1000.0):
        limiter.allow("client-a")
    with patch("time.monotonic", return_value=1000.0 + 20):
        assert limiter.retry_after_seconds("client-a") == pytest.approx(40.0)


def test_retry_after_seconds_zero_once_window_fully_elapsed():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    with patch("time.monotonic", return_value=1000.0):
        limiter.allow("client-a")
    with patch("time.monotonic", return_value=1000.0 + 90):
        assert limiter.retry_after_seconds("client-a") == 0.0
