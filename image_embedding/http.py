"""Shared HTTP session: identifying User-Agent, rate limiting, retry with backoff."""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Optional

import requests

logger = logging.getLogger("wikivoyage.images.http")

RETRY_STATUSES = {429, 500, 502, 503, 504}
NO_RETRY_STATUSES = {400, 401, 403, 404, 410}
BACKOFF_SECONDS = (1, 2, 4, 8, 16, 32)


class HttpError(Exception):
    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def reason(self) -> str:
        return f"http_{self.status}" if self.status else "network_error"


class RateLimiter:
    """Token bucket shared by threads; ``rate`` requests per second (<=0 disables)."""

    def __init__(self, rate: float, burst: Optional[float] = None) -> None:
        self.rate = rate
        self.capacity = burst if burst is not None else max(1.0, rate)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.rate <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait = (1 - self._tokens) / self.rate
            time.sleep(wait)


def make_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = user_agent
    return s


def _retry_after(resp: requests.Response) -> Optional[float]:
    v = resp.headers.get("Retry-After")
    if not v:
        return None
    try:
        return max(0.0, float(v))
    except ValueError:
        return None


def _sleep_backoff(attempt: int, retry_after: Optional[float], sleep=time.sleep) -> None:
    if retry_after is not None:
        delay = retry_after
    else:
        base = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
        delay = base * random.uniform(0.8, 1.2)
    sleep(delay)


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    limiter: Optional[RateLimiter] = None,
    max_attempts: int = 6,
    timeout: tuple[float, float] = (10, 60),
    sleep=time.sleep,
    **kwargs,
) -> requests.Response:
    """Send a request, retrying 429/5xx/network errors with exponential backoff.

    429/503 honour ``Retry-After``. 4xx in NO_RETRY_STATUSES raise immediately.
    The caller owns the returned response (close it if ``stream=True``).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_attempts):
        if limiter is not None:
            limiter.acquire()
        try:
            resp = session.request(method, url, timeout=timeout, **kwargs)
        except requests.RequestException as e:
            last_exc = e
            logger.warning("%s %s failed (attempt %d/%d): %s", method, url, attempt + 1, max_attempts, e)
            if attempt + 1 < max_attempts:
                _sleep_backoff(attempt, None, sleep)
            continue

        if resp.status_code < 400:
            return resp
        if resp.status_code in RETRY_STATUSES and attempt + 1 < max_attempts:
            ra = _retry_after(resp)
            logger.warning(
                "%s %s → HTTP %d (attempt %d/%d, retry-after=%s)",
                method, url, resp.status_code, attempt + 1, max_attempts, ra,
            )
            resp.close()
            _sleep_backoff(attempt, ra, sleep)
            continue
        status = resp.status_code
        resp.close()
        raise HttpError(f"{method} {url} → HTTP {status}", status)

    raise HttpError(f"{method} {url} failed after {max_attempts} attempts: {last_exc}", None)
