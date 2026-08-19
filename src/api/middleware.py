"""Cross-cutting HTTP concerns: CORS, API keys, rate limiting, timing.

The rate limiter and key check are intentionally simple and in-process. This
platform is meant to be deployable by a ministry IT team on one server, not to
require an API gateway; anything heavier is opt-in via a reverse proxy.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Optional

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import get_settings
from src.core.logging import get_logger

log = get_logger("api.middleware")

#: Paths that never require an API key, so a monitoring probe and the docs work.
PUBLIC_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"}


class TimingMiddleware(BaseHTTPMiddleware):
    """Attach a server-timing header and warn on slow responses.

    Acceptance criterion #10 requires predictions in under 2 seconds; this is
    what tells us when that stops being true.
    """

    SLOW_SECONDS = 2.0

    async def dispatch(self, request: Request, call_next: Callable):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        response.headers["X-Response-Time-Ms"] = f"{elapsed * 1000:.1f}"
        if elapsed > self.SLOW_SECONDS:
            log.warning(
                "slow response: %s %s took %.2fs (target < %.1fs)",
                request.method, request.url.path, elapsed, self.SLOW_SECONDS,
            )
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require `X-API-Key` when one is configured; open otherwise.

    Defaulting to open is deliberate: the platform must be usable by an agency
    that has not set up secrets management yet, and every deployment guide entry
    says to set `API_KEY` before exposing it beyond a private network.
    """

    async def dispatch(self, request: Request, call_next: Callable):
        settings = get_settings()
        if not settings.api_key or request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        provided = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if provided != settings.api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing or invalid X-API-Key header", "code": "unauthorized"},
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window per-client rate limit."""

    def __init__(self, app, requests_per_minute: Optional[int] = None) -> None:
        super().__init__(app)
        self.limit = requests_per_minute or get_settings().rate_limit_per_minute
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: Callable):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        client = request.headers.get("X-API-Key") or (
            request.client.host if request.client else "unknown"
        )
        now = time.time()
        window = self._hits[client]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.limit:
            retry_after = int(60 - (now - window[0])) + 1
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Rate limit of {self.limit} requests/minute exceeded",
                    "code": "rate_limited",
                },
                headers={"Retry-After": str(retry_after)},
            )
        window.append(now)
        return await call_next(request)


def install_middleware(app) -> None:
    """Attach every middleware in the right order."""
    from fastapi.middleware.cors import CORSMiddleware

    settings = get_settings()
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Response-Time-Ms"],
    )
