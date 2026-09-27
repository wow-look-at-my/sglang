"""Event-driven detection of an HTTP client going away mid-request."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from starlette.requests import ClientDisconnect, Request

logger = logging.getLogger(__name__)


async def _wait_for_disconnect(request: Request) -> None:
    # Drain (or reuse the cached) request body first, so body chunks of a large
    # upload are never consumed here and never mistaken for a disconnect.
    try:
        await request.body()
    except ClientDisconnect:
        return
    except RuntimeError:
        # Another reader consumed the stream; the endpoint runs after it ends.
        pass
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


def watch_client_disconnect(
    request: object, on_disconnect: Callable[[], None]
) -> Optional[asyncio.Task]:
    """Call ``on_disconnect`` as soon as the ASGI server reports the client gone.

    ``http.disconnect`` also arrives once the response is complete, so the
    caller cancels the returned task when the request finishes. Returns None
    for anything that is not a Starlette request (Engine calls, test doubles).
    """
    if not isinstance(request, Request):
        return None

    async def _run() -> None:
        await _wait_for_disconnect(request)
        try:
            on_disconnect()
        except Exception:
            logger.exception("Failed to handle a client disconnect")

    return asyncio.create_task(_run())
