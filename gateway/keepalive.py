"""Keep-alive: silently ping the upstream to refresh cache TTL during active hours."""

import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from gateway.config import (
    KEEPALIVE_ENABLED, KEEPALIVE_START_HOUR, KEEPALIVE_END_HOUR,
    KEEPALIVE_INTERVAL, TIMEZONE,
)
from gateway import state
from gateway.http_client import get_client
from gateway.upstream import get_active_upstream

logger = logging.getLogger("gateway.keepalive")

_CHECK_INTERVAL = 60  # check every 60 seconds


def _in_active_hours() -> bool:
    hour = datetime.now(ZoneInfo(TIMEZONE)).hour
    return KEEPALIVE_START_HOUR <= hour < KEEPALIVE_END_HOUR


async def _ping(upstream, system, model: str):
    """Send a minimal request to hit the cache without generating meaningful output."""
    headers = {
        "x-api-key": upstream.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model or "claude-haiku-4-5-20251001",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "k"}],
    }
    if system:
        payload["system"] = system

    try:
        client = get_client()
        resp = await client.post(
            f"{upstream.base_url.rstrip('/')}/v1/messages",
            headers=headers,
            json=payload,
            timeout=30,
        )
        data = resp.json()
        usage = data.get("usage", {})
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_write = usage.get("cache_creation_input_tokens", 0)
        logger.info(
            "Keep-alive ping OK | cache_read=%d cache_write=%d",
            cache_read, cache_write,
        )
        state.touch()  # update last_request time
    except Exception as e:
        logger.warning("Keep-alive ping failed: %s", e)


async def keepalive_loop():
    if not KEEPALIVE_ENABLED:
        logger.info("Keep-alive disabled")
        return

    logger.info(
        "Keep-alive started: active %02d:00–%02d:00, interval %d min",
        KEEPALIVE_START_HOUR, KEEPALIVE_END_HOUR, KEEPALIVE_INTERVAL,
    )
    while True:
        await asyncio.sleep(_CHECK_INTERVAL)
        try:
            if not _in_active_hours():
                continue

            idle_seconds = state.seconds_since_last()
            threshold = KEEPALIVE_INTERVAL * 60

            if idle_seconds < threshold:
                continue

            upstream = get_active_upstream()
            if not upstream or upstream.api_format != "anthropic":
                continue  # only Anthropic format benefits from prompt cache

            system = state.last_request.get("system")
            model  = state.last_request.get("model", "")

            if not system and not model:
                continue  # no prior request to base the ping on

            logger.info("Idle %.0fs > threshold %ds, sending keep-alive ping", idle_seconds, threshold)
            await _ping(upstream, system, model)

        except Exception:
            logger.exception("Keep-alive loop error")
