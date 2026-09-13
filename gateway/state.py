"""Shared in-process state: last request info for keep-alive."""

import time

# Updated by proxy functions after each successful Anthropic request
last_request: dict = {
    "time": 0.0,          # monotonic timestamp of last request
    "system": None,       # system message sent (after preprocessing, with cache_control)
    "model": "",          # model name used
}


def touch(system=None, model: str = ""):
    last_request["time"] = time.monotonic()
    if system is not None:
        last_request["system"] = system
    if model:
        last_request["model"] = model


def seconds_since_last() -> float:
    if last_request["time"] == 0:
        return float("inf")
    return time.monotonic() - last_request["time"]
