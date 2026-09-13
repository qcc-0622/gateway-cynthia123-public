"""Cache breakpoint pipeline：BP1-BP4 cache_control 放置、TTL、4 槽上限。

2026-09-02（REFACTOR_ROADMAP P2.1 第 1 批）：自 hooks.py 绞杀式搬迁而来，
函数体零改动；hooks.py 保留 import 转发兼容旧引用。本模块不得反向 import
hooks（会成环）。

职责：网关独占地管理 Anthropic 的 4 个 cache_control 槽位——
- BP1  system 末块断点（TTL 跟随上游配置）
- BP3  冻结窗口边界断点（稳定前缀，TTL 跟随上游）
- BP4  最后一条 assistant 断点（滑动，固定 5m）
- tools 列表单断点（末位覆盖）
- 客户端自带断点全深度剥离 + 4 槽超限裁剪
"""

import logging

logger = logging.getLogger("gateway.pipeline.cache")

_CACHE_5M = {"type": "ephemeral", "ttl": "5m"}
_CACHE_1H = {"type": "ephemeral", "ttl": "1h"}
_EPHEMERAL = _CACHE_5M
_MAX_CACHE_BREAKPOINTS = 4


def _cache_control_for_ttl(ttl: str | None) -> dict | None:
    ttl = (ttl or "5m").strip().lower()
    if ttl == "off":
        return None
    if ttl == "1h":
        return dict(_CACHE_1H)
    return dict(_CACHE_5M)


def _count_cache_control(obj) -> int:
    """Count cache_control fields recursively."""
    if isinstance(obj, dict):
        return (1 if "cache_control" in obj else 0) + sum(
            _count_cache_control(v) for k, v in obj.items() if k != "cache_control"
        )
    if isinstance(obj, list):
        return sum(_count_cache_control(x) for x in obj)
    return 0


def _has_cache_control(obj) -> bool:
    return _count_cache_control(obj) > 0


def _cache_slots_left(body: dict, messages: list | None = None) -> int:
    count = _count_cache_control(body.get("tools"))
    count += _count_cache_control(body.get("system"))
    count += _count_cache_control(messages if messages is not None else body.get("messages"))
    return max(0, _MAX_CACHE_BREAKPOINTS - count)


def _add_bp1_system(body: dict, cache_control: dict | None = None) -> dict:
    if cache_control is None:
        return body
    system = body.get("system")
    if not system:
        return body
    if _cache_slots_left(body) <= 0 or _has_cache_control(system):
        return body
    body = dict(body)
    if isinstance(system, str):
        body["system"] = [{"type": "text", "text": system, "cache_control": cache_control}]
    elif isinstance(system, list) and system:
        system = list(system)
        last = dict(system[-1])
        last["cache_control"] = cache_control
        system[-1] = last
        body["system"] = system
    return body


# ──────────────────────────────────────────────
# BP3: freeze window
# ──────────────────────────────────────────────

def _count_pairs(messages: list) -> int:
    """Count complete assistant turns (excluding the trailing user turn if present)."""
    counting = messages[:-1] if messages and messages[-1].get("role") == "user" else messages
    return sum(1 for m in counting if m.get("role") == "assistant")


def _add_cache_control_at(messages: list, idx: int, cache_control: dict | None = None) -> list:
    """Add cache_control: ephemeral to the last content block of messages[idx]."""
    if idx < 0 or idx >= len(messages):
        return messages
    messages = list(messages)
    msg = dict(messages[idx])
    if _has_cache_control(msg):
        return messages
    cache_control = cache_control or _CACHE_5M
    content = msg.get("content", "")
    if isinstance(content, str) and content:
        msg["content"] = [{"type": "text", "text": content, "cache_control": cache_control}]
    elif isinstance(content, list) and content:
        content = list(content)
        last = dict(content[-1])
        last["cache_control"] = cache_control
        content[-1] = last
        msg["content"] = content
    messages[idx] = msg
    return messages


def _summary_prefix_len(messages: list) -> int:
    """Gateway BP2 summaries are injected as user summary + assistant ack."""
    if len(messages) < 2:
        return 0
    first = messages[0]
    second = messages[1]
    if first.get("role") != "user" or second.get("role") != "assistant":
        return 0
    content = first.get("content", "")
    text = content if isinstance(content, str) else " ".join(
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    )
    return 2 if text.lstrip().startswith("[以下是我们之前对话的摘要") else 0


def _strategy_rounds() -> tuple[int, int]:
    """Return (frozen_rounds, live_rounds) for the fixed cache strategy."""
    from gateway.settings import get as _get
    frozen = int(_get("bp3_frozen_rounds", 8))
    live = int(_get("bp3_live_rounds", 24))
    return max(1, frozen), max(1, live)


def _add_bp3_freeze(
    messages: list,
    body: dict | None = None,
    cache_control: dict | None = None,
) -> list:
    """
    BP3: add cache_control at the freeze boundary.
    The frozen window begins after the optional BP2 summary pair and stays fixed
    until BP2 is rewritten after enough live turns accumulate.
    """
    if cache_control is None:
        return messages
    if body is not None and _cache_slots_left(body, messages) <= 0:
        return messages
    frozen_rounds, _ = _strategy_rounds()
    prefix_len = _summary_prefix_len(messages)
    retained_pairs = _count_pairs(messages[prefix_len:])
    if retained_pairs < frozen_rounds:
        return messages
    bp3_end_idx = prefix_len + frozen_rounds * 2 - 1
    logger.debug("BP3: end_idx=%d retained_pairs=%d frozen_rounds=%d",
                 bp3_end_idx, retained_pairs, frozen_rounds)
    return _add_cache_control_at(messages, bp3_end_idx, cache_control=cache_control)


# ──────────────────────────────────────────────
# BP4: sliding breakpoint on last assistant message
# ──────────────────────────────────────────────

def _add_bp4_last_assistant(messages: list, body: dict | None = None) -> list:
    """
    BP4: cache_control on last assistant message.
    Put this BEFORE the current user message — the timestamp appended later
    must NOT be inside the cached prefix.
    """
    if body is not None and _cache_slots_left(body, messages) <= 0:
        return messages
    messages = list(messages)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") != "assistant":
            continue
        msg = dict(messages[i])
        if _has_cache_control(msg):
            break
        content = msg.get("content", "")
        if isinstance(content, str) and content:
            msg["content"] = [{"type": "text", "text": content, "cache_control": _CACHE_5M}]
        elif isinstance(content, list) and content:
            content = list(content)
            last = dict(content[-1])
            last["cache_control"] = _CACHE_5M
            content[-1] = last
            msg["content"] = content
        messages[i] = msg
        break
    return messages


def _deep_strip_cache_control(obj):
    """Recursively remove every cache_control field from a dict/list structure.
    Returns a new object (does not mutate)."""
    if isinstance(obj, dict):
        return {
            k: _deep_strip_cache_control(v)
            for k, v in obj.items()
            if k != "cache_control"
        }
    if isinstance(obj, list):
        return [_deep_strip_cache_control(x) for x in obj]
    return obj


def _strip_client_cache_control(messages: list) -> list:
    """Remove any cache_control the client already set — gateway owns all 4 slots.
    Strips at every depth (tool_result.content blocks, etc.)."""
    return _deep_strip_cache_control(messages)


def _strip_client_cache_control_body(body: dict) -> dict:
    """Same as above but for the whole body — also catches system/tools cache_control."""
    body = dict(body)
    for key in ("system", "tools"):
        if key in body:
            body[key] = _deep_strip_cache_control(body[key])
    return body


def _normalize_tool_cache_control(body: dict, cache_control: dict | None = None) -> dict:
    """
    Tool definitions are a single stable prefix. Strip client choices and put one
    breakpoint on the last tool; that covers the whole tools list.
    """
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return body
    if cache_control is None:
        body = dict(body)
        body["tools"] = _deep_strip_cache_control(tools)
        return body

    normalized = []
    for tool in tools:
        if isinstance(tool, dict):
            normalized.append({k: v for k, v in tool.items() if k != "cache_control"})
        else:
            normalized.append(tool)

    for i in range(len(normalized) - 1, -1, -1):
        if isinstance(normalized[i], dict):
            normalized[i] = {**normalized[i], "cache_control": cache_control}
            break

    body = dict(body)
    body["tools"] = normalized
    return body


def _ensure_tool_cache_control(body: dict, cache_control: dict | None = None) -> dict:
    """If tools are present and no tool breakpoint exists, add one on the last tool."""
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return body
    if cache_control is None:
        return body
    if _has_cache_control(tools) or _cache_slots_left(body) <= 0:
        return body
    normalized = list(tools)
    for i in range(len(normalized) - 1, -1, -1):
        if isinstance(normalized[i], dict):
            normalized[i] = {**normalized[i], "cache_control": cache_control}
            body = dict(body)
            body["tools"] = normalized
            return body
    return body


def _limit_cache_controls(obj, remaining: list[int]):
    """Keep cache_control fields while slots remain; strip the overflow."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k == "cache_control":
                if remaining[0] > 0:
                    result[k] = v
                    remaining[0] -= 1
                continue
            result[k] = _limit_cache_controls(v, remaining)
        return result
    if isinstance(obj, list):
        return [_limit_cache_controls(x, remaining) for x in obj]
    return obj


def _enforce_cache_control_limit(body: dict) -> dict:
    """
    Anthropic allows at most four cache breakpoints across tools/system/messages.
    Preserve client-provided breakpoints first in canonical prefix order, then any
    gateway-added BP slots that still fit.
    """
    body = dict(body)
    remaining = [_MAX_CACHE_BREAKPOINTS]
    for key in ("tools", "system", "messages"):
        if key in body:
            body[key] = _limit_cache_controls(body[key], remaining)
    return body
