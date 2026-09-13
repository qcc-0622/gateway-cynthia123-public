"""Core proxy logic: forward requests to upstream, handle streaming, accumulate responses."""

import asyncio
import json
import os
import time
import uuid
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator

import httpx

from gateway.upstream import Upstream
from gateway import state as _state
from gateway.http_client import get_client
from gateway.converter import (
    openai_to_anthropic, anthropic_to_openai,
    anthropic_response_to_openai, openai_response_to_anthropic,
    anthropic_sse_to_openai_sse, openai_sse_to_anthropic_sse,
    strip_gateway_private_fields,
)
from gateway.config import UPSTREAM_TIMEOUT

logger = logging.getLogger("gateway.proxy")

# Raw upstream dump (debug). Set GATEWAY_RAW_DUMP=1 to capture forward_body +
# unprocessed SSE lines into data/raw_dumps/. Useful when diagnosing thinking
# leak issues where we need to see exactly what the upstream returned before
# our rewriter touched it.
_RAW_DUMP_DIR = Path("data/raw_dumps") if os.getenv("GATEWAY_RAW_DUMP") == "1" else None


def _cache_ttl_from_body(body: dict) -> str:
    ttl = str(body.get("_gateway_cache_ttl") or "1h").strip().lower()
    return ttl if ttl in {"1h", "5m", "off"} else "1h"


def _downgrade_cache_ttl(obj, ttl: str = "5m"):
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k == "cache_control" and isinstance(v, dict):
                result[k] = {**v, "ttl": ttl}
            else:
                result[k] = _downgrade_cache_ttl(v, ttl=ttl)
        return result
    if isinstance(obj, list):
        return [_downgrade_cache_ttl(x, ttl=ttl) for x in obj]
    return obj


def _is_extended_cache_error_text(status_code: int, text: str) -> bool:
    if status_code != 400:
        return False
    text = (text or "").lower()
    cache_signal = (
        "cache_control" in text
        or "ttl" in text
        or "extended-cache-ttl" in text
        or "extended cache" in text
        or "anthropic-beta" in text
    )
    return cache_signal and ("1h" in text or "ttl" in text or "extended" in text)


def _is_extended_cache_error(resp: httpx.Response) -> bool:
    return _is_extended_cache_error_text(resp.status_code, resp.text or "")


def _open_raw_dump(tag: str, forward_body: dict) -> "object | None":
    if _RAW_DUMP_DIR is None:
        return None
    try:
        _RAW_DUMP_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = _RAW_DUMP_DIR / f"{ts}_{tag}_{uuid.uuid4().hex[:6]}.txt"
        fp = open(path, "w", encoding="utf-8")
        fp.write("=== REQUEST BODY (sent to upstream, post-hooks) ===\n")
        fp.write(json.dumps(forward_body, ensure_ascii=False, indent=2))
        fp.write("\n\n=== UPSTREAM RAW SSE (verbatim, pre-rewriter) ===\n")
        fp.flush()
        logger.info(f"raw_dump: writing to {path}")
        return fp
    except Exception as e:
        logger.warning(f"raw_dump: failed to open file: {e}")
        return None


class _ThinkingRewriter:
    """
    Rewrite inline thinking markers into proper Anthropic thinking content blocks
    so RikkaHub renders them as 深度思考.

    Handles two modes:
    - Normal mode: detects <prior_reasoning>, <think>, or <thinking> open tags in text
    - force_think_mode: first text block is assumed to be thinking content (the open
      tag was consumed by the assistant prefill injected by force_think)

    State machine:
      scan      – haven't seen a text content_block_start yet
      buf       – buffering text to detect an open tag (normal mode only)
      thinking  – inside thinking block, emitting thinking_delta events
      text      – after close tag, emitting text_delta on shifted index
      pass      – no thinking tag detected, pure pass-through (normal mode only)
    """

    _OPEN_TAGS = ["<prior_reasoning>", "<think>", "<thinking>", "<thought>"]
    _CLOSE_TAGS = ["</prior_reasoning>", "</think>", "</thinking>", "</thought>"]

    def __init__(self, force_think_mode: bool = False):
        self._force_think_mode = force_think_mode
        self._state = "scan"
        self._ti = None          # index of first text block
        self._pending = None     # buffered content_block_start event
        self._buf = ""           # accumulated text (during buf/thinking states)
        self._shifted = False    # True once we inserted an extra block

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sse(ev: dict) -> str:
        event_type = ev.get("type", "message")
        return f"event: {event_type}\ndata: {json.dumps(ev, ensure_ascii=False)}\n"

    def _shifted_index(self, idx: int) -> int:
        if self._shifted and idx > self._ti:
            return idx + 1
        return idx

    def _thinking_delta(self, text: str) -> str:
        return self._sse({"type": "content_block_delta", "index": self._ti,
                          "delta": {"type": "thinking_delta", "thinking": text}})

    def _text_delta(self, text: str) -> str:
        return self._sse({"type": "content_block_delta", "index": self._ti + 1,
                          "delta": {"type": "text_delta", "text": text}})

    def _close_thinking_open_text(self) -> list[str]:
        self._state = "text"
        self._shifted = True
        return [
            self._sse({"type": "content_block_stop", "index": self._ti}),
            self._sse({"type": "content_block_start", "index": self._ti + 1,
                       "content_block": {"type": "text", "text": ""}}),
        ]

    def _find_close_tag(self, buf: str):
        """Find the earliest close tag in buf. Returns (pos, tag_len) or (-1, 0)."""
        best_pos, best_len = -1, 0
        for tag in self._CLOSE_TAGS:
            pos = buf.find(tag)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos, best_len = pos, len(tag)
        return best_pos, best_len

    # ── thinking-state text processor ────────────────────────────────────────

    def _process_thinking(self, text: str) -> list[str]:
        """Emit thinking_delta lines, detect any close tag and transition."""
        out = []
        self._buf += text
        while True:
            ci, close_len = self._find_close_tag(self._buf)
            if ci == -1:
                # Check for partial close tag at end of buffer
                lt = self._buf.rfind('<')
                if lt != -1 and any(tag.startswith(self._buf[lt:]) for tag in self._CLOSE_TAGS):
                    if self._buf[:lt]:
                        out.append(self._thinking_delta(self._buf[:lt]))
                    self._buf = self._buf[lt:]
                else:
                    if self._buf:
                        out.append(self._thinking_delta(self._buf))
                    self._buf = ""
                break
            # Found close tag
            if self._buf[:ci]:
                out.append(self._thinking_delta(self._buf[:ci]))
            out += self._close_thinking_open_text()
            remaining = self._buf[ci + close_len:].lstrip('\n')
            self._buf = ""
            if remaining:
                out.append(self._text_delta(remaining))
            break
        return out

    # ── main entry point ─────────────────────────────────────────────────────

    def process(self, event: dict) -> list[str]:
        t = event.get("type", "")

        # content_block_start
        if t == "content_block_start":
            cb = event.get("content_block", {})
            if cb.get("type") == "text" and self._state == "scan":
                self._ti = event.get("index", 0)
                if self._force_think_mode:
                    # Open tag was consumed by prefill – treat block as thinking immediately
                    self._state = "thinking"
                    return [self._sse({"type": "content_block_start", "index": self._ti,
                                       "content_block": {"type": "thinking", "thinking": ""}})]
                self._state = "buf"
                self._pending = event
                return []
            idx = self._shifted_index(event.get("index", 0))
            if idx != event.get("index", 0):
                event = dict(event); event["index"] = idx
            return [self._sse(event)]

        # content_block_delta
        if t == "content_block_delta":
            idx = event.get("index", 0)
            delta = event.get("delta", {})
            dtype = delta.get("type", "")

            if idx == self._ti and dtype == "text_delta":
                text = delta.get("text", "")

                if self._state == "buf":
                    self._buf += text
                    tb = self._buf

                    # Check if any open tag is confirmed
                    confirmed_tag = next((tag for tag in self._OPEN_TAGS if tb.startswith(tag)), None)
                    if confirmed_tag:
                        self._state = "thinking"
                        # ⚠️ rest 必须在清空 buf 之后再交给 _process_thinking——
                        # 它内部会先 self._buf += text，若把带残留的 buf 原样传入，
                        # open tag 跨 chunk 时标签后的首段 thinking 文字会翻倍。
                        rest = tb[len(confirmed_tag):]
                        self._buf = ""
                        out = [self._sse({"type": "content_block_start", "index": self._ti,
                                          "content_block": {"type": "thinking", "thinking": ""}})]
                        if rest:
                            out += self._process_thinking(rest)
                        return out

                    # Still a possible prefix of some open tag – keep buffering
                    if any(tag.startswith(tb) for tag in self._OPEN_TAGS):
                        return []

                    # Not any open tag – flush as normal text
                    self._state = "pass"
                    out = [self._sse(self._pending)]
                    self._pending = None
                    e = dict(event); e["delta"] = {"type": "text_delta", "text": tb}
                    self._buf = ""
                    return out + [self._sse(e)]

                if self._state == "thinking":
                    return self._process_thinking(text)

                if self._state == "text":
                    e = dict(event); e["index"] = self._ti + 1
                    return [self._sse(e)]

            # Other delta or different index
            new_idx = self._shifted_index(idx)
            if new_idx != idx:
                event = dict(event); event["index"] = new_idx
            return [self._sse(event)]

        # content_block_stop
        if t == "content_block_stop":
            idx = event.get("index", 0)
            if idx == self._ti:
                if self._state == "buf":
                    out = [self._sse(self._pending), self._sse(event)]
                    self._pending = None; self._state = "pass"
                    return out
                if self._state == "thinking":
                    # Stream ended inside thinking block – close it
                    return [self._sse({"type": "content_block_stop", "index": self._ti})]
                if self._state == "text":
                    e = dict(event); e["index"] = self._ti + 1
                    return [self._sse(e)]
            new_idx = self._shifted_index(idx)
            if new_idx != idx:
                event = dict(event); event["index"] = new_idx
            return [self._sse(event)]

        return [self._sse(event)]


def _split_orphan_close_think(content: list[dict]) -> list[dict]:
    """
    Normalize thinking-tag layouts that arrive embedded in a text block.

    relay-a.example.com Kiro thinking models do one of two things depending on the day:
      (1) swallow the open tag, leave only "[thinking]</think>\\n[answer]"
      (2) keep a full pair, "<thought>[thinking]</thought>\\n[answer]"
    Either way, a downstream client like RikkaHub renders it as polluted
    plaintext instead of a folded thinking block.

    Handles both layouts by scanning the first text block in `content`:
      - find earliest close tag
      - find earliest open tag (if any) that precedes it
      - emit  [optional prefix text] + [thinking block] + [trailing text]

    No-op if there is no close tag or layout looks fine.
    """
    if not content:
        return content
    open_tags = _ThinkingRewriter._OPEN_TAGS
    close_tags = _ThinkingRewriter._CLOSE_TAGS
    new_blocks = []
    splitted = False
    for b in content:
        if splitted or b.get("type") != "text":
            new_blocks.append(b)
            continue
        text = b.get("text", "")
        # Find earliest close tag
        close_pos, close_len = -1, 0
        for tag in close_tags:
            pos = text.find(tag)
            if pos != -1 and (close_pos == -1 or pos < close_pos):
                close_pos, close_len = pos, len(tag)
        if close_pos == -1:
            new_blocks.append(b)
            continue
        # Find earliest open tag that precedes the close tag (full pair case)
        open_pos, open_len = -1, 0
        for tag in open_tags:
            pos = text.find(tag)
            if pos != -1 and pos < close_pos and (open_pos == -1 or pos < open_pos):
                open_pos, open_len = pos, len(tag)
        # Compute the three slices
        if open_pos >= 0:
            prefix_text = text[:open_pos]
            thinking_text = text[open_pos + open_len:close_pos]
        else:
            prefix_text = ""
            thinking_text = text[:close_pos]
        trailing_text = text[close_pos + close_len:]
        # Strip enclosing whitespace/newlines from thinking + trailing for cleanliness
        thinking_text = thinking_text.strip("\n")
        trailing_text = trailing_text.lstrip("\n")
        if prefix_text:
            new_blocks.append({"type": "text", "text": prefix_text})
        if thinking_text:
            new_blocks.append({"type": "thinking", "thinking": thinking_text})
        if trailing_text:
            new_blocks.append({"type": "text", "text": trailing_text})
        splitted = True
    return new_blocks


def _error_sse(client_format: str, message: str) -> str:
    """Return an SSE error event in the appropriate format."""
    if client_format == "openai":
        chunk = {"id": "err", "object": "chat.completion.chunk", "created": int(time.time()),
                 "model": "", "choices": [{"index": 0, "delta": {"content": f"\n\n[网关错误] {message}"},
                                           "finish_reason": "stop"}]}
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n"
    else:
        return (f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':f'[网关错误] {message}'}}, ensure_ascii=False)}\n\n"
                f"event: message_stop\ndata: {json.dumps({'type':'message_stop'}, ensure_ascii=False)}\n\n")


async def _check_upstream_error(resp: httpx.Response, upstream_name: str) -> str | None:
    """If response is an error, log body and return error message. Returns None if OK."""
    if resp.is_error:
        body = await _read_error_text(resp)
        logger.error("Upstream %s returned %s: %s", upstream_name, resp.status_code, body[:500])
        return f"上游 {resp.status_code}: {body[:200]}"
    return None


async def _read_error_text(resp: httpx.Response) -> str:
    try:
        await resp.aread()
        return resp.text
    except Exception:
        return "(unreadable)"


async def guarded_sse_stream(
    gen: AsyncGenerator[str, None], client_format: str, upstream_name: str
) -> AsyncGenerator[str, None]:
    """流式最外层保险（Issue #39，2026-08-28）：上游 ConnectError/超时/读断等异常
    若直接穿透 StreamingResponse，客户端只会看到 HTTP/2 流被重置
    （okhttp StreamResetException: INTERNAL_ERROR），完全不可读——200 已发出，
    想返回错误状态码已经来不及，只能在流内吐一条 _error_sse 再正常收流。
    各 forward_stream_* 内部已处理"上游返回非 2xx"的情况，这里兜的是异常。
    只影响失败路径，正常转发零开销。"""
    try:
        async for chunk in gen:
            yield chunk
    except httpx.HTTPStatusError as e:
        try:
            text = (e.response.text or "")[:200]
        except Exception:
            text = "(unreadable)"
        logger.warning("Upstream %s returned %s (stream): %s",
                       upstream_name, e.response.status_code, text)
        yield _error_sse(client_format, f"上游 {upstream_name} 返回 {e.response.status_code}: {text}")
    except Exception as e:
        logger.exception("Stream to upstream %s failed", upstream_name)
        yield _error_sse(client_format, f"上游 {upstream_name} 连接失败: {type(e).__name__}")


class UpstreamPreStreamError(Exception):
    """上游在吐出第一个字节之前失败（连接级错误 / 401 / 429 / 5xx）。

    stream_with_failover 据此换下一个候选；候选耗尽后把 message 作为
    _error_sse 带内报错（与无 failover 时代行为一致）。"""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _failover_worthy(status_code: int) -> bool:
    """坏 key / 限流 / 上游 5xx / 连接级失败 → 值得换上游；
    4xx 客户端错误（400/403/404 等）说明请求本身有问题，换站也会重复失败。"""
    return status_code == 0 or status_code in (401, 408, 429) or status_code >= 500


async def stream_with_failover(
    gen_factory, candidates: list, client_format: str
) -> AsyncGenerator[str, None]:
    """多上游 failover（2026-09-06）：按序尝试候选，仅当上游在**第一个字节
    之前**失败（UpstreamPreStreamError / 连接级异常）才换下一个候选。

    gen_factory(upstream) -> 未启动的 forward_stream_* 生成器；工厂负责把
    共用的 StreamArchiveState 重指到当前候选（失败的尝试没有产出内容，无副作用）。
    - 某候选开始吐内容 → 独占该流转发到底（客户端已收到字节，不再换站）
    - 候选耗尽 → yield 最后一次失败的 _error_sse（无 failover 时代行为）
    - 4xx 客户端错误不值得换站 → 直接报错
    """
    last_error: UpstreamPreStreamError | None = None
    for index, up in enumerate(candidates):
        try:
            gen = gen_factory(up)
            first = await gen.__anext__()
        except StopAsyncIteration:
            return
        except UpstreamPreStreamError as e:
            last_error = e
            if not _failover_worthy(e.status_code):
                logger.warning("Upstream %s pre-stream %s (not failover-worthy, reporting error directly)",
                               up.name, e.message[:120])
                yield _error_sse(client_format, e.message)
                return
            logger.warning("Upstream %s pre-stream failure (%s), failover to next candidate (%d/%d)",
                           up.name, e.message[:120], index + 1, len(candidates))
            continue
        except httpx.HTTPError as e:
            last_error = UpstreamPreStreamError(f"上游连接失败: {type(e).__name__}", 0)
            logger.warning("Upstream %s connection-level failure (%s), failover to next candidate (%d/%d)",
                           up.name, type(e).__name__, index + 1, len(candidates))
            continue
        yield first
        async for chunk in gen:
            yield chunk
        return
    if last_error is not None:
        yield _error_sse(client_format, last_error.message)


def _build_headers(upstream: Upstream, cache_ttl: str = "1h") -> dict[str, str]:
    headers = {"content-type": "application/json"}
    model = getattr(upstream, "_gateway_selected_model", "")
    if upstream.api_format == "anthropic":
        headers["x-api-key"] = upstream.get_key("chat", model=model)
        headers["anthropic-version"] = "2023-06-01"
        if cache_ttl == "1h":
            headers["anthropic-beta"] = "extended-cache-ttl-2025-04-11"
    else:
        headers["authorization"] = f"Bearer {upstream.get_key('chat', model=model)}"
    return headers


def _extract_user_content(body: dict, client_format: str) -> str:
    """Extract the last user message content for archival."""
    messages = body.get("messages", [])
    for msg in reversed(messages):
        role = msg.get("role", "")
        if role == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                return " ".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )
            return str(content)
    return ""


def _get_conversation_id(body: dict) -> str:
    return body.get("metadata", {}).get("conversation_id") or str(uuid.uuid4())


@dataclass
class StreamArchiveState:
    """流式归档累积器（2026-09-01，修客户端断连漏归档）。

    forward_stream_* 边收上游流边把 assistant 文本 / token 用量 / 模型名写进
    这里；正常结束时走原来的 __ARCHIVE__ 哨兵交给 routers 归档。而客户端中途
    断开时生成器被 cancel，哨兵永远到不了——以前这一轮就彻底丢了（无记录、
    无成本统计、BP2 也少一块）。现在 routers 在 finally 里用这里已累积的
    数据后台补归档（archive_stream_state_in_background），断连不再丢轮。

    failed=True（上游报错走了 _error_sse）时不补归档，与哨兵路径的
    stream_failed 早退行为一致。"""

    conversation_id: str = ""
    user_content: str = ""
    assistant_content: str = ""
    model: str = ""
    client_model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    upstream_name: str = ""
    api_format: str = ""
    started_at: float = field(default_factory=time.monotonic)
    duration_ms: int = 0
    failed: bool = False
    archive_dispatched: bool = False

    @classmethod
    def for_body(cls, body: dict, upstream: Upstream, client_format: str) -> "StreamArchiveState":
        return cls(
            conversation_id=_get_conversation_id(body),
            user_content=_extract_user_content(body, client_format),
            client_model=body.get("model", ""),
            model=body.get("model", ""),
            upstream_name=upstream.name,
            api_format=client_format,
        )

    def finalize(self) -> None:
        if not self.duration_ms:
            self.duration_ms = int((time.monotonic() - self.started_at) * 1000)

    def worth_archiving(self) -> bool:
        """上游至少开始吐内容（有文本或已计 token）才值得补归档；
        连 message_start 都没收到的断连是纯废请求，归档只会污染账本。"""
        return not self.failed and bool(
            self.assistant_content or self.tokens_out or self.tokens_in
        )

    def to_archive_info(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "user_content": self.user_content,
            "assistant_content": self.assistant_content,
            "model": self.model,
            "client_model": self.client_model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "upstream_name": self.upstream_name,
            "api_format": self.api_format,
            "duration_ms": self.duration_ms,
        }


async def forward_nonstream(
    body: dict,
    upstream: Upstream,
    client_format: str,
    body_format: str = "",
) -> tuple[dict, dict]:
    """Forward a non-streaming request. Returns (response_dict, archive_info).

    body_format：传入 body 的实际消息格式，默认等于 client_format（历史行为）。
    openai 客户端在网关内预转换 + 完整预处理后 body 已是 anthropic 格式
    （见 routers/openai_proxy.py），此时 body_format="anthropic" 而
    client_format="openai"：请求侧不再二次转换，响应侧仍转回 openai 格式。
    """
    body_format = body_format or client_format
    need_req_convert = body_format != upstream.api_format
    need_resp_convert = client_format != upstream.api_format

    if need_req_convert:
        if body_format == "openai":
            forward_body = openai_to_anthropic(body)
        else:
            forward_body = anthropic_to_openai(body)
    else:
        forward_body = dict(body)

    cache_ttl = _cache_ttl_from_body(body)
    forward_body = strip_gateway_private_fields(forward_body)
    forward_body["stream"] = False
    setattr(upstream, "_gateway_selected_model", forward_body.get("model", body.get("model", "")))
    headers = _build_headers(upstream, cache_ttl=cache_ttl)

    start = time.monotonic()
    client = get_client()
    resp = await client.post(upstream.messages_url, headers=headers, json=forward_body, timeout=UPSTREAM_TIMEOUT)
    if (
        upstream.api_format == "anthropic"
        and cache_ttl == "1h"
        and _is_extended_cache_error(resp)
    ):
        logger.warning(
            "Upstream %s rejected 1h cache TTL; retrying once with 5m",
            upstream.name,
        )
        forward_body = _downgrade_cache_ttl(forward_body, ttl="5m")
        headers = _build_headers(upstream, cache_ttl="5m")
        resp = await client.post(upstream.messages_url, headers=headers, json=forward_body, timeout=UPSTREAM_TIMEOUT)
    resp.raise_for_status()
    upstream_data = resp.json()

    duration_ms = int((time.monotonic() - start) * 1000)

    # Fix orphan </think> close tag (some upstreams swallow the opening tag).
    # Applies only to anthropic-format upstreams whose response has a content array.
    if upstream.api_format == "anthropic":
        original_content = upstream_data.get("content")
        if isinstance(original_content, list):
            fixed = _split_orphan_close_think(original_content)
            if fixed is not original_content:
                upstream_data = dict(upstream_data)
                upstream_data["content"] = fixed

    if need_resp_convert:
        if upstream.api_format == "anthropic":
            response_data = anthropic_response_to_openai(upstream_data)
        else:
            response_data = openai_response_to_anthropic(upstream_data)
    else:
        response_data = upstream_data

    # Extract archival info
    cache_write_tokens = 0
    cache_read_tokens = 0
    if upstream.api_format == "anthropic":
        usage = upstream_data.get("usage", {})
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)
        cache_write_tokens = usage.get("cache_creation_input_tokens", 0)
        cache_read_tokens = usage.get("cache_read_input_tokens", 0)
        text_blocks = upstream_data.get("content", [])
        assistant_text = "".join(b.get("text", "") for b in text_blocks if b.get("type") == "text")
    else:
        usage = upstream_data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
        choice = upstream_data.get("choices", [{}])[0]
        assistant_text = choice.get("message", {}).get("content", "")

    archive_info = {
        "conversation_id": _get_conversation_id(body),
        "user_content": _extract_user_content(body, client_format),
        "assistant_content": assistant_text,
        "model": upstream_data.get("model", body.get("model", "")),
        "client_model": body.get("model", ""),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cache_write_tokens": cache_write_tokens,
        "cache_read_tokens": cache_read_tokens,
        "upstream_name": upstream.name,
        "raw_request": body,
        "raw_response": json.dumps(upstream_data, ensure_ascii=False),
        "api_format": client_format,
        "duration_ms": duration_ms,
    }

    return response_data, archive_info


async def forward_stream_anthropic_to_anthropic(
    body: dict,
    upstream: Upstream,
    archive_state: "StreamArchiveState | None" = None,
) -> AsyncGenerator[str, None]:
    """Client=Anthropic, Upstream=Anthropic: pass-through SSE."""
    st = archive_state or StreamArchiveState.for_body(body, upstream, "anthropic")
    cache_ttl = _cache_ttl_from_body(body)
    forward_body = dict(body)
    forward_body = strip_gateway_private_fields(forward_body)
    forward_body["stream"] = True
    setattr(upstream, "_gateway_selected_model", forward_body.get("model", body.get("model", "")))
    headers = _build_headers(upstream, cache_ttl=cache_ttl)

    rewriter = _ThinkingRewriter(force_think_mode=getattr(upstream, "force_think", False))

    dump_fp = _open_raw_dump(f"a2a_{upstream.name}", forward_body)

    async def emit_response(resp: httpx.Response):
        err = await _check_upstream_error(resp, upstream.name)
        if err:
            st.failed = True
            raise UpstreamPreStreamError(err, resp.status_code)
        async for line in resp.aiter_lines():
            if dump_fp is not None:
                try:
                    dump_fp.write(line + "\n")
                except Exception:
                    pass
            if not line.strip():
                yield "\n"
                continue
            if line.startswith("event:"):
                # Rewriter may split one upstream data event into multiple Anthropic
                # events (thinking start/delta/stop + shifted text). Drop the
                # upstream event name and re-emit frames from the rewritten JSON
                # type, otherwise clients can see an event/data mismatch.
                continue
            if not line.startswith("data:"):
                yield line + "\n"
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                yield line + "\n"
                continue
            try:
                event = json.loads(raw)
                etype = event.get("type", "")
                # Extract usage stats before rewriting
                if etype == "message_start":
                    msg = event.get("message", {})
                    st.model = msg.get("model", st.model)
                    usage = msg.get("usage", {})
                    st.tokens_in = usage.get("input_tokens", 0)
                    st.cache_write_tokens = usage.get("cache_creation_input_tokens", 0)
                    st.cache_read_tokens = usage.get("cache_read_input_tokens", 0)
                elif etype == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        st.assistant_content += delta.get("text", "")
                elif etype == "message_delta":
                    usage = event.get("usage", {})
                    st.tokens_out = usage.get("output_tokens", st.tokens_out)
                    if usage.get("input_tokens"):
                        st.tokens_in = usage["input_tokens"]
                    if usage.get("cache_creation_input_tokens"):
                        st.cache_write_tokens = usage["cache_creation_input_tokens"]
                    if usage.get("cache_read_input_tokens"):
                        st.cache_read_tokens = usage["cache_read_input_tokens"]
                # Rewrite and emit
                for out_line in rewriter.process(event):
                    yield out_line
                    yield "\n"
            except json.JSONDecodeError:
                yield line + "\n"

    try:
        client = get_client()
        async with client.stream("POST", upstream.messages_url, headers=headers, json=forward_body, timeout=UPSTREAM_TIMEOUT) as resp:
            if (
                resp.is_error
                and upstream.api_format == "anthropic"
                and cache_ttl == "1h"
            ):
                body_text = await _read_error_text(resp)
                if _is_extended_cache_error_text(resp.status_code, body_text):
                    logger.warning(
                        "Upstream %s rejected 1h cache TTL for stream; retrying once with 5m",
                        upstream.name,
                    )
                    retry_body = _downgrade_cache_ttl(forward_body, ttl="5m")
                    headers = _build_headers(upstream, cache_ttl="5m")
                    async with client.stream("POST", upstream.messages_url, headers=headers, json=retry_body, timeout=UPSTREAM_TIMEOUT) as retry_resp:
                        async for chunk in emit_response(retry_resp):
                            yield chunk
                    return
                logger.error("Upstream %s returned %s: %s", upstream.name, resp.status_code, body_text[:500])
                st.failed = True
                raise UpstreamPreStreamError(f"上游 {resp.status_code}: {body_text[:200]}", resp.status_code)

            async for chunk in emit_response(resp):
                yield chunk
    finally:
        if dump_fp is not None:
            try:
                dump_fp.close()
            except Exception:
                pass
    if st.failed:
        return

    st.finalize()
    _state.touch(system=body.get("system"), model=st.model)
    yield "\n__ARCHIVE__:" + json.dumps(st.to_archive_info(), ensure_ascii=False)


async def forward_stream_openai_to_openai(
    body: dict,
    upstream: Upstream,
    archive_state: "StreamArchiveState | None" = None,
) -> AsyncGenerator[str, None]:
    """Client=OpenAI, Upstream=OpenAI: pass-through SSE."""
    st = archive_state or StreamArchiveState.for_body(body, upstream, "openai")
    forward_body = dict(body)
    forward_body = strip_gateway_private_fields(forward_body)
    forward_body["stream"] = True
    setattr(upstream, "_gateway_selected_model", forward_body.get("model", body.get("model", "")))
    headers = _build_headers(upstream, cache_ttl=_cache_ttl_from_body(body))

    client = get_client()
    async with client.stream("POST", upstream.messages_url, headers=headers, json=forward_body, timeout=UPSTREAM_TIMEOUT) as resp:
        err = await _check_upstream_error(resp, upstream.name)
        if err:
            st.failed = True
            raise UpstreamPreStreamError(err, resp.status_code)
        async for line in resp.aiter_lines():
            if not line.strip():
                yield "\n"
                continue
            yield line + "\n"
            if line.startswith("data:"):
                raw = line[5:].strip()
                if raw and raw != "[DONE]":
                    try:
                        event = json.loads(raw)
                        st.model = event.get("model", st.model)
                        choices = event.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if "content" in delta and delta["content"]:
                                st.assistant_content += delta["content"]
                        usage = event.get("usage")
                        if usage:
                            st.tokens_in = usage.get("prompt_tokens", st.tokens_in)
                            st.tokens_out = usage.get("completion_tokens", st.tokens_out)
                    except json.JSONDecodeError:
                        pass
    st.finalize()
    yield "\n__ARCHIVE__:" + json.dumps(st.to_archive_info(), ensure_ascii=False)


async def forward_stream_anthropic_to_openai(
    body: dict,
    upstream: Upstream,
    archive_state: "StreamArchiveState | None" = None,
) -> AsyncGenerator[str, None]:
    """Client=Anthropic, Upstream=OpenAI: convert Anthropic request, convert SSE events back."""
    st = archive_state or StreamArchiveState.for_body(body, upstream, "anthropic")
    forward_body = anthropic_to_openai(body)
    forward_body = strip_gateway_private_fields(forward_body)
    forward_body["stream"] = True
    setattr(upstream, "_gateway_selected_model", forward_body.get("model", body.get("model", "")))
    headers = _build_headers(upstream, cache_ttl=_cache_ttl_from_body(body))

    is_first = True

    client = get_client()
    async with client.stream("POST", upstream.messages_url, headers=headers, json=forward_body, timeout=UPSTREAM_TIMEOUT) as resp:
        err = await _check_upstream_error(resp, upstream.name)
        if err:
            st.failed = True
            raise UpstreamPreStreamError(err, resp.status_code)
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                if raw == "[DONE]":
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue

            st.model = event.get("model", st.model)
            events = openai_sse_to_anthropic_sse(event, is_first=is_first, model=st.model)
            is_first = False
            for e in events:
                yield e

            choices = event.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                if "content" in delta and delta["content"]:
                    st.assistant_content += delta["content"]

    st.finalize()
    yield "\n__ARCHIVE__:" + json.dumps(st.to_archive_info(), ensure_ascii=False)


async def forward_stream_openai_to_anthropic(
    body: dict,
    upstream: Upstream,
    archive_state: "StreamArchiveState | None" = None,
) -> AsyncGenerator[str, None]:
    """Client=OpenAI, Upstream=Anthropic: convert OpenAI request, convert SSE events back."""
    forward_body = openai_to_anthropic(body)
    # openai body 上的缓存 TTL 标记会被 converter 丢掉，搬运到转换后的 body 上
    if "_gateway_cache_ttl" in body:
        forward_body["_gateway_cache_ttl"] = body["_gateway_cache_ttl"]
    async for chunk in forward_stream_anthropic_body_to_openai(forward_body, upstream, archive_state=archive_state):
        yield chunk


async def forward_stream_anthropic_body_to_openai(
    body: dict,
    upstream: Upstream,
    archive_state: "StreamArchiveState | None" = None,
) -> AsyncGenerator[str, None]:
    """Body 已是 anthropic 格式 → anthropic 上游，SSE 转回 openai chunk 流。

    openai 客户端（mikeko 等）在网关内预转换 + 走完整 preprocess_anthropic
    管道后由 routers/openai_proxy.py 调用这里；不再做 openai_to_anthropic
    二次转换，缓存断点等预处理结果原样送达上游。"""
    st = archive_state or StreamArchiveState.for_body(body, upstream, "openai")
    cache_ttl = _cache_ttl_from_body(body)
    forward_body = strip_gateway_private_fields(body)
    forward_body["stream"] = True
    if "metadata" in forward_body:
        # openai 路径历史行为：metadata 不透传给上游（converter 本来就会丢掉它）
        forward_body.pop("metadata", None)
    setattr(upstream, "_gateway_selected_model", forward_body.get("model", body.get("model", "")))
    headers = _build_headers(upstream, cache_ttl=cache_ttl)

    client = get_client()
    async with client.stream("POST", upstream.messages_url, headers=headers, json=forward_body, timeout=UPSTREAM_TIMEOUT) as resp:
        err = await _check_upstream_error(resp, upstream.name)
        if err:
            st.failed = True
            raise UpstreamPreStreamError(err, resp.status_code)
        current_event_type = ""
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                current_event_type = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue

            etype = current_event_type or event.get("type", "")
            if etype == "message_start":
                msg = event.get("message", {})
                st.model = msg.get("model", st.model)
                st.tokens_in = msg.get("usage", {}).get("input_tokens", 0)
            elif etype == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    st.assistant_content += delta.get("text", "")
            elif etype == "message_delta":
                st.tokens_out = event.get("usage", {}).get("output_tokens", 0)

            converted = anthropic_sse_to_openai_sse(etype, event)
            if converted:
                yield converted

    st.finalize()
    yield "\n__ARCHIVE__:" + json.dumps(st.to_archive_info(), ensure_ascii=False)


# ── 流最外层归档包装（2026-09-01，修客户端断连漏归档） ──────────────────

_BG_ARCHIVE_TASKS: set[asyncio.Task] = set()


def _spawn_disconnect_archive(body: dict, archive_kwargs: dict, st: "StreamArchiveState"):
    """断连补归档：不能用 await（GeneratorExit/CancelledError 的 finally 里
    一旦挂起会直接 RuntimeError），用 create_task 派生独立后台任务——它不属于
    被取消的请求作用域，能安全跑完 DB 写入。任务引用挂在模块级集合防 GC。"""
    st.finalize()
    info = st.to_archive_info()

    async def _run():
        try:
            from gateway.services.archiver import archive  # 函数内导入防循环依赖
            await archive(
                raw_request=strip_gateway_private_fields(body),
                raw_response="[streamed-disconnect]",
                **archive_kwargs,
                **info,
            )
            logger.warning(
                "客户端断开，已补归档 conv=%s model=%s tokens_out=%d chars=%d",
                st.conversation_id, st.model, st.tokens_out, len(st.assistant_content),
            )
        except Exception:
            logger.exception("断连补归档失败 conv=%s", st.conversation_id)

    task = asyncio.create_task(_run())
    _BG_ARCHIVE_TASKS.add(task)
    task.add_done_callback(_BG_ARCHIVE_TASKS.discard)


async def archiving_sse_stream(
    gen: AsyncGenerator[str, None],
    *,
    body: dict,
    archive_state: "StreamArchiveState",
    archive_kwargs: dict,
) -> AsyncGenerator[str, None]:
    """包住 forward_stream_* 的最外层，两条归档路径：

    1. 正常结束：拦截 __ARCHIVE__ 哨兵就地归档（历史行为不变，哨兵不透传给客户端）
    2. 客户端断开/流被取消：哨兵到不了，finally 里用 archive_state 已累积的
       数据后台补归档——以前这种情况整轮丢失（无记录、无成本统计）

    failed 路径（上游报错）与收到任何内容前的断连不补归档，避免污染账本。
    """
    try:
        async for chunk in gen:
            if chunk.startswith("\n__ARCHIVE__:"):
                info = json.loads(chunk.split(":", 1)[1])
                from gateway.services.archiver import archive  # 函数内导入防循环依赖
                await archive(
                    raw_request=strip_gateway_private_fields(body),
                    raw_response="[streamed]",
                    **archive_kwargs,
                    **info,
                )
                archive_state.archive_dispatched = True
            else:
                yield chunk
    finally:
        if not archive_state.archive_dispatched and archive_state.worth_archiving():
            _spawn_disconnect_archive(body, archive_kwargs, archive_state)
