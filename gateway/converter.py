"""OpenAI <-> Anthropic format conversion."""

import json
import uuid
import time
from typing import Any


def _openai_content_to_anthropic(content):
    """OpenAI 消息 content → Anthropic content。

    string 统一归一化为单个 text block：网关预处理管道（缓存断点 BP1-BP4、
    Seamless/BP2 摘要注入、孤儿清理）面向 block 结构设计与实测，归一化后
    openai 客户端走的是与橘瓣完全相同的代码路径。None（assistant 只有
    tool_calls 时）归一化为 ""，后续由 _strip_empty_text_blocks 兜底丢弃。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        if not content:
            return ""
        return [{"type": "text", "text": content}]
    return content  # list：text parts 本就与 anthropic text block 同形，原样透传


def openai_to_anthropic(body: dict) -> dict:
    """Convert OpenAI Chat Completions request to Anthropic Messages request."""
    messages = body.get("messages", [])
    system = None
    converted = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            if system is None:
                system = content
            else:
                system += "\n\n" + content
        elif role == "user":
            converted.append({"role": "user", "content": _openai_content_to_anthropic(content)})
        elif role == "assistant":
            converted.append({"role": "assistant", "content": _openai_content_to_anthropic(content)})

    result: dict[str, Any] = {
        "model": body.get("model", ""),
        "messages": converted,
        "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens") or 4096,
        "stream": body.get("stream", False),
    }
    if system:
        result["system"] = system
    if body.get("temperature") is not None:
        result["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        result["top_p"] = body["top_p"]
    if body.get("stop"):
        result["stop_sequences"] = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
    if "thinking" in body:
        result["thinking"] = body["thinking"]
    if "reasoning_effort" in body:
        result["reasoning_effort"] = body["reasoning_effort"]
    return result


def strip_gateway_private_fields(body: dict) -> dict:
    return {
        k: v for k, v in body.items()
        if not str(k).startswith("_gateway_")
    }


def anthropic_to_openai(body: dict) -> dict:
    """Convert Anthropic Messages request to OpenAI Chat Completions request."""
    messages = []
    system = body.get("system")
    if system:
        if isinstance(system, list):
            system_text = " ".join(
                block.get("text", "") for block in system if block.get("type") == "text"
            )
        else:
            system_text = str(system)
        messages.append({"role": "system", "content": system_text})

    for msg in body.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [
                block.get("text", "") for block in content if block.get("type") == "text"
            ]
            content = " ".join(text_parts)
        messages.append({"role": role, "content": content})

    result: dict[str, Any] = {
        "model": body.get("model", ""),
        "messages": messages,
        "max_tokens": body.get("max_tokens", 4096),
        "stream": body.get("stream", False),
    }
    if body.get("temperature") is not None:
        result["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        result["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        result["stop"] = body["stop_sequences"]
    if "thinking" in body:
        result["thinking"] = body["thinking"]
    if "reasoning_effort" in body:
        result["reasoning_effort"] = body["reasoning_effort"]
    return result


def anthropic_response_to_openai(data: dict) -> dict:
    """Convert Anthropic non-stream response to OpenAI format."""
    content_blocks = data.get("content", [])
    text = ""
    for block in content_blocks:
        if block.get("type") == "text":
            text += block.get("text", "")

    usage = data.get("usage", {})
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model", ""),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": _map_stop_reason(data.get("stop_reason", "end_turn")),
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


def openai_response_to_anthropic(data: dict) -> dict:
    """Convert OpenAI non-stream response to Anthropic format."""
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage = data.get("usage", {})
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": message.get("content", "")}],
        "model": data.get("model", ""),
        "stop_reason": _map_finish_reason(choice.get("finish_reason", "stop")),
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def anthropic_sse_to_openai_sse(event_type: str, data: dict) -> str | None:
    """Convert a single Anthropic SSE event to OpenAI SSE chunk. Returns None to skip."""
    if event_type == "message_start":
        return None
    elif event_type == "content_block_start":
        return None
    elif event_type == "content_block_delta":
        delta = data.get("delta", {})
        if delta.get("type") == "text_delta":
            chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "",
                "choices": [{
                    "index": 0,
                    "delta": {"content": delta.get("text", "")},
                    "finish_reason": None,
                }],
            }
            return f"data: {json.dumps(chunk)}\n\n"
    elif event_type == "message_delta":
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": _map_stop_reason(data.get("delta", {}).get("stop_reason", "end_turn")),
            }],
        }
        return f"data: {json.dumps(chunk)}\n\n"
    elif event_type == "message_stop":
        return "data: [DONE]\n\n"
    return None


def openai_sse_to_anthropic_sse(data: dict, is_first: bool = False, model: str = "") -> list[str]:
    """Convert a single OpenAI SSE chunk to Anthropic SSE event(s)."""
    events = []
    if is_first:
        events.append(
            f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': f'msg_{uuid.uuid4().hex[:24]}', 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
        )
        events.append(
            f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        )

    choices = data.get("choices", [])
    if not choices:
        return events

    choice = choices[0]
    delta = choice.get("delta", {})
    finish = choice.get("finish_reason")

    if "content" in delta and delta["content"]:
        events.append(
            f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta['content']}})}\n\n"
        )

    if finish:
        events.append(
            f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        )
        events.append(
            f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': _map_finish_reason(finish)}, 'usage': {'output_tokens': 0}})}\n\n"
        )
        events.append(
            f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        )

    return events


def _map_stop_reason(reason: str) -> str:
    mapping = {"end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop"}
    return mapping.get(reason, "stop")


def _map_finish_reason(reason: str) -> str:
    mapping = {"stop": "end_turn", "length": "max_tokens"}
    return mapping.get(reason, "end_turn")
