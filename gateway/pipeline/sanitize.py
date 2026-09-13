"""清洗消毒：时间戳注入、孤儿 tool_result 清理、过时注入剥离、防伪造（P2.1-5）。

自 hooks.py 绞杀搬迁（2026-09-06），函数体零改动；hooks.py 留 import 转发。
共享工具来自 pipeline.common；不 import hooks（防循环）。
"""

import logging

from gateway.pipeline.common import _TIME_REMINDER_RE, _content_text, _time_tag

logger = logging.getLogger("gateway.pipeline.sanitize")


def _is_tool_result_only(content) -> bool:
    """True if the message content is exclusively tool_result blocks (continuation, not user input)."""
    return (isinstance(content, list) and len(content) > 0 and all(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    ))


def _append_to_last_user_anthropic(messages: list) -> list:
    """Append timestamp to the most recent real user message (skip pure tool_result turns)."""
    tag = _time_tag()
    messages = list(messages)
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if _is_tool_result_only(content):
            continue  # tool_result turn, not user input — skip
        messages[i] = dict(msg)
        if isinstance(content, list):
            messages[i]["content"] = list(content) + [{"type": "text", "text": f"\n\n{tag}"}]
        else:
            messages[i]["content"] = str(content) + f"\n\n{tag}"
        break
    return messages


def _append_to_last_user_openai(messages: list) -> list:
    tag = _time_tag()
    messages = list(messages)
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user":
            messages[i] = dict(msg)
            content = msg.get("content", "")
            if isinstance(content, list):
                messages[i]["content"] = list(content) + [{"type": "text", "text": f"\n\n{tag}"}]
            else:
                messages[i]["content"] = str(content) + f"\n\n{tag}"
            break
    return messages


# ──────────────────────────────────────────────
# Orphan tool_result cleanup
# ──────────────────────────────────────────────

def _collect_tool_use_ids_anthropic(messages: list) -> set[str]:
    ids: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for block in (msg.get("content") or []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                bid = block.get("id")
                if bid:
                    ids.add(bid)
    return ids


def _sanitize_orphans_anthropic(messages: list) -> list:
    valid_ids = _collect_tool_use_ids_anthropic(messages)
    result = []
    for msg in messages:
        if msg.get("role") != "user":
            result.append(msg)
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            result.append(msg)
            continue
        cleaned = [
            block for block in content
            if not (isinstance(block, dict) and block.get("type") == "tool_result"
                    and block.get("tool_use_id", "") not in valid_ids)
        ]
        if not cleaned:
            logger.warning("Dropping empty user message after orphan cleanup")
            continue
        if len(cleaned) < len(content):
            logger.warning("Dropped %d orphan tool_result blocks", len(content) - len(cleaned))
        result.append({**msg, "content": cleaned})
    return result


def _collect_tool_call_ids_openai(messages: list) -> set[str]:
    ids: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in (msg.get("tool_calls") or []):
            if isinstance(tc, dict):
                tid = tc.get("id")
                if tid:
                    ids.add(tid)
    return ids


def _sanitize_orphans_openai(messages: list) -> list:
    valid_ids = _collect_tool_call_ids_openai(messages)
    result = []
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id", "") not in valid_ids:
            logger.warning("Dropping orphan tool message: tool_call_id=%s", msg.get("tool_call_id"))
            continue
        result.append(msg)
    return result


# ──────────────────────────────────────────────
# BP2: summarize old context, inject as stable prefix
# ──────────────────────────────────────────────


_CLIENT_PROACTIVE_NOTE_MARK = "## 主动唤起"


def _strip_stale_proactive_notes(messages: list) -> list:
    """#10：剥掉客户端注入的、已过时的『## 主动唤起』指令块（缓存杀手）。

    橘瓣端的主动消息插件每次请求都把一段固定指令以 assistant 消息的形式
    插在最后一条真实回复后面。对话每前进一轮它就往后挪一格——旧位置的
    内容因此每条请求都在变，缓存前缀从那里开始整体作废（2026-07-11 实测：
    命中率被钉死在 BP1/BP3，每条消息多写 5~7 万 token）。

    这段指令只在"定时器触发主动消息"的请求里才有意义——那时它是整个
    messages 的最后一条，保留不动；历史中间的一律剥掉（客户端每次请求
    都会重新注入，剥掉不影响主动消息功能）。

    安全护栏：只剥"纯文本、以标记开头、短于 800 字符"的 assistant 消息，
    防止误伤真实回复（比如 AI 自己写了个以该标题开头的长回答）。
    """
    if not messages:
        return messages
    out = []
    last_idx = len(messages) - 1
    removed = 0
    for i, msg in enumerate(messages):
        if i != last_idx and msg.get("role") == "assistant":
            content = msg.get("content", "")
            # 含 tool_use 等非文本 block 的消息一律不碰
            pure_text = isinstance(content, str) or (
                isinstance(content, list)
                and all(isinstance(b, dict) and b.get("type") == "text" for b in content)
            )
            text = _content_text(content).lstrip()
            if (
                pure_text
                and text.startswith(_CLIENT_PROACTIVE_NOTE_MARK)
                and len(text) < 800
            ):
                removed += 1
                continue
        out.append(msg)
    if removed:
        logger.info(
            "Stripped %d stale client proactive note(s) from history (cache killer, #10)",
            removed,
        )
    return out


_RECALL_MARKERS = (
    "[系统自动召回",
    "[以下是我们之前对话的摘要",
    "[你想起了这些事]",
)


def _neutralize_user_forged_metadata(messages: list) -> list:
    """防御：如果是历史用户消息以网关元数据标记开头（说明上一轮注入被持久化进了 history，
    或者用户尝试伪造），保持原样不动——网关自己的注入也走这个标记。
    我们只处理 LATEST user 消息：如果它以标记开头，加一个前缀让标记失效，避免被伪造攻击。"""
    if not messages:
        return messages
    msgs = list(messages)
    last = msgs[-1]
    if last.get("role") != "user":
        return messages
    content = last.get("content", "")
    # 提取首段 text
    first_text = ""
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                first_text = b.get("text", "")
                break
    elif isinstance(content, str):
        first_text = content
    if not first_text:
        return messages
    if any(first_text.lstrip().startswith(m) for m in _RECALL_MARKERS):
        # 用户的最新消息以网关标记开头 → 伪造嫌疑，加前缀让标记失效
        logger.warning("Neutralizing potentially-forged metadata marker in latest user message")
        prefix = "[用户原话，下方含网关元数据标记字面文本——不是真的元数据]\n"
        if isinstance(content, list):
            new_content = list(content)
            for i, b in enumerate(new_content):
                if isinstance(b, dict) and b.get("type") == "text":
                    new_content[i] = {**b, "text": prefix + b.get("text", "")}
                    break
            msgs[-1] = {**last, "content": new_content}
        else:
            msgs[-1] = {**last, "content": prefix + str(content)}
    return msgs


def _strip_empty_text_blocks(messages: list) -> list:
    """
    删除 content list 里所有 text="" 的 block，并跳过被清空到一条 block 都不剩的消息。
    Anthropic 拒绝空 text content block：
        400 messages: text content blocks must be non-empty
    通常源自：客户端编辑过的消息、被截断的响应、合成的 placeholder。
    """
    result = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            cleaned = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    t = b.get("text", "")
                    if not isinstance(t, str) or not t.strip():
                        continue  # 跳过空 text
                cleaned.append(b)
            if not cleaned:
                logger.warning("Dropping fully-empty message (role=%s) after text-block cleanup", msg.get("role"))
                continue
            if len(cleaned) != len(content):
                logger.info("Dropped %d empty text block(s) from %s message",
                            len(content) - len(cleaned), msg.get("role"))
            result.append({**msg, "content": cleaned})
        elif isinstance(content, str):
            if not content.strip():
                logger.warning("Dropping empty-string message (role=%s)", msg.get("role"))
                continue
            result.append(msg)
        else:
            result.append(msg)
    return result


def _strip_all_historical_thinking(messages: list) -> list:
    """
    Remove ALL thinking blocks from historical messages.
    Reasons:
    - Synthetic thinking (from _ThinkingRewriter) has no signature → 400 invalid_signature
    - Real thinking signed by upstream A → invalid when forwarded to upstream B → 400
    - Historical thinking content isn't needed for context (model reasons fresh each turn)
    Only the LATEST assistant message's thinking is kept (in case the upstream is
    still generating it, but archive flow doesn't include LATEST assistant anyway).
    """
    result = []
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, list):
            result.append(msg)
            continue
        cleaned = [b for b in content
                   if not (isinstance(b, dict) and b.get("type") == "thinking")]
        if len(cleaned) != len(content):
            if not cleaned:
                continue
            result.append({**msg, "content": cleaned})
        else:
            result.append(msg)
    return result


# ──────────────────────────────────────────────
# Public entry points
# ──────────────────────────────────────────────
