"""BP2: call upstream API to summarize old conversation context."""

import hashlib
import json
import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from gateway.config import UPSTREAM_TIMEOUT, TIMEZONE
from gateway.http_client import get_client

logger = logging.getLogger("gateway.summarizer")

_DEFAULT_SYSTEM_PROMPT = (
    "你是一个对话摘要助手。请将以下对话写成详细的中文摘要（目标 10000 字符以上），"
    "保留：主要话题与事件、情感状态、已建立的重要信息或约定、对话中的关键细节和原文引用。"
    "宁可写长也不要遗漏重要细节。直接输出摘要内容，不要标题或前缀。"
)
_FALLBACK_MODEL = "claude-haiku-4-5-20251001"


def derive_conv_fingerprint(messages: list) -> str:
    """Stable conversation ID from the first few messages (never changes for a given conversation)."""
    anchor = messages[:4] if len(messages) >= 4 else messages
    text = json.dumps(
        [{"role": m.get("role"), "content": str(m.get("content", ""))[:300]} for m in anchor],
        ensure_ascii=False, sort_keys=True,
    )
    return "fp_" + hashlib.md5(text.encode()).hexdigest()[:16]


# 归档正文里的网关时间戳（注入在 user 消息末尾）。摘要提取时把它转成
# 行首紧凑戳：长消息被截断到 500 字时尾部时间戳会丢，模型看不到日期只能
# 脑补时间线（2026-07-13 实测漂移 +2 天、多个会话经继承扩散）。行首戳
# 保证每条消息无论截断与否都带真实日期。
_EXTRACT_TIME_RE = re.compile(
    r"<time_reminder>.*?(\d{4})年(\d{1,2})月(\d{1,2})日\s+(\d{1,2}:\d{2}).*?</time_reminder>",
    re.S,
)


def _extract_text(messages: list) -> str:
    """Convert messages (possibly with tool calls, images, thinking) to readable text."""
    lines = []
    label = {"user": "用户", "assistant": "助手"}
    for msg in messages:
        role = label.get(msg.get("role", ""), msg.get("role", ""))
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                t = block.get("type", "")
                if t == "text":
                    parts.append(block.get("text", ""))
                elif t == "thinking":
                    pass  # skip internal thinking
                elif t == "tool_use":
                    parts.append(f"[调用工具: {block.get('name', '')}]")
                elif t == "tool_result":
                    rc = block.get("content", "")
                    if isinstance(rc, list):
                        rc = " ".join(b.get("text", "") for b in rc if isinstance(b, dict))
                    parts.append(f"[工具结果: {str(rc)[:100]}]")
                elif t == "image":
                    parts.append("[图片]")
            content = "\n".join(parts)
        text = str(content)
        # 时间戳前置：从尾部 time_reminder 提取日期挪到行首（截断也不丢日期）
        stamp = ""
        m = _EXTRACT_TIME_RE.search(text)
        if m:
            y, mo, d, hm = m.groups()
            stamp = f"[{y}-{int(mo):02d}-{int(d):02d} {hm}] "
            text = _EXTRACT_TIME_RE.sub("", text).strip()
        lines.append(f"{stamp}{role}：{text[:500]}")
    return "\n\n".join(lines)


def _resolve_upstream(default_upstream):
    """Return the upstream to use for summarization (may differ from conversation upstream)."""
    from gateway.settings import get as _get
    from gateway.upstream import load_upstreams
    name = _get("summary_upstream", "")
    if name:
        for u in load_upstreams():
            if u.name == name and u.is_active:
                return u
        logger.warning("Configured summary_upstream '%s' not found/inactive, falling back", name)
    return default_upstream


def _resolve_summary_target(default_upstream):
    """Return (upstream, clean_model) for the configured summary target."""
    from gateway.settings import get as _get
    from gateway.upstream import load_upstreams

    configured_model = _get("summary_model", "") or ""
    configured_upstream = _get("summary_upstream", "") or ""
    clean_model = configured_model
    upstream = None

    if "::" in configured_model:
        upstream_name, clean = configured_model.split("::", 1)
        clean_model = clean.strip()
        for u in load_upstreams():
            if u.name == upstream_name and u.is_active:
                upstream = u
                break

    if upstream is None and configured_upstream:
        for u in load_upstreams():
            if u.name == configured_upstream and u.is_active:
                upstream = u
                break
        if upstream is None:
            logger.warning("Configured summary_upstream '%s' not found/inactive, falling back", configured_upstream)

    upstream = upstream or default_upstream
    model = clean_model or upstream.default_model or _FALLBACK_MODEL
    return upstream, model


async def summarize_messages(
    new_turns: list,
    default_upstream,
    prev_summary: str = "",
) -> str:
    """
    Rolling summary: combine previous summary + new turns into a fresh summary.
    Cost is ~constant regardless of total conversation length.

    Args:
        new_turns:     the new batch of messages to incorporate (one freeze cycle worth)
        default_upstream: conversation upstream (used as fallback)
        prev_summary:  text of the previous BP2 summary (empty on first cycle)
    """
    from gateway.settings import get as _get

    upstream, model = _resolve_summary_target(default_upstream)
    new_text = _extract_text(new_turns)

    # 注入当前日期，让模型能算"X 天前"做分层归档
    today_str = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M %A")

    if prev_summary:
        user_content = (
            f"【今天日期】{today_str}\n\n"
            f"【已有摘要】（之前已经写好的，请保留所有四层结构与原有内容，仅按规则下沉/晋升/追加）\n"
            f"{prev_summary}\n\n"
            f"【新增对话】（按当前日期归入对应层级）\n"
            f"{new_text}\n\n"
            "任务：\n"
            "⚠️ 【最高优先级·字符范围】总输出目标 8000-10000 字符（硬上限 12000）。低于 6000 说明细节丢失过多；超过 10000 需要压缩 # 远期 和 # 古老 层。\n\n"
            "⚠️ 【最高优先级·禁止编造】摘要只能包含对话中明确出现的事实。禁止把对话中的某个词语（人名、地名、比喻、口头禅）推测展开为具体场景或事件。含义模糊的句子保留原文引用，不要脑补上下文。如果无法确认某事是否真实发生，不写入摘要。\n\n"
            "⚠️ 【日期纪律】摘要中出现的所有日期，必须是【已有摘要】或【新增对话】原文里出现过的日期，"
            "禁止推算或编造日期；任何晚于【今天日期】的日期都是错误，写出即视为失败。\n\n"
            "分层规则：\n"
            "1. 输出仍是分层结构（锚定 / 近期 / 远期 / 古老）的完整文档\n"
            "2. 把【新增对话】的内容写入 # 近期 层（最近的放在最下面，保持时间线）\n"
            "3. # 近期 中超过 7 天的事件，提炼后下沉到 # 远期\n"
            "4. # 远期 中超过 30 天的事件，进一步压缩或合并\n"
            "5. 在已有摘要中反复出现的稳定事实，可以提升到 # 锚定（# 锚定 完整保留，仅追加新晋升项）\n"
            "6. 允许压缩 # 远期 和 # 古老 层的冗余细节：合并同类事件、删除重复描述、只保留骨架和情绪标记\n"
            "7. 新增的情感密集段落，仍附【氛围快照】+【原文 sample】\n"
            "\n"
            "⚠️ 【结构硬约束】4 个标题必须齐全：# 锚定 / # 近期 / # 远期 / # 古老。\n"
            "   该层暂时无内容时也必须写出标题，下面接一行『暂无』。\n"
            "   缺任何一层视为失败，整个摘要会被丢弃。"
        )
    else:
        user_content = (
            f"【今天日期】{today_str}\n\n"
            f"【对话内容】\n{new_text}\n\n"
            "任务：按 system prompt 的分层结构（锚定 / 近期 / 远期 / 古老）写出第一版摘要。\n"
            "本次所有内容默认都进 # 近期 层。# 锚定 层提取反复出现的稳定事实。\n"
            "⚠️ 【日期纪律】摘要中出现的所有日期必须是【对话内容】原文里出现过的日期，"
            "禁止推算或编造；任何晚于【今天日期】的日期都是错误。\n"
            "\n"
            "⚠️ 【结构硬约束】4 个标题必须齐全：# 锚定 / # 近期 / # 远期 / # 古老。\n"
            "   首版摘要中 # 远期 和 # 古老 通常无内容，此时也必须写出标题，下面接一行『暂无』。\n"
            "   缺任何一层视为失败，整个摘要会被丢弃。"
        )

    system_prompt = _get("summary_prompt", "") or _DEFAULT_SYSTEM_PROMPT

    if upstream.api_format == "openai":
        headers = {
            "authorization": f"Bearer {upstream.get_key('summary', model=model)}",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 64000,
            "stream": True,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
    else:
        headers = {
            "x-api-key": upstream.get_key("summary", model=model),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 64000,
            "stream": True,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}],
        }

    url = upstream.messages_url
    logger.info("Summarizing (stream) via upstream='%s' format='%s' model='%s' (prev_summary=%d chars, new_turns=%d)",
                upstream.name, upstream.api_format, model, len(prev_summary), len(new_turns))
    # 通知：摘要开始
    try:
        from gateway.notifier import fire_notify as _fn
        _fn(
            f"⏳ BP2 摘要生成中 [{upstream.name}]",
            f"上游：{upstream.name}\n模型：{model}\n新增对话：{len(new_turns)} 轮\n旧摘要：{len(prev_summary)} 字符",
            dedup_key=f"sum_start_{upstream.name}_{model}",
            cooldown=120,  # 同一上游+模型 2 分钟只通知一次
        )
    except Exception:
        pass

    async def _do_stream():
        """流式调用：捕获各种中断异常并 raise 让上层 fall back 到非流式。"""
        text = ""
        stop_reason = ""
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        client = get_client()
        async with client.stream("POST", url, headers=headers, json=stream_payload, timeout=1200) as resp:
            if resp.is_error:
                await resp.aread()
                logger.error("Summarize stream error %s: %s", resp.status_code, resp.text[:500])
                # 通知：429 限流
                if resp.status_code == 429:
                    try:
                        from gateway.notifier import fire_notify as _fn
                        _fn(
                            f"🚫 上游限流 429 [{upstream.name}]",
                            f"上游【{upstream.name}】返回 429 Too Many Requests。\n摘要暂停，请检查配额或切换上游。\n错误：{resp.text[:200]}",
                            dedup_key=f"429_{upstream.name}",
                            cooldown=300,
                        )
                    except Exception:
                        pass
                resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if upstream.api_format == "openai":
                    choices = event.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            text += delta["content"]
                        fr = choices[0].get("finish_reason")
                        if fr:
                            stop_reason = fr
                else:
                    etype = event.get("type")
                    if etype == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text += delta.get("text", "")
                    elif etype == "message_delta":
                        sr = event.get("delta", {}).get("stop_reason")
                        if sr:
                            stop_reason = sr
        return text, stop_reason

    async def _do_nonstream():
        """非流式调用：read 超时设大（DeepSeek 直连无 48s 限制）。中转站走流式那路。"""
        ns_payload = dict(payload)
        ns_payload["stream"] = False
        # connect=30s, read=1200s, write=30s, pool=30s
        ns_timeout = httpx.Timeout(1200.0, connect=30.0, write=30.0, pool=30.0)
        client = get_client()
        resp = await client.post(url, headers=headers, json=ns_payload, timeout=ns_timeout)
        if resp.is_error:
            logger.error("Summarize nonstream error %s: %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
        data = resp.json()
        text = ""
        stop_reason = ""
        if upstream.api_format == "openai":
            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                c = msg.get("content", "")
                if isinstance(c, str):
                    text = c
                elif isinstance(c, list):
                    text = "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
                stop_reason = choices[0].get("finish_reason", "") or ""
        else:
            for block in data.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    text += block.get("text", "")
            stop_reason = data.get("stop_reason", "") or ""
        return text, stop_reason

    # 主路径：流式（兼容中转站 ~48s 非流式硬限制）
    # 异常路径：流式中断 → 回落非流式重试一次（DeepSeek 等直连场景没有 48s 限制）
    _STREAM_FALLBACK_EXC = (
        httpx.ReadError,
        httpx.RemoteProtocolError,
        httpx.ReadTimeout,
        httpx.ConnectError,
    )
    try:
        text, stop_reason = await _do_stream()
        if not text:
            # 流式返回 200 但无内容（DeepSeek 偶发）→ 也走回落
            raise ValueError("Empty summary response (stream returned no text)")
    except _STREAM_FALLBACK_EXC as e:
        logger.warning(
            "BP2 stream interrupted (%s: %s), falling back to non-stream retry (upstream='%s')",
            type(e).__name__, str(e)[:200], upstream.name,
        )
        text, stop_reason = await _do_nonstream()
        if not text:
            raise ValueError("Empty summary response (non-stream fallback also returned no text)")
        logger.info("BP2 non-stream fallback succeeded (%d chars)", len(text))
    except ValueError:
        # 流式 200 但空 → 也回落试一次
        logger.warning("BP2 stream returned empty, falling back to non-stream retry (upstream='%s')", upstream.name)
        text, stop_reason = await _do_nonstream()
        if not text:
            raise ValueError("Empty summary response (non-stream fallback also returned no text)")
        logger.info("BP2 non-stream fallback succeeded (%d chars)", len(text))

    # 检测输出被截断或拒绝——两种情况都不能存入 DB
    bad_stop = stop_reason in ("length", "max_tokens")
    if bad_stop:
        logger.warning(
            "⚠️ BP2 summary rejected: stop_reason=%s len=%d. "
            "Will NOT save to prevent corrupting prev_summary.",
            stop_reason, len(text),
        )
        raise ValueError(f"BP2 summary rejected (stop_reason={stop_reason}, len={len(text)})")

    logger.info("Generated BP2 summary (%d chars, stop_reason=%s) from %d turns",
                len(text), stop_reason or "?", len(new_turns))
    return text.strip()


# ────────────────────────────────────────────────────────────────
# compress_summary: 压缩超长摘要（Issue #18 修复）
# 当 hooks.py BG rebuild 发现新摘要 > 15000 字符时调用，把它压回 8-12k。
# 复用流式/非流式回落逻辑。
# ────────────────────────────────────────────────────────────────
async def compress_summary(text: str, default_upstream) -> str:
    """压缩超长摘要到 18000 字符以内，保留分层结构。"""
    from gateway.settings import get as _get

    upstream, model = _resolve_summary_target(default_upstream)

    sys_prompt = (
        "你是摘要压缩助手。严格按要求压缩，不添加任何新内容，不编造，不推测。"
    )
    user_content = (
        f"以下摘要超长（{len(text)} 字符），请压缩到 8000-10000 字符以内。\n\n"
        "规则：\n"
        "1. 保留完整的分层结构（锚定/近期/远期/古老）\n"
        "2. # 锚定 层硬约束 ≤2500 字符。如果超过，淘汰最不核心的条目：技术细节先丢，关系核心保留。不要『完整保留』\n"
        "3. 压缩 # 近期 中超过 7 天的内容（下沉到远期），近期层目标 ≤5000 字符\n"
        "4. 大幅压缩 # 远期 和 # 古老：合并同类事件、删除重复描述、只保留骨架和情绪标记\n"
        "5. ⚠️ 禁止添加任何原文中没有的内容；禁止把词语展开成场景；禁止脑补\n"
        "6. ⚠️ 4 个标题必须齐全（# 锚定 / # 近期 / # 远期 / # 古老），无内容的层写『暂无』占位，缺层视为失败\n"
        "7. ⚠️ 总长目标 8000-10000 字符（绝对上限 12000），分层硬约束：锚定≤2500/近期≤5000/远期≤3000/古老≤1500\n\n"
        "直接输出压缩后的完整摘要，不要前言后语。\n\n"
        "──────── 原摘要 ────────\n"
        f"{text}"
    )

    if upstream.api_format == "openai":
        headers = {
            "authorization": f"Bearer {upstream.get_key('summary', model=model)}",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 64000,
            "stream": True,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
        }
    else:
        headers = {
            "x-api-key": upstream.get_key("summary", model=model),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 64000,
            "stream": True,
            "system": sys_prompt,
            "messages": [{"role": "user", "content": user_content}],
        }

    url = upstream.messages_url
    logger.info("Compressing summary via upstream='%s' model='%s' (input=%d chars)",
                upstream.name, model, len(text))

    async def _do_stream():
        out = ""
        stop = ""
        sp = dict(payload)
        sp["stream"] = True
        client = get_client()
        async with client.stream("POST", url, headers=headers, json=sp, timeout=1200) as resp:
            if resp.is_error:
                await resp.aread()
                logger.error("compress_summary stream error %s: %s", resp.status_code, resp.text[:500])
                resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if upstream.api_format == "openai":
                    choices = event.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            out += delta["content"]
                        fr = choices[0].get("finish_reason")
                        if fr:
                            stop = fr
                else:
                    etype = event.get("type")
                    if etype == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            out += delta.get("text", "")
                    elif etype == "message_delta":
                        sr = event.get("delta", {}).get("stop_reason")
                        if sr:
                            stop = sr
        return out, stop

    async def _do_nonstream():
        ns_payload = dict(payload)
        ns_payload["stream"] = False
        ns_timeout = httpx.Timeout(1200.0, connect=30.0, write=30.0, pool=30.0)
        client = get_client()
        resp = await client.post(url, headers=headers, json=ns_payload, timeout=ns_timeout)
        if resp.is_error:
            logger.error("compress_summary nonstream error %s: %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()
        data = resp.json()
        out = ""
        stop = ""
        if upstream.api_format == "openai":
            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                c = msg.get("content", "")
                if isinstance(c, str):
                    out = c
                elif isinstance(c, list):
                    out = "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
                stop = choices[0].get("finish_reason", "") or ""
        else:
            for block in data.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    out += block.get("text", "")
            stop = data.get("stop_reason", "") or ""
        return out, stop

    # 流式 + 失败回落非流式
    _FALLBACK_EXC = (httpx.ReadError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError)
    try:
        out, stop = await _do_stream()
        if not out:
            raise ValueError("Empty compress_summary response")
    except _FALLBACK_EXC as e:
        logger.warning("compress_summary stream interrupted (%s), falling back to non-stream", type(e).__name__)
        out, stop = await _do_nonstream()
        if not out:
            raise ValueError("compress_summary non-stream fallback also empty")
    except ValueError:
        logger.warning("compress_summary stream returned empty, falling back to non-stream")
        out, stop = await _do_nonstream()
        if not out:
            raise ValueError("compress_summary non-stream fallback also empty")

    if stop in ("length", "max_tokens"):
        logger.warning("compress_summary truncated (stop=%s, len=%d), returning original", stop, len(out))
        return text  # 压缩失败回退原文，让上层 sanity check 拒绝
    logger.info("Compressed summary: %d → %d chars", len(text), len(out))
    return out.strip()
