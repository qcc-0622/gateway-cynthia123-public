"""
Memory recall: 在请求处理流程中自动从 ombre-brain 检索相关记忆并注入上下文。

四路检索（小红书 4 维记忆 × ombre 现有能力 融合方案）：
    1) Vector + Keyword 检索 → ombre 的 breath 工具一次性返回（自带评分）
    2) 随机注入 → 直接扫 /opt/ombre-brain/buckets/dynamic/ 排除 resolved
    3) 合并去重，按 score 截断到 top 4
    4) 仅当检测到「新话题」时才触发，避免每条消息都打扰

注入位置：消息最前，类似 Seamless Session。
"""

import asyncio
import json
import logging
import os
import random
import re
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from gateway.http_client import get_client

logger = logging.getLogger("gateway.memory")

# ── 默认配置（可被 settings.json 覆盖） ─────────────────────────
DEFAULT_OMBRE_URL    = "http://127.0.0.1:18001/mcp"
DEFAULT_OMBRE_TOKEN  = ""  # 内网调用通常不需要，外网时会带 X-Brain-Token
DEFAULT_BUCKETS_DIR  = "/opt/ombre-brain/buckets"
DEFAULT_EMBED_URL    = "https://api.siliconflow.cn/v1/embeddings"
DEFAULT_EMBED_MODEL  = "BAAI/bge-m3"
DEFAULT_EMBED_KEY    = ""
DEFAULT_TOPIC_THRESHOLD = 0.58  # 新旧消息 cosine < 该值视为新话题
DEFAULT_FORCE_EVERY = 15        # 即使话题延续，每 N 条 user 消息也强制触发一次（之前 5 太频繁）
DEFAULT_MAX_RECALL = 4          # 最多注入几条
DEFAULT_RANDOM_K = 2            # 随机注入几条
DEFAULT_BREATH_MAX_RESULTS = 8
DEFAULT_BREATH_MAX_TOKENS = 2000

_YAML_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)


def _cfg(key: str, default):
    """从 settings.json 读配置，没有就用默认。"""
    try:
        from gateway.settings import get as _get
        val = _get(key, None)
        if val is None or val == "":
            return default
        return val
    except Exception:
        return default


# ────────────────────────────────────────────────────────────────
# 工具：YAML frontmatter 解析（不引入 PyYAML 重依赖，手动 parse 基础字段）
# ────────────────────────────────────────────────────────────────

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """返回 (metadata_dict, content_body)。"""
    m = _YAML_RE.match(text)
    if not m:
        return {}, text
    meta_raw, body = m.group(1), m.group(2)
    meta: dict = {}
    current_list_key = None
    for line in meta_raw.splitlines():
        if not line.strip():
            current_list_key = None
            continue
        # 列表项 (- xxx)
        if line.startswith("-") or line.startswith("  -"):
            if current_list_key:
                val = line.lstrip(" -").strip().strip("'\"")
                meta.setdefault(current_list_key, []).append(val)
            continue
        # key: value
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip().strip("'\"")
            if v:
                # 简单类型推断
                if v in ("true", "True"):
                    meta[k] = True
                elif v in ("false", "False"):
                    meta[k] = False
                else:
                    try:
                        if "." in v:
                            meta[k] = float(v)
                        else:
                            meta[k] = int(v)
                    except ValueError:
                        meta[k] = v
                current_list_key = None
            else:
                current_list_key = k
                meta[k] = []
    return meta, body.strip()


# ────────────────────────────────────────────────────────────────
# 新话题检测：用 bge-m3 算最新两条 user 消息的 cosine 相似度
# ────────────────────────────────────────────────────────────────

def _extract_user_text(msg: dict) -> str:
    """从单条 user 消息提取纯文本（去掉 time_reminder/tool_result）。"""
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
        text = " ".join(parts)
    else:
        text = str(content)
    # 剥离 time_reminder
    text = re.sub(r"<time_reminder>.*?</time_reminder>", "", text, flags=re.DOTALL).strip()
    return text


def _last_two_user_messages(messages: list) -> tuple[str, str]:
    """从消息列表里取最后两条「实质性」user 消息（跳过 tool_result）。"""
    user_texts: list[str] = []
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        text = _extract_user_text(msg)
        # 跳过纯 tool_result 或空消息
        if not text or len(text) < 3:
            continue
        user_texts.append(text)
        if len(user_texts) >= 2:
            break
    if len(user_texts) == 0:
        return "", ""
    if len(user_texts) == 1:
        return user_texts[0], ""
    return user_texts[0], user_texts[1]  # (latest, previous)


async def _embed(text: str, http: httpx.AsyncClient) -> list[float] | None:
    """调 SiliconFlow bge-m3 embedding。"""
    if not text:
        return None
    url    = _cfg("embed_url",   DEFAULT_EMBED_URL)
    model  = _cfg("embed_model", DEFAULT_EMBED_MODEL)
    apikey = _cfg("embed_api_key", DEFAULT_EMBED_KEY)
    if not apikey:
        return None
    try:
        resp = await http.post(
            url,
            json={"model": model, "input": text[:1500]},
            headers={"Authorization": f"Bearer {apikey}"},
            timeout=10,
        )
        if resp.is_error:
            logger.warning("Embed failed %s: %s", resp.status_code, resp.text[:200])
            return None
        return resp.json()["data"][0]["embedding"]
    except Exception:
        logger.exception("Embed exception")
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _count_user_messages_since_last_recall(messages: list) -> int:
    """从尾部往前数，到上一次记忆召回注入为止有几条 user 消息。
    没找到注入痕迹则返回总 user 消息数。兼容新旧两种注入标记。"""
    markers = ("[系统自动召回", "[你想起了这些事]")
    count = 0
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        t = b.get("text", "")
                        if any(t.startswith(p) for p in markers):
                            return count
            elif isinstance(content, str):
                if any(content.startswith(p) for p in markers):
                    return count
            count += 1
    return count


# 2026-09-06：省外呼。每请求固定 2 次 embedding（本次+上条 user 消息）是热路径
# 最大的外部延迟来源。两级保守减负——只短路"非新话题"判定，绝不反向短路：
# 1) (本次, 上条) 完全相同的消息对直接复用上次判定（重试 / 重复触发场景）
# 2) 字符 bigram 余弦 ≥ max(0.9, 阈值+0.2) 的近似重复文本，embedding 余弦
#    必然远高于新话题阈值 → 直接判"非新话题"，省掉 2 次 RTT
_topic_check_cache: dict[tuple[str, str], tuple[bool, str]] = {}
_TOPIC_CHECK_CACHE_MAX = 512


def _bigram_cosine(a: str, b: str) -> float:
    """字符 bigram 余弦相似度（0~1）。任一文本 <2 字返回 0（不做短路）。"""
    import math
    if len(a) < 2 or len(b) < 2:
        return 0.0
    def grams(t):
        c = {}
        for i in range(len(t) - 1):
            g = t[i:i + 2]
            c[g] = c.get(g, 0) + 1
        return c
    ga, gb = grams(a), grams(b)
    dot = sum(n * gb.get(g, 0) for g, n in ga.items())
    if not dot:
        return 0.0
    na = math.sqrt(sum(n * n for n in ga.values()))
    nb = math.sqrt(sum(n * n for n in gb.values()))
    return dot / (na * nb)


async def is_new_topic(messages: list, http: httpx.AsyncClient) -> tuple[bool, str]:
    """判断当前消息是否触发记忆召回。两个触发条件任一：
        1. 与上一条 user 消息的语义 cosine < 阈值 → 新话题
        2. 距离上一次召回已 ≥ N 条 user 消息 → 强制刷新
    返回 (是否触发, 查询文本)。"""
    latest, prev = _last_two_user_messages(messages)
    if not latest:
        return False, ""
    if not prev:
        # 第一条消息，总是触发
        return True, latest

    # 触发条件 2：长间隔强制
    force_every = int(_cfg("memory_force_every", DEFAULT_FORCE_EVERY))
    msgs_since = _count_user_messages_since_last_recall(messages)
    if force_every > 0 and msgs_since >= force_every:
        logger.info("Memory force-trigger: %d user msgs since last recall (≥%d)", msgs_since, force_every)
        return True, latest

    # 触发条件 1：语义跳变（带两级省外呼短路，见上方注释）
    threshold = float(_cfg("memory_new_topic_threshold", DEFAULT_TOPIC_THRESHOLD))
    pair_key = (latest, prev)
    cached = _topic_check_cache.get(pair_key)
    if cached is not None:
        return cached

    skip_thr = max(0.9, threshold + 0.2)
    if _bigram_cosine(latest, prev) >= skip_thr:
        logger.info("Memory new-topic check: bigram shortcut (near-identical msgs) → continue")
        result = (False, latest)
    else:
        emb_new, emb_old = await asyncio.gather(_embed(latest, http), _embed(prev, http))
        if emb_new is None or emb_old is None:
            logger.info("Memory: embedding unavailable, skipping recall")
            return False, latest   # 失败结果不进缓存，下次重试
        sim = _cosine(emb_new, emb_old)
        is_new = sim < threshold
        logger.info("Memory new-topic check: sim=%.3f threshold=%.3f msgs_since_recall=%d → %s",
                    sim, threshold, msgs_since, "NEW" if is_new else "continue")
        result = (is_new, latest)

    if len(_topic_check_cache) >= _TOPIC_CHECK_CACHE_MAX:
        _topic_check_cache.clear()   # 个人网关量级：简单清空即可
    _topic_check_cache[pair_key] = result
    return result


# ────────────────────────────────────────────────────────────────
# Ombre breath 调用（MCP streamable-http）
# ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _ombre_session():
    """与 ombre MCP 建立短期 session。"""
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError:
        logger.warning("mcp library not installed")
        yield None
        return

    url   = _cfg("ombre_url",   DEFAULT_OMBRE_URL)
    token = _cfg("ombre_token", DEFAULT_OMBRE_TOKEN)
    headers = {"X-Brain-Token": token} if token else {}

    try:
        async with streamablehttp_client(url, headers=headers) as (read, write, _):
            async with ClientSession(read, write) as sess:
                await sess.initialize()
                yield sess
    except Exception:
        logger.exception("Ombre MCP session failed")
        yield None


async def query_ombre_breath(query: str) -> list[dict]:
    """调 ombre.breath，返回检索到的记忆列表。"""
    if not query:
        return []
    async with _ombre_session() as sess:
        if sess is None:
            return []
        try:
            result = await sess.call_tool("breath", {
                "query": query,
                "max_results": int(_cfg("memory_breath_max_results", DEFAULT_BREATH_MAX_RESULTS)),
                "max_tokens":  int(_cfg("memory_breath_max_tokens", DEFAULT_BREATH_MAX_TOKENS)),
            })
            # FastMCP 返回的 result.content 是 list[TextContent]
            blocks = result.content or []
            return _parse_breath_output(blocks)
        except Exception:
            logger.exception("Ombre breath call failed")
            return []


def _parse_breath_output(blocks: list) -> list[dict]:
    """从 breath 返回的 TextContent 列表里解析出记忆条目。
    ombre 的 breath 输出格式是 markdown 文本，每条记忆以标题分隔。
    我们尽量解析；解析不出来就当一整块塞进去。"""
    items: list[dict] = []
    for blk in blocks:
        text = getattr(blk, "text", str(blk))
        if not text or not text.strip():
            continue
        # 按 markdown 标题切分
        parts = re.split(r"\n(?=##? )", text)
        for p in parts:
            p = p.strip()
            if not p or len(p) < 10:
                continue
            # 提取标题作为 name
            name_m = re.match(r"^##? (.+?)\n", p)
            name = name_m.group(1).strip() if name_m else "(no title)"
            items.append({
                "source": "breath",
                "name": name,
                "content": p,
            })
    return items


# ────────────────────────────────────────────────────────────────
# 随机注入：直接读 buckets/dynamic/**/*.md
# ────────────────────────────────────────────────────────────────

async def random_inject_from_dynamic(k: int = 2) -> list[dict]:
    """从 dynamic 桶里随机抽 k 条，排除 resolved。"""
    base = Path(_cfg("ombre_buckets_dir", DEFAULT_BUCKETS_DIR)) / "dynamic"
    if not base.exists():
        return []

    def _scan():
        candidates = []
        for path in base.rglob("*.md"):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                meta, body = _parse_frontmatter(text)
                if meta.get("resolved"):
                    continue
                if meta.get("digested"):
                    continue
                candidates.append({
                    "source": "random",
                    "id":      meta.get("id", path.stem),
                    "name":    meta.get("name", path.stem),
                    "content": body[:600],  # 截断避免太长
                    "importance": meta.get("importance", 5),
                })
            except Exception:
                continue
        return candidates

    all_candidates = await asyncio.to_thread(_scan)
    if not all_candidates:
        return []
    return random.sample(all_candidates, min(k, len(all_candidates)))


# ────────────────────────────────────────────────────────────────
# 合并去重 + 截断
# ────────────────────────────────────────────────────────────────

def _dedupe_and_cap(items: list[dict], cap: int) -> list[dict]:
    seen_names = set()
    seen_content_prefix = set()
    out = []
    for it in items:
        name = (it.get("name") or "").strip()
        prefix = (it.get("content") or "")[:80]
        if name and name in seen_names:
            continue
        if prefix and prefix in seen_content_prefix:
            continue
        if name:
            seen_names.add(name)
        if prefix:
            seen_content_prefix.add(prefix)
        out.append(it)
        if len(out) >= cap:
            break
    return out


# ────────────────────────────────────────────────────────────────
# 注入到消息列表
# ────────────────────────────────────────────────────────────────

def _format_recall_text(items: list[dict]) -> str:
    # 用明确的 [系统注入] 标记 + 自然语言说明，让 AI 知道这是网关自动召回的记忆
    # 而不是用户主动发的消息（防止 AI 回应"你想起这些事"这种话）
    lines = [
        "[系统自动召回 · 非用户消息]",
        "网关从你的长期记忆库（Ombre）按当前话题语义检索到以下相关记忆。",
        "用户没有发这段，是你的「想起来」。请把它当作脑海中浮现的画面继续对话，",
        "不要回复或讨论这段本身。",
        "",
        "你想起了这些事：",
    ]
    for i, it in enumerate(items, 1):
        name = it.get("name", "")
        content = it.get("content", "").strip()
        # content 已经可能含 markdown 标题，去掉重复的
        if content.startswith(f"## {name}") or content.startswith(f"# {name}"):
            content = content.split("\n", 1)[-1].strip() if "\n" in content else ""
        tag = " (随机抽取)" if it.get("source") == "random" else ""
        lines.append(f"\n{i}. {name}{tag}\n{content}")
    return "\n".join(lines)


def _is_tool_result_message(msg: dict) -> bool:
    """该 user 消息是否纯粹是 tool_result（属于 tool 调用链路，不能在前面插东西）。"""
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    if not content:
        return False
    return all(
        isinstance(b, dict) and b.get("type") == "tool_result"
        for b in content
    )


def inject_into_messages(messages: list, recalled: list[dict]) -> list:
    """在「最新一条真实 user 消息之前」注入一对 user/assistant 提供回忆。
    跳过 tool_result 消息——它们必须紧跟在 tool_use 后面，中间不能插东西。"""
    if not recalled:
        return messages
    msgs = list(messages)
    # 从后往前找最后一条「真实 user 消息」（非 tool_result）
    target_idx = -1
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.get("role") != "user":
            continue
        if _is_tool_result_message(m):
            continue  # 跳过 tool_result
        target_idx = i
        break
    if target_idx < 0:
        # 没找到合适的位置（极端情况），不注入更安全
        logger.info("Memory inject: no real user message slot found, skipping")
        return messages
    text = _format_recall_text(recalled)
    recall_user = {"role": "user", "content": [{"type": "text", "text": text}]}
    # assistant 确认收到了这些"想起来的"，自然过渡到下一句用户消息
    recall_asst = {"role": "assistant", "content": "（脑海中浮现这些画面。）"}
    return msgs[:target_idx] + [recall_user, recall_asst] + msgs[target_idx:]


# ────────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────────

async def maybe_recall_memories(messages: list, tag: str = "") -> list:
    """主入口：判断是否新话题，决定是否检索 + 注入。
    返回（可能被修改过的）messages。
    总时间预算：DEFAULT 8 秒；超时直接放弃注入，让主请求继续走。"""
    # 关闭开关
    if not _cfg("memory_recall_enabled", True):
        return messages
    # 技术频道跳过
    if tag:
        return messages
    # 工具调用链路中跳过：最后一条是 tool_result 意味着 AI 正在用工具
    # 这一轮不能被记忆注入打断（会破坏 tool_use → tool_result 链）
    if messages and messages[-1].get("role") == "user" and _is_tool_result_message(messages[-1]):
        logger.info("Memory recall: skipping (current turn is tool_result)")
        return messages
    # 已经注入过的检测：扫整个 messages，看是否有系统注入标记
    # （兼容新旧两种格式）
    _RECALL_MARKERS = ("[系统自动召回", "[你想起了这些事]")
    for m in messages[-6:]:  # 只看最后几条，避免历史里的旧注入误判
        if m.get("role") != "user":
            continue
        c = m.get("content", "")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    t = b.get("text", "")
                    if any(t.startswith(p) for p in _RECALL_MARKERS):
                        return messages
        elif isinstance(c, str) and any(c.startswith(p) for p in _RECALL_MARKERS):
            return messages

    budget = float(_cfg("memory_recall_timeout", 10.0))

    try:
        return await asyncio.wait_for(
            _do_recall(messages),
            timeout=budget,
        )
    except asyncio.TimeoutError:
        logger.warning("Memory recall timed out (>%.1fs), skipping injection", budget)
        return messages
    except Exception:
        logger.exception("Memory recall failed (non-fatal)")
        return messages


async def _do_recall(messages: list) -> list:
    """实际检索逻辑，被 maybe_recall_memories 用 wait_for 包住做超时保护。
    breath 有自己的短超时，超时则只用 random（仍然能注入），避免整体一无所获。"""
    http = get_client()
    is_new, query = await is_new_topic(messages, http)
    if not is_new:
        return messages

    logger.info("Memory recall: triggered for query=%r", query[:80])

    breath_budget = float(_cfg("memory_breath_timeout", 6.0))

    async def _safe_breath():
        try:
            return await asyncio.wait_for(
                query_ombre_breath(query), timeout=breath_budget
            )
        except asyncio.TimeoutError:
            logger.warning("ombre.breath timed out (>%.1fs), falling back to random only", breath_budget)
            return []
        except Exception:
            logger.exception("ombre.breath failed")
            return []

    # 并行：breath（带短超时）+ random（极快，本地文件）
    breath_items, random_items = await asyncio.gather(
        _safe_breath(),
        random_inject_from_dynamic(int(_cfg("memory_random_k", DEFAULT_RANDOM_K))),
    )

    candidates = list(breath_items) + list(random_items)
    if not candidates:
        logger.info("Memory recall: no candidates")
        return messages

    cap = int(_cfg("memory_max_recall", DEFAULT_MAX_RECALL))
    final = _dedupe_and_cap(candidates, cap)
    logger.info("Memory recall: injecting %d items (breath=%d random=%d)",
                len(final), len(breath_items), len(random_items))
    return inject_into_messages(messages, final)


# ────────────────────────────────────────────────────────────────
# Phase 2：自动喂记忆
# ────────────────────────────────────────────────────────────────

async def feed_to_ombre(text: str, importance: int = 5) -> bool:
    """调 ombre.hold 存一条记忆。"""
    async with _ombre_session() as sess:
        if sess is None:
            return False
        try:
            await sess.call_tool("hold", {
                "content": text,
                "importance": int(importance),
            })
            return True
        except Exception:
            logger.exception("Ombre hold failed")
            return False


async def feed_pair_to_ombre(
    fingerprint: str,
    user_content: str,
    assistant_content: str,
    tag: str = "",
):
    """Phase 2 v2（archive 时机触发）：把这一轮真实的 user+assistant 对喂给 ombre。

    优势 vs 旧版 chunk_and_feed_recent：
    - 输入是 archive() 拿到的真实新对话内容，不含 Seamless 注入污染
    - 用 chunk_hash 去重，重复 archive / re-roll 不会重复喂
    - 技术频道（tag != ""）跳过
    """
    if not _cfg("memory_autofeed_enabled", True):
        return
    # 技术频道不喂
    if tag:
        return
    # 用户消息或助手回复任一为空 → 跳过
    user_content = (user_content or "").strip()
    assistant_content = (assistant_content or "").strip()
    if len(user_content) < 10 or len(assistant_content) < 10:
        return
    # 内容含网关注入标记 → 跳过（防止 archive 异常情况下喂到注入内容）
    skip_markers = (
        "[系统自动召回",
        "[以下是我们之前对话的摘要",
        "[你想起了这些事]",
        "[网关主动唤起",
    )
    if any(user_content.startswith(m) for m in skip_markers):
        logger.info("Memory feed_pair: skip (user starts with injection marker)")
        return

    # 算 chunk hash 去重
    # hash 只含 fingerprint + user_content，不含 assistant_content。
    # 否则 re-roll 时 assistant 变 → hash 变 → dedup 失效 → ombre 出现重复桶。
    import hashlib
    _fp = fingerprint or "no_fp"
    _sep = chr(10) + "---" + chr(10)
    payload = _fp + _sep + user_content[:300]
    chunk_hash = hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]

    try:
        from gateway import db as _db
        if await _db.is_chunk_already_fed(chunk_hash):
            logger.info("Memory feed_pair: skip (already fed) hash=%s", chunk_hash)
            return
    except Exception:
        logger.exception("Memory feed_pair: dedup check failed (continuing)")

    # 拼成 ombre 喜欢的格式
    feed_text = (
        f"【用户】{user_content[:800]}\n\n"
        f"【助手】{assistant_content[:1200]}"
    )

    ok = await feed_to_ombre(feed_text)
    logger.info("Memory feed_pair fp=%s hash=%s ok=%s len=%d",
                fingerprint[:8], chunk_hash, ok, len(feed_text))

    if ok:
        try:
            from gateway import db as _db
            await _db.mark_chunk_fed(chunk_hash, fingerprint)
        except Exception:
            logger.exception("Memory feed_pair: mark_chunk_fed failed (non-fatal)")


# ── 已废弃 ─────────────────────────────────────────────────────
# chunk_and_feed_recent 的旧实现：从 BP2 rolling 时机喂，从 messages 切片取数据。
# 缺陷：messages 含 Seamless 注入的旧对话，被反复喂导致 ombre 出现 135 个重复桶。
# 新做法：archive 时机触发 feed_pair_to_ombre（见上）。
# 保留这个函数仅为向后兼容（如果还有别处调用），实际逻辑等同 no-op。
async def chunk_and_feed_recent(text: str):
    """⚠️ 已废弃，不再喂记忆。改用 feed_pair_to_ombre（在 archiver 里触发）。"""
    logger.debug("chunk_and_feed_recent called (no-op, deprecated)")
    return
