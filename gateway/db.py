import aiosqlite
from pathlib import Path
from gateway.config import DB_PATH

_db: aiosqlite.Connection | None = None


async def init_db():
    global _db
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    _db = await aiosqlite.connect(DB_PATH)
    _db.row_factory = aiosqlite.Row
    await _db.execute("PRAGMA journal_mode=WAL")
    await _db.execute("PRAGMA busy_timeout=5000")
    await _db.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            model TEXT,
            tokens_in INTEGER DEFAULT 0,
            tokens_out INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            upstream_name TEXT,
            raw_request TEXT,
            raw_response TEXT,
            api_format TEXT DEFAULT 'anthropic',
            duration_ms INTEGER DEFAULT 0
        )
    """)
    await _db.execute("""
        CREATE TABLE IF NOT EXISTS context_summaries (
            conv_fingerprint TEXT NOT NULL,
            freeze_cycle     INTEGER NOT NULL,
            summary          TEXT NOT NULL,
            model            TEXT DEFAULT '',
            created_at       TEXT NOT NULL,
            PRIMARY KEY (conv_fingerprint, freeze_cycle)
        )
    """)

    # 迁移：旧数据库补字段
    for col, definition in [
        ("cache_write_tokens", "INTEGER DEFAULT 0"),
        ("cache_read_tokens",  "INTEGER DEFAULT 0"),
    ]:
        try:
            await _db.execute(f"ALTER TABLE conversations ADD COLUMN {col} {definition}")
        except Exception:
            pass

    # context_summaries: 加 messages_covered 字段（用消息覆盖范围做主索引，
    # 而不是 freeze_cycle —— 这样 cycle_size 改变后老摘要也能被找到）
    try:
        await _db.execute("ALTER TABLE context_summaries ADD COLUMN messages_covered INTEGER DEFAULT 0")
    except Exception:
        pass

    # conversations: 加 tag 字段，用于区分日常/技术等不同对话流
    try:
        await _db.execute("ALTER TABLE conversations ADD COLUMN tag TEXT DEFAULT ''")
    except Exception:
        pass
    # Re-roll detection: hash of messages[:-1] (history excluding latest user msg)
    # + fingerprint to scope detection per-conversation
    try:
        await _db.execute("ALTER TABLE conversations ADD COLUMN fingerprint TEXT DEFAULT ''")
    except Exception:
        pass
    try:
        await _db.execute("ALTER TABLE conversations ADD COLUMN history_hash TEXT DEFAULT ''")
    except Exception:
        pass
    # client_model: 客户端请求的模型名（带中转站前缀，如 [aws量] claude-opus-4-6-thinking）
    # 跟 model 字段区分（后者是上游返回的标准化名字）
    try:
        await _db.execute("ALTER TABLE conversations ADD COLUMN client_model TEXT DEFAULT ''")
    except Exception:
        pass
    # context_fingerprint: 共享摘要上下文；fingerprint 仍表示当前窗口/session
    try:
        await _db.execute("ALTER TABLE conversations ADD COLUMN context_fingerprint TEXT DEFAULT ''")
    except Exception:
        pass
    await _db.execute(
        "CREATE INDEX IF NOT EXISTS idx_reroll ON conversations(fingerprint, history_hash, timestamp)"
    )
    await _db.execute(
        "CREATE INDEX IF NOT EXISTS idx_context_fp ON conversations(context_fingerprint, timestamp)"
    )
    await _db.execute(
        "CREATE INDEX IF NOT EXISTS idx_summary_position ON context_summaries(conv_fingerprint, messages_covered)"
    )

    # One-time cleanup 2026-06-05:
    # Remove dirty summaries generated for the new session before seamless
    # handoff used inherited BP2 as initial_prev_summary. Keep old fp summaries.
    await _db.execute(
        """DELETE FROM context_summaries
           WHERE conv_fingerprint = 'fp_db95d3c44eb20d4c'
             AND messages_covered IN (48, 96)
             AND created_at < '2026-06-05T10:30:00'"""
    )
    # cost_usd / saved_usd: 归档时算好存入，避免改价后历史数据失真
    for col, definition in [
        ("cost_usd",  "REAL"),
        ("saved_usd", "REAL"),
    ]:
        try:
            await _db.execute(f"ALTER TABLE conversations ADD COLUMN {col} {definition}")
        except Exception:
            pass

    await _db.execute(
        "CREATE INDEX IF NOT EXISTS idx_conv_id ON conversations(conversation_id)"
    )
    await _db.execute(
        "CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)"
    )
    # Phase 2 自动喂记忆去重表：每对 user+assistant 算 hash，喂过的不再喂
    await _db.execute("""
        CREATE TABLE IF NOT EXISTS memory_fed_chunks (
            chunk_hash       TEXT PRIMARY KEY,
            conv_fingerprint TEXT NOT NULL,
            fed_at           TEXT NOT NULL
        )
    """)
    await _db.execute(
        "CREATE INDEX IF NOT EXISTS idx_fed_fp ON memory_fed_chunks(conv_fingerprint, fed_at)"
    )
    await _db.execute("""
        CREATE TABLE IF NOT EXISTS context_session_progress (
            context_fingerprint TEXT NOT NULL,
            session_fingerprint TEXT NOT NULL,
            messages_covered    INTEGER DEFAULT 0,
            updated_at          TEXT NOT NULL,
            PRIMARY KEY (context_fingerprint, session_fingerprint)
        )
    """)
    await _db.commit()


async def close_db():
    global _db
    if _db:
        await _db.close()
        _db = None


def get_db() -> aiosqlite.Connection:
    if _db is None:
        raise RuntimeError("Database not initialized")
    return _db


async def save_conversation(
    conversation_id: str,
    role: str,
    content: str,
    model: str = "",
    tokens_in: int = 0,
    tokens_out: int = 0,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    upstream_name: str = "",
    raw_request: str = "",
    raw_response: str = "",
    api_format: str = "anthropic",
    duration_ms: int = 0,
    timestamp: str = "",
    tag: str = "",
    fingerprint: str = "",
    history_hash: str = "",
    client_model: str = "",
    context_fingerprint: str = "",
    cost_usd: float | None = None,
    saved_usd: float | None = None,
):
    db = get_db()
    cursor = await db.execute(
        """INSERT INTO conversations
        (conversation_id, timestamp, role, content, model, tokens_in, tokens_out,
         cache_write_tokens, cache_read_tokens,
         upstream_name, raw_request, raw_response, api_format, duration_ms,
         tag, fingerprint, history_hash, client_model, context_fingerprint,
         cost_usd, saved_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            conversation_id, timestamp, role, content, model,
            tokens_in, tokens_out, cache_write_tokens, cache_read_tokens,
            upstream_name, raw_request, raw_response, api_format, duration_ms,
            tag, fingerprint, history_hash, client_model, context_fingerprint,
            cost_usd, saved_usd,
        ),
    )
    await db.commit()
    # 返回插入行的自增 id（= 全局时间线序号）。统一大脑的归档水位线需要它；
    # 现有调用方都忽略返回值，完全向后兼容。
    return cursor.lastrowid


async def find_and_delete_reroll(fingerprint: str, history_hash: str,
                                  tag: str = "", within_minutes: int = 30) -> int:
    """Detect re-roll: rows with same (fingerprint, history_hash, tag) within time window
    represent a previous attempt at the same turn. Delete them.
    Returns the number of rows deleted."""
    if not fingerprint or not history_hash:
        return 0
    from datetime import datetime, timedelta, timezone
    db = get_db()
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=within_minutes)).isoformat()
    # Find the conversation_ids of the predecessor pairs
    cursor = await db.execute(
        """SELECT DISTINCT conversation_id FROM conversations
           WHERE fingerprint=? AND history_hash=?
             AND COALESCE(tag,'')=COALESCE(?,'')
             AND timestamp >= ?""",
        (fingerprint, history_hash, tag, cutoff),
    )
    rows = await cursor.fetchall()
    if not rows:
        return 0
    conv_ids = [r[0] for r in rows]
    placeholders = ",".join("?" * len(conv_ids))
    cur2 = await db.execute(
        f"DELETE FROM conversations WHERE conversation_id IN ({placeholders})",
        conv_ids,
    )
    await db.commit()
    return cur2.rowcount


async def get_cache_stats(days: int = 7):
    """Per-day token and cache stats for the monitoring page."""
    db = get_db()
    cursor = await db.execute("""
        SELECT
            date(timestamp) as day,
            SUM(tokens_in)          as input_tokens,
            SUM(tokens_out)         as output_tokens,
            SUM(cache_write_tokens) as cache_write,
            SUM(cache_read_tokens)  as cache_read,
            COUNT(CASE WHEN role='assistant' THEN 1 END) as messages
        FROM conversations
        WHERE timestamp >= date('now', ?, 'localtime')
        GROUP BY day
        ORDER BY day DESC
    """, (f"-{days} days",))
    return await cursor.fetchall()


async def get_cache_stats_by_day_and_model(days: int = 7):
    """Per-day, per-(upstream, client_model) token and cache stats.
    Used to compute accurate daily costs when different models / relays
    with different prices are mixed."""
    db = get_db()
    cursor = await db.execute("""
        SELECT
            date(timestamp) as day,
            COALESCE(NULLIF(upstream_name, ''), '(unknown)') as upstream,
            COALESCE(NULLIF(client_model, ''), NULLIF(model, ''), '(unknown)') as model,
            SUM(tokens_in)          as input_tokens,
            SUM(tokens_out)         as output_tokens,
            SUM(cache_write_tokens) as cache_write,
            SUM(cache_read_tokens)  as cache_read,
            COUNT(*) as messages,
            SUM(COALESCE(cost_usd, 0))  as stored_cost,
            SUM(COALESCE(saved_usd, 0)) as stored_saved
        FROM conversations
        WHERE timestamp >= date('now', ?, 'localtime')
          AND role = 'assistant'
        GROUP BY day, upstream, model
        ORDER BY day DESC, messages DESC
    """, (f"-{days} days",))
    return await cursor.fetchall()


async def get_cache_stats_by_model(days: int = 7):
    """Per-(upstream, client_model) token and cache stats.
    Groups by (upstream_name, client_model) so the same model from different
    relay stations is counted separately (different prices/caches)."""
    db = get_db()
    cursor = await db.execute("""
        SELECT
            COALESCE(NULLIF(upstream_name, ''), '(unknown)') as upstream,
            COALESCE(NULLIF(client_model, ''), NULLIF(model, ''), '(unknown)') as display_model,
            COALESCE(NULLIF(model, ''), '(unknown)') as upstream_model,
            SUM(tokens_in)          as input_tokens,
            SUM(tokens_out)         as output_tokens,
            SUM(cache_write_tokens) as cache_write,
            SUM(cache_read_tokens)  as cache_read,
            COUNT(*) as messages,
            MAX(timestamp)          as last_used,
            SUM(COALESCE(cost_usd, 0))  as stored_cost,
            SUM(COALESCE(saved_usd, 0)) as stored_saved
        FROM conversations
        WHERE timestamp >= date('now', ?, 'localtime')
          AND role = 'assistant'
        GROUP BY upstream, display_model, upstream_model
        ORDER BY last_used DESC
    """, (f"-{days} days",))
    return await cursor.fetchall()


async def get_conversations(page: int = 1, per_page: int = 20, search: str = ""):
    db = get_db()
    offset = (page - 1) * per_page
    if search:
        count_sql = """
            SELECT COUNT(DISTINCT conversation_id) as cnt
            FROM conversations WHERE content LIKE ?
        """
        rows_sql = """
            SELECT conversation_id,
                   MIN(timestamp) as first_time,
                   MAX(timestamp) as last_time,
                   COUNT(*) as msg_count,
                   MAX(model) as model
            FROM conversations
            WHERE content LIKE ?
            GROUP BY conversation_id
            ORDER BY last_time DESC
            LIMIT ? OFFSET ?
        """
        param = f"%{search}%"
        cursor = await db.execute(count_sql, (param,))
        total = (await cursor.fetchone())[0]
        cursor = await db.execute(rows_sql, (param, per_page, offset))
    else:
        count_sql = "SELECT COUNT(DISTINCT conversation_id) as cnt FROM conversations"
        rows_sql = """
            SELECT conversation_id,
                   MIN(timestamp) as first_time,
                   MAX(timestamp) as last_time,
                   COUNT(*) as msg_count,
                   MAX(model) as model
            FROM conversations
            GROUP BY conversation_id
            ORDER BY last_time DESC
            LIMIT ? OFFSET ?
        """
        cursor = await db.execute(count_sql)
        total = (await cursor.fetchone())[0]
        cursor = await db.execute(rows_sql, (per_page, offset))
    rows = await cursor.fetchall()
    return rows, total


async def get_conversation_messages(conversation_id: str):
    db = get_db()
    cursor = await db.execute(
        """SELECT * FROM conversations
        WHERE conversation_id = ?
        ORDER BY id ASC""",
        (conversation_id,),
    )
    return await cursor.fetchall()


async def get_stats():
    db = get_db()
    cursor = await db.execute("SELECT COUNT(*) FROM conversations")
    total = (await cursor.fetchone())[0]
    cursor = await db.execute(
        "SELECT COUNT(*) FROM conversations WHERE timestamp >= date('now')"
    )
    today = (await cursor.fetchone())[0]
    cursor = await db.execute(
        "SELECT COUNT(DISTINCT conversation_id) FROM conversations"
    )
    conv_count = (await cursor.fetchone())[0]
    cursor = await db.execute(
        "SELECT COALESCE(SUM(tokens_in), 0), COALESCE(SUM(tokens_out), 0) FROM conversations"
    )
    row = await cursor.fetchone()
    return {
        "total_messages": total,
        "today_messages": today,
        "total_conversations": conv_count,
        "total_tokens_in": row[0],
        "total_tokens_out": row[1],
    }


async def get_summary_at_or_before(conv_fingerprint: str, target_position: int):
    """Return (summary_text, messages_covered) for the most recent summary that
    covers up to ≤ target_position messages. (None, 0) if no summary exists."""
    db = get_db()
    cursor = await db.execute(
        """SELECT summary, messages_covered FROM context_summaries
           WHERE conv_fingerprint=? AND messages_covered<=? AND messages_covered>0
           ORDER BY messages_covered DESC LIMIT 1""",
        (conv_fingerprint, target_position),
    )
    row = await cursor.fetchone()
    if row:
        return row["summary"], row["messages_covered"]
    return None, 0


async def save_summary_at(conv_fingerprint: str, messages_covered: int,
                          summary: str, model: str = "", freeze_cycle: int = 0):
    from datetime import datetime
    db = get_db()
    # The legacy primary key is (fingerprint, freeze_cycle), but modern lookup is
    # by messages_covered. When chunk sizes change, a newly computed cycle can
    # collide with an old row that covers a different position. Reuse the row for
    # the same coverage if present; otherwise pick a fresh cycle on collision.
    cursor = await db.execute(
        """SELECT freeze_cycle FROM context_summaries
           WHERE conv_fingerprint=? AND messages_covered=? LIMIT 1""",
        (conv_fingerprint, messages_covered),
    )
    existing_same_pos = await cursor.fetchone()
    if existing_same_pos:
        freeze_cycle = existing_same_pos["freeze_cycle"]
    else:
        cursor = await db.execute(
            """SELECT messages_covered FROM context_summaries
               WHERE conv_fingerprint=? AND freeze_cycle=? LIMIT 1""",
            (conv_fingerprint, freeze_cycle),
        )
        existing_cycle = await cursor.fetchone()
        if existing_cycle and int(existing_cycle["messages_covered"] or 0) != int(messages_covered):
            cursor = await db.execute(
                """SELECT COALESCE(MAX(freeze_cycle), 0) + 1 AS next_cycle
                   FROM context_summaries WHERE conv_fingerprint=?""",
                (conv_fingerprint,),
            )
            row = await cursor.fetchone()
            freeze_cycle = int(row["next_cycle"] or 1)
    # Use UPSERT keyed on (fingerprint, messages_covered): same coverage replaces
    await db.execute(
        """INSERT INTO context_summaries
           (conv_fingerprint, freeze_cycle, messages_covered, summary, model, created_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(conv_fingerprint, freeze_cycle) DO UPDATE SET
             summary=excluded.summary,
             messages_covered=excluded.messages_covered,
             model=excluded.model,
             created_at=excluded.created_at""",
        (conv_fingerprint, freeze_cycle, messages_covered, summary, model,
         datetime.utcnow().isoformat()),
    )
    await db.commit()


# Legacy compatibility (still used by some callers)
async def get_context_summary(conv_fingerprint: str, cycle: int) -> str | None:
    db = get_db()
    cursor = await db.execute(
        "SELECT summary FROM context_summaries WHERE conv_fingerprint=? AND freeze_cycle=?",
        (conv_fingerprint, cycle),
    )
    row = await cursor.fetchone()
    return row[0] if row else None


async def save_context_summary(conv_fingerprint: str, cycle: int, summary: str, model: str = ""):
    """Legacy: caller doesn't know messages_covered. Stores with messages_covered=0
    which makes it unfindable by new lookup — use save_summary_at instead."""
    from datetime import datetime
    db = get_db()
    await db.execute(
        """INSERT OR REPLACE INTO context_summaries
           (conv_fingerprint, freeze_cycle, summary, model, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (conv_fingerprint, cycle, summary, model, datetime.utcnow().isoformat()),
    )
    await db.commit()


async def get_recent_archived_turns(
    limit: int = 40,
    tag: str = "",
    context_fingerprint: str = "",
    exclude_fingerprint: str = "",
):
    """Return up to `limit` most recent archived rows (with given tag) in chronological order."""
    db = get_db()
    conditions = [
        "content IS NOT NULL",
        "content != ''",
        "COALESCE(tag, '') = ?",
    ]
    params = [tag]
    if context_fingerprint:
        conditions.append("COALESCE(context_fingerprint, '') = ?")
        params.append(context_fingerprint)
    if exclude_fingerprint:
        conditions.append("COALESCE(fingerprint, '') != ?")
        params.append(exclude_fingerprint)
    params.append(limit)
    cursor = await db.execute(
        f"""SELECT role, content, timestamp FROM conversations
            WHERE {' AND '.join(conditions)}
            ORDER BY id DESC LIMIT ?""",
        tuple(params),
    )
    rows = await cursor.fetchall()
    return list(reversed(rows))


async def get_conversation_tag(conversation_id: str) -> str:
    """Return the tag of an existing conversation, or '' if not found."""
    db = get_db()
    cursor = await db.execute(
        "SELECT tag FROM conversations WHERE conversation_id=? LIMIT 1",
        (conversation_id,),
    )
    row = await cursor.fetchone()
    return (row[0] or "") if row else ""


async def is_chunk_already_fed(chunk_hash: str) -> bool:
    """Phase 2 dedup: 检查 chunk hash 是否已经喂过 ombre。"""
    db = get_db()
    cursor = await db.execute(
        "SELECT 1 FROM memory_fed_chunks WHERE chunk_hash=? LIMIT 1",
        (chunk_hash,),
    )
    row = await cursor.fetchone()
    return row is not None


async def mark_chunk_fed(chunk_hash: str, fingerprint: str):
    """Phase 2 dedup: 记录该 chunk hash 已喂过。"""
    from datetime import datetime, timezone
    db = get_db()
    await db.execute(
        "INSERT OR IGNORE INTO memory_fed_chunks (chunk_hash, conv_fingerprint, fed_at) VALUES (?, ?, ?)",
        (chunk_hash, fingerprint, datetime.now(timezone.utc).isoformat()),
    )
    await db.commit()


async def find_predecessor_summary(exclude_fingerprint: str, within_hours: int = 48):
    """寻找「前驱 session」：最近 N 小时内，不是当前 fp、有 BP2 摘要的其他 fp 中
    最新更新的那个的最大 messages_covered 摘要。
    用于：新 session 第一次触发 BP2 时，继承上一个 session 的最新摘要作为起点。
    返回 (summary_text, source_fingerprint, source_messages_covered) 或 (None, None, 0)。

    2026-06-09: 加 `LIKE 'fp_%'` 过滤——只继承 session 摘要，跳过共享池摘要 ctx_xxx。
    背景：codex 加共享 BP2 后 DB 混入 conv_fingerprint=ctx_xxx 记录，
    cold start inherit 会拉到一份跨 session 累积的"别人摘要"塞给当前 session，
    导致 AI 串味。详见 Issue #36。"""
    db = get_db()
    cursor = await db.execute("""
        SELECT cs.summary, cs.conv_fingerprint, cs.messages_covered, cs.created_at
        FROM context_summaries cs
        WHERE cs.conv_fingerprint != ?
          AND cs.conv_fingerprint LIKE 'fp_%'
          AND cs.created_at >= datetime('now', ?)
          AND cs.messages_covered > 0
        ORDER BY cs.created_at DESC, cs.messages_covered DESC
        LIMIT 1
    """, (exclude_fingerprint, f"-{within_hours} hours"))
    row = await cursor.fetchone()
    if not row:
        return None, None, 0
    return row["summary"], row["conv_fingerprint"], row["messages_covered"]


async def get_latest_summary(conv_fingerprint: str):
    """Return the latest summary for a fingerprint regardless of target position."""
    if not conv_fingerprint:
        return None, 0
    db = get_db()
    cursor = await db.execute(
        """SELECT summary, messages_covered FROM context_summaries
           WHERE conv_fingerprint=? AND messages_covered>0
           ORDER BY messages_covered DESC, freeze_cycle DESC LIMIT 1""",
        (conv_fingerprint,),
    )
    row = await cursor.fetchone()
    if not row:
        return None, 0
    return row["summary"], row["messages_covered"]


async def get_max_summary_position(conv_fingerprint: str) -> int:
    """返回该 fingerprint 下已保存摘要的最大 messages_covered。没有返回 0。"""
    db = get_db()
    cursor = await db.execute(
        """SELECT MAX(messages_covered) AS m FROM context_summaries
           WHERE conv_fingerprint=? AND messages_covered>0""",
        (conv_fingerprint,),
    )
    row = await cursor.fetchone()
    if not row or row["m"] is None:
        return 0
    return int(row["m"])


async def get_context_session_position(context_fingerprint: str, session_fingerprint: str) -> int:
    if not context_fingerprint or not session_fingerprint:
        return 0
    db = get_db()
    cursor = await db.execute(
        """SELECT messages_covered FROM context_session_progress
           WHERE context_fingerprint=? AND session_fingerprint=?""",
        (context_fingerprint, session_fingerprint),
    )
    row = await cursor.fetchone()
    if not row:
        return 0
    return int(row["messages_covered"] or 0)


async def update_context_session_position(
    context_fingerprint: str,
    session_fingerprint: str,
    messages_covered: int,
):
    if not context_fingerprint or not session_fingerprint:
        return
    from datetime import datetime
    db = get_db()
    await db.execute(
        """INSERT INTO context_session_progress
           (context_fingerprint, session_fingerprint, messages_covered, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(context_fingerprint, session_fingerprint) DO UPDATE SET
             messages_covered=MAX(messages_covered, excluded.messages_covered),
             updated_at=excluded.updated_at""",
        (context_fingerprint, session_fingerprint, max(0, int(messages_covered)), datetime.utcnow().isoformat()),
    )
    await db.commit()


async def list_context_summaries():
    db = get_db()
    cursor = await db.execute(
        """SELECT conv_fingerprint, freeze_cycle, messages_covered, summary, model, created_at
           FROM context_summaries
           ORDER BY conv_fingerprint ASC, messages_covered ASC, freeze_cycle ASC"""
    )
    return await cursor.fetchall()


# ════════════════════════════════════════════════════════════════
# 统一大脑（PLAN_UNIFIED_BRAIN.md 第 1 步）：全局时间线 + 水位线
#
# 设计要点（全部包在 gateway/settings.py 的 unified_brain_enabled 开关背后，
# 本文件这几个函数本身不读开关——是否调用由 hooks.py 决定）：
#
# 1. 不建新表存对话内容。"全局时间线"只是对现有 conversations 表的一个查询
#    视角：跨所有 fingerprint，按插入顺序（自增主键 id）排序。
#    用 id 而不是 timestamp 排序，是因为两个平台并发写入时 timestamp 可能相同
#    (同一秒)，而 id 严格递增，天然就是"全局到达顺序"，不会有并列歧义。
# 2. 水位线（每个 session 归档到了全局第几条）复用 context_session_progress 表，
#    用固定的 context_fingerprint='fp_global' 这一行区分"这是全局水位线"，
#    session_fingerprint 仍是各 session 自己的 fp_xxx。这样不用建新表，
#    也不会和「共享摘要上下文 context_session_progress」原有用法冲突——
#    原有用法的 context_fingerprint 是 ctx_xxx / daily-main 等，"fp_global"
#    是专门为统一大脑保留的命名空间，不会撞车。
# 3. 全局滚动摘要复用 context_summaries 表，conv_fingerprint 固定为 'fp_global'，
#    messages_covered = 全局序号（即 conversations.id）。
# ════════════════════════════════════════════════════════════════

UNIFIED_GLOBAL_CONTEXT_KEY = "fp_global"  # 水位线 context_fingerprint 命名空间
UNIFIED_GLOBAL_SUMMARY_FP = "fp_global"   # context_summaries 的全局摘要 fingerprint


async def get_global_timeline_count() -> int:
    """全局时间线当前总条数（= conversations 表里带来源标签的有效行数上限判断用）。
    直接用 MAX(id) 作为"全局序号上界"，比 COUNT(*) 更贴合"id 即序号"的设计。
    没有数据时返回 0。"""
    db = get_db()
    cursor = await db.execute("SELECT COALESCE(MAX(id), 0) AS m FROM conversations")
    row = await cursor.fetchone()
    return int(row["m"] or 0)


_RAW_PACK_PREFIX = "gz64:"
_RAW_PACK_CAP = 2_000_000  # 压缩前原文上限 2MB，防病态大请求撑爆行


def pack_raw(text: str) -> str:
    """raw_request/raw_response 全量存储（REFACTOR_ROADMAP P0.3）。

    旧做法 [:50000] 截断让 0712 排障两次抓瞎（关键部位全在截断线之后）。
    新做法：gzip + base64，前缀 "gz64:" 标记。90k token 的中文请求
    （~30 万字符）压后约 3-6 万字符，比截断版还小，且信息无损。
    读取用 unpack_raw()，兼容历史明文行（无前缀原样返回）。"""
    if not text:
        return text
    import gzip as _gzip
    import base64 as _b64
    raw = text[:_RAW_PACK_CAP].encode("utf-8")
    return _RAW_PACK_PREFIX + _b64.b64encode(_gzip.compress(raw, 6)).decode("ascii")


def unpack_raw(stored: str) -> str:
    """解包 pack_raw 存储的内容；历史明文行（截断版）原样返回。"""
    if not stored or not stored.startswith(_RAW_PACK_PREFIX):
        return stored or ""
    import gzip as _gzip
    import base64 as _b64
    try:
        return _gzip.decompress(_b64.b64decode(stored[len(_RAW_PACK_PREFIX):])).decode("utf-8")
    except Exception:
        return stored  # 损坏时原样返回，不抛错


async def get_last_archived_at(fingerprint: str):
    """该 session 最近一条归档行的时间戳（ISO 字符串），没有归档则 None。
    用于主动消息"形状 B"的冷场判定（hooks._detect_and_rewrite_repeated_proactive）。"""
    if not fingerprint:
        return None
    db = get_db()
    cursor = await db.execute(
        """SELECT timestamp FROM conversations
           WHERE fingerprint=? AND role IN ('user','assistant')
             AND content IS NOT NULL AND content != ''
           ORDER BY id DESC LIMIT 1""",
        (fingerprint,),
    )
    row = await cursor.fetchone()
    return row["timestamp"] if row else None


async def get_global_timeline_after(
    after_id: int = 0,
    limit: int = 0,
    exclude_tags: tuple[str, ...] = (),
):
    """跨所有 fingerprint、按全局到达顺序（id 升序）查询归档对话，带来源标签。

    Args:
        after_id: 只返回 id > after_id 的行（用于"摘要位置之后的尾巴"）
        limit:    最多返回多少行（0 = 不限制）；如果指定，取「最新的 limit 行」
                  再按时间正序返回（和 get_recent_archived_turns 语义一致）
        exclude_tags: 排除哪些 tag 的对话进入全局时间线。默认**不排除任何频道**
                      ——用户拍板"全部都进"（PLAN_UNIFIED_BRAIN.md 开头）；
                      如果以后技术频道冲淡生活记忆，再改由 settings 配置

    Returns: list of rows，每行含 id / role / content / timestamp / tag /
             fingerprint / context_fingerprint（可用来推断来源平台标签）
    """
    db = get_db()
    conditions = ["content IS NOT NULL", "content != ''", "id > ?"]
    params: list = [after_id]
    if exclude_tags:
        placeholders = ",".join("?" * len(exclude_tags))
        conditions.append(f"COALESCE(tag, '') NOT IN ({placeholders})")
        params.extend(exclude_tags)

    where = " AND ".join(conditions)
    if limit and limit > 0:
        # 先取最新 limit 行（按 id 降序），再翻回正序，语义等价于「时间线尾巴」
        cursor = await db.execute(
            f"""SELECT id, role, content, timestamp, tag, fingerprint, context_fingerprint
                FROM conversations
                WHERE {where}
                ORDER BY id DESC LIMIT ?""",
            (*params, limit),
        )
        rows = await cursor.fetchall()
        return list(reversed(rows))
    cursor = await db.execute(
        f"""SELECT id, role, content, timestamp, tag, fingerprint, context_fingerprint
            FROM conversations
            WHERE {where}
            ORDER BY id ASC""",
        tuple(params),
    )
    return await cursor.fetchall()


async def get_global_timeline_for_session(
    session_fingerprint: str,
    after_id: int = 0,
):
    """全局时间线里属于「某个 session」的那些行（用于增量识别/重roll/编辑检测对账）。
    按 id 升序返回，只含 role/content/id，不做 limit（对账需要看到该 session
    水位线之后的全部归档记录，不能截断）。

    TODO(性能，延后优化)：长期运行的 session 归档量大时，这里每次请求都
    全量加载该 session 的所有归档行做对账。等全局摘要节奏跑稳后，可以改成
    只加载"水位线附近 + 客户端历史长度 × 2"的窗口。当前先接受全量加载
    （个人网关流量小，fingerprint 有索引，可承受）。"""
    if not session_fingerprint:
        return []
    db = get_db()
    cursor = await db.execute(
        """SELECT id, role, content FROM conversations
           WHERE fingerprint=? AND id>?
             AND content IS NOT NULL AND content != ''
           ORDER BY id ASC""",
        (session_fingerprint, after_id),
    )
    return await cursor.fetchall()


async def get_global_watermark(session_fingerprint: str) -> int:
    """读取某 session 在全局时间线上的水位线（已归档到全局第几条 id）。没有则 0。"""
    return await get_context_session_position(UNIFIED_GLOBAL_CONTEXT_KEY, session_fingerprint)


async def update_global_watermark(session_fingerprint: str, global_id: int):
    """更新某 session 的全局水位线。内部用 MAX() 保证水位线只增不减
    （沿用 update_context_session_position 的 UPSERT 语义）。"""
    await update_context_session_position(UNIFIED_GLOBAL_CONTEXT_KEY, session_fingerprint, global_id)


async def force_set_global_watermark(session_fingerprint: str, global_id: int):
    """强制设置水位线（不走 MAX() 只增不减的保护）。
    用于「编辑历史/重roll 导致时间线上游数据被删除」之后，水位线需要**回退**
    到分叉点，这种情况不能用 update_global_watermark（它只会取更大值，回退不了）。"""
    if not session_fingerprint:
        return
    from datetime import datetime
    db = get_db()
    await db.execute(
        """INSERT INTO context_session_progress
           (context_fingerprint, session_fingerprint, messages_covered, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(context_fingerprint, session_fingerprint) DO UPDATE SET
             messages_covered=excluded.messages_covered,
             updated_at=excluded.updated_at""",
        (UNIFIED_GLOBAL_CONTEXT_KEY, session_fingerprint, max(0, int(global_id)), datetime.utcnow().isoformat()),
    )
    await db.commit()


async def delete_conversations_by_ids(ids: list[int]) -> int:
    """按主键 id 批量删除 conversations 行。用于编辑历史检测时"只删本 session
    分叉点之后的归档"——绝不能用 fingerprint 之外的条件误删别的 session。"""
    if not ids:
        return 0
    db = get_db()
    placeholders = ",".join("?" * len(ids))
    cursor = await db.execute(
        f"DELETE FROM conversations WHERE id IN ({placeholders})", ids
    )
    await db.commit()
    return cursor.rowcount


async def delete_conversations_after_id(session_fingerprint: str, after_id: int) -> int:
    """删除某 session 在全局时间线上、id > after_id 的所有归档行。
    仅限定 fingerprint=session_fingerprint，绝不会碰到别的 session 的数据。
    用于编辑历史（分叉）场景：分叉点之后的旧归档要整体撤下，等待重新入账。"""
    if not session_fingerprint:
        return 0
    db = get_db()
    cursor = await db.execute(
        "DELETE FROM conversations WHERE fingerprint=? AND id>?",
        (session_fingerprint, after_id),
    )
    await db.commit()
    return cursor.rowcount


async def get_global_summary():
    """返回全局滚动摘要 (summary_text, messages_covered)。没有则 (None, 0)。
    messages_covered 语义 = 全局序号（conversations.id 上界）。"""
    return await get_latest_summary(UNIFIED_GLOBAL_CONTEXT_KEY)


async def save_global_summary(messages_covered: int, summary: str, model: str = ""):
    """保存全局滚动摘要，messages_covered = 全局序号（conversations.id 上界）。"""
    await save_summary_at(UNIFIED_GLOBAL_CONTEXT_KEY, messages_covered, summary, model=model)
