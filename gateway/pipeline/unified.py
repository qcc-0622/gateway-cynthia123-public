"""统一大脑：全平台共享时间线对账 + 全局滚动摘要（P2.1，2026-09-06）。

自 hooks.py 绞杀搬迁，函数体零改动；hooks.py 留 import 转发。
总开关 unified_brain_enabled（settings），关闭时本模块全部不参与请求路径。
不 import hooks（防循环）；共享工具来自 pipeline.common。
"""

import asyncio
import logging

from gateway.pipeline.common import _TIME_REMINDER_RE, _content_text
from gateway.pipeline.common import _rebuild_locks
from gateway.pipeline.identity import _clean_context_id

logger = logging.getLogger("gateway.pipeline.unified")


def _bp2_with_summary(summary: str, tail_messages: list) -> list:
    bp2_user = {
        "role": "user",
        "content": [{
            "type": "text",
            "text": f"[以下是我们之前对话的摘要，请基于此继续]\n\n{summary}",
        }],
    }
    bp2_asst = {
        "role": "assistant",
        "content": "好的，我已了解之前的对话内容，继续。",
    }
    return [bp2_user, bp2_asst] + list(tail_messages)


_UNIFIED_ANCHOR_LEN = 5  # 平台标签兜底：没有 gateway_context_id 时用 fingerprint 前几位


def _unified_platform_label(context_id: str, session_fingerprint: str) -> str:
    """给一条消息打的"来源平台"标签。优先用显式 gateway_context_id
    （将来橘瓣配 X-Channel 自定义 header 会走到这里）；没有的话退化用
    session fingerprint 的前几位，保证不同 session 之间仍能在时间线里区分开，
    即使叫不出真实平台名字。"""
    cid = _clean_context_id(context_id)
    if cid:
        return cid
    fp = (session_fingerprint or "").replace("fp_", "")
    return f"session-{fp[:_UNIFIED_ANCHOR_LEN]}" if fp else "unknown"


def _unified_label_for_row(row) -> str:
    """从一条已归档行（sqlite Row，含 fingerprint / context_fingerprint）推断平台标签。"""
    ctx_fp = ""
    try:
        ctx_fp = row["context_fingerprint"] or ""
    except (IndexError, KeyError):
        ctx_fp = ""
    if ctx_fp:
        return ctx_fp
    try:
        fp = row["fingerprint"] or ""
    except (IndexError, KeyError):
        fp = ""
    fp = fp.replace("fp_", "")
    return f"session-{fp[:_UNIFIED_ANCHOR_LEN]}" if fp else "unknown"


def _tag_text_with_platform(label: str, text: str) -> str:
    """把平台标签内嵌进消息正文前面（用于喂给摘要模型 / 拼进尾巴时让 AI 能区分来源）。
    只加前缀，不改变原文内容，摘要 prompt 完全复用，不用改 summarizer.py。"""
    text = text or ""
    prefix = f"[{label}] "
    if text.startswith(prefix):
        return text  # 避免重复打标（如摘要 rebuild 对同一批消息重复调用）
    return prefix + text


def _unified_row_to_message(row, label: str | None = None) -> dict:
    """把一条 conversations 行转换成 messages 里的一条，正文带平台标签前缀。"""
    role = row["role"]
    content = row["content"] or ""
    lbl = label if label is not None else _unified_label_for_row(row)
    return {"role": role, "content": _tag_text_with_platform(lbl, content)}


def _unified_normalize_text(content) -> str:
    """把消息内容规约成可比较的纯文本：提取 text block、剥离 <time_reminder>、strip。

    ⚠️ 剥离 <time_reminder> 是必须的（和 detect_tag 的处理一致）：
    archiver 归档的 user_content 取自**预处理后**的 body（anthropic_proxy.py 先
    preprocess 再 forward，_append_to_last_user_anthropic 在管道第 24 步注入
    时间戳），所以已归档行的 user 正文**带** <time_reminder> 尾巴；而客户端
    下次发来的历史是它自己存的原文，**不带**注入。不剥离的话对账两边逐条
    失配，永远匹配不上。"""
    return _TIME_REMINDER_RE.sub("", _content_text(content)).strip()


def _unified_message_key(role: str, content) -> tuple:
    """把一条 message 规约成可比较的 (role, text) 元组，用于对账匹配。
    只取纯文本内容比较（tool_use/image 等复杂 block 极少出现在已归档的
    user/assistant 文本里，_extract_text 同样只关心 text），保持和
    archiver._compute_history_hash 类似的粒度但更宽容——只要文字对得上即可。"""
    return (role, _unified_normalize_text(content))


class UnifiedReconcileResult:
    """对账结果：增量消息 / 需要在时间线上删除的行 id / 新水位线。"""

    __slots__ = ("new_messages", "delete_ids", "new_watermark", "kind")

    def __init__(self, new_messages, delete_ids, new_watermark, kind):
        self.new_messages = new_messages      # list[dict]：本 session 尚未归档的增量消息（时间正序）
        self.delete_ids = delete_ids          # list[int]：需要从 conversations 删除的行 id（只属于本 session）
        self.new_watermark = new_watermark    # int：对账后应该写回的全局水位线（本 session 在时间线上的位置）
        self.kind = kind                      # "append" | "reroll" | "edit" | "noop"


def _common_prefix_len(client_keys: list[tuple], archived_keys: list[tuple]) -> int:
    """最长公共前缀长度 p：client_keys 和 archived_keys 从头开始逐条比较，
    直到第一个不同的位置为止。假设 client 的历史"从 session 一开始就延续
    自 archived"——常规追加 / 中途编辑分叉都满足这个假设。"""
    limit = min(len(client_keys), len(archived_keys))
    p = 0
    while p < limit and client_keys[p] == archived_keys[p]:
        p += 1
    return p


def _tail_truncated_len(client_keys: list[tuple], archived_keys: list[tuple]) -> int:
    """如果 client_keys 整体等于 archived_keys 的某段尾巴（橘瓣 contextMessageSize
    截断场景），返回 archived 里对应的重叠长度（= len(client_keys)）；否则返回 0。
    只有 client 完全被尾部吃掉、一条不多一条不少时才算数——这是"纯截断"，
    不含"截断之后又追加了新内容"的组合场景（那种场景由
    `_tail_truncated_then_appended` 单独处理）。"""
    m, n = len(client_keys), len(archived_keys)
    if m == 0 or m > n:
        return 0
    return m if archived_keys[n - m:] == client_keys else 0


def _tail_truncated_then_appended_split(client_keys: list[tuple], archived_keys: list[tuple]) -> int | None:
    """处理"截断尾巴 + 后面又追加了新内容"的组合场景（常见于长期使用
    contextMessageSize 的橘瓣 session）：client_keys 的**前面一段**恰好等于
    archived_keys 的尾巴，后面剩下的是真正的新内容。

    返回 L = 这一段重叠的长度（即 client_keys[:L] == archived_keys[n-L:]），
    没有找到有意义的重叠（L 必须 > 0 且严格小于 m，否则退化成纯追加/纯截断，
    由调用方前面的分支处理）时返回 None。

    从大到小尝试 L 是为了在有多种可能对齐时，优先选择"保留更多 archived 尾巴"
    的那个对齐（更符合"session 内容只会变多不会突然对不上"的直觉）。
    """
    m, n = len(client_keys), len(archived_keys)
    for L in range(min(m, n) - 1, 0, -1):  # L < m（留出至少 1 条新内容），L <= n
        if archived_keys[n - L:] == client_keys[:L]:
            return L
    return None


def reconcile_session_history(
    client_history: list,
    archived_rows: list,
    current_user_text: str = "",
) -> UnifiedReconcileResult:
    """核心对账算法（PLAN 第 3 步 + 二.5 第 2 条）。

    Args:
        client_history: 客户端发来的完整历史，**已去掉本轮正在提问的最后一条 user**
                         （即 messages[:-1]）。可能是完整历史，也可能因为橘瓣
                         `contextMessageSize` 被截断成尾巴，也可能是"截断尾巴 +
                         本 session 还没来得及归档的新一轮"的组合。
        archived_rows:  db.get_global_timeline_for_session() 返回的该 session
                         已归档行，按 id 升序，每行含 id / role / content。
        current_user_text: 本轮正在提问的最后一条 user 消息的规范化文本
                         （调用方传 _unified_normalize_text(messages[-1]["content"])）。
                         重roll 判定必须用它：橘瓣重roll 时 messages 最后一条就是
                         被重新问的那条 user，所以 client_history（= messages[:-1]）
                         里**不含**这条 user——只看 client_history 永远探测不到重roll。

    Returns: UnifiedReconcileResult

    五种判定依次尝试，顺序很重要（更具体/更安全的判断排在前面）：

    第一步・重roll 特判（放在纯截断检查之前）：
        条件全部满足才算 reroll：
          - n >= 2，archived 最后两条是 (user, assistant) 完整一轮
          - current_user_text == archived 倒数第二条（旧 user）的文本
          - client_history 能对齐到 archived[:n-2]：
              全量：client_keys == archived_keys[:n-2]
              截断：client_keys 等于 archived_keys[:n-2] 的尾巴（m < n-2）
              空：m == 0（首轮重roll，或截断到 0；统一写法
                  client_keys == archived_keys[n-2-m : n-2] 天然涵盖）
        → 删除 archived 最后**两条**（旧 user + 旧 assistant 都删！之后
          archiver 会重新归档 (user, assistant')，如果只删 assistant，
          这条 user 会在时间线上重复入账），水位线回退到 n-2。

        误判分析：用户"新发一条和上次问题一模一样的话"不会被误判——那种
        情况下 client_history 含有 archived[-1]（旧 assistant 还在客户端
        历史里），上面的对齐条件不成立，会落到后面的纯截断/追加分支。

    第二步・纯截断尾巴：
        client_history 整体等于 archived 的某段尾巴（一条不多一条不少）
        → **noop**：橘瓣 contextMessageSize 截断，archived 前面那部分虽然
        没出现在这次请求里，但依然有效，不能删除。

    第三步・"最长公共前缀"（识别常规追加 / 编辑分叉，假设 client 从 session
    开头延续自 archived）：
        p = client_history 和 archived 的最长公共前缀长度
        - p == n（archived 整体是 client 的前缀）→ **append**：
          增量 = client_history[n:]
        - p < n 且 p < m：进入第四步。

    第四步・区分"编辑分叉"和"截断尾巴+追加"：
        如果能找到 L（0 < L < m）使得 client_history[:L] 恰好等于 archived
        的某段尾巴（`_tail_truncated_then_appended_split`），说明 client
        发来的不是"从头延续"，而是"看不到 archived 的前面部分，只从中间
        某处开始 + 后面追加了新内容"——**append**：不删除 archived 任何
        东西，增量 = client_history[L:]。

    第五步・编辑分叉（带 p >= 1 最后防线）：
        - p >= 1（至少一条锚点对上，证明客户端看到的历史确实从头开始、
          不是截断的）→ **edit**：删除 archived[p:]（只删本 session 的），
          增量 = client_history[p:]。
        - p == 0 且所有对齐尝试都失败 → **拒绝删除**，logger.warning 后
          按 append 处理（增量 = 整个 client_history，不删任何行）。
          取舍说明：宁可时间线短暂出现重复内容，绝不允许因为"疑似截断的
          历史无法对齐"而误删该 session 全部归档（灾难级）。代价是"编辑
          第一条消息"这种 p=0 的合法编辑场景会被降级为 append（旧内容
          留在时间线里 + 新内容重复入账），可接受。

    注意：本函数只做**决策**（要删哪些 id、增量是什么），不执行任何 DB 写入，
    调用方负责真正执行删除 + 更新水位线，保持函数纯粹、方便单元测试。
    """
    archived_keys = [_unified_message_key(r["role"], r["content"]) for r in archived_rows]
    client_keys = [_unified_message_key(m.get("role", ""), m.get("content", "")) for m in client_history]

    m, n = len(client_keys), len(archived_keys)

    if n == 0:
        # 该 session 在全局时间线上还没有任何归档记录：客户端历史（如果有）
        # 全部当增量，等待被归档。没有可删除的行。
        return UnifiedReconcileResult(
            new_messages=list(client_history),
            delete_ids=[],
            new_watermark=0,
            kind="append" if client_history else "noop",
        )

    # 第一步：重roll 特判（用 current_user_text 对齐 archived 倒数第二条旧 user）
    if (
        n >= 2
        and archived_rows[-1]["role"] == "assistant"
        and archived_rows[-2]["role"] == "user"
        and current_user_text
        and current_user_text == archived_keys[-2][1]
        and m <= n - 2
        and client_keys == archived_keys[n - 2 - m : n - 2]
    ):
        # 旧 user + 旧 assistant 两条都删：archiver 之后会重新归档 (user, assistant')，
        # 只删 assistant 的话这条 user 会在时间线上重复。
        delete_ids = [archived_rows[n - 2]["id"], archived_rows[n - 1]["id"]]
        return UnifiedReconcileResult(
            new_messages=[], delete_ids=delete_ids, new_watermark=n - 2, kind="reroll",
        )

    # 第二步：纯截断尾巴（client 完全被 archived 尾部吃掉，没有多余内容）
    if _tail_truncated_len(client_keys, archived_keys) == m:
        return UnifiedReconcileResult(new_messages=[], delete_ids=[], new_watermark=n, kind="noop")

    # 第三步：最长公共前缀
    p = _common_prefix_len(client_keys, archived_keys)

    if p == n:
        # archived 整体是 client 的前缀，client 后面多出来的是新增量。
        new_msgs = list(client_history[n:])
        return UnifiedReconcileResult(
            new_messages=new_msgs, delete_ids=[], new_watermark=n,
            kind="append" if new_msgs else "noop",
        )

    # 第四步：p < n（且必然 p < m，否则上面已经 return）——
    # 先排除"截断尾巴 + 追加"的组合场景，避免真正的正常场景被误判为编辑分叉。
    combo_L = _tail_truncated_then_appended_split(client_keys, archived_keys)
    if combo_L is not None:
        new_msgs = list(client_history[combo_L:])
        return UnifiedReconcileResult(new_messages=new_msgs, delete_ids=[], new_watermark=n, kind="append")

    # 第五步：编辑分叉。最后防线：p >= 1 才允许删除（至少一条锚点对上，
    # 证明客户端历史是从头开始的，不是截断到无法对齐的历史）。
    if p == 0:
        # 所有对齐尝试都失败——最可能是截断历史 + 内容漂移（图片被剥、
        # 特殊 block 序列化差异等），也可能是编辑了第一条消息。无法区分时
        # 绝不删除：宁可时间线短暂重复，不允许误删全史。
        logger.warning(
            "Unified brain reconcile: p=0, all alignment attempts failed "
            "(m=%d n=%d) — 疑似截断历史无法对齐，拒绝删除，降级为 append",
            m, n,
        )
        return UnifiedReconcileResult(
            new_messages=list(client_history), delete_ids=[], new_watermark=n, kind="append",
        )

    # p >= 1：真正的编辑分叉，archived[p:] 对不上，删除（只删本 session 的）。
    delete_ids = [r["id"] for r in archived_rows[p:]]
    new_msgs = list(client_history[p:])
    return UnifiedReconcileResult(new_messages=new_msgs, delete_ids=delete_ids, new_watermark=p, kind="edit")


async def _reconcile_and_update_timeline(
    session_fingerprint: str,
    client_history: list,
    current_user_text: str = "",
) -> UnifiedReconcileResult:
    """读取该 session 已归档行 + 现有水位线，跑对账算法，并**立即执行**
    删除/水位线回退（reroll / edit 场景需要在组装上下文前先把旧数据摘掉，
    否则这一轮的上下文还是错的）。返回对账结果供调用方决定要不要把
    new_messages 当作"本 session 未归档增量"拼进上下文。

    注意：new_messages 本身不会在这里写入 conversations —— 归档仍由现有
    archiver.py 在收到完整回复后完成（PLAN 第 4 步范围，本次不做）。这里
    只负责"清掉过时/被重roll的旧数据"，让下一次 archive 能干净地写入。"""
    from gateway import db as _db

    archived_rows = await _db.get_global_timeline_for_session(session_fingerprint, after_id=0)
    result = reconcile_session_history(client_history, archived_rows, current_user_text=current_user_text)

    # 语义换算：result.new_watermark 是"保留了 archived 前多少行"（计数，
    # 纯函数便于测试），而 DB 里水位线存的是"全局 conversations.id"。
    # 交错场景下两者不同（本 session 第 4 行的全局 id 可能是 6），必须换算：
    # 保留的最后一行的 id；一行不留则 0。
    if result.new_watermark > 0 and archived_rows:
        watermark_id = archived_rows[min(result.new_watermark, len(archived_rows)) - 1]["id"]
    else:
        watermark_id = 0

    if result.delete_ids:
        # 删除留痕：把被删行的内容摘要打进 INFO 日志——万一误删，
        # 还能从 journalctl 里找回被删的内容。
        delete_set = set(result.delete_ids)
        for r in archived_rows:
            if r["id"] in delete_set:
                logger.info(
                    "Unified brain reconcile[%s]: deleting row id=%d role=%s content=%.200s",
                    result.kind, r["id"], r["role"], (r["content"] or "").replace("\n", "\\n"),
                )
        removed = await _db.delete_conversations_by_ids(result.delete_ids)
        logger.info(
            "Unified brain reconcile[%s]: session=%s deleted %d rows (ids=%s), new_watermark=%d (id=%d)",
            result.kind, session_fingerprint, removed, result.delete_ids[:10],
            result.new_watermark, watermark_id,
        )
        # 编辑/重roll 都需要把水位线**回退**到分叉点——用强制写入，
        # 不能用 update_global_watermark（它只会取更大值，回退不了）。
        await _db.force_set_global_watermark(session_fingerprint, watermark_id)
    elif result.kind == "append" and watermark_id:
        # 正常追加场景：archived 没变，只是确保水位线不落后于 archived 现状
        # （比如水位线因为某种原因是 0，但其实已经有 archived 数据）。
        await _db.update_global_watermark(session_fingerprint, watermark_id)

    return result


async def _unified_global_tail(
    after_id: int,
    limit: int,
) -> list:
    """全局时间线尾巴：摘要位置(after_id)之后的所有对话，带平台标签，
    上限 limit 条（unified_tail_max）。按到达顺序（id 升序）返回。"""
    from gateway import db as _db

    rows = await _db.get_global_timeline_after(after_id=after_id, limit=limit)
    msgs = [_unified_row_to_message(r) for r in rows]
    # 保持首尾都是完整轮次：掐头去尾半轮，避免孤立的 assistant 开头 / user 结尾
    while msgs and msgs[0]["role"] == "assistant":
        msgs.pop(0)
    while msgs and msgs[-1]["role"] == "user":
        msgs.pop()
    return msgs


async def _maybe_rebuild_global_summary(upstream):
    """检查全局时间线新增条数是否 ≥ unified_summary_interval，是则后台滚动压缩。
    复用现有 _rebuild_locks 去重机制（用固定 key 'fp_global' 区分全局 rebuild
    和各 session/context 自己的 rebuild，不会互相冲突）和现有摘要 prompt
    （summarize_messages 完全不用改，只是喂给它的 new_turns 正文已经带了
    [平台标签] 前缀）。"""
    from gateway import db as _db
    from gateway.settings import get as _get

    interval = int(_get("unified_summary_interval", 30))
    global_max_id = await _db.get_global_timeline_count()
    _, summary_pos = await _db.get_global_summary()

    if global_max_id - summary_pos < max(1, interval):
        return  # 还没攒够，跳过

    lock_key = "fp_global"
    if _rebuild_locks.get(lock_key):
        logger.info("Unified global summary rebuild already running, skipping")
        return
    _rebuild_locks[lock_key] = True
    asyncio.create_task(_rebuild_global_summary_background(upstream, summary_pos, global_max_id))


async def _rebuild_global_summary_background(upstream, start_pos: int, target_position: int):
    """后台滚动重建全局摘要，chunk 大小 = unified_summary_interval（不像 session
    级 BP2 那样按轮数 ×2，因为全局时间线里 user/assistant 交错来自不同平台，
    没有"轮"这个概念，直接按条数分块）。"""
    from gateway import db as _db
    from gateway.summarizer import summarize_messages, compress_summary
    from gateway.settings import get as _get

    lock_key = "fp_global"
    chunk = max(1, int(_get("unified_summary_interval", 30)))
    prev_summary, pos = await _db.get_global_summary()
    pos = max(pos, start_pos)
    try:
        while pos < target_position:
            end = min(pos + chunk, target_position)
            rows = await _db.get_global_timeline_after(after_id=pos, limit=0)
            new_turns_rows = [r for r in rows if r["id"] <= end]
            if not new_turns_rows:
                break
            new_turns = [_unified_row_to_message(r) for r in new_turns_rows]
            try:
                summary = await summarize_messages(new_turns, upstream, prev_summary=prev_summary)
                if len(summary) > int(_get("summary_compress_threshold", 12000)):
                    try:
                        compressed = await compress_summary(summary, upstream)
                        if compressed and len(compressed) < len(summary):
                            summary = compressed
                    except Exception:
                        logger.exception("Unified global summary: compress_summary failed, saving uncompressed")
                await _db.save_global_summary(end, summary)
                prev_summary = summary
                pos = end
                logger.info("Unified global summary: saved pos %d (%d chars)", end, len(summary))
            except Exception:
                logger.exception("Unified global summary: rebuild failed at pos %d, stopping", end)
                break
        logger.info("Unified global summary: finished, pos=%d", pos)
    finally:
        _rebuild_locks.pop(lock_key, None)


async def _apply_bp2_unified(
    messages: list,
    upstream,
    fingerprint: str,
    context_id: str = "",
) -> list:
    """统一大脑上下文组装（PLAN 第 3 步核心）。只在 unified_brain_enabled=True
    时被调用。

    发给 AI 的内容 =
        客户端 system prompt（原样保留，system 字段不在这里处理，由 hooks 别处负责）
      + 全局滚动摘要（fp_global 最新一条）
      + 全局时间线尾巴（摘要位置之后，带 [平台标签]，上限 unified_tail_max）
      + 本 session 未归档的增量消息（对账算出来的，不是整段客户端历史）
    """
    from gateway import db as _db
    from gateway.settings import get as _get

    if not messages:
        return messages

    # 当前正在提问的最后一条 user 消息永远保留在最末尾，不参与对账；
    # 但它的文本要传给对账算法做重roll判定（重roll 时 messages 最后一条
    # 就是被重新问的那条 user，client_history 里探测不到）。
    #
    # 主动消息免疫（PLAN_PROACTIVE_FIX.md）：橘瓣主动触发请求的合成 user
    # 消息永远是 messages[-1]，所以它天然落进 current_user（AI 需要看到它
    # 才知道要主动说话）、永远不进 client_history → 不可能出现在
    # result.new_messages 里被当增量。归档端由 archiver 的 proactive flag
    # 负责跳过它。reroll 判定也不受影响：合成文本带固定尾缀且从不入库，
    # 不可能等于 archived 里任何一条真实旧 user。
    if messages[-1].get("role") == "user":
        client_history = messages[:-1]
        current_user = [messages[-1]]
        current_user_text = _unified_normalize_text(messages[-1].get("content", ""))
    else:
        # 理论上不应该发生（preprocess 前面已经保证以 user 结尾），保守兜底
        client_history = list(messages)
        current_user = []
        current_user_text = ""

    # 1) 对账：识别增量 / 重roll / 编辑，需要删除的立即执行
    result = await _reconcile_and_update_timeline(
        fingerprint, client_history, current_user_text=current_user_text,
    )

    # 2) 取全局摘要 + 尾巴
    summary_text, summary_pos = await _db.get_global_summary()
    tail_max = int(_get("unified_tail_max", 60))
    tail_msgs = await _unified_global_tail(after_id=summary_pos, limit=tail_max)

    # 3) 本 session 增量消息（对账结果），带平台标签
    label = _unified_platform_label(context_id, fingerprint)
    incremental = [
        {"role": m.get("role", ""), "content": _tag_text_with_platform(label, _content_text(m.get("content", "")))}
        for m in result.new_messages
    ]

    assembled = list(tail_msgs) + incremental + current_user
    # 掐头：绝不能让 assistant 落在最前面（tail_msgs 已经掐过，但 incremental
    # 拼接后如果 tail_msgs 为空且 incremental 以 assistant 开头，仍要兜底）
    while assembled and assembled[0]["role"] == "assistant":
        assembled.pop(0)

    if summary_text:
        out = _bp2_with_summary(summary_text, assembled)
    else:
        out = assembled

    # 4) 检查是否需要触发全局摘要滚动压缩（后台异步，不阻塞本次请求）
    try:
        await _maybe_rebuild_global_summary(upstream)
    except Exception:
        logger.exception("Unified brain: failed to check/trigger global summary rebuild (non-fatal)")

    logger.info(
        "Unified brain context assembled: session=%s label=%s reconcile=%s "
        "summary_pos=%d tail=%d incremental=%d",
        fingerprint, label, result.kind, summary_pos, len(tail_msgs), len(incremental),
    )
    return out


# ──────────────────────────────────────────────
# Tag detection: distinguish 日常 / 技术 / etc. conversations
# Detected from prefix on FIRST user message — RikkaHub sends full history each
# request, so first message stays stable across the session.
# ──────────────────────────────────────────────
