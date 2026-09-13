"""BP2 滚动摘要 + Seamless Session 接续（P2.1 收尾，2026-09-06）。

自 hooks.py 绞杀搬迁，函数体零改动；hooks.py 留 import 转发。
依赖 pipeline.common / identity / cache / unified 与 summarizer；不 import hooks。
"""

import asyncio
import hashlib
import logging

from gateway.pipeline.cache import _count_pairs, _strategy_rounds
from gateway.pipeline.common import _rebuild_locks, _seen_fingerprints
from gateway.pipeline.common import _TIME_REMINDER_RE, _content_text, _time_tag
from gateway.pipeline.identity import _clean_context_id, _should_seed_shared_context
from gateway.pipeline.unified import _bp2_with_summary

logger = logging.getLogger("gateway.pipeline.summary")



async def _shared_context_tail(
    context_fingerprint: str,
    session_fingerprint: str,
    tag: str = "",
    limit: int = 60,
) -> list:
    """Recent archived turns from the same shared context, excluding this session."""
    if not context_fingerprint:
        return []
    from gateway import db as _db
    try:
        rows = await _db.get_recent_archived_turns(
            limit=limit,
            tag=tag,
            context_fingerprint=context_fingerprint,
            exclude_fingerprint=session_fingerprint,
        )
    except Exception:
        logger.exception("Shared context tail: failed to load archive")
        return []
    recent_msgs = []
    for r in rows:
        role = r["role"]
        content = r["content"] or ""
        if role in ("user", "assistant") and content:
            recent_msgs.append({"role": role, "content": content})
    while recent_msgs and recent_msgs[-1]["role"] == "user":
        recent_msgs.pop()
    while recent_msgs and recent_msgs[0]["role"] == "assistant":
        recent_msgs.pop(0)
    return recent_msgs


def _messages_after_position(messages: list, position: int) -> list:
    if position <= 0:
        return list(messages)
    if position >= len(messages):
        # Some clients send a clipped history. Never drop the current user turn.
        return list(messages[-1:]) if messages else []
    return list(messages[position:])


async def _load_or_seed_shared_summary(context_fingerprint: str, context_id: str):
    """Return latest summary for a shared context, optionally seeding daily-main."""
    from gateway import db as _db

    summary, pos = await _db.get_latest_summary(context_fingerprint)
    if summary:
        return summary, pos
    if not _should_seed_shared_context(context_id):
        return None, 0
    inherited, src_fp, src_pos = await _db.find_predecessor_summary(
        exclude_fingerprint=context_fingerprint,
        within_hours=48,
    )
    if inherited and src_pos > 0:
        await _db.save_summary_at(
            context_fingerprint,
            messages_covered=src_pos,
            summary=inherited,
            model=f"seed:{src_fp}",
        )
        logger.info(
            "BP2 shared context seeded: context=%s id=%s from=%s pos=%d",
            context_fingerprint, context_id, src_fp, src_pos,
        )
        return inherited, src_pos
    return None, 0


async def _inject_seamless_context(
    messages: list,
    upstream,
    body: dict,
    tag: str = "",
    context_fingerprint: str = "",
    context_id: str = "",
) -> list:
    """
    Issue #5: Seamless session.
    Keep injecting the latest BP2 summary (daily channel only) until session
    grows large enough for natural BP2 to take over.
    """
    from gateway import db as _db
    # Tagged conversations (tech/etc.) get a clean slate — no daily injection.
    if tag:
        return messages

    # If session has already grown past the natural BP2 threshold, stop injecting.
    frozen_rounds, live_rounds = _strategy_rounds()
    threshold_pairs = frozen_rounds + live_rounds
    n_pairs = _count_pairs(messages)
    if n_pairs >= threshold_pairs:
        return messages  # natural BP2 will handle it
    if context_fingerprint:
        # Shared context is handled by _apply_bp2 using per-session progress.
        # Injecting here would add two synthetic messages before that accounting.
        return messages

    # Check opt-out keyword (only on the latest user message)
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_user:
        content = last_user.get("content", "")
        text = content if isinstance(content, str) else " ".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
        if text.strip().lower() == "reset":
            logger.info("Seamless session: reset keyword detected, skipping injection")
            return messages

    # Check if a summary is already injected (first message looks like our injection)
    if messages and isinstance(messages[0].get("content"), list):
        first_text = ""
        for b in messages[0].get("content", []):
            if isinstance(b, dict) and b.get("type") == "text":
                first_text = b.get("text", "")
                break
        if first_text.startswith("[以下是我们之前对话的摘要"):
            return messages  # already has our injection — RikkaHub stored it from a previous turn

    # Look up the most relevant summary. With an explicit shared context, only
    # inject from that namespace; otherwise keep the old "latest daily" behavior.
    try:
        if context_fingerprint:
            summary_text, covered = await _load_or_seed_shared_summary(
                context_fingerprint,
                context_id,
            )
            fp = context_fingerprint
            if not summary_text:
                return messages
        else:
            summaries = await _db.list_context_summaries()
            if not summaries:
                return messages
            # Pick the most-recently-active fingerprint (by max created_at),
            # then within that fingerprint, pick the summary covering the most messages.
            latest_fp = max(summaries, key=lambda r: r["created_at"] or "")["conv_fingerprint"]
            same_fp = [r for r in summaries if r["conv_fingerprint"] == latest_fp]
            best = max(same_fp, key=lambda r: (r["messages_covered"] or 0, r["freeze_cycle"]))
            summary_text = best["summary"]
            fp = best["conv_fingerprint"]
            covered = best["messages_covered"] or 0
    except Exception:
        logger.exception("Seamless session: failed to load summary")
        return messages

    # Fetch recent archived turns to reconstruct conversation continuity
    recent_msgs = []
    try:
        # 30 pairs ≈ 60 rows; tune via this constant if needed
        rows = await _db.get_recent_archived_turns(
            limit=60,
            tag=tag,
            context_fingerprint=context_fingerprint,
        )
        for r in rows:
            role = r["role"]
            content = r["content"] or ""
            if role in ("user", "assistant") and content:
                recent_msgs.append({"role": role, "content": content})
        # Drop trailing assistant rows so the sequence ends on a complete pair
        while recent_msgs and recent_msgs[-1]["role"] == "user":
            recent_msgs.pop()
        # Drop leading assistant rows (must start with user)
        while recent_msgs and recent_msgs[0]["role"] == "assistant":
            recent_msgs.pop(0)
    except Exception:
        logger.exception("Seamless session: failed to load archive")

    logger.info(
        "Seamless session: injecting summary fp=%s covered=%d + %d archived turns "
        "(session n_pairs=%d/threshold=%d)",
        fp, covered, len(recent_msgs), n_pairs, threshold_pairs,
    )
    bp2_user = {
        "role": "user",
        "content": [{
            "type": "text",
            "text": f"[以下是我们之前对话的摘要，新窗口继续]\n\n{summary_text}",
        }],
    }
    bp2_asst = {
        "role": "assistant",
        "content": "好的，我已了解之前的对话，继续。",
    }
    return [bp2_user, bp2_asst] + recent_msgs + list(messages)


async def _rebuild_summaries_background(
    messages: list, upstream, fingerprint: str, target_position: int, cycle_size: int,
    initial_prev_summary: str = "",
    session_fingerprint: str = "",
):
    """Build incremental summaries up to target_position in coarse rolling chunks.

    initial_prev_summary is used for seamless new-session handoff: the new
    fingerprint starts at pos=0, but its first rolling summary should inherit
    the previous session's BP2 summary as prev_summary, then summarize only the
    new session's live messages on top of it.
    """
    from gateway import db as _db
    from gateway.summarizer import summarize_messages

    chunk = max(1, cycle_size) * 2  # messages per rolling cycle
    # 修复 (Issue #30)：从 max_pos 开始 catchup，不从 pos=0 扫。
    # 否则 inherit 模式下（DB 里只有一条孤立的 max_pos 摘要、没有 pos=16, 32, ..., max_pos
    # 这些中间 cycle），从 0 开始会把每个 chunk 都当成 hole 调上游生成摘要，浪费 LLM 调用甚至 429。
    start_pos = 0
    save_base_pos = 0
    prev_summary = ""
    shared_context = bool(session_fingerprint and session_fingerprint != fingerprint)
    try:
        if shared_context:
            start_pos = await _db.get_context_session_position(fingerprint, session_fingerprint)
            save_base_pos = await _db.get_max_summary_position(fingerprint)
            if initial_prev_summary:
                prev_summary = initial_prev_summary
            elif save_base_pos > 0:
                existing, _ = await _db.get_latest_summary(fingerprint)
                if existing:
                    prev_summary = existing
        else:
            start_pos = await _db.get_max_summary_position(fingerprint)
            if start_pos > 0:
                existing, _ = await _db.get_summary_at_or_before(fingerprint, start_pos)
                if existing:
                    prev_summary = existing
            elif initial_prev_summary:
                prev_summary = initial_prev_summary
    except Exception:
        logger.exception("BG rebuild: failed to load start_pos, falling back to 0")
        start_pos = 0
        save_base_pos = 0
        prev_summary = initial_prev_summary or ""

    logger.info(
        "BG rebuild: starting for %s from pos %d up to pos %d "
        "(chunk=%d, inherited_prev=%s, shared_context=%s, session=%s, save_base=%d)",
        fingerprint, start_pos, target_position, chunk,
        bool(initial_prev_summary and start_pos == 0), shared_context,
        session_fingerprint or "", save_base_pos,
    )
    pos = start_pos
    cycle = max(0, (save_base_pos if shared_context else start_pos) // chunk)  # 估算 cycle 编号
    try:
        while pos < target_position:
            cycle += 1
            end = min(pos + chunk, target_position)
            # Resume: if a summary already covers up to `end`, use it as prev and skip
            if not shared_context:
                existing, existing_pos = await _db.get_summary_at_or_before(fingerprint, end)
                if existing and existing_pos == end:
                    prev_summary = existing
                    pos = end
                    logger.info("BG rebuild: pos %d already covered, skipping", end)
                    continue
            new_turns = messages[pos:end]
            if not new_turns:
                break
            try:
                summary = await summarize_messages(new_turns, upstream, prev_summary=prev_summary)
                raw_size = sum(len(str(m.get("content", ""))) for m in new_turns)
                prev_size = len(prev_summary) if prev_summary else 0
                # Sanity check fixed (2026-05-27 v2):
                # 1) 截断检测：新摘要比 prev_summary 小一半以上 = 模型丢了分层结构（DeepSeek 输出衰减常见）
                if prev_size > 5000 and len(summary) < prev_size * 0.5:
                    logger.warning(
                        "BG rebuild: pos %d truncated summary, skipping "
                        "(prev_sum=%d new_sum=%d raw=%d)",
                        end, prev_size, len(summary), raw_size,
                    )
                    pos = end
                    continue
                # 2) raw copy 检测：增量超过 raw × 5 = 模型把原文复制了一遍
                increment = len(summary) - prev_size
                if raw_size > 0 and increment > raw_size * 5.0:
                    logger.warning(
                        "BG rebuild: pos %d looks like raw copy, skipping "
                        "(raw=%d prev_sum=%d new_sum=%d increment=%d)",
                        end, raw_size, prev_size, len(summary), increment,
                    )
                    pos = end
                    continue
                # 3) 硬上限：超过 summary_compress_threshold（默认 12000）→ 调用
                # compress_summary 压到 8000-10000（Issue #18；2026-07-12 用户定
                # 目标 8000-10000，阈值从 35000 收紧——滚动继承会让摘要一代代变肥，
                # 14000-23000 的摘要卡在旧阈值下永远不被压缩）
                from gateway.settings import get as _sget_compress
                _compress_threshold = int(_sget_compress("summary_compress_threshold", 12000))
                if len(summary) > _compress_threshold:
                    logger.warning(
                        "BG rebuild: pos %d summary too long (%d chars > %d), "
                        "requesting compression retry", end, len(summary), _compress_threshold,
                    )
                    try:
                        from gateway.summarizer import compress_summary
                        compressed = await compress_summary(summary, upstream)
                        if compressed and len(compressed) < len(summary):
                            logger.info(
                                "BG rebuild: compressed pos %d: %d → %d chars",
                                end, len(summary), len(compressed),
                            )
                            summary = compressed
                        else:
                            logger.warning(
                                "BG rebuild: compress_summary did not shrink (got %d), "
                                "saving original", len(compressed) if compressed else 0,
                            )
                    except Exception:
                        logger.exception("BG rebuild: compress_summary failed, saving uncompressed")
                # 倒退保护：当前 fp 已有更大 messages_covered 时，跳过保存（防并发或 fp 漂移覆盖更新摘要）
                try:
                    _max_pos = await _db.get_max_summary_position(fingerprint)
                except Exception:
                    _max_pos = 0
                save_position = save_base_pos + (end - start_pos) if shared_context else end
                if _max_pos > save_position:
                    logger.warning(
                        "BG rebuild: pos %d regression detected (already have pos %d for %s), skipping save",
                        save_position, _max_pos, fingerprint,
                    )
                    pos = end
                    continue
                await _db.save_summary_at(fingerprint, save_position, summary, freeze_cycle=cycle)
                if shared_context:
                    await _db.update_context_session_position(
                        fingerprint, session_fingerprint, end
                    )
                prev_summary = summary
                pos = end
                logger.info("BG rebuild: saved pos %d (%d chars)", save_position, len(summary))
                # ── 通知：摘要 cycle 已生成 ──
                try:
                    from gateway.notifier import notify as _notify
                    await _notify(
                        f"📝 BP2 摘要已生成 [cycle {cycle}]",
                        (
                            f"对话指纹：{fingerprint}\n"
                            f"覆盖消息数：{save_position}\n"
                            f"摘要长度：{len(summary)} 字符\n"
                            f"上游：{upstream.name if upstream else '?'}"
                        ),
                        dedup_key=f"rebuild_done_{fingerprint}_{save_position}",
                        cooldown=300,  # 同一段摘要 5 分钟内只通知一次
                    )
                except Exception:
                    pass
            except Exception as exc:
                logger.exception("BG rebuild: failed at pos %d, stopping", end)
                # ── 通知：rebuild 失败 ──
                try:
                    from gateway.notifier import notify as _notify
                    exc_str = str(exc)
                    is_balance = "402" in exc_str or "Insufficient" in exc_str or "balance" in exc_str.lower()
                    if is_balance:
                        title = f"❌ 余额耗尽，摘要停止 [{upstream.name if upstream else '?'}]"
                        body = (
                            f"上游【{upstream.name if upstream else '?'}】返回 402 余额不足。\n"
                            f"BP2 摘要卡在 pos={end}，记忆更新停止。\n"
                            f"请立即充值后重启网关。"
                        )
                    else:
                        title = f"⚠️ BP2 rebuild 失败 [pos={end}]"
                        body = (
                            f"对话指纹：{fingerprint}\n"
                            f"失败位置：pos={end}\n"
                            f"错误：{exc_str[:300]}"
                        )
                    await _notify(title, body,
                                  dedup_key=f"rebuild_fail_{fingerprint}_{end}",
                                  cooldown=1800)
                except Exception:
                    pass
                break
        logger.info("BG rebuild: finished for %s", fingerprint)
    finally:
        # Issue #9: 无论正常退出还是异常都要清锁
        _rebuild_locks.pop(fingerprint, None)


async def _apply_bp2(
    messages: list,
    upstream,
    fingerprint: str | None = None,
    context_fingerprint: str = "",
    context_id: str = "",
) -> list:
    """
    Replace old live history with a rolling summary once enough live turns have
    accumulated, while retaining a fixed frozen window as original text.

    Layout after injection:
        [BP2 user: summary]
        [BP2 asst: ack]
        [BP3: frozen turns, last has cache_control = 1h]
        [Live turns, last asst has cache_control = BP4/5m]
        [Current user turn]
    """
    from gateway import db as _db
    from gateway.summarizer import summarize_messages, derive_conv_fingerprint

    frozen_rounds, live_rounds = _strategy_rounds()
    n_pairs = _count_pairs(messages)

    if fingerprint is None:
        fingerprint = derive_conv_fingerprint(messages)
    summary_fingerprint = context_fingerprint or fingerprint
    session_fingerprint = fingerprint

    # Summary should cover everything before the retained frozen window.
    # First write happens after live_rounds have accumulated beyond frozen.
    target_position = max(0, (n_pairs - frozen_rounds) * 2)
    if target_position < live_rounds * 2:
        if context_fingerprint:
            latest_summary, latest_position = await _load_or_seed_shared_summary(
                context_fingerprint,
                context_id,
            )
            if latest_summary:
                session_position = await _db.get_context_session_position(
                    context_fingerprint, fingerprint
                )
                if latest_position > session_position and len(messages) > latest_position:
                    session_position = latest_position
                    await _db.update_context_session_position(
                        context_fingerprint, fingerprint, session_position
                    )
                archived_tail = await _shared_context_tail(
                    context_fingerprint,
                    fingerprint,
                    tag="",
                )
                if archived_tail:
                    logger.info(
                        "BP2 shared context: injecting %d archived tail turns for short session %s",
                        len(archived_tail), fingerprint,
                    )
                return _bp2_with_summary(
                    latest_summary,
                    archived_tail + _messages_after_position(messages, session_position),
                )
        return messages

    if summary_fingerprint and summary_fingerprint not in _seen_fingerprints:
        _seen_fingerprints.add(summary_fingerprint)
        try:
            sig_lines = []
            for i, m in enumerate(messages[:5]):
                role = m.get("role", "?")
                text = _content_text(m.get("content", ""))
                h8 = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:8]
                preview = text[:60].replace("\n", "\\n")
                sig_lines.append(f"  [{i}] {role} md5={h8} len={len(text)} | {preview}")
            logger.info(
                "FP first-seen: session=%s summary=%s n_msgs=%d signatures:\n%s",
                session_fingerprint, summary_fingerprint, len(messages), "\n".join(sig_lines),
            )
        except Exception:
            logger.exception("FP signature dump failed (non-fatal)")
    pre_bp3 = messages[:target_position]  # replaced by summary
    rest = messages[target_position:]     # frozen + live + current user
    summary_cycle = target_position // max(1, frozen_rounds * 2)

    if context_fingerprint:
        session_position = await _db.get_context_session_position(
            summary_fingerprint, session_fingerprint
        )
        latest_summary, latest_position = await _load_or_seed_shared_summary(
            summary_fingerprint,
            context_id,
        )
        logger.info(
            "BP2 shared context: context=%s session=%s session_pos=%d target_pos=%d latest_pos=%d",
            summary_fingerprint, session_fingerprint, session_position, target_position, latest_position,
        )
        if latest_summary and latest_position > session_position and len(messages) > latest_position:
            session_position = latest_position
            await _db.update_context_session_position(
                summary_fingerprint, session_fingerprint, session_position
            )
            logger.info(
                "BP2 shared context continuation aligned: context=%s session=%s pos=%d",
                summary_fingerprint, session_fingerprint, session_position,
            )
        if latest_summary and target_position - session_position < live_rounds * 2:
            archived_tail = await _shared_context_tail(
                summary_fingerprint,
                session_fingerprint,
                tag="",
            )
            if archived_tail:
                logger.info(
                    "BP2 shared context: injecting %d archived tail turns for session %s",
                    len(archived_tail), session_fingerprint,
                )
            return _bp2_with_summary(
                latest_summary,
                archived_tail + _messages_after_position(messages, session_position),
            )
        if target_position > session_position:
            if not _rebuild_locks.get(summary_fingerprint):
                _rebuild_locks[summary_fingerprint] = True
                asyncio.create_task(
                    _rebuild_summaries_background(
                        list(messages),
                        upstream,
                        summary_fingerprint,
                        target_position,
                        live_rounds,
                        initial_prev_summary=latest_summary or "",
                        session_fingerprint=session_fingerprint,
                    )
                )
            else:
                logger.info("BP2 shared-context rebuild already running for %s, skipping", summary_fingerprint)
        if latest_summary:
            archived_tail = await _shared_context_tail(
                summary_fingerprint,
                session_fingerprint,
                tag="",
            )
            if archived_tail:
                logger.info(
                    "BP2 shared context: injecting %d archived tail turns while rebuild catches up for %s",
                    len(archived_tail), session_fingerprint,
                )
            return _bp2_with_summary(
                latest_summary,
                archived_tail + _messages_after_position(messages, session_position),
            )
        return messages

    # Look up the most recent summary covering ≤ target_position
    found_summary, found_position = await _db.get_summary_at_or_before(summary_fingerprint, target_position)
    logger.info(
        "BP2 lookup: summary_fp=%s session_fp=%s target_pos=%d found_pos=%d n_pairs=%d frozen=%d live=%d",
        summary_fingerprint, session_fingerprint, target_position, found_position, n_pairs, frozen_rounds, live_rounds,
    )

    if found_summary and found_position == target_position:
        # Perfect match — use it directly
        summary = found_summary
    elif found_summary and found_position > 0:
        # Keep using the previous summary until at least live_rounds new turns
        # have accumulated. This avoids expensive rewrite churn.
        if target_position - found_position < live_rounds * 2:
            logger.info(
                "BP2 using previous summary for %s: found_pos=%d target_pos=%d "
                "(waiting for %d live rounds)",
                fingerprint, found_position, target_position, live_rounds,
            )
            bp2_user = {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"[以下是我们之前对话的摘要，请基于此继续]\n\n{found_summary}",
                }],
            }
            bp2_asst = {
                "role": "assistant",
                "content": "好的，我已了解之前的对话内容，继续。",
            }
            return [bp2_user, bp2_asst] + list(messages[found_position:])

        # Rolling needed but DON'T block this request — always defer to background.
        # Use the previous (smaller-covered) summary if available so this turn at
        # least gets SOME BP2 compression; new summary will be ready for next turn.
        new_turns = pre_bp3[found_position:target_position]
        if not new_turns:
            # 边界 case：刚好对齐
            summary = found_summary
        else:
            logger.info(
                "BP2 rolling deferred to background for %s: %d→%d (%d new msgs) — "
                "using prev_summary @ %d this turn, full rebuild scheduled",
                fingerprint, found_position, target_position, len(new_turns), found_position,
            )
            # Issue #9: 启动后台滚动 rebuild（带去重锁，避免并发轰炸上游）
            if not _rebuild_locks.get(fingerprint):
                _rebuild_locks[fingerprint] = True
                asyncio.create_task(
                    _rebuild_summaries_background(
                        list(messages), upstream, fingerprint, target_position, live_rounds
                    )
                )
            else:
                logger.info("BP2 rebuild already running for %s, skipping", fingerprint)
            # Phase 2 v2：自动喂记忆已经移到 archive 时机触发（archiver.py），
            # 不再在 BP2 这里从 messages 切片取——避免被 Seamless 注入的旧对话污染。
            # 本轮先用旧 summary 压住前半段，避免等后台重建期间主请求突然回到全量上下文。
            bp2_user = {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"[以下是我们之前对话的摘要，请基于此继续]\n\n{found_summary}",
                }],
            }
            bp2_asst = {
                "role": "assistant",
                "content": "好的，我已了解之前的对话内容，继续。",
            }
            return [bp2_user, bp2_asst] + list(messages[found_position:])
    else:
        # Cold start — no summary exists for current fingerprint.
        # 先尝试「跨 session 继承」：查找最近 48h 内其他 fp 的最新摘要作为起点。
        # 这样 RikkaHub 切了新窗口（产生新 fp）时，AI 不会失忆。
        inherited, src_fp, src_pos = await _db.find_predecessor_summary(
            exclude_fingerprint=fingerprint, within_hours=48
        )
        if inherited:
            is_continuation = len(messages) > src_pos
            logger.info(
                "BP2 cross-session inherit: %s adopting summary from %s "
                "(src_pos=%d, len=%d, continuation=%s, recv_msgs=%d)",
                fingerprint, src_fp, src_pos, len(inherited), is_continuation, len(messages),
            )
            if is_continuation:
                # 老场景：fp 漂移，新 fp 实际有完整对话历史；摘要标签贴对到 src_pos
                await _db.save_summary_at(
                    fingerprint, messages_covered=src_pos,
                    summary=inherited, freeze_cycle=summary_cycle,
                )
                live_msgs = list(messages[src_pos:])
            else:
                # 新场景：独立新 session，messages 是全新的（远短于 src_pos）
                # 不写 DB（避免 messages_covered=src_pos 标签远超实际、后续 get_summary_at_or_before 查不到）
                # 每次请求都会再 inherit 一次（轻微浪费但语义正确），直到新 session 自己长大触发正常 BP2
                live_msgs = list(messages)
            # 本轮直接注入继承的摘要给 AI 看（让伴侣不失忆）
            bp2_user = {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"[以下是我们之前对话的摘要，请基于此继续]\n\n{inherited}",
                }],
            }
            bp2_asst = {
                "role": "assistant",
                "content": "好的，我已了解之前的对话内容，继续。",
            }
            logger.info(
                "BP2 injected (inherited from %s): fingerprint=%s "
                "summary_covers=%d live_msgs=%d continuation=%s",
                src_fp, fingerprint, src_pos if is_continuation else 0, len(live_msgs), is_continuation,
            )

            # 独立新 session 继承旧摘要后，也启动当前 fingerprint 自己的
            # 后台 BP2 rebuild。否则 found_pos 会长期为 0，新窗口自己的对话
            # 无法滚入摘要。
            if not is_continuation and target_position >= live_rounds * 2:
                if not _rebuild_locks.get(fingerprint):
                    logger.info(
                        "BP2 inherited new-session rebuild scheduled for %s up to pos %d",
                        fingerprint, target_position,
                    )
                    _rebuild_locks[fingerprint] = True
                    asyncio.create_task(
                        _rebuild_summaries_background(
                            list(messages), upstream, fingerprint, target_position, live_rounds,
                            initial_prev_summary=inherited,
                        )
                    )
                else:
                    logger.info("BP2 rebuild already running for %s, skipping", fingerprint)

            return [bp2_user, bp2_asst] + live_msgs

        # 没找到前驱，走原来的冷启动逻辑（Issue #9: 去重锁防并发轰炸）
        logger.info(
            "BP2 cold start for %s: no predecessor, launching background rebuild up to pos %d",
            fingerprint, target_position,
        )
        if not _rebuild_locks.get(fingerprint):
            _rebuild_locks[fingerprint] = True
            asyncio.create_task(
                _rebuild_summaries_background(
                    list(messages), upstream, fingerprint, target_position, live_rounds
                )
            )
        else:
            logger.info("BP2 rebuild already running for %s, skipping", fingerprint)
        return messages

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

    logger.info("BP2 injected: fingerprint=%s cycle=%d (replaced %d messages with summary)",
                fingerprint, summary_cycle, len(pre_bp3))
    return [bp2_user, bp2_asst] + list(rest)


# ════════════════════════════════════════════════════════════════
# 统一大脑（PLAN_UNIFIED_BRAIN.md 第 2~3 步）
#
# 全部逻辑只在 `settings.unified_brain_enabled=True` 时被调用（调用点在
# preprocess_anthropic 里，见文件末尾）。开关关闭时本区块的函数完全不会
# 被执行，现有行为一个字节都不变——这是 Issue #36 之后定的铁律。
#
# 架构（对应 PLAN 二、核心架构转变）：
#   发给 AI 的内容 = 客户端 system prompt（原样保留）
#                  + 全局滚动摘要（fp_global 最新一条）
#                  + 全局时间线尾巴（摘要位置之后，带 [平台标签]，上限 unified_tail_max）
#                  + 本 session 未归档的增量消息
#
#   客户端发来的全量历史只用来"对账"（增量识别 / 重roll / 编辑检测），
#   绝不直接拼进发给 AI 的上下文——这是和 Issue #36 失败版本的本质区别。
# ════════════════════════════════════════════════════════════════
