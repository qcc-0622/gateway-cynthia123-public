"""Conversation archival: save complete request/response pairs to SQLite."""

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from gateway.config import TIMEZONE
from gateway.db import save_conversation, find_and_delete_reroll, pack_raw
from gateway.costs import calculate_cost
from gateway.summarizer import derive_conv_fingerprint

logger = logging.getLogger("gateway.archiver")

# 橘瓣主动消息（ProactiveMessageService）AI 回复里的 [JUMP] 跳转标记。
# 客户端会把它从正文剥掉再存本地（源码 578-596 行：rawText.replace("\\[JUMP]",
# ignoreCase).trim() 后 updateOrAppendAiMessage）——网关归档时必须做同样的
# 剥离，否则网关存的带 [JUMP]、橘瓣本地存的不带，两边账本又分叉。
_PROACTIVE_JUMP_RE = re.compile(r"\[JUMP\]", re.IGNORECASE)


def _compute_history_hash(messages: list) -> str:
    """Hash of all messages except the latest user message.
    Stable across re-rolls (which only change the trailing user content + new response)."""
    if not messages:
        return ""
    # Drop the trailing user message (the one being responded to)
    history = messages[:-1] if messages[-1].get("role") == "user" else messages
    payload = json.dumps(history, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()[:16]


async def archive(
    conversation_id: str,
    user_content: str,
    assistant_content: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    upstream_name: str = "",
    raw_request: dict = None,
    raw_response: str = "",
    api_format: str = "anthropic",
    duration_ms: int = 0,
    tag: str = "",
    client_model: str = "",
    fingerprint: str = "",
    history_hash: str = "",
    context_fingerprint: str = "",
    proactive: bool = False,
):
    # ── 主动消息归档纪律（PLAN_PROACTIVE_FIX.md）────────────────────
    # proactive=True 表示 hooks 确定性识别出这是橘瓣 ProactiveMessageService
    # 的主动触发请求。归档规则和普通请求不同：
    #   1) 合成 user 消息**跳过不归档**——它不会存进橘瓣本地对话，网关归档它
    #      就会账本分叉（Issue #36 修 5 次失败的病根之一）
    #   2) AI 回复剥离 [JUMP] 标记后归档（镜像客户端行为，见 _PROACTIVE_JUMP_RE）
    #   3) 剥离后 trim、大小写不敏感等于 [PASS]（或为空）→ **整条不归档**
    #      （客户端也删了气泡，两边一致地当这次没发生过）
    #   4) 不喂 ombre（合成 user 无意义；assistant 单条缺配对上下文）
    if proactive:
        cleaned_reply = _PROACTIVE_JUMP_RE.sub("", assistant_content or "").strip()
        if not cleaned_reply or cleaned_reply.lower() == "[pass]":
            logger.info(
                "Proactive archive skipped entirely: reply is %s (conv=%s)",
                "[PASS]" if cleaned_reply else "blank", conversation_id,
            )
            return
        assistant_content = cleaned_reply

    now = datetime.now(ZoneInfo(TIMEZONE)).isoformat()
    # 优先使用 hooks 算好的 fingerprint（基于原始 messages，稳定）。
    # 如果调用方没传，fallback 用 raw_request 重算——但要注意 raw_request 这里是
    # **预处理后**的 body，messages 已被 BP2/Seamless/Memory recall 改过，
    # 算出来的 fingerprint 会漂移，所以只是保底兜底。
    if not fingerprint and raw_request:
        msgs = raw_request.get("messages", []) if isinstance(raw_request, dict) else []
        if msgs:
            try:
                fingerprint = derive_conv_fingerprint(msgs)
            except Exception:
                logger.exception("Failed to compute fingerprint (fallback)")
    if not history_hash and raw_request:
        msgs = raw_request.get("messages", []) if isinstance(raw_request, dict) else []
        if msgs:
            try:
                history_hash = await asyncio.to_thread(_compute_history_hash, msgs)
            except Exception:
                logger.exception("Failed to compute history_hash (fallback)")

    # 统一大脑总开关（PLAN_UNIFIED_BRAIN.md 第 4 步）。默认 False，
    # 关闭时本函数所有新增逻辑都不执行，现状完全不变。
    from gateway.settings import get as _get_setting
    _unified_enabled = bool(_get_setting("unified_brain_enabled", False))

    try:
        # Detect & remove re-roll predecessor (same fingerprint + history + tag, recent)
        if fingerprint and history_hash:
            removed = await find_and_delete_reroll(fingerprint, history_hash, tag=tag,
                                                    within_minutes=30)
            if removed:
                logger.info(
                    "Re-roll dedup: removed %d rows (fp=%s hash=%s)",
                    removed, fingerprint, history_hash,
                )
                # 统一大脑：conversations 表就是全局时间线（同一张表），行删了
                # 时间线自然就没了，不需要额外同步内容。但该 session 的全局
                # 水位线可能还指着被删的行 id，要回退修正到剩余最大 id。
                # 说明：统一大脑开启时重roll 一般已在 hooks 对账阶段被删掉，
                # 这里的 find_and_delete_reroll 大多是 no-op；只有对账漏判
                # （如 p==0 防线降级）且 30 分钟内同 hash 时才会走到这里。
                if _unified_enabled and fingerprint:
                    try:
                        from gateway.db import (
                            get_global_timeline_for_session,
                            force_set_global_watermark,
                        )
                        remaining = await get_global_timeline_for_session(fingerprint, after_id=0)
                        last_id = remaining[-1]["id"] if remaining else 0
                        await force_set_global_watermark(fingerprint, last_id)
                        logger.info(
                            "Unified brain: watermark corrected after reroll dedup (fp=%s → id=%d)",
                            fingerprint, last_id,
                        )
                    except Exception:
                        logger.exception("Unified brain: watermark correction failed (non-fatal)")

        # 主动请求跳过合成 user 消息（见函数开头的归档纪律说明）；
        # 统一大脑时间线允许 assistant 单行（相邻 assistant 由橘瓣端合并，
        # 网关侧组装上下文时 _unified_global_tail 只掐头去尾不查配对）。
        if not proactive:
            await save_conversation(
                conversation_id=conversation_id,
                role="user",
                content=user_content,
                model=model,
                timestamp=now,
                api_format=api_format,
                tag=tag,
                fingerprint=fingerprint,
                history_hash=history_hash,
                client_model=client_model,
                context_fingerprint=context_fingerprint,
            )
        # 归档时算好费用存入，保证历史数据不受价格表变动影响
        _display_model = client_model or model
        try:
            _cost, _saved = calculate_cost(
                upstream_name, _display_model,
                tokens_in, tokens_out, cache_write_tokens, cache_read_tokens,
                msgs=1,
            )
        except Exception:
            _cost, _saved = None, None

        # P0.3：gzip 全量存储（旧 [:50000] 截断让排障抓瞎），读取用 db.unpack_raw。
        # dumps+gzip 双份 2MB 上限是纯同步 CPU，丢线程池防卡事件循环（2026-09-06）
        packed_raw_request, packed_raw_response = await asyncio.to_thread(
            lambda: (
                pack_raw(json.dumps(raw_request or {}, ensure_ascii=False)),
                pack_raw(raw_response or ""),
            )
        )

        assistant_row_id = await save_conversation(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_content,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
            upstream_name=upstream_name,
            # P0.3：gzip 全量存储（旧 [:50000] 截断让排障抓瞎），读取用 db.unpack_raw。
            # dumps+gzip 双份 2MB 上限是纯同步 CPU，丢线程池防卡事件循环（2026-09-06）
            raw_request=packed_raw_request,
            raw_response=packed_raw_response,
            api_format=api_format,
            duration_ms=duration_ms,
            timestamp=now,
            tag=tag,
            fingerprint=fingerprint,
            history_hash=history_hash,
            client_model=client_model,
            context_fingerprint=context_fingerprint,
            cost_usd=_cost,
            saved_usd=_saved,
        )
        logger.info(
            "Archived %s tag=%s | in=%d out=%d cache_w=%d cache_r=%d",
            conversation_id, tag or "(daily)", tokens_in, tokens_out, cache_write_tokens, cache_read_tokens,
        )
        # 统一大脑（PLAN 第 4 步）：归档成功后把该 session 的全局水位线推进到
        # 刚插入的 assistant 行 id（水位线语义 = 全局 conversations.id，
        # 不是行数计数——和 hooks._reconcile_and_update_timeline 的换算一致）。
        # update_global_watermark 内部是 MAX() 只增语义，重复调用无害。
        if _unified_enabled and fingerprint and assistant_row_id:
            try:
                from gateway.db import update_global_watermark
                await update_global_watermark(fingerprint, int(assistant_row_id))
                logger.info(
                    "Unified brain: watermark advanced (fp=%s → id=%d)",
                    fingerprint, int(assistant_row_id),
                )
            except Exception:
                logger.exception("Unified brain: watermark advance failed (non-fatal)")
        # Phase 2 v2：archive 成功后异步把这对真实对话喂给 ombre（带 hash 去重 + tag 过滤）
        # 主动请求跳过：user_content 是合成指令不是真实用户消息，
        # assistant 单条缺配对上下文，喂进记忆库只会污染
        if not proactive:
            try:
                from gateway.memory import feed_pair_to_ombre
                asyncio.create_task(
                    feed_pair_to_ombre(
                        fingerprint=context_fingerprint or fingerprint or "",
                        user_content=user_content,
                        assistant_content=assistant_content,
                        tag=tag,
                    )
                )
            except Exception:
                logger.exception("Memory feed_pair scheduling failed (non-fatal)")
    except Exception:
        logger.exception("Failed to archive conversation %s", conversation_id)
