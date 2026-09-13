"""Message preprocessing hooks: BP1-BP4 cache_control, timestamp, orphan cleanup, BP2 summarization."""

import asyncio
import hashlib
import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from gateway.config import TIMEZONE

logger = logging.getLogger("gateway.hooks")

# ── REFACTOR_ROADMAP P2.1 绞杀搬迁（2026-09-06 起）────────────────────
# 管道各块正分批搬到 gateway/pipeline/*，原位置留 import 转发兼容旧引用；
# 全部引用迁完后删转发。common 是叶子工具模块（防循环导入）。
from gateway.pipeline.common import (  # noqa: F401
    _GATEWAY_PROTOCOL,
    _TIME_REMINDER_RE,
    _add_gateway_protocol,
    _content_text,
    _pipe_trace,
    _time_tag,
)
from gateway.pipeline.identity import (  # noqa: F401
    _DEFAULT_DAILY_CONTEXT_ID,
    _GATEWAY_CONTEXT_KEYS,
    _TAG_PREFIXES,
    _clean_context_id,
    _context_fingerprint_from_id,
    _default_context_id_for_tag,
    _extract_context_id_from_text,
    _extract_gateway_context_id,
    _should_seed_shared_context,
    _strip_gateway_metadata,
    detect_tag,
)
from gateway.pipeline.proactive import (  # noqa: F401
    _PROACTIVE_PLACEHOLDERS,
    _PROACTIVE_PREFIX,
    _PROACTIVE_REPEAT_IDLE_MINUTES_DEFAULT,
    _PROACTIVE_SYNTHETIC_SUFFIX,
    _assistant_has_tool_use,
    _detect_and_rewrite_repeated_proactive,
    _drop_stale_assistant_tail,
    _format_proactive_trigger,
    _infer_proactive_from_assistant_tail,
    _is_proactive_placeholder_text,
    _is_proactive_synthetic_user,
    _is_proactive_trigger_text,
    _normalize_proactive_trigger,
    _proactive_tail_is_current,
    _proactive_user_message,
)
from gateway.pipeline.unified import (  # noqa: F401
    UnifiedReconcileResult,
    _UNIFIED_ANCHOR_LEN,
    _apply_bp2_unified,
    _bp2_with_summary,
    _common_prefix_len,
    _maybe_rebuild_global_summary,
    _rebuild_global_summary_background,
    _reconcile_and_update_timeline,
    _tag_text_with_platform,
    _tail_truncated_len,
    _tail_truncated_then_appended_split,
    _unified_global_tail,
    _unified_label_for_row,
    _unified_message_key,
    _unified_normalize_text,
    _unified_platform_label,
    _unified_row_to_message,
    reconcile_session_history,
)
from gateway.pipeline.summary import (  # noqa: F401
    _apply_bp2,
    _inject_seamless_context,
    _load_or_seed_shared_summary,
    _messages_after_position,
    _rebuild_summaries_background,
    _shared_context_tail,
)
from gateway.pipeline.sanitize import (  # noqa: F401
    _CLIENT_PROACTIVE_NOTE_MARK,
    _RECALL_MARKERS,
    _append_to_last_user_anthropic,
    _append_to_last_user_openai,
    _collect_tool_call_ids_openai,
    _collect_tool_use_ids_anthropic,
    _is_tool_result_only,
    _neutralize_user_forged_metadata,
    _sanitize_orphans_anthropic,
    _sanitize_orphans_openai,
    _strip_all_historical_thinking,
    _strip_empty_text_blocks,
    _strip_stale_proactive_notes,
)


# ── 2026-09-02（REFACTOR_ROADMAP P2.1 第 1 批绞杀搬迁）──────────────────
# cache_control 系列（BP1-BP4 断点、TTL、4 槽上限、客户端断点剥离）已搬到
# gateway/pipeline/cache.py，函数体零改动。此处保留 import 转发：hooks 内部
# 其余函数（preprocess / _apply_bp2 / _inject_seamless_context 等）与外部旧
# 引用照常工作；全部引用迁移完毕后再删这层转发。cache.py 不得反向 import
# hooks（会成环）。
from gateway.pipeline.cache import (  # noqa: F401
    _CACHE_5M,
    _CACHE_1H,
    _EPHEMERAL,
    _MAX_CACHE_BREAKPOINTS,
    _add_bp1_system,
    _add_bp3_freeze,
    _add_bp4_last_assistant,
    _add_cache_control_at,
    _cache_control_for_ttl,
    _cache_slots_left,
    _count_cache_control,
    _count_pairs,
    _deep_strip_cache_control,
    _enforce_cache_control_limit,
    _ensure_tool_cache_control,
    _has_cache_control,
    _limit_cache_controls,
    _normalize_tool_cache_control,
    _strategy_rounds,
    _strip_client_cache_control,
    _strip_client_cache_control_body,
    _summary_prefix_len,
)

# Issue #9: rebuild deduplication lock — 防止同 fingerprint 多次并发 rebuild
# 每条用户消息进来都会触发 _apply_bp2，如果 cold start 或 rolling 需要后台 rebuild，
# 没去重会导致 N 条消息 = N 个并发任务同时跑，狂打上游 API。
async def preprocess_anthropic(body: dict, upstream=None) -> dict:
    """Full Anthropic preprocessing pipeline——P2.2 声明式管道入口。

    2026-09-06 完成绞杀迁移（PLAN_PIPELINE_DECLARATIVE.md）：全部 20 步在
    pipeline/steps.py 的 PIPELINE_ANTHROPIC 步骤表里（名称/函数/after 依赖
    声明一目了然，新增功能 = 插一行）。_legacy_preprocess.py 保留作 AB
    对比基准（tests/test_pipeline_ab.py），稳定后可删。"""
    from gateway.pipeline.runner import PipelineContext, run_pipeline
    from gateway.pipeline.steps import PIPELINE_ANTHROPIC
    ctx = PipelineContext(body=dict(body), upstream=upstream)
    # 与旧实现一致：msgs 与 body["messages"] 是同一个 list 对象（浅拷贝语义）
    ctx.messages = ctx.body.get("messages", [])
    await run_pipeline(PIPELINE_ANTHROPIC, ctx)
    # 收尾兜底同步（各步骤已按原位置同步过，这里保证最终一致）
    ctx.body["messages"] = ctx.messages
    return ctx.body


def preprocess_openai(body: dict, upstream=None) -> dict:
    from gateway.summarizer import derive_conv_fingerprint
    from gateway.services.archiver import _compute_history_hash

    body = dict(body)
    msgs = body.get("messages", [])
    # 2026-06-09: 关闭主动消息处理（同 preprocess_anthropic）。
    # allow_proactive_tail_inference = body.get("stream") is False
    # msgs, proactive_trigger = _normalize_proactive_trigger(
    #     msgs,
    #     allow_placeholder_tail_inference=allow_proactive_tail_inference,
    # )
    # if not proactive_trigger:
    #     msgs = _drop_stale_assistant_tail(msgs)
    # 2026-07-11（PLAN_PROACTIVE_FIX.md）：确定性识别，同 preprocess_anthropic。
    proactive_trigger = _is_proactive_synthetic_user(msgs)
    if proactive_trigger:
        logger.info("Proactive trigger detected (deterministic suffix match, openai path)")
    body["_gateway_proactive_trigger"] = proactive_trigger
    fingerprint = derive_conv_fingerprint(msgs)
    history_hash = _compute_history_hash(msgs)
    tag = detect_tag(msgs)
    gateway_context_id = _extract_gateway_context_id(body, msgs) or _default_context_id_for_tag(tag)
    context_fingerprint = _context_fingerprint_from_id(gateway_context_id)
    body["_gateway_fingerprint"] = fingerprint
    body["_gateway_context_fingerprint"] = context_fingerprint
    body["_gateway_context_id"] = gateway_context_id
    body["_gateway_history_hash"] = history_hash
    selected_model = body.get("model", "")
    body["_gateway_cache_ttl"] = (
        upstream.get_cache_ttl("chat", model=selected_model)
        if upstream is not None and hasattr(upstream, "get_cache_ttl")
        else "1h"
    )
    body = _strip_gateway_metadata(body)
    msgs = _sanitize_orphans_openai(msgs)
    msgs = _strip_empty_text_blocks(msgs)
    msgs = _append_to_last_user_openai(msgs)
    body["messages"] = msgs
    return body
