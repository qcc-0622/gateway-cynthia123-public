"""preprocess_anthropic 原版实现（P2.2 迁移期的 AB 对比基准 / oracle）。

2026-09-06 从 hooks.py 原样提取（函数体零改动）。作用：
tests/test_pipeline_ab.py 用它与声明式管道（pipeline/steps.py + runner.py）
做同输入对比，两者输出必须完全一致；全部批次通过且稳定运行一段时间后，
本文件可整体删除（它不参与生产路径——hooks.preprocess_anthropic 是唯一入口）。

不 import hooks（hooks 反向依赖本模块，防循环）；依赖各 pipeline 子模块。
"""

import asyncio
import logging

from gateway.pipeline.cache import (
    _add_bp1_system,
    _add_bp3_freeze,
    _add_bp4_last_assistant,
    _cache_control_for_ttl,
    _enforce_cache_control_limit,
    _normalize_tool_cache_control,
    _strip_client_cache_control,
    _strip_client_cache_control_body,
)
from gateway.pipeline.common import _add_gateway_protocol, _pipe_trace
from gateway.pipeline.identity import (
    _context_fingerprint_from_id,
    _default_context_id_for_tag,
    _extract_gateway_context_id,
    _strip_gateway_metadata,
    detect_tag,
)
from gateway.pipeline.proactive import _detect_and_rewrite_repeated_proactive, _is_proactive_synthetic_user
from gateway.pipeline.sanitize import (
    _append_to_last_user_anthropic,
    _neutralize_user_forged_metadata,
    _sanitize_orphans_anthropic,
    _strip_all_historical_thinking,
    _strip_empty_text_blocks,
    _strip_stale_proactive_notes,
)
from gateway.pipeline.summary import _apply_bp2, _inject_seamless_context
from gateway.pipeline.unified import _apply_bp2_unified

logger = logging.getLogger("gateway.pipeline.legacy")


async def _preprocess_anthropic_legacy(body: dict, upstream=None) -> dict:
    """Full Anthropic preprocessing pipeline (async for BP2 summarization)."""
    from gateway.summarizer import derive_conv_fingerprint
    body = dict(body)
    msgs = body.get("messages", [])
    # 2026-06-09: 关闭主动消息处理（_normalize_proactive_trigger / _drop_stale_assistant_tail）。
    # 原因：codex 反复迭代 5 次仍"已知不稳定"（CLAUDE.md 第 546 行），用户决定放弃。
    # 网关原样转发 messages，由上游/客户端自行处理 assistant-prefill 续写。
    # 若以后要恢复：取消下方注释即可。
    # allow_proactive_tail_inference = body.get("stream") is False
    # msgs, proactive_trigger = _normalize_proactive_trigger(
    #     msgs,
    #     allow_placeholder_tail_inference=allow_proactive_tail_inference,
    # )
    # if not proactive_trigger:
    #     msgs = _drop_stale_assistant_tail(msgs)
    #
    # 2026-07-11（PLAN_PROACTIVE_FIX.md）：改用**确定性识别**——橘瓣
    # ProactiveMessageService 的合成 user 消息带固定尾缀（源码常量），
    # 尾缀匹配即为主动触发请求，不再靠语义猜测。识别在原始 messages 上做
    # （time_reminder 注入在管道第 24 步，此处消息未被注入）。
    proactive_trigger = _is_proactive_synthetic_user(msgs)
    if proactive_trigger:
        logger.info("Proactive trigger detected (deterministic suffix match)")
    # P3.2（2026-09-06）：橘瓣端将来带 X-Proactive: true 显式标记时直接认定
    # 主动触发（网关侧先行就绪；header 缺失时行为与原来完全一致）
    if body.pop("_gateway_proactive_header", False):
        proactive_trigger = True
        logger.info("Proactive trigger forced by X-Proactive header")

    # Compute fingerprint from ORIGINAL messages BEFORE any cleaning,
    # so it stays stable across preprocessing changes.
    from gateway.services.archiver import _compute_history_hash
    from gateway.settings import get as _get_setting
    fingerprint = derive_conv_fingerprint(msgs)
    # 对全量历史做 json.dumps+md5（带图请求 MB 级）是纯同步 CPU，
    # 丢线程池防大请求卡事件循环（2026-09-06；sync 的 preprocess_openai 冷路径不包）
    history_hash = await asyncio.to_thread(_compute_history_hash, msgs)
    # 形状 B（旧版橘瓣）：末尾 user 与上一条 user 完全相同 + 冷场 ≥ N 分钟
    # → 主动触发，重发文本被替换为网关主动唤起指令。需要 fingerprint 查
    # 冷场时间，所以放在 fingerprint 计算之后。注意：替换发生在 history_hash
    # 计算之后，re-roll 去重不受影响。
    if not proactive_trigger:
        proactive_trigger, msgs = await _detect_and_rewrite_repeated_proactive(msgs, fingerprint)
    body["_gateway_proactive_trigger"] = proactive_trigger
    _pipe_trace("proactive_detect", msgs)
    tag = detect_tag(msgs)
    gateway_context_id = _extract_gateway_context_id(body, msgs) or _default_context_id_for_tag(tag)
    context_fingerprint = _context_fingerprint_from_id(gateway_context_id)
    # 统一大脑总开关：提前算好，因为它同时影响下面的 Seamless 注入是否跳过，
    # 以及 BP2 是否走新分支。默认 False，不影响现状。
    _unified_enabled = bool(_get_setting("unified_brain_enabled", False))
    body["_gateway_tag"] = tag                  # passed to archiver downstream
    body["_gateway_fingerprint"] = fingerprint  # 稳定的 fingerprint（基于原始 messages）
    body["_gateway_context_fingerprint"] = context_fingerprint
    body["_gateway_context_id"] = gateway_context_id
    body["_gateway_history_hash"] = history_hash
    selected_model = body.get("model", "")
    cache_ttl = (
        upstream.get_cache_ttl("chat", model=selected_model)
        if upstream is not None and hasattr(upstream, "get_cache_ttl")
        else "1h"
    )
    cache_control = _cache_control_for_ttl(cache_ttl)
    body["_gateway_cache_ttl"] = cache_ttl
    body = _strip_gateway_metadata(body)
    if context_fingerprint:
        logger.info(
            "Gateway shared context: id=%s context_fp=%s session_fp=%s",
            gateway_context_id, context_fingerprint, fingerprint,
        )

    # #6: gateway owns the four cache slots deterministically.
    # Stable breakpoints follow upstream cache_ttl; BP4 stays 5m when caching is enabled.
    body = _strip_client_cache_control_body(body)
    msgs = _strip_client_cache_control(msgs)
    body["messages"] = msgs
    body = _normalize_tool_cache_control(body, cache_control=cache_control)

    # Normalise non-standard thinking-effort levels that some upstreams reject.
    # Map 'xhigh' → 'max' since not all upstreams support 'xhigh'.
    if "reasoning_effort" in body and body["reasoning_effort"] == "xhigh":
        body["reasoning_effort"] = "max"
    if "thinking" in body and isinstance(body["thinking"], dict):
        th = body["thinking"]
        t = th.get("type")
        if t in ("enabled", "disabled", "adaptive"):
            # Anthropic 原生合法 type：只规范化 effort（如果有）
            if th.get("effort") == "xhigh":
                body["thinking"] = {**th, "effort": "max"}
        else:
            # 其他未知扩展 → 兜底规范化为 enabled（保留这条兜底，万一未来有更怪的）
            budget = th.get("budget_tokens") or th.get("budget") or 10000
            try:
                budget = int(budget)
            except (TypeError, ValueError):
                budget = 10000
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            logger.info(
                "thinking normalized: type=%r → enabled (budget_tokens=%d) "
                "(unknown thinking type, falling back to standard enabled)",
                t, budget,
            )
    msgs = _strip_all_historical_thinking(msgs) # #7: remove ALL historical thinking blocks
    msgs = _strip_empty_text_blocks(msgs)       # #8: remove empty text blocks → avoid 400
    msgs = _neutralize_user_forged_metadata(msgs)  # #9: 防伪——用户不能伪造网关 metadata 标记
    msgs = _sanitize_orphans_anthropic(msgs)
    msgs = _strip_stale_proactive_notes(msgs)   # #10: 剥掉历史中间过时的主动唤起注入（缓存杀手）
    _pipe_trace("strips_done", msgs)

    # Issue #5: seamless session — if new session AND daily channel, inject summary
    #
    # 统一大脑打开时跳过旧 Seamless 注入：_apply_bp2_unified 会自己组装完整
    # 上下文（全局摘要 + 全局尾巴 + 增量），如果这里先注入了旧摘要 + 60 条
    # 归档，会把这些"合成消息"当成客户端历史的一部分传给下面的对账算法，
    # 污染增量识别 / 重roll / 编辑检测（合成消息不是真的客户端发来的内容）。
    if upstream is not None and not _unified_enabled:
        msgs = await _inject_seamless_context(
            msgs,
            upstream,
            body,
            tag=tag,
            context_fingerprint=context_fingerprint,
            context_id=gateway_context_id,
        )
    _pipe_trace("seamless_done", msgs)

    # BP2: compress old context into summary (only if upstream available + ≥18 turns)
    # Skip BP2 entirely for tagged conversations — they live in their own channel.
    #
    # 统一大脑总开关（PLAN_UNIFIED_BRAIN.md）：默认 False，关闭时下面这个
    # if 分支永远不会进入，_apply_bp2() 原样执行，现状完全不变。
    if _unified_enabled and upstream is not None and not tag:
        msgs = await _apply_bp2_unified(
            msgs,
            upstream,
            fingerprint=fingerprint,
            context_id=gateway_context_id,
        )
        msgs = _sanitize_orphans_anthropic(msgs)
    elif upstream is not None and not tag:
        msgs = await _apply_bp2(
            msgs,
            upstream,
            fingerprint=fingerprint,
            context_fingerprint=context_fingerprint,
            context_id=gateway_context_id,
        )
        # 2026-06-09: 关闭 BP2 后的 stale tail 清理（同上）。
        # if not proactive_trigger:
        #     msgs = _drop_stale_assistant_tail(msgs)
        # Fix TOOL_USE_RESULT_MISMATCH: BP2 切割可能落在 tool_use/tool_result 之间，
        # 切完后 rest 里会留下孤立的 tool_result，必须再清一次防止 Bedrock 400。
        msgs = _sanitize_orphans_anthropic(msgs)
    _pipe_trace("bp2_done", msgs)

    # Memory recall (ombre): 新话题时检索 ombre 相关记忆，注入到「最新 user 消息之前」
    # 注意：必须放在 BP2 之后，否则 BP2 截断 pre_bp3 时会把注入吞掉
    try:
        from gateway.memory import maybe_recall_memories
        if not proactive_trigger:
            msgs = await maybe_recall_memories(msgs, tag=tag)
        else:
            logger.info("Memory recall skipped for proactive trigger")
    except Exception:
        logger.exception("Memory recall failed (non-fatal, continuing)")
    _pipe_trace("memory_recall_done", msgs)

    body["messages"] = msgs
    body = _add_gateway_protocol(body)      # 在 system 末尾说明网关注入协议（让 AI 信任注入）
    body = _add_bp1_system(body, cache_control=cache_control)  # BP1: system cache
    msgs = _add_bp3_freeze(msgs, body=body, cache_control=cache_control)  # BP3: stable freeze window
    body["messages"] = msgs
    if cache_control is not None:
        msgs = _add_bp4_last_assistant(msgs, body=body)    # BP4: sliding breakpoint
    msgs = _append_to_last_user_anthropic(msgs)  # timestamp (outside cached prefix)
    _pipe_trace("final", msgs)

    body["messages"] = msgs
    body = _enforce_cache_control_limit(body)

    # 强制思考：在消息列表末尾追加助手预填充（<think>
    if upstream is not None and getattr(upstream, "force_think", False):
        tag = getattr(upstream, "think_tag", "<think>\n") or "<think>\n"
        msgs = body.get("messages", [])
        # 只有最后一条是 user 消息时才注入（避免重复）
        if msgs and msgs[-1].get("role") == "user":
            msgs = list(msgs) + [{"role": "assistant", "content": tag}]
            body["messages"] = msgs
            logger.debug("force_think: injected prefill for upstream %s", upstream.name)

    return body
