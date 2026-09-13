"""声明式预处理管道的步骤表（P2.2，PLAN_PIPELINE_DECLARATIVE.md）。

每个 step_xxx 是"旧 preprocess_anthropic 中一段逻辑"的等价包装，函数体
按原顺序逐字搬迁；PIPELINE_ANTHROPIC 是步骤表——新增功能 = 插一行 + 声明
after。2026-09-06 迁移完成：preprocess_anthropic 全部 23 步入表，
hooks 入口只剩 run_pipeline 调用；_legacy_preprocess.py 保留作 AB
对比基准（tests/test_pipeline_ab.py 持续守护），稳定后可删。

不 import hooks（hooks 反向依赖 steps）；依赖各 pipeline 子模块与
summarizer/archiver/memory。
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
from gateway.pipeline.identity import (
    _context_fingerprint_from_id,
    _default_context_id_for_tag,
    _extract_gateway_context_id,
    _strip_gateway_metadata,
    detect_tag,
)
from gateway.pipeline.common import _add_gateway_protocol, _pipe_trace
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
from gateway.pipeline.proactive import _detect_and_rewrite_repeated_proactive, _is_proactive_synthetic_user
from gateway.pipeline.runner import PipelineContext, Step
from gateway.services.archiver import _compute_history_hash
from gateway.summarizer import derive_conv_fingerprint

logger = logging.getLogger("gateway.pipeline.steps")


# ────────────────────────────────────────────────
# 批次 1：阶段 A（身份与元数据提取）
# ────────────────────────────────────────────────
# 批次 2：阶段 B（清洗消毒）
# ────────────────────────────────────────────────

def step_strip_metadata(ctx: PipelineContext) -> None:
    ctx.cache_control = _cache_control_for_ttl(ctx.cache_ttl)
    ctx.body = _strip_gateway_metadata(ctx.body)
    if ctx.context_fingerprint:
        logger.info(
            "Gateway shared context: id=%s context_fp=%s session_fp=%s",
            ctx.context_id, ctx.context_fingerprint, ctx.fingerprint,
        )


def step_strip_client_cache(ctx: PipelineContext) -> None:
    """#6: gateway owns the four cache slots deterministically.
    Stable breakpoints follow upstream cache_ttl; BP4 stays 5m when caching is enabled."""
    ctx.body = _strip_client_cache_control_body(ctx.body)
    ctx.messages = _strip_client_cache_control(ctx.messages)
    ctx.body["messages"] = ctx.messages


def step_tools_cache(ctx: PipelineContext) -> None:
    """工具是稳定前缀：剥掉客户端选择，末位覆盖一个断点（档位随上游 TTL）。"""
    ctx.body = _normalize_tool_cache_control(ctx.body, cache_control=ctx.cache_control)


def step_normalize_thinking(ctx: PipelineContext) -> None:
    """Normalise non-standard thinking-effort levels that some upstreams reject.
    Map 'xhigh' → 'max' since not all upstreams support 'xhigh'."""
    body = ctx.body
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


def step_sanitize_strips(ctx: PipelineContext) -> None:
    """#7/#8/#9/#10 四连：剥历史 thinking → 剥空 text 块 → 防伪造元数据 →
    孤儿 tool_result 清理 → 剥过时主动唤起注入（缓存杀手）。顺序敏感：
    先剥 thinking 再清空块（剥完可能产生新空块）。"""
    msgs = _strip_all_historical_thinking(ctx.messages)  # #7: remove ALL historical thinking blocks
    msgs = _strip_empty_text_blocks(msgs)                # #8: remove empty text blocks → avoid 400
    msgs = _neutralize_user_forged_metadata(msgs)        # #9: 防伪——用户不能伪造网关 metadata 标记
    msgs = _sanitize_orphans_anthropic(msgs)
    msgs = _strip_stale_proactive_notes(msgs)            # #10: 剥掉历史中间过时的主动唤起注入（缓存杀手）
    ctx.messages = msgs


# ────────────────────────────────────────────────
# 批次 3：阶段 C（上下文注入）
# ────────────────────────────────────────────────

async def step_seamless(ctx: PipelineContext) -> None:
    """Issue #5: seamless session — if new session AND daily channel, inject summary.

    统一大脑打开时跳过旧 Seamless 注入（condition 控制）：_apply_bp2_unified
    会自己组装完整上下文，若先注入旧摘要+归档尾巴，会把合成消息当成客户端
    历史传给对账算法，污染增量识别 / 重roll / 编辑检测。"""
    ctx.messages = await _inject_seamless_context(
        ctx.messages,
        ctx.upstream,
        ctx.body,
        tag=ctx.tag,
        context_fingerprint=ctx.context_fingerprint,
        context_id=ctx.context_id,
    )


async def step_bp2_unified(ctx: PipelineContext) -> None:
    """统一大脑分支：reconcile 对账 + 统一 BP2。"""
    ctx.messages = await _apply_bp2_unified(
        ctx.messages,
        ctx.upstream,
        fingerprint=ctx.fingerprint,
        context_id=ctx.context_id,
    )
    ctx.messages = _sanitize_orphans_anthropic(ctx.messages)


async def step_bp2(ctx: PipelineContext) -> None:
    """常规 BP2：compress old context into summary (only if upstream available + ≥18 turns)。
    Skip BP2 entirely for tagged conversations — they live in their own channel.

    BP2 切割可能落在 tool_use/tool_result 之间，切完后必须再清一次孤儿
    tool_result 防止 Bedrock 400。"""
    ctx.messages = await _apply_bp2(
        ctx.messages,
        ctx.upstream,
        fingerprint=ctx.fingerprint,
        context_fingerprint=ctx.context_fingerprint,
        context_id=ctx.context_id,
    )
    ctx.messages = _sanitize_orphans_anthropic(ctx.messages)


async def step_memory_recall(ctx: PipelineContext) -> None:
    """Memory recall (ombre): 新话题时检索相关记忆，注入到「最新 user 消息之前」。
    必须放在 BP2 之后，否则 BP2 截断 pre_bp3 时会把注入吞掉。
    主动触发请求跳过（合成 user 无真实语义）。异常不致命，继续。"""
    try:
        from gateway.memory import maybe_recall_memories
        if not ctx.proactive_trigger:
            ctx.messages = await maybe_recall_memories(ctx.messages, tag=ctx.tag)
        else:
            logger.info("Memory recall skipped for proactive trigger")
    except Exception:
        logger.exception("Memory recall failed (non-fatal, continuing)")


# ────────────────────────────────────────────────
# 批次 4：阶段 D（收尾）
# ────────────────────────────────────────────────

def step_gateway_protocol(ctx: PipelineContext) -> None:
    """同步 messages 回 body 后，在 system 末尾追加网关注入协议
    （让 AI 信任网关注入的摘要/记忆等合成消息）。"""
    ctx.body["messages"] = ctx.messages
    ctx.body = _add_gateway_protocol(ctx.body)


def step_bp1_system(ctx: PipelineContext) -> None:
    """BP1: system 末块缓存断点（TTL 跟随上游）。"""
    ctx.body = _add_bp1_system(ctx.body, cache_control=ctx.cache_control)


def step_bp3_freeze(ctx: PipelineContext) -> None:
    """BP3: 冻结窗口边界断点（稳定前缀）。"""
    ctx.messages = _add_bp3_freeze(ctx.messages, body=ctx.body, cache_control=ctx.cache_control)
    ctx.body["messages"] = ctx.messages


def step_bp4_last_assistant(ctx: PipelineContext) -> None:
    """BP4: 最后一条 assistant 的滑动断点（固定 5m）。缓存关闭时不注入。"""
    if ctx.cache_control is not None:
        ctx.messages = _add_bp4_last_assistant(ctx.messages, body=ctx.body)


def step_timestamp(ctx: PipelineContext) -> None:
    """时间戳注入在最后一条真实 user 上（在缓存断点之外，见 BP4 注释）。"""
    ctx.messages = _append_to_last_user_anthropic(ctx.messages)


def step_cache_limit(ctx: PipelineContext) -> None:
    """Anthropic 最多 4 个 cache 槽位：保留前 4 个，剥掉溢出。"""
    ctx.body["messages"] = ctx.messages
    ctx.body = _enforce_cache_control_limit(ctx.body)


def step_force_think(ctx: PipelineContext) -> None:
    """强制思考：在消息列表末尾追加助手预填充（<think>）。只有最后一条是
    user 消息时才注入（避免重复）。"""
    if ctx.upstream is not None and getattr(ctx.upstream, "force_think", False):
        think_tag = getattr(ctx.upstream, "think_tag", "<think>\n") or "<think>\n"
        msgs = ctx.body.get("messages", [])
        if msgs and msgs[-1].get("role") == "user":
            ctx.body["messages"] = list(msgs) + [{"role": "assistant", "content": think_tag}]
            logger.debug("force_think: injected prefill for upstream %s", ctx.upstream.name)
# ────────────────────────────────────────────────

def step_proactive_detect(ctx: PipelineContext) -> None:
    """确定性识别橘瓣主动触发；X-Proactive header 强制置位（P3.2）。"""
    proactive_trigger = _is_proactive_synthetic_user(ctx.messages)
    if proactive_trigger:
        logger.info("Proactive trigger detected (deterministic suffix match)")
    # P3.2（2026-09-06）：橘瓣端将来带 X-Proactive: true 显式标记时直接认定
    # 主动触发（网关侧先行就绪；header 缺失时行为与原来完全一致）
    if ctx.body.pop("_gateway_proactive_header", False):
        proactive_trigger = True
        logger.info("Proactive trigger forced by X-Proactive header")
    ctx.proactive_trigger = proactive_trigger


async def step_identity_fingerprint(ctx: PipelineContext) -> None:
    """指纹与历史哈希——必须基于清洗前的原始 messages（重roll 去重依赖稳定性）。"""
    ctx.fingerprint = derive_conv_fingerprint(ctx.messages)
    # 对全量历史做 json.dumps+md5（带图请求 MB 级）是纯同步 CPU，
    # 丢线程池防大请求卡事件循环（2026-09-06）
    ctx.history_hash = await asyncio.to_thread(_compute_history_hash, ctx.messages)


async def step_proactive_rewrite(ctx: PipelineContext) -> None:
    """形状 B（旧版橘瓣）：末尾 user 与上一条 user 完全相同 + 冷场 ≥ N 分钟
    → 主动触发，重发文本被替换为网关主动唤起指令。需要 fingerprint 查
    冷场时间，所以放在指纹计算之后。注意：替换发生在 history_hash 计算
    之后，re-roll 去重不受影响。"""
    if not ctx.proactive_trigger:
        ctx.proactive_trigger, ctx.messages = await _detect_and_rewrite_repeated_proactive(
            ctx.messages, ctx.fingerprint)


def step_set_proactive_key(ctx: PipelineContext) -> None:
    ctx.body["_gateway_proactive_trigger"] = ctx.proactive_trigger


def step_identity_tag_context(ctx: PipelineContext) -> None:
    ctx.tag = detect_tag(ctx.messages)
    ctx.context_id = _extract_gateway_context_id(ctx.body, ctx.messages) or _default_context_id_for_tag(ctx.tag)
    ctx.context_fingerprint = _context_fingerprint_from_id(ctx.context_id)
    # 统一大脑总开关：提前算好，因为它同时影响 Seamless 注入是否跳过，
    # 以及 BP2 是否走新分支。默认 False，不影响现状。
    from gateway.settings import get as _get_setting
    ctx.unified_enabled = bool(_get_setting("unified_brain_enabled", False))


def step_identity_gateway_keys(ctx: PipelineContext) -> None:
    # passed to archiver downstream（routers 在 preprocess 后 pop 这些键）
    ctx.body["_gateway_tag"] = ctx.tag
    ctx.body["_gateway_fingerprint"] = ctx.fingerprint       # 稳定指纹（基于原始 messages）
    ctx.body["_gateway_context_fingerprint"] = ctx.context_fingerprint
    ctx.body["_gateway_context_id"] = ctx.context_id
    ctx.body["_gateway_history_hash"] = ctx.history_hash


def step_cache_ttl(ctx: PipelineContext) -> None:
    selected_model = ctx.body.get("model", "")
    ctx.cache_ttl = (
        ctx.upstream.get_cache_ttl("chat", model=selected_model)
        if ctx.upstream is not None and hasattr(ctx.upstream, "get_cache_ttl")
        else "1h"
    )
    ctx.body["_gateway_cache_ttl"] = ctx.cache_ttl


# 步骤表：按列表顺序执行；after 只做校验（文档第五节"为什么不用拓扑排序"）
PIPELINE_ANTHROPIC: list[Step] = [
    Step("proactive_detect", step_proactive_detect, after=[]),
    Step("identity_fingerprint", step_identity_fingerprint, after=["proactive_detect"]),
    Step("proactive_rewrite", step_proactive_rewrite, after=["identity_fingerprint"]),
    Step("set_proactive_key", step_set_proactive_key, after=["proactive_rewrite"]),
    Step("identity_tag_context", step_identity_tag_context, after=["proactive_rewrite"]),
    Step("identity_gateway_keys", step_identity_gateway_keys, after=["identity_tag_context"]),
    Step("cache_ttl", step_cache_ttl, after=["identity_tag_context"]),
    # ── 批次 2：清洗消毒 ──
    Step("strip_metadata", step_strip_metadata, after=["cache_ttl"]),
    Step("strip_client_cache", step_strip_client_cache, after=["strip_metadata"]),
    Step("tools_cache", step_tools_cache, after=["strip_client_cache"]),
    Step("normalize_thinking", step_normalize_thinking, after=["tools_cache"]),
    Step("sanitize_strips", step_sanitize_strips, after=["normalize_thinking"]),
    # ── 批次 3：上下文注入 ──
    Step("seamless", step_seamless, after=["sanitize_strips"],
         condition=lambda c: c.upstream is not None and not c.unified_enabled),
    Step("bp2_unified", step_bp2_unified, after=["sanitize_strips"],
         condition=lambda c: c.unified_enabled and c.upstream is not None and not c.tag),
    Step("bp2", step_bp2, after=["sanitize_strips"],
         condition=lambda c: not c.unified_enabled and c.upstream is not None and not c.tag),
    Step("memory_recall", step_memory_recall, after=["bp2", "bp2_unified"]),
    # ── 批次 4：收尾 ──
    Step("gateway_protocol", step_gateway_protocol, after=["memory_recall"]),
    Step("bp1_system", step_bp1_system, after=["gateway_protocol"]),
    Step("bp3_freeze", step_bp3_freeze, after=["bp1_system"]),
    Step("bp4_last_assistant", step_bp4_last_assistant, after=["bp3_freeze"]),
    Step("timestamp", step_timestamp, after=["bp4_last_assistant"]),
    Step("cache_limit", step_cache_limit, after=["timestamp"]),
    Step("force_think", step_force_think, after=["cache_limit"]),
]
