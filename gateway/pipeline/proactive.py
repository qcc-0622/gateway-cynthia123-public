"""主动消息：橘瓣 ProactiveMessageService 的确定性识别与改写（P2.1，2026-09-06）。

自 hooks.py 绞杀搬迁，函数体零改动；hooks.py 留 import 转发。
依赖共享工具（pipeline.common）；不 import hooks（防循环）。
"""

import logging
import re

from gateway.pipeline.common import _TIME_REMINDER_RE, _content_text, _pipe_trace
from gateway.pipeline.sanitize import _is_tool_result_only

logger = logging.getLogger("gateway.pipeline.proactive")


_PROACTIVE_PREFIX = "[网关主动唤起 · 非用户消息]"
# 橘瓣 ProactiveMessageService 合成 user 消息的固定尾缀（常量，确定性标记）。
# 源码：_temp/orangechat/.../data/service/ProactiveMessageService.kt 482-487 行：
#   contextStr + "\n\n如果你觉得现在没什么好说的，或者没什么有趣的话题，
#                 请只回复 [PASS] 即可，不要强行找话题。"
# 合成消息不会存进橘瓣本地对话（只有 AI 回复会存），所以正常历史里不会出现
# 这句话——用它做尾缀匹配是确定性的，不是猜测（区别于 Issue #36 注释掉的旧路径）。
_PROACTIVE_SYNTHETIC_SUFFIX = "请只回复 [PASS] 即可，不要强行找话题。"
_PROACTIVE_PLACEHOLDERS = {
    "",
    ".",
    "。",
    "hi",
    "hello",
    "hey",
    "嗨",
    "你好",
    "哈喽",
}


def _is_proactive_synthetic_user(messages: list) -> bool:
    """确定性识别橘瓣 ProactiveMessageService 的主动触发请求（PLAN_PROACTIVE_FIX.md）。

    判定：messages 最后一条是 user，且其文本（剥离 <time_reminder> 后 strip）
    以固定尾缀 `_PROACTIVE_SYNTHETIC_SUFFIX` 结尾。

    为什么剥 time_reminder：识别在管道第 3 步附近执行、时间戳注入在第 24 步，
    正常情况识别时消息还没被注入；但保留剥离作为防御（万一调用顺序变动，
    或未来有路径拿到注入后的 messages 再调本函数）。

    宽容处理：橘瓣源码 490-496 行显示合成消息还会过 inputTransformers +
    templateTransformer（Pebble 模板渲染、正则转换、lorebook 注入等），
    某些 transformer 可能在消息**尾部追加**内容导致 endswith 不中。
    所以 endswith 不中时退一步做包含匹配（整句固定尾缀出现在正文任何位置）——
    这句话是常量长句，用户正常聊天不可能打出来，且合成消息不会存进橘瓣
    本地历史（不会随历史重发），包含匹配依然是确定性的，误判概率可忽略。
    """
    if not messages or messages[-1].get("role") != "user":
        return False
    text = _TIME_REMINDER_RE.sub("", _content_text(messages[-1].get("content", ""))).strip()
    if not text:
        return False
    if text.endswith(_PROACTIVE_SYNTHETIC_SUFFIX):
        return True
    if _PROACTIVE_SYNTHETIC_SUFFIX in text:
        logger.warning(
            "Proactive synthetic user detected by containment (not suffix) — "
            "an input transformer may have appended content after the fixed suffix"
        )
        return True
    return False


# 2026-07-12 实测修正：用户的触发器间隔是 1~2 分钟（21:56/21:57/21:59 三连重发），
# 原默认 20 分钟把真实触发全挡住了。降到 1 分钟——"手动原样重发同一句且隔 1 分钟
# 以上"极罕见，且误判后果只是 AI 主动接话而非机械重答，可接受。
_PROACTIVE_REPEAT_IDLE_MINUTES_DEFAULT = 1


async def _detect_and_rewrite_repeated_proactive(msgs: list, fingerprint: str) -> tuple[bool, list]:
    """形状 B 主动触发识别（2026-07-12，用户实测截图证实）。

    旧版橘瓣的主动消息不发 [PASS] 合成消息（那是形状 A / 新版源码的行为），
    而是把用户的最后一句**原样重发**来触发 AI——客户端不会把这条重发存进
    本地历史，模型看到重复消息只会傻乎乎再答一遍（用户 0709 凌晨的实测截图，
    连续重复回答四五遍）。

    确定性判定 = 重复 + 冷场，两个条件缺一不可：
      1. 最后一条 user 与它之前最近的一条 user 文本完全相同（剥 time_reminder
         后比较），且中间隔着 assistant 回复；
      2. 该 session 距最近一次归档已闲置 ≥ proactive_repeat_idle_minutes
         （默认 20 分钟）——排除用户手动连发两遍相同内容（那种间隔是秒级的，
         而主动消息定时器最小间隔就是几十分钟）。

    命中后把重发文本**替换**为网关主动唤起指令（带 _PROACTIVE_PREFIX 前缀：
    memory.py skip_markers 会跳过它不喂 ombre；archiver 的 proactive flag
    负责不归档它），并返回 (True, 重写后的 msgs)。

    已知误判边界：用户在冷场 20+ 分钟后手动重发同一句话，会被当成主动触发
    ——AI 不再机械重答而是主动接话，语义上可接受，且有 INFO 日志可查。
    """
    if len(msgs) < 2:
        return False, msgs

    # ── 形状判定 ─────────────────────────────────────────────────
    # 形状 C（2026-07-12 探针实测，用户 0709 手写规则 v1 描述的就是它）：
    #   请求以 assistant 自己的最后回复结尾（continuation/prefill 式触发）。
    #   正常聊天请求永远以 user 结尾；以 assistant 结尾只有两种可能——
    #   主动触发，或用户点了"继续生成"（后者发生在回复刚断掉的几秒内，
    #   被下面的冷场判定天然排除）。附带收益：把它转成"追加 user 指令"
    #   之后，不再以 assistant 结尾，治好部分上游 400
    #   "This model does not support assistant message prefill"（0709 截图）。
    # 形状 B（保留，其他版本可能用）：末尾 user 与上一条 user 完全相同。
    shape = ""
    if msgs[-1].get("role") == "assistant":
        shape = "C"
    elif msgs[-1].get("role") == "user" and msgs[-2].get("role") == "assistant" and len(msgs) >= 3:
        cur = _TIME_REMINDER_RE.sub("", _content_text(msgs[-1].get("content", ""))).strip()
        prev = ""
        for m in reversed(msgs[:-1]):
            if m.get("role") == "user":
                prev = _TIME_REMINDER_RE.sub("", _content_text(m.get("content", ""))).strip()
                break
        if cur and prev == cur:
            shape = "B"
    if not shape:
        return False, msgs

    # ── 冷场判定（两种形状共用）────────────────────────────────────
    from gateway.settings import get as _sget
    idle_threshold = int(_sget("proactive_repeat_idle_minutes", _PROACTIVE_REPEAT_IDLE_MINUTES_DEFAULT))
    idle_minutes = None
    try:
        from gateway import db as _db
        last_at = await _db.get_last_archived_at(fingerprint)
        if last_at:
            from datetime import datetime as _dt
            last_dt = _dt.fromisoformat(last_at)
            now_dt = _dt.now(last_dt.tzinfo) if last_dt.tzinfo else _dt.now()
            idle_minutes = (now_dt - last_dt).total_seconds() / 60.0
    except Exception:
        logger.exception("Proactive shape %s: idle lookup failed, treating as normal message", shape)
        return False, msgs

    if idle_minutes is None:
        # 该 fingerprint 下查无归档（主动触发请求的消息列表开头和正常聊天不同，
        # fingerprint 是独立的，首次触发必然查无归档）：
        #   形状 C：以 assistant 结尾本身就是极强信号，放行判定为主动触发
        #           （误判面只剩"新会话里点继续生成"，罕见且后果可接受）
        #   形状 B：重复消息信号较弱，保守放弃
        if shape != "C":
            return False, msgs
        idle_minutes = -1.0  # 标记"未知"，note 里不写具体分钟数
    elif idle_minutes < idle_threshold:
        logger.info(
            "Proactive shape %s: signal matched but idle %.1f min < %d min threshold, "
            "treating as normal message (continue-generation / manual resend)",
            shape, idle_minutes, idle_threshold,
        )
        return False, msgs

    # 连发感知（2026-07-13，用户反馈"AI 对之前的主动消息感知很弱"）：
    # 数时间线末尾连续的 assistant 行。她最后一条消息之后的第一条是正常回复，
    # 再往后每一条都是之前的主动消息 → 之前主动发送数 = 连续数 - 1。
    prev_proactive_count = 0
    try:
        _tail_rows = await _db.get_global_timeline_for_session(fingerprint, after_id=0)
        _consec = 0
        for _r in reversed(_tail_rows):
            if _r["role"] == "assistant":
                _consec += 1
            else:
                break
        prev_proactive_count = max(0, _consec - 1)
    except Exception:
        logger.exception("Proactive: consecutive-count lookup failed (non-fatal)")

    idle_desc = f"约 {int(idle_minutes)} 分钟" if idle_minutes >= 0 else "一段时间"
    if prev_proactive_count >= 1:
        streak_note = (
            f"特别注意：她未回复期间你已经主动发过 {prev_proactive_count} 条消息"
            "（就是对话末尾连续的那几条，都是你自己发的）。这次严禁重复其中任何"
            "话题、句式或关心点；要么换一个明显不同且简短的角度，要么只说一句"
            "轻轻的陪伴话。她一直没回就要克制，宁可少说。"
        )
    else:
        streak_note = "不要回答或重复她之前说过的话，也不要续写或重复你自己的上一条回复。"
    note = (
        f"{_PROACTIVE_PREFIX} 这是定时器触发的主动消息窗口：她已经{idle_desc}没有说话，"
        f"并没有发新消息。{streak_note}"
        "基于当前时间和你们最近聊的内容，自然地主动开口；"
        "实在没有想说的就说一句简短的陪伴式问候，不要强行找话题。"
    )
    msgs = list(msgs)
    if shape == "C":
        # 追加 user 指令：既告知模型主动开口，又让请求不再以 assistant 结尾
        msgs.append({"role": "user", "content": note})
    else:
        # 形状 B：重发的 user 文本直接替换为指令
        msgs[-1] = {"role": "user", "content": note}
    logger.info(
        "Proactive trigger detected (shape %s, idle=%s, threshold=%d min)",
        shape, f"{idle_minutes:.0f}min" if idle_minutes >= 0 else "unknown", idle_threshold,
    )
    return True, msgs


def _is_proactive_trigger_text(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    compact = "".join(text.split())
    has_marker = (
        "主动唤起" in text
        or "主动开口" in text
        or "主动" in compact
        or "proactive" in lower
    )
    has_semantics = (
        "定时器" in text
        or "没说话" in text
        or "没有发" in text
        or "空字符串" in text
        or "last message" in lower
        or "timer" in lower
        or "empty string" in lower
    )
    # RikkaHub/橘瓣主动消息常见形态：最后一条 assistant 是一个
    # "## 主动唤起" 提醒，而不是用户新消息。保守要求：像系统提醒
    # 标题 + 主动语义 + 定时/空回复语义同时出现。
    looks_like_notice = text.lstrip().startswith(("##", "#", "[", "【"))
    return has_marker and has_semantics and (looks_like_notice or "proactive" in lower)


def _format_proactive_trigger(original_text: str) -> str:
    return (
        f"{_PROACTIVE_PREFIX}\n"
        "前端定时器正在请求你主动开口。用户此刻没有发送新消息；"
        "不要把历史里最后一条 user 消息当成刚刚收到的问题来回答，"
        "也不要续写或重复历史里最后一条 assistant 消息。"
        "不要复述、解释或确认前端原始提醒本身。\n"
        "请基于当前时间、BP2 摘要和最近上下文，自然地主动说一句新的话。"
        "如果确实没有合适内容，输出空字符串。\n\n"
        "前端原始提醒：\n"
        f"{original_text.strip()}"
    )


def _proactive_user_message(original_text: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "text", "text": _format_proactive_trigger(original_text)}],
    }


def _is_proactive_placeholder_text(text: str) -> bool:
    if text is None:
        return True
    cleaned = _TIME_REMINDER_RE.sub("", str(text)).strip()
    compact = re.sub(r"[\s\u200b\u200c\u200d\ufeff]+", "", cleaned).lower()
    return compact in _PROACTIVE_PLACEHOLDERS


def _assistant_has_tool_use(message: dict) -> bool:
    content = message.get("content", "")
    if isinstance(content, list):
        return any(isinstance(block, dict) and block.get("type") == "tool_use" for block in content)
    return bool(message.get("tool_calls"))


def _proactive_tail_is_current(
    messages: list,
    idx: int,
    allow_placeholder_user_tail: bool = False,
) -> bool:
    """Accept proactive markers only when they are the current tail, not history."""
    tail = messages[idx + 1:]
    if not tail:
        return True

    first = tail[0]
    if first.get("role") == "assistant":
        return all(msg.get("role") == "assistant" for msg in tail)

    if not allow_placeholder_user_tail:
        return False
    if first.get("role") != "user":
        return False
    if _is_tool_result_only(first.get("content", "")):
        return False
    if not _is_proactive_placeholder_text(_content_text(first.get("content", ""))):
        return False

    for msg in tail[1:]:
        role = msg.get("role")
        if role == "assistant":
            continue
        if role != "user":
            return False
        content = msg.get("content", "")
        if _is_tool_result_only(content):
            return False
        if not _is_proactive_placeholder_text(_content_text(content)):
            return False

    return True


def _infer_proactive_from_assistant_tail(
    messages: list,
    allow_completed_tail: bool = False,
) -> tuple[list, bool]:
    """Infer Orange/RikkaHub timer wakeups from non-stream placeholder tails."""
    if not messages:
        return messages, False
    tail_idx = len(messages) - 1
    trailing_assistants = 0
    while tail_idx >= 0 and messages[tail_idx].get("role") == "assistant":
        if _assistant_has_tool_use(messages[tail_idx]):
            return messages, False
        trailing_assistants += 1
        tail_idx -= 1
    if trailing_assistants <= 0 or tail_idx < 0:
        return messages, False
    last_user = messages[tail_idx]
    if last_user.get("role") != "user":
        return messages, False
    if _is_tool_result_only(last_user.get("content", "")):
        return messages, False
    placeholder = _content_text(last_user.get("content", ""))
    if not _is_proactive_placeholder_text(placeholder):
        if not allow_completed_tail:
            return messages, False
        last_assistant_text = _content_text(messages[-1].get("content", ""))
        if not last_assistant_text.strip():
            return messages, False
        fallback_notice = (
            "## 主动唤起（网关已完成尾部识别）\n"
            "前端这次请求没有新增用户消息；请求尾部已经包含上一轮 user 和 assistant 完整回复。\n"
            "这通常表示前端正在尝试定时主动发消息。\n"
            "请把上一轮 user 当作已经回答过的历史，不要再次回答、解释或补写那一轮；"
            "尤其不要再确认消息是否完整、是否收到、是否看到日志。"
            "只基于当前时间、摘要和最近上下文，自然地主动说一句新的话。"
        )
        history = list(messages[:tail_idx + 1])
        if trailing_assistants:
            history.append(messages[-1])
        logger.info(
            "Proactive trigger inferred from non-stream completed assistant tail: "
            "user_idx=%d assistants=%d",
            tail_idx,
            trailing_assistants,
        )
        return history + [_proactive_user_message(fallback_notice)], True

    fallback_notice = (
        "## 主动唤起（网关兜底识别）\n"
        "前端没有保留原始主动唤起提醒，但请求尾部以 assistant 预填充结尾，"
        f"且最近 user 只有占位文本 {placeholder.strip()!r}。\n"
        "这通常表示前端正在尝试主动发消息。"
    )
    logger.info(
        "Proactive trigger inferred from non-stream assistant tail: user_idx=%d assistants=%d",
        tail_idx,
        trailing_assistants,
    )
    return list(messages[:tail_idx]) + [_proactive_user_message(fallback_notice)], True


def _drop_stale_assistant_tail(messages: list) -> list:
    """Drop client-persisted assistant replies after the current user turn."""
    if not messages:
        return messages
    tail_idx = len(messages) - 1
    trailing_assistants = 0
    while tail_idx >= 0 and messages[tail_idx].get("role") == "assistant":
        if _assistant_has_tool_use(messages[tail_idx]):
            return messages
        trailing_assistants += 1
        tail_idx -= 1
    if trailing_assistants <= 0 or tail_idx < 0:
        return messages
    last_user = messages[tail_idx]
    if last_user.get("role") != "user":
        return messages
    if _is_tool_result_only(last_user.get("content", "")):
        return messages

    logger.warning(
        "Dropped stale assistant tail after user turn: user_idx=%d dropped=%d",
        tail_idx,
        trailing_assistants,
    )
    return list(messages[:tail_idx + 1])


def _normalize_proactive_trigger(
    messages: list,
    allow_placeholder_tail_inference: bool = False,
) -> tuple[list, bool]:
    if not messages:
        return messages, False
    msgs = list(messages)
    for idx in range(len(msgs) - 1, -1, -1):
        msg = msgs[idx]
        text = _content_text(msg.get("content", ""))
        if not _is_proactive_trigger_text(text):
            continue
        if not _proactive_tail_is_current(
            msgs,
            idx,
            allow_placeholder_user_tail=allow_placeholder_tail_inference,
        ):
            logger.info(
                "Historical proactive marker ignored: idx=%d role=%s",
                idx,
                msg.get("role"),
            )
            continue
        dropped = len(msgs) - idx - 1
        logger.info(
            "Proactive trigger normalized: idx=%d role=%s dropped_trailing=%d",
            idx,
            msg.get("role"),
            dropped,
        )
        return list(msgs[:idx]) + [_proactive_user_message(text)], True

    if allow_placeholder_tail_inference:
        return _infer_proactive_from_assistant_tail(
            msgs,
            allow_completed_tail=True,
        )

    return msgs, False
