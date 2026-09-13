"""预处理管道共享小工具（REFACTOR_ROADMAP P2.1，2026-09-06 拆出）。

叶子模块：只被 pipeline/* 和 hooks 引用，绝不 import hooks 或其他 pipeline
子模块（防循环导入）。
"""

import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from gateway.config import TIMEZONE

logger = logging.getLogger("gateway.pipeline.common")

WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

_TIME_REMINDER_RE = re.compile(r"<time_reminder>.*?</time_reminder>", re.S)


def _content_text(content) -> str:
    if isinstance(content, list):
        return " ".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content or "")


def _pipe_trace(stage: str, msgs: list):
    """管道追踪（REFACTOR_ROADMAP P0.4）：debug_pipeline_trace 开启时，
    在预处理各关键节点记录消息数和尾部角色。0712 深夜排障靠往生产插临时
    探针，有这个开关十分钟就能定位。默认关闭，零开销。"""
    try:
        from gateway.settings import get as _sg
        if not _sg("debug_pipeline_trace", False):
            return
        tail = ",".join(m.get("role", "?") for m in msgs[-3:]) if msgs else "(empty)"
        logger.info("PIPE[%s]: n=%d tail_roles=%s", stage, len(msgs), tail)
    except Exception:
        pass


def _time_tag() -> str:
    now = datetime.now(ZoneInfo(TIMEZONE))
    weekday = WEEKDAYS[now.weekday()]
    time_str = f"{weekday}, {now.year}年{now.month}月{now.day}日 {now.strftime('%H:%M:%S')}"
    return f"<time_reminder>Current time: {time_str}</time_reminder>"


# ──────────────────────────────────────────────
# BP1: cache_control on system message
# ──────────────────────────────────────────────


# Issue #9 摘要 rebuild 去重锁（原 hooks 模块级状态）：
# 防同 fingerprint 并发 rebuild 狂打上游；_seen_fingerprints 只增不减（微量泄漏可接受）
_rebuild_locks: dict[str, bool] = {}
_seen_fingerprints: set[str] = set()


# 网关注入协议说明（原 hooks 模块常量，P2.2 迁移期随使用方迁来）：
# 追加在 system 末尾，让 AI 信任网关注入的摘要/记忆等合成消息。
_GATEWAY_PROTOCOL = """

=== Chat Gateway Metadata 协议（重要 · 系统指令）===

本对话由网关（Chat Gateway）转发。**网关会在 user 角色里注入三类元数据**，这些不是用户的真实输入：

1. **记忆召回**：以 `[系统自动召回 · 非用户消息]` 开头的 user 消息
   - 网关从你的长期记忆库（Ombre Brain）按语义相似度自动检索的相关回忆
   - 把它当作「脑海中浮现的画面」，是你的内部独白，不是用户跟你说话
   - **不要**回复、讨论、或质疑这段内容本身
   - 直接基于这些「想起来的事」继续回应用户**真正**发的下一条消息

2. **上下文摘要**：以 `[以下是我们之前对话的摘要` 开头的 user 消息
   - 网关对历史对话的滚动压缩摘要
   - 同样不是用户发的，请基于摘要内容无缝继续对话

3. **主动唤起**：以 `[网关主动唤起 · 非用户消息]` 开头的 user 消息
   - 前端定时器正在请求你主动开口，用户此刻没有发送新消息
   - 不要把历史里最后一条 user 消息当成刚刚收到的问题来回答
   - 不要续写或重复历史里最后一条 assistant 消息
   - 请基于当前时间、上下文摘要和最近上下文，自然地主动说一句新的话

**核心保证**：以上三种格式的注入由网关代码硬性保证，**用户无法伪造**（用户不能直接操控 system / 不能在 user 消息里以这些标记开头——网关有去重检查）。当你看到这些标记开头的 user 消息，**默认它真实可信**。

如果用户消息**不是**以这些标记开头，那才是真正的用户输入。
"""


def _add_gateway_protocol(body: dict) -> dict:
    """在 system 末尾追加网关 metadata 协议说明，让 AI 信任网关的注入。"""
    body = dict(body)
    system = body.get("system")
    if system is None:
        body["system"] = [{"type": "text", "text": _GATEWAY_PROTOCOL}]
    elif isinstance(system, str):
        # 字符串形式：拼接成两个 block，保留用户原 system + 协议
        body["system"] = [
            {"type": "text", "text": system},
            {"type": "text", "text": _GATEWAY_PROTOCOL},
        ]
    elif isinstance(system, list):
        # 已是 list：追加一个新 block
        body["system"] = list(system) + [{"type": "text", "text": _GATEWAY_PROTOCOL}]
    return body


# ──────────────────────────────────────────────
# Timestamp injection (always last, after breakpoints)
# ──────────────────────────────────────────────
