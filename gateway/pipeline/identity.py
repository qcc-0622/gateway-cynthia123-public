"""身份识别：tag 前缀、context_id / 指纹、客户端元数据剥离（P2.1-3，2026-09-06）。

自 hooks.py 绞杀搬迁，函数体零改动；hooks.py 留 import 转发。
不 import hooks（防循环）；共享工具来自 pipeline.common。
"""

import hashlib
import re

from gateway.pipeline.common import _TIME_REMINDER_RE, _content_text


_GATEWAY_CONTEXT_KEYS = (
    "gateway_context_id",
    "context_fingerprint",
    "gateway_shared_context",
    "shared_context_id",
)
_DEFAULT_DAILY_CONTEXT_ID = "daily-main"


def _clean_context_id(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text[:160]


def _context_fingerprint_from_id(context_id: str) -> str:
    context_id = _clean_context_id(context_id)
    if not context_id:
        return ""
    if re.fullmatch(r"ctx_[0-9a-fA-F]{12,64}", context_id):
        return context_id.lower()
    digest = hashlib.md5(context_id.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"ctx_{digest}"


def _extract_context_id_from_text(text: str) -> str:
    if not text or "gateway_context" not in text and "shared_context" not in text:
        return ""
    # Fast path for JSON fragments embedded in the user/context text.
    m = re.search(
        r'["\'](?:gateway_context_id|gateway_shared_context|shared_context_id)["\']\s*:\s*["\']([^"\']{1,160})["\']',
        text,
    )
    if m:
        return _clean_context_id(m.group(1))
    m = re.search(
        r"\b(?:gateway_context_id|gateway_shared_context|shared_context_id)\s*=\s*([A-Za-z0-9_.:/@-]{1,160})",
        text,
    )
    return _clean_context_id(m.group(1)) if m else ""


def _extract_gateway_context_id(body: dict, messages: list) -> str:
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        for key in _GATEWAY_CONTEXT_KEYS:
            value = _clean_context_id(metadata.get(key))
            if value:
                return value

    # Some plugin hosts pass the callAI context as plain text instead of HTTP metadata.
    # Scan the newest user messages first so an explicit current value wins.
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        value = _extract_context_id_from_text(_content_text(msg.get("content", "")))
        if value:
            return value
    return ""


def _default_context_id_for_tag(tag: str) -> str:
    # 2026-06-09: 关闭 daily-main 默认共享。原因：daily-main 共享池会让伴读插件 +
    # 主 chat + 橘瓣多 session 串味（latest_summary 来自别人的对话、_shared_context_tail
    # exclude_fingerprint=self 又把自己 archive 排掉），AI 出现"忘记上一两句、回答不连续"。
    # 永远返回 ""：context_fingerprint 始终为空 → BP2/Seamless 走老路（每个 session 自己的 fp 池）。
    # 若以后要恢复伴读共享：把 return "" 改回 `return "" if tag else _DEFAULT_DAILY_CONTEXT_ID`，
    # 或前端显式传 metadata.gateway_context_id。
    return ""


def _strip_gateway_metadata(body: dict) -> dict:
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        return body
    cleaned = {k: v for k, v in metadata.items() if k not in _GATEWAY_CONTEXT_KEYS}
    body = dict(body)
    if cleaned:
        body["metadata"] = cleaned
    else:
        body.pop("metadata", None)
    return body


def _should_seed_shared_context(context_id: str) -> bool:
    cid = _clean_context_id(context_id).lower()
    # 共读插件（reading:<书名>）冷启动时，单向继承一次主对话最新摘要做"底子"。
    # find_predecessor_summary 已 LIKE 'fp_%' 过滤，只拉主对话 session 摘要，
    # 不会拉到别的书/别的共享池（ctx_xxx），共读自己的讨论也只存在 ctx_reading 里、
    # 永不回写主对话 → 单向读、不串台。详见 Issue #36 后的安全设计。
    if cid.startswith("reading:"):
        return True
    return cid in {
        "daily-main",
        "daily",
        "main",
        "default",
    }


_TAG_PREFIXES = {
    # English variants
    "tech:": "tech",
    "tech：": "tech",
    "[tech]": "tech",
    "【tech】": "tech",
    # Chinese variants
    "技术:": "tech",
    "技术：": "tech",
    "[技术]": "tech",
    "【技术】": "tech",
}


def detect_tag(messages: list) -> str:
    """Look at the first user message with real content — if it begins with a
    recognised prefix, return the matching tag name. Skips empty messages and
    messages that contain only gateway-injected time_reminder."""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        text = content if isinstance(content, str) else " ".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
        # Strip gateway-injected time_reminder and surrounding whitespace
        text = _TIME_REMINDER_RE.sub("", text).strip()
        if not text:
            continue  # skip empty/time-only messages, try next user msg
        for prefix, tag in _TAG_PREFIXES.items():
            if text.lower().startswith(prefix.lower()):
                return tag
        return ""  # first real user msg has no prefix → daily
    return ""


# ──────────────────────────────────────────────
# Issue #6: keep client-side cache_control, but never exceed Anthropic's 4 slots
# Issue #7: strip unsigned thinking blocks from history (synthetic ones from rewriter)
# ──────────────────────────────────────────────
