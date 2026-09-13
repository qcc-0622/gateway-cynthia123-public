"""
轻量通知模块 — 支持 PushPlus（免费微信推送）+ ntfy（Android 原生推送）
配置：在管理后台「网关设置」的 settings.json 里加：
  "notify_pushplus_token": "你的token"   ← 从 pushplus.plus 注册后获取
  "notify_ntfy_url": "https://ntfy.sh/你的频道名"  ← ntfy Android 推送

触发点：
  1. BP2 后台 rebuild 成功完成（摘要做好了）
  2. BP2 rebuild 失败（尤其是 402 余额不足）
  3. 余额监控低于阈值（balance_monitor.py）
"""

import asyncio
import logging
import time

from gateway.http_client import get_client

logger = logging.getLogger("gateway.notifier")

# 同 dedup_key 的最小发送间隔（秒），防止同类通知轰炸
_COOLDOWN_DEFAULT = 1800          # 30 分钟
_last_sent: dict[str, float] = {}


def _get_token() -> str:
    try:
        from gateway.settings import get as _get
        return _get("notify_pushplus_token", "") or ""
    except Exception:
        return ""


def _get_ntfy_url() -> str:
    try:
        from gateway.settings import get as _get
        return (_get("notify_ntfy_url", "") or "").strip()
    except Exception:
        return ""


async def _send_ntfy(title: str, body: str, dedup_key: str = "") -> bool:
    """发送 ntfy 推送（Android 原生通知）。"""
    ntfy_url = _get_ntfy_url()
    if not ntfy_url:
        return False

    key = "ntfy_" + (dedup_key or title)
    now = time.monotonic()
    if now - _last_sent.get(key, 0) < 60:   # ntfy 冷却 60s 即可，PushPlus 冷却更长
        return True

    try:
        # HTTP header 只能是 ASCII，去掉 emoji 前缀
        import re as _re
        safe_title = _re.sub(r'[^\x00-\x7F]+', '', title).strip() or title.encode('ascii', 'ignore').decode() or "Chat Gateway"
        client = get_client()
        resp = await client.post(
            ntfy_url,
            content=body.encode("utf-8"),
            headers={
                "Title": safe_title,
                "Priority": "high",
                "Tags": "bell",
                "Content-Type": "text/plain; charset=utf-8",
            },
            timeout=10,
        )
        if resp.status_code < 300:
            _last_sent[key] = now
            logger.info("ntfy sent: %s", title)
            return True
        else:
            logger.warning("ntfy failed: status=%s", resp.status_code)
            return False
    except Exception as exc:
        logger.warning("ntfy error: %s", exc)
        return False


async def notify(title: str, body: str, dedup_key: str = "",
                 cooldown: int = _COOLDOWN_DEFAULT) -> bool:
    """
    发送 PushPlus 微信通知。
    - dedup_key: 相同 key 的消息在 cooldown 秒内只发一条
    - cooldown: 覆盖默认冷却时间（秒）
    返回 True 表示成功发出（或已在冷却期跳过）。
    """
    # 同时尝试发 ntfy（不等待结果，不阻塞 PushPlus）
    asyncio.create_task(_send_ntfy(title, body, dedup_key))

    token = _get_token()
    if not token:
        return False  # 未配置，静默跳过

    key = dedup_key or title
    now = time.monotonic()
    if now - _last_sent.get(key, 0) < cooldown:
        logger.debug("notify: dedup skip key=%s", key)
        return True  # 算作"成功"，不是失败

    try:
        client = get_client()
        resp = await client.post(
            "https://www.pushplus.plus/send",
            json={
                "token": token,
                "title": title,
                "content": body,
                "template": "txt",
            },
            timeout=12,
        )
        data = resp.json()
        if data.get("code") == 200:
            _last_sent[key] = now
            logger.info("notify sent: %s", title)
            # 同时发 Web Push
            try:
                import asyncio as _asyncio
                from gateway.webpush_manager import send_push as _sp
                _asyncio.create_task(_sp(title, body))
            except Exception:
                pass
            return True
        else:
            logger.warning("notify failed: code=%s msg=%s", data.get("code"), data.get("msg"))
            return False
    except Exception as exc:
        logger.warning("notify error: %s", exc)
        return False


def fire_notify(title: str, body: str, dedup_key: str = "",
                cooldown: int = _COOLDOWN_DEFAULT):
    """
    非 async 场景用：创建 asyncio task 发送通知，不阻塞调用方。
    只在已有 event loop 的情况下有效（FastAPI 运行期间均满足）。
    """
    try:
        asyncio.create_task(notify(title, body, dedup_key=dedup_key, cooldown=cooldown))
    except RuntimeError:
        pass  # 没有 running loop，跳过

    # 无论有没有 PushPlus token，都尝试发 Web Push
    try:
        from gateway.webpush_manager import send_push as _sp
        asyncio.create_task(_sp(title, body))
    except Exception:
        pass
