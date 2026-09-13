"""
Web Push 管理器
- 生成/加载 VAPID 密钥
- 保存/删除订阅
- 发送推送
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger("gateway.webpush")

_DATA_DIR = Path(__file__).parent.parent / "data"
_VAPID_FILE = _DATA_DIR / "vapid_keys.json"
_SUBS_FILE = _DATA_DIR / "push_subscriptions.json"

_vapid_cache: dict | None = None


def get_vapid_keys() -> dict:
    """获取 VAPID 密钥，不存在则生成。返回 {private_key, public_key}"""
    global _vapid_cache
    if _vapid_cache:
        return _vapid_cache

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    if _VAPID_FILE.exists():
        data = json.loads(_VAPID_FILE.read_text())
        _vapid_cache = data
        return data

    # 生成新密钥
    from py_vapid import Vapid
    v = Vapid()
    v.generate_keys()

    keys = {
        "private_key": v.private_pem().decode(),
        "public_key": v.public_key.public_bytes(
            __import__("cryptography.hazmat.primitives.serialization",
                       fromlist=["Encoding", "PublicFormat"]).Encoding.X962,
            __import__("cryptography.hazmat.primitives.serialization",
                       fromlist=["Encoding", "PublicFormat"]).PublicFormat.UncompressedPoint
        ).hex(),
    }

    # 同时保存 base64url 格式（前端需要）
    import base64
    raw_pub = bytes.fromhex(keys["public_key"])
    keys["public_key_b64"] = base64.urlsafe_b64encode(raw_pub).rstrip(b"=").decode()

    _VAPID_FILE.write_text(json.dumps(keys, indent=2))
    _vapid_cache = keys
    logger.info("VAPID keys generated and saved")
    return keys


def get_subscriptions() -> list:
    if not _SUBS_FILE.exists():
        return []
    try:
        return json.loads(_SUBS_FILE.read_text())
    except Exception:
        return []


def save_subscription(sub: dict):
    subs = get_subscriptions()
    endpoint = sub.get("endpoint", "")
    # 去重
    subs = [s for s in subs if s.get("endpoint") != endpoint]
    subs.append(sub)
    _SUBS_FILE.write_text(json.dumps(subs, indent=2))
    logger.info("Push subscription saved: %s…", endpoint[:60])


def remove_subscription(endpoint: str):
    subs = get_subscriptions()
    subs = [s for s in subs if s.get("endpoint") != endpoint]
    _SUBS_FILE.write_text(json.dumps(subs, indent=2))


async def send_push(title: str, body: str, url: str = "/chat-gateway/admin/",
                    icon: str = "/chat-gateway/static/icon-192.png",
                    tag: str = "gateway-event"):
    """向所有已订阅设备发推送。"""
    import asyncio
    subs = get_subscriptions()
    if not subs:
        return

    keys = get_vapid_keys()

    dead = []
    for sub in subs:
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                _send_one,
                sub,
                {"title": title, "body": body, "url": url, "icon": icon, "tag": tag},
                keys,
            )
        except Exception as e:
            err = str(e)
            if "410" in err or "404" in err:
                dead.append(sub.get("endpoint", ""))
                logger.info("Subscription expired/gone: %s…", sub.get("endpoint","")[:60])
            else:
                logger.warning("Push send error: %s", err[:200])

    # 清理失效订阅
    for ep in dead:
        remove_subscription(ep)


def _send_one(subscription: dict, data: dict, keys: dict):
    import json as _json
    from pywebpush import webpush, WebPushException

    webpush(
        subscription_info=subscription,
        data=_json.dumps(data),
        vapid_private_key=keys["private_key"],
        vapid_claims={"sub": "mailto:admin@chat.gateway"},
        ttl=86400,
    )
