"""
余额监控 — 定期查各上游 API 余额，低于阈值推送微信通知

支持的上游：
  - DeepSeek 官方 (base_url 含 deepseek.com): GET /user/balance
  - 其他上游：暂不支持，跳过

配置（settings.json）：
  "notify_balance_threshold":       5.0    ← 低于多少元/USD 触发告警（默认 5）
  "notify_balance_interval_hours":  6      ← 多少小时查一次（默认 6）
  "notify_pushplus_token":          "..."  ← PushPlus token（notifier.py 共用）
"""

import asyncio
import logging

from gateway.http_client import get_client

logger = logging.getLogger("gateway.balance_monitor")


def _get(key, default):
    try:
        from gateway.settings import get as _g
        v = _g(key, None)
        return v if v is not None else default
    except Exception:
        return default


async def _check_deepseek(api_key: str, upstream_name: str) -> dict | None:
    """
    查询 DeepSeek 余额。
    返回 {"currency": "CNY", "balance": 12.34} 或 None（失败/不支持）。
    """
    try:
        client = get_client()
        resp = await client.get(
            "https://api.deepseek.com/user/balance",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=12,
        )
        if resp.status_code != 200:
            logger.warning("balance check %s: HTTP %d", upstream_name, resp.status_code)
            return None
        data = resp.json()
        infos = data.get("balance_infos", [])
        if not infos:
            return None
        info = infos[0]
        return {
            "currency": info.get("currency", "?"),
            "balance": float(info.get("total_balance", 0)),
        }
    except Exception as exc:
        logger.warning("balance check %s error: %s", upstream_name, exc)
        return None


async def balance_check_once() -> list[str]:
    """
    做一次全量余额检查，返回各上游结果摘要列表。
    如有余额低于阈值，同时触发通知。
    """
    from gateway.upstream import load_upstreams
    from gateway.notifier import notify

    threshold = float(_get("notify_balance_threshold", 5.0))
    upstreams = load_upstreams()
    results = []

    for u in upstreams:
        if not u.is_active:
            continue

        # 只检查 DeepSeek 官方
        if "deepseek.com" not in u.base_url:
            continue

        info = await _check_deepseek(u.api_key, u.name)
        if info is None:
            continue

        bal = info["balance"]
        cur = info["currency"]
        results.append(f"{u.name}: {bal:.2f} {cur}")
        logger.info("balance %s: %.2f %s (threshold=%.2f)", u.name, bal, cur, threshold)

        if bal < threshold:
            await notify(
                f"⚠️ 网关余额不足 [{u.name}]",
                (
                    f"上游【{u.name}】余额剩余 {bal:.2f} {cur}\n"
                    f"告警阈值：{threshold} {cur}\n\n"
                    f"‼️ 余额耗尽后 BP2 摘要将卡死，对话记忆停止更新。\n"
                    f"请登录 platform.deepseek.com 充值。"
                ),
                dedup_key=f"balance_low_{u.name}",
                cooldown=21600,  # 6 小时内只提醒一次
            )

    return results


async def balance_monitor_loop():
    """
    后台无限循环，每 notify_balance_interval_hours 小时检查一次余额。
    由 main.py lifespan 启动。
    """
    # 启动后等 2 分钟再第一次检查（给网关初始化时间）
    await asyncio.sleep(120)
    while True:
        try:
            results = await balance_check_once()
            if results:
                logger.info("balance check done: %s", " | ".join(results))
        except Exception:
            logger.exception("balance_monitor_loop unexpected error")

        interval_h = float(_get("notify_balance_interval_hours", 6))
        await asyncio.sleep(max(1.0, interval_h) * 3600)
