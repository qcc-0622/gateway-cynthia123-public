"""上游模型列表同步（2026-08-28）：拉取 → 更新 upstreams.json 的模型缓存。

两个触发源共用同一套逻辑（唯一真相源，避免双份漂移）：
- 管理后台"一键拉取所有上游模型"（routers/admin.py /upstreams/refresh-all-models）
- 客户端在橘瓣/RikkaHub 点"拉取模型"= GET /v1/models，网关借此自动刷新
  （routers/models.py，带冷却期 + 并发锁，settings 可关）

背景：上游（中转站）上新模型很快，而 /v1/models 旧逻辑只要 cached_models
非空就返回缓存、从不实拉，新模型必须去后台手动点拉取才能看见。
"""

import asyncio
import logging
import time
import traceback

from gateway.http_client import get_client, proxy_hint
from gateway.settings import get
from gateway.upstream import (
    Upstream,
    load_upstreams,
    update_cached_models,
    update_extra_key_cached_models,
    update_primary_cached_models,
)

logger = logging.getLogger("gateway.model_sync")

# ── 自动刷新冷却状态（进程内；重启后视为已冷却，首次 /v1/models 即触发） ──
_last_auto_refresh = 0.0
_refresh_lock = asyncio.Lock()


def reset_auto_refresh_state() -> None:
    """测试用：清零冷却计时。"""
    global _last_auto_refresh
    _last_auto_refresh = 0.0


def model_headers(api_key: str, api_format: str) -> dict:
    headers = {"authorization": f"Bearer {api_key}"}
    if api_format == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    return headers


def parse_models_payload(data) -> set[str]:
    """兼容三种返回：OpenAI 风格 {"data":[{"id":..}]}、Anthropic 风格 {"models":[..]}、纯列表。"""
    if isinstance(data, dict) and "data" in data:
        return {m.get("id", "") for m in data["data"] if isinstance(m, dict) and m.get("id")}
    if isinstance(data, dict) and "models" in data:
        items = data["models"]
        return {m.get("id", m) if isinstance(m, dict) else str(m) for m in items}
    if isinstance(data, list):
        return {m.get("id", m) if isinstance(m, dict) else str(m) for m in data}
    return set()


async def fetch_models_for_key(models_url: str, api_key: str, api_format: str) -> set[str]:
    """拉取单个令牌可见的模型 id 集合。失败返回空集（不抛出，由调用方决定降级）。"""
    try:
        client = get_client()
        resp = await client.get(models_url, headers=model_headers(api_key, api_format), timeout=15)
        resp.raise_for_status()
        return parse_models_payload(resp.json()) - {""}
    except Exception as e:
        # proxy_hint：忘补 NO_PROXY 时 httpx 会静默走代理，报出来的是
        # ConnectError(str 为空)、traceback 落在 http_proxy.py —— 光看日志
        # 像是 TLS/证书问题，实际是白名单漏配（Issue #39/#41/2026-09-19 relay-d 三次）。
        hint = proxy_hint(models_url)
        logger.warning(
            "fetch_models_for_key FAILED: url=%s err=%s(%s)%s\n%s",
            models_url, type(e).__name__, e,
            ("\n" + hint) if hint else "",
            traceback.format_exc()
        )
        return set()


async def refresh_one_upstream_models(u: Upstream) -> dict:
    """刷新单个上游：主令牌 → primary_cached_models（并入 cached_models），
    额外令牌按 key 分组 → 各自 cached_models，最后汇总写 cached_models。
    每步都即时落盘（update_* 内部 save_upstreams）。异常向上抛给调用方。"""
    primary = sorted(await fetch_models_for_key(u.models_url, u.api_key, u.api_format))
    update_primary_cached_models(u.name, primary)

    groups = [{
        "key_index": "primary",
        "label": "主令牌",
        "purpose": "chat",
        "count": len(primary),
        "models": primary,
        "ok": True,
    }]
    all_models = set(primary)

    for idx, ek in enumerate(u.extra_keys or []):
        if not (isinstance(ek, dict) and ek.get("key")):
            continue
        models = sorted(await fetch_models_for_key(u.models_url, ek["key"], u.api_format))
        update_extra_key_cached_models(u.name, idx, models)
        all_models |= set(models)
        groups.append({
            "key_index": idx,
            "label": ek.get("label") or f"额外令牌 {idx + 1}",
            "purpose": ek.get("purpose", "chat"),
            "count": len(models),
            "models": models,
            "ok": True,
        })

    update_cached_models(u.name, sorted(all_models))
    logger.info("Refreshed %d grouped models from %s", len(all_models), u.name)
    return {"count": len(all_models), "groups": groups, "ok": True}


async def refresh_all_upstream_models() -> dict:
    """顺序刷新所有 active 上游（含 hidden，摘要令牌也要跟上新模型）。
    必须顺序不能并行：update_* 每次都是 读 upstreams.json → 改 → 写回，
    并行会互相覆盖丢更新。单上游失败不影响其它上游。"""
    result = {}
    for u in [u for u in load_upstreams() if u.is_active]:
        try:
            result[u.name] = await refresh_one_upstream_models(u)
        except Exception as e:
            logger.warning("Failed to fetch models from %s: %s", u.name, e)
            result[u.name] = {"count": 0, "ok": False, "error": str(e)[:200]}
    return result


def should_auto_refresh(now: float | None = None) -> bool:
    """距上次自动刷新是否已过冷却期（且总开关开启）。"""
    if not get("models_auto_refresh_on_client_pull", True):
        return False
    now = time.monotonic() if now is None else now
    cooldown = int(get("models_auto_refresh_cooldown_min", 30)) * 60
    return (now - _last_auto_refresh) >= cooldown


async def auto_refresh_if_due() -> dict | None:
    """客户端触发点（GET /v1/models）：冷却期已到才刷新，正在刷就直接放行。
    刷新结果即本次响应的新列表；失败只记日志（保留旧缓存），绝不阻塞响应结构。"""
    global _last_auto_refresh
    if not should_auto_refresh():
        return None
    if _refresh_lock.locked():
        return None  # 另一个并发请求已在刷新，本次直接用现有缓存响应
    async with _refresh_lock:
        if not should_auto_refresh():  # double-check：拿到锁前可能已被刷过
            return None
        try:
            result = await refresh_all_upstream_models()
            logger.info("Auto-refreshed upstream models on client pull: %s",
                        {k: v.get("count", 0) for k, v in result.items()})
            return result
        except Exception:
            logger.exception("Auto model refresh failed (keep old caches)")
            return None
        finally:
            _last_auto_refresh = time.monotonic()  # 失败也计冷却，防止故障上游被打爆
