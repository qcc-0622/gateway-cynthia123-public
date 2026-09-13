"""GET /v1/models — merge model list from all active upstreams."""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from gateway.auth import verify_api_key
from gateway.http_client import get_client
from gateway.services import model_sync
from gateway.upstream import get_all_active_upstreams, key_has_model_filter, key_matches_model

logger = logging.getLogger("gateway.models")
router = APIRouter()


def _build_headers(upstream) -> dict:
    if upstream.api_format == "anthropic":
        return {"x-api-key": upstream.api_key, "anthropic-version": "2023-06-01"}
    return {"authorization": f"Bearer {upstream.api_key}"}


def _visible_models_for_upstream(upstream) -> list[str]:
    """Return cached models visible to clients, respecting key-level filters."""
    models = list(upstream.cached_models or [])
    if upstream.primary_cached_models:
        models += list(upstream.primary_cached_models or [])
    for key_info in upstream.extra_keys or []:
        if not isinstance(key_info, dict):
            continue
        models += list(key_info.get("cached_models") or [])
        models += list(key_info.get("models") or [])
    if not models:
        return []
    result = []
    seen = set()
    for model in models:
        if not model or model in seen:
            continue
        allowed_by_filtered_key = any(
            isinstance(k, dict) and key_has_model_filter(k) and key_matches_model(k, model)
            for k in (upstream.extra_keys or [])
        )
        if allowed_by_filtered_key or model in (upstream.primary_cached_models or upstream.cached_models or []):
            seen.add(model)
            result.append(model)
    return result


def _parse_models(data) -> list[str]:
    if isinstance(data, dict) and "data" in data:
        return [m.get("id", "") for m in data["data"] if isinstance(m, dict) and m.get("id")]
    if isinstance(data, list):
        return [(m.get("id", m) if isinstance(m, dict) else str(m)) for m in data]
    return []


async def _fetch_one(upstream) -> list[str]:
    # Use cached models if available (avoids extra latency)
    cached = _visible_models_for_upstream(upstream)
    if cached:
        return cached
    try:
        client = get_client()
        resp = await client.get(upstream.models_url, headers=_build_headers(upstream), timeout=10)
        if resp.is_error:
            return []
        return _parse_models(resp.json())
    except Exception as e:
        logger.warning("Failed to fetch models from %s: %s", upstream.name, e)
        return []


@router.get("/v1/models")
async def list_models(_=Depends(verify_api_key)):
    upstreams = get_all_active_upstreams()
    if not upstreams:
        raise HTTPException(502, "No active upstream configured")

    # 客户端在橘瓣/RikkaHub 点"拉取模型"本身就是 GET /v1/models——借此机会自动
    # 刷新所有上游的模型缓存（冷却期 + 并发锁见 model_sync），本次响应即为新列表，
    # 不用再去后台手动"一键拉取"（2026-08-28）。
    try:
        await model_sync.auto_refresh_if_due()
    except Exception:
        logger.exception("auto model refresh hook failed (ignore, serve cache)")

    # 返回带 upstream:: 前缀的模型列表，RikkaHub 选模型时可明确指定上游
    # 格式：upstream_name::model_name，网关路由时会自动解析前缀
    # hidden 上游（如摘要专用令牌）不出现在此列表
    visible_upstreams = [u for u in upstreams if not u.hidden]
    results = await asyncio.gather(*[_fetch_one(u) for u in visible_upstreams])
    merged = []
    for upstream, model_list in zip(visible_upstreams, results):
        seen_in_upstream = set()
        for m in model_list:
            if not m or m in seen_in_upstream:
                continue
            seen_in_upstream.add(m)
            prefixed = f"{upstream.name}::{m}"
            merged.append({"id": prefixed, "object": "model"})

    return JSONResponse({"object": "list", "data": merged})
