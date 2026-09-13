"""POST /v1/messages — Anthropic Messages API proxy."""

import logging
import time

import httpx
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from gateway.auth import verify_api_key
from gateway.hooks import preprocess_anthropic
from gateway.upstream import get_upstream_for_model, resolve_model_and_upstream, resolve_model_candidates
from gateway.services.proxy import (
    forward_nonstream,
    forward_stream_anthropic_to_anthropic,
    forward_stream_anthropic_to_openai,
    guarded_sse_stream,
    archiving_sse_stream,
    stream_with_failover,
    StreamArchiveState,
)
from gateway.converter import strip_gateway_private_fields
from gateway.services.archiver import archive

logger = logging.getLogger("gateway.anthropic")
router = APIRouter()


@router.post("/v1/messages")
async def messages_proxy(request: Request, _=Depends(verify_api_key)):
    body = await request.json()
    client_model = body.get("model", "")
    clean_model, upstream = resolve_model_and_upstream(client_model)
    # failover 候选（2026-09-06）：首选取首选上游，后续为同样能服务该模型的
    # 活跃上游（显式 upstream::model 点名时只有首选一个，不自动换站）
    candidates = resolve_model_candidates(client_model)
    if not upstream or not candidates:
        raise HTTPException(502, "No active upstream configured")
    if clean_model != client_model:
        body["model"] = clean_model  # 剥掉 upstream:: 前缀再发给上游
    # P3 网关侧显式身份（2026-09-06）：橘瓣配好自定义 header 后即生效；
    # header 缺失时零影响。X-Channel → gateway_context_id（metadata 路径）；
    # X-Proactive: true → 强制主动触发标记（preprocess 内识别）。
    _x_channel = (request.headers.get("x-channel") or "").strip()
    if _x_channel:
        body.setdefault("metadata", {})["gateway_context_id"] = _x_channel[:160]
    if (request.headers.get("x-proactive") or "").lower() == "true":
        body["_gateway_proactive_header"] = True
    logger.info("Anthropic request: model=%s (client=%s) stream=%s", clean_model, client_model, body.get("stream"))

    body = await preprocess_anthropic(body, upstream=upstream)
    is_stream = body.get("stream", False)

    tag = body.pop("_gateway_tag", "")  # set by preprocess_anthropic
    # fingerprint / history_hash 由 hooks 用「原始 messages」算好，这里直接传给 archive，
    # 不要让 archive 用「预处理后的 messages」重算（那样 fingerprint 会因 BP2 注入漂移）
    fingerprint = body.pop("_gateway_fingerprint", "")
    context_fingerprint = body.pop("_gateway_context_fingerprint", "")
    body.pop("_gateway_context_id", None)
    # 主动触发标记（PLAN_PROACTIVE_FIX.md）：hooks 确定性识别的橘瓣主动消息
    # 请求，归档时要跳过合成 user、[PASS] 回复整条不归档
    proactive_trigger = bool(body.pop("_gateway_proactive_trigger", False))
    history_hash = body.pop("_gateway_history_hash", "")
    archive_kwargs = {
        "tag": tag,
        "fingerprint": fingerprint,
        "history_hash": history_hash,
        "context_fingerprint": context_fingerprint,
        "proactive": proactive_trigger,
    }

    if not is_stream:
        try:
            response_data, archive_info = await forward_nonstream(body, upstream, "anthropic")
            await archive(raw_request=strip_gateway_private_fields(body), raw_response=archive_info["raw_response"],
                          **archive_kwargs,
                          **{k: v for k, v in archive_info.items() if k not in ("raw_request", "raw_response")})
            return JSONResponse(content=response_data)
        except httpx.HTTPStatusError as e:
            # Upstream returned a non-2xx — pass its real status code through to
            # the client instead of folding everything into 502. RikkaHub and
            # other clients can then surface "上游 503" rather than "网关 502".
            status = e.response.status_code
            body = (e.response.text or "")[:500]
            logger.warning("Upstream %s returned %s (passthrough): %s",
                           upstream.name, status, body[:200])
            raise HTTPException(status, body or f"Upstream {status}")
        except Exception as e:
            logger.exception("Upstream error")
            raise HTTPException(502, f"Upstream error: {e}")

    archive_state = StreamArchiveState.for_body(body, upstream, "anthropic")

    def gen_factory(u):
        # failover 换上游时复用同一个 state：失败的尝试没有产出内容，只需
        # 重指上游名并重置计时/失败标记（stream_with_failover 保证逐候选启动）
        archive_state.upstream_name = u.name
        archive_state.started_at = time.monotonic()
        archive_state.failed = False
        if u.api_format == "anthropic":
            return forward_stream_anthropic_to_anthropic(body, u, archive_state=archive_state)
        return forward_stream_anthropic_to_openai(body, u, archive_state=archive_state)

    return StreamingResponse(
        guarded_sse_stream(
            archiving_sse_stream(
                stream_with_failover(gen_factory, candidates, "anthropic"),
                body=body, archive_state=archive_state, archive_kwargs=archive_kwargs,
            ),
            "anthropic", upstream.name,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
