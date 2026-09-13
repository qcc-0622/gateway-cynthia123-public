"""POST /v1/chat/completions — OpenAI Chat Completions API proxy."""

import logging
import time

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from gateway.auth import verify_api_key
from gateway.hooks import preprocess_anthropic, preprocess_openai
from gateway.converter import openai_to_anthropic, strip_gateway_private_fields
from gateway.upstream import get_upstream_for_model, resolve_model_and_upstream, resolve_model_candidates
from gateway.services.proxy import (
    forward_nonstream,
    forward_stream_openai_to_openai,
    forward_stream_openai_to_anthropic,
    forward_stream_anthropic_body_to_openai,
    guarded_sse_stream,
    archiving_sse_stream,
    stream_with_failover,
    StreamArchiveState,
)
from gateway.services.archiver import archive

logger = logging.getLogger("gateway.openai")
router = APIRouter()


@router.post("/v1/chat/completions")
async def completions_proxy(request: Request, _=Depends(verify_api_key)):
    body = await request.json()
    client_model = body.get("model", "")
    clean_model, upstream = resolve_model_and_upstream(client_model)
    # failover 候选（语义同 anthropic_proxy，2026-09-06）
    candidates = resolve_model_candidates(client_model)
    if not upstream or not candidates:
        raise HTTPException(502, "No active upstream configured")
    if clean_model != client_model:
        body["model"] = clean_model
    # P3 网关侧显式身份（2026-09-06）：橘瓣配好自定义 header 后即生效；
    # header 缺失时零影响。X-Channel → gateway_context_id（metadata 路径）；
    # X-Proactive: true → 强制主动触发标记（preprocess 内识别）。
    _x_channel = (request.headers.get("x-channel") or "").strip()
    if _x_channel:
        body.setdefault("metadata", {})["gateway_context_id"] = _x_channel[:160]
    if (request.headers.get("x-proactive") or "").lower() == "true":
        body["_gateway_proactive_header"] = True
    logger.info("OpenAI request: model=%s (client=%s) stream=%s", clean_model, client_model, body.get("stream"))

    if upstream.api_format == "anthropic":
        # 2026-08-31: OpenAI 格式前端（mikeko 等）接入完整预处理管道。
        # 旧路径是 preprocess_openai 裸转发（只加时间戳），Seamless 接续、
        # BP2 滚动摘要、Ombre 记忆召回、缓存断点全都没有 → 助手拿不到任何
        # 摘要注入且每轮全价计费。现在先把 body 转成 anthropic 格式，再走
        # 和 /v1/messages 完全相同的 preprocess_anthropic；发上游前不再二次
        # 转换（body_format="anthropic"），响应侧仍转回 openai 格式。
        body = openai_to_anthropic(body)
        body = await preprocess_anthropic(body, upstream=upstream)
        tag = body.pop("_gateway_tag", "")
        fingerprint = body.pop("_gateway_fingerprint", "")
        context_fingerprint = body.pop("_gateway_context_fingerprint", "")
        body.pop("_gateway_context_id", None)
        # 主动触发标记（PLAN_PROACTIVE_FIX.md），语义同 anthropic_proxy
        proactive_trigger = bool(body.pop("_gateway_proactive_trigger", False))
        history_hash = body.pop("_gateway_history_hash", "")
        body_format = "anthropic"
    else:
        # openai 格式上游（当前无活跃上游，保留旧路径兜底）
        from gateway.hooks import detect_tag
        tag = detect_tag(body.get("messages", []))
        body = preprocess_openai(body, upstream=upstream)
        fingerprint = body.pop("_gateway_fingerprint", "")
        context_fingerprint = body.pop("_gateway_context_fingerprint", "")
        body.pop("_gateway_context_id", None)
        proactive_trigger = bool(body.pop("_gateway_proactive_trigger", False))
        history_hash = body.pop("_gateway_history_hash", "")
        body_format = "openai"
    archive_kwargs = {
        "tag": tag,
        "fingerprint": fingerprint,
        "history_hash": history_hash,
        "context_fingerprint": context_fingerprint,
        "proactive": proactive_trigger,
    }
    is_stream = body.get("stream", False)

    if not is_stream:
        try:
            response_data, archive_info = await forward_nonstream(body, upstream, "openai", body_format=body_format)
            await archive(raw_request=strip_gateway_private_fields(body), raw_response=archive_info["raw_response"], **archive_kwargs,
                           **{k: v for k, v in archive_info.items() if k not in ("raw_request", "raw_response")})
            return JSONResponse(content=response_data)
        except Exception as e:
            logger.exception("Upstream error")
            raise HTTPException(502, f"Upstream error: {e}")

    archive_state = StreamArchiveState.for_body(body, upstream, "openai")

    def gen_factory(u):
        # 语义同 anthropic_proxy.gen_factory：failover 时复用 state，重指上游
        archive_state.upstream_name = u.name
        archive_state.started_at = time.monotonic()
        archive_state.failed = False
        if u.api_format == "anthropic":
            if body_format == "anthropic":
                return forward_stream_anthropic_body_to_openai(body, u, archive_state=archive_state)
            return forward_stream_openai_to_anthropic(body, u, archive_state=archive_state)
        return forward_stream_openai_to_openai(body, u, archive_state=archive_state)

    return StreamingResponse(
        guarded_sse_stream(
            archiving_sse_stream(
                stream_with_failover(gen_factory, candidates, "openai"),
                body=body, archive_state=archive_state, archive_kwargs=archive_kwargs,
            ),
            "openai", upstream.name,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
