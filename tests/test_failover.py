"""多上游 failover 测试（2026-09-06）。

设计：stream_with_failover 按序尝试候选上游，仅当**第一个字节之前**失败
（UpstreamPreStreamError / 连接级异常）才换下一个；4xx 客户端错误不换站；
候选耗尽后带内报最后一条错误。MockTransport 按 URL host 区分两个上游。

既可以直接 `python tests/test_failover.py` 跑，也可以 pytest 跑。
"""

import asyncio
import sys
from contextlib import contextmanager
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import httpx  # noqa: E402

from gateway import http_client as _hc  # noqa: E402
from gateway.services.proxy import (  # noqa: E402
    forward_stream_anthropic_to_anthropic,
    stream_with_failover,
    StreamArchiveState,
)
from gateway.upstream import Upstream  # noqa: E402

_UPSTREAM_SSE = (
    'event: message_start\n'
    'data: {"type":"message_start","message":{"id":"msg_x","type":"message","role":"assistant",'
    '"model":"claude-x","usage":{"input_tokens":100,"output_tokens":0}}}\n\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"来自正确上游"}}\n\n'
    'event: message_stop\n'
    'data: {"type":"message_stop"}\n\n'
).encode("utf-8")

_BODY = {
    "model": "claude-x", "stream": True, "max_tokens": 100,
    "messages": [{"role": "user", "content": [{"type": "text", "text": "你好"}]}],
}


def _upstream(name: str, host: str) -> Upstream:
    return Upstream(name=name, base_url=f"http://{host}", api_key="sk-test",
                    api_format="anthropic", cache_ttl="1h")


def _make_state() -> StreamArchiveState:
    return StreamArchiveState.for_body(_BODY, _upstream("upA", "upa.test"), "anthropic")


def _factory_for(st: StreamArchiveState):
    """模拟 router 的 gen_factory：state 重指当前候选 + 构造 a2a 转发。"""
    def factory(u: Upstream):
        st.upstream_name = u.name
        st.failed = False
        return forward_stream_anthropic_to_anthropic(_BODY, u, archive_state=st)
    return factory


@contextmanager
def _mock_hosts(handler):
    real_client = httpx.AsyncClient

    class _PatchedClient(real_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    _hc.reset_client()
    httpx.AsyncClient = _PatchedClient
    try:
        yield
    finally:
        httpx.AsyncClient = real_client
        _hc.reset_client()


def _collect(agen):
    async def run():
        out = []
        async for chunk in agen:
            out.append(chunk)
        return out
    return asyncio.run(run())


def test_failover_on_500_second_upstream_serves():
    """首选 500 → 自动换第二家；客户端拿到正确内容，state 归属正确。"""
    hits = {"a": 0, "b": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        hits["a" if host == "upa.test" else "b"] += 1
        if host == "upa.test":
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, content=_UPSTREAM_SSE,
                              headers={"content-type": "text/event-stream"})

    upA, upB = _upstream("upA", "upa.test"), _upstream("upB", "upb.test")
    st = _make_state()

    with _mock_hosts(handler):
        chunks = _collect(stream_with_failover(_factory_for(st), [upA, upB], "anthropic"))

    stream = "".join(chunks)
    assert "来自正确上游" in stream
    assert "[网关错误]" not in stream
    assert hits == {"a": 1, "b": 1}, f"应各打一次，实际 {hits}"
    assert st.upstream_name == "upB", "state 应归属真正服务请求的上游"


def test_no_failover_on_client_error_4xx():
    """400 客户端错误不值得换站：直接报错，第二家不被打扰。"""
    hits = {"a": 0, "b": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        hits["a" if host == "upa.test" else "b"] += 1
        if host == "upa.test":
            return httpx.Response(400, json={"error": "bad request"})
        return httpx.Response(200, content=_UPSTREAM_SSE,
                              headers={"content-type": "text/event-stream"})

    upA, upB = _upstream("upA", "upa.test"), _upstream("upB", "upb.test")
    st = _make_state()

    with _mock_hosts(handler):
        chunks = _collect(stream_with_failover(_factory_for(st), [upA, upB], "anthropic"))

    stream = "".join(chunks)
    assert "400" in stream and "[网关错误]" in stream
    assert hits == {"a": 1, "b": 0}, "4xx 不应 failover"


def test_all_candidates_fail_reports_last_error():
    """候选全挂：带内报最后一条错误，每个候选各试一次。"""
    hits = {"a": 0, "b": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        hits["a" if host == "upa.test" else "b"] += 1
        return httpx.Response(429, json={"error": "rate limited"})

    upA, upB = _upstream("upA", "upa.test"), _upstream("upB", "upb.test")
    st = _make_state()

    with _mock_hosts(handler):
        chunks = _collect(stream_with_failover(_factory_for(st), [upA, upB], "anthropic"))

    stream = "".join(chunks)
    assert "429" in stream and "[网关错误]" in stream
    assert hits == {"a": 1, "b": 1}


def test_first_success_never_touches_second():
    """首选成功：第二家零请求。"""
    hits = {"a": 0, "b": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        hits["a" if host == "upa.test" else "b"] += 1
        return httpx.Response(200, content=_UPSTREAM_SSE,
                              headers={"content-type": "text/event-stream"})

    upA, upB = _upstream("upA", "upa.test"), _upstream("upB", "upb.test")
    st = _make_state()

    with _mock_hosts(handler):
        chunks = _collect(stream_with_failover(_factory_for(st), [upA, upB], "anthropic"))

    assert "来自正确上游" in "".join(chunks)
    assert hits == {"a": 1, "b": 0}
    assert st.upstream_name == "upA"


def test_failover_kill_switch():
    """upstream_failover_enabled=False（默认）→ 只返回首选一个候选，不会撞备用上游的钱。"""
    import json
    import tempfile
    from gateway import settings as settings_module
    from gateway import upstream as upstream_module

    tmp = Path(tempfile.mkdtemp(prefix="failover_switch_"))
    ups_file = tmp / "upstreams.json"
    ups_file.write_text(json.dumps([
        {"name": "main", "base_url": "http://main.test", "api_key": "k",
         "api_format": "anthropic", "cached_models": ["model-x"]},
        {"name": "backup", "base_url": "http://backup.test", "api_key": "k",
         "api_format": "anthropic", "cached_models": ["model-x"]},
    ]), encoding="utf-8")
    settings_file = tmp / "settings.json"
    settings_file.write_text("{}", encoding="utf-8")

    old_up, old_cache = upstream_module.UPSTREAMS_FILE, upstream_module._upstream_cache
    old_sf, old_sc = settings_module._SETTINGS_FILE, settings_module._cache
    upstream_module.UPSTREAMS_FILE = str(ups_file)
    upstream_module._upstream_cache = None
    settings_module._SETTINGS_FILE = settings_file
    settings_module._cache = None
    try:
        # 默认关闭：只有首选
        cands = upstream_module.resolve_model_candidates("model-x")
        assert [u.name for u in cands] == ["main"], f"关闭时只应返回首选，实际 {[u.name for u in cands]}"
        # 点名也不换（本来就只有首选）
        cands = upstream_module.resolve_model_candidates("main::model-x")
        assert [u.name for u in cands] == ["main"]

        # 开启：两个候选，首选在前
        settings_file.write_text(json.dumps({"upstream_failover_enabled": True}), encoding="utf-8")
        settings_module._cache = None
        cands = upstream_module.resolve_model_candidates("model-x")
        assert [u.name for u in cands] == ["main", "backup"]
        # 点名即使开启也只一个
        cands = upstream_module.resolve_model_candidates("main::model-x")
        assert [u.name for u in cands] == ["main"]
    finally:
        upstream_module.UPSTREAMS_FILE = old_up
        upstream_module._upstream_cache = old_cache
        settings_module._SETTINGS_FILE = old_sf
        settings_module._cache = old_sc


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
