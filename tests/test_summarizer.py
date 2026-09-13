"""summarizer 流式回落 / 截断拒收测试（REFACTOR_ROADMAP P1.1，2026-09-02）。

summarizer.py 此前零测试，但它承担 BP2 的核心产出——摘要一旦写坏会经
滚动继承污染之后每一代（历史上咬人三次）。覆盖的关键行为：

- 主路径流式解析（anthropic / openai 两种上游格式）
- 流式空响应 → 回落非流式重试一次（DeepSeek 偶发 200 空响应）
- stop_reason=length/max_tokens → 拒收入库（ValueError），且**不再回落**
  （截断的摘要存进去会污染 prev_summary，重试也是白花钱）
- compress_summary 截断 → 回退原文交给上层 sanity check

铁律：MockTransport 模拟上游（同 test_openai_pipeline 的 _mock_upstream 形状），
共享 client 需先 reset 才会用补丁类重建。
既可以 `python tests/test_summarizer.py` 直接跑，也可以 pytest 跑。
"""

import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import httpx  # noqa: E402

from gateway import settings as settings_module  # noqa: E402
from gateway import http_client as _hc  # noqa: E402
from gateway.summarizer import summarize_messages, compress_summary  # noqa: E402
from gateway.upstream import Upstream  # noqa: E402


@contextmanager
def _tmp_settings():
    """临时 settings.json（summary_model/summary_upstream 留空 → 用传入的 upstream）。"""
    old_file = settings_module._SETTINGS_FILE
    old_cache = settings_module._cache
    tmp = Path(tempfile.mkdtemp(prefix="summarizer_test_")) / "settings.json"
    tmp.write_text("{}", encoding="utf-8")
    settings_module._SETTINGS_FILE = tmp
    settings_module._cache = None
    try:
        yield
    finally:
        settings_module._SETTINGS_FILE = old_file
        settings_module._cache = old_cache


@contextmanager
def _mock_upstream(handler):
    """同 test_openai_pipeline：patch httpx.AsyncClient + reset 共享 client。"""
    import gateway.services.proxy as proxy_module
    real_client = proxy_module.httpx.AsyncClient

    class _PatchedClient(real_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    _hc.reset_client()
    proxy_module.httpx.AsyncClient = _PatchedClient
    try:
        yield
    finally:
        proxy_module.httpx.AsyncClient = real_client
        _hc.reset_client()


def _make_upstream(api_format: str = "anthropic") -> Upstream:
    return Upstream(
        name="fake-upstream", base_url="http://upstream.test",
        api_key="sk-test", api_format=api_format, default_model="sum-model",
    )


def _sse_bytes(text: str, stop_reason: str = "end_turn") -> bytes:
    """anthropic 格式 SSE：message_start → 两个 text_delta → message_delta → stop。"""
    frames = [
        'event: message_start\ndata: {"type":"message_start","message":{"id":"m","usage":{"input_tokens":10}}}',
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    ]
    mid = max(1, len(text) // 2)
    for part in (text[:mid], text[mid:]):
        ev = {"type": "content_block_delta", "index": 0,
              "delta": {"type": "text_delta", "text": part}}
        frames.append("data: " + json.dumps(ev, ensure_ascii=False))
    md = {"type": "message_delta", "delta": {"stop_reason": stop_reason},
          "usage": {"output_tokens": 20}}
    frames.append("data: " + json.dumps(md, ensure_ascii=False))
    return ("\n\n".join(frames) + "\n\n").encode("utf-8")


def _openai_sse_bytes(text: str, finish_reason: str = "stop") -> bytes:
    frames = [
        'data: {"id":"c","object":"chat.completion.chunk","model":"sum-model",'
        '"choices":[{"index":0,"delta":{"role":"assistant"}}]}',
    ]
    mid = max(1, len(text) // 2)
    for part in (text[:mid], text[mid:]):
        ev = {"id": "c", "object": "chat.completion.chunk", "model": "sum-model",
              "choices": [{"index": 0, "delta": {"content": part}}]}
        frames.append("data: " + json.dumps(ev, ensure_ascii=False))
    ev = {"id": "c", "object": "chat.completion.chunk", "model": "sum-model",
          "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}
    frames.append("data: " + json.dumps(ev, ensure_ascii=False))
    frames.append("data: [DONE]")
    return ("\n\n".join(frames) + "\n\n").encode("utf-8")


def _ns_json(text: str, stop_reason: str = "end_turn") -> dict:
    return {"content": [{"type": "text", "text": text}], "stop_reason": stop_reason,
            "usage": {"input_tokens": 10, "output_tokens": 20}}


_TURNS = [{"role": "user", "content": "第一轮问题"},
          {"role": "assistant", "content": "第一轮回答"}]


def test_summarize_stream_happy_path_anthropic():
    async def case():
        async def _run():
            def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, content=_sse_bytes("# 锚定\n稳定事实\n# 近期\n用户问了问题\n# 远期\n暂无\n# 古老\n暂无"),
                                      headers={"content-type": "text/event-stream"})
            with _tmp_settings(), _mock_upstream(handler):
                return await summarize_messages(_TURNS, _make_upstream(), prev_summary="")
        out = await _run()
        assert out.startswith("# 锚定")
        assert "用户问了问题" in out

    import asyncio
    asyncio.run(case())


def test_stream_empty_falls_back_to_nonstream():
    """流式 200 空响应（DeepSeek 偶发）→ 回落非流式重试一次并成功。"""
    calls = []

    async def case():
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            calls.append(body.get("stream"))
            if body.get("stream"):
                return httpx.Response(200, content=b"")   # 空 SSE → 文本为空
            return httpx.Response(200, json=_ns_json("非流式摘要内容"))

        with _tmp_settings(), _mock_upstream(handler):
            out = await summarize_messages(_TURNS, _make_upstream(), prev_summary="")
        assert out == "非流式摘要内容"
        assert calls == [True, False], f"应先流式后非流式，实际 {calls}"

    import asyncio
    asyncio.run(case())


def test_stop_reason_length_rejected_and_no_fallback():
    """stop_reason=length：拒收（ValueError），且绝不能回落非流式再烧一次。"""
    calls = []

    async def case():
        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(200, content=_sse_bytes("被截断的摘要", stop_reason="length"),
                                  headers={"content-type": "text/event-stream"})

        with _tmp_settings(), _mock_upstream(handler):
            raised = False
            try:
                await summarize_messages(_TURNS, _make_upstream(), prev_summary="")
            except ValueError as e:
                raised = True
                assert "stop_reason=length" in str(e)
            assert raised, "截断摘要必须拒收"
        assert len(calls) == 1, "拒收路径不得触发非流式重试"

    import asyncio
    asyncio.run(case())


def test_summarize_openai_format_upstream():
    async def case():
        def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url).endswith("/v1/chat/completions")
            body = json.loads(request.content)
            assert isinstance(body.get("messages"), list)   # openai 格式：system 是 message
            return httpx.Response(200, content=_openai_sse_bytes("openai 格式摘要"),
                                  headers={"content-type": "text/event-stream"})

        with _tmp_settings(), _mock_upstream(handler):
            out = await summarize_messages(_TURNS, _make_upstream("openai"), prev_summary="")
        assert out == "openai 格式摘要"

    import asyncio
    asyncio.run(case())


def test_compress_summary_truncated_returns_original():
    """压缩被截断：回退原文（上层 sanity check 会拒绝），不抛错不落半截结果。"""
    original = "原始超长摘要" * 100

    async def case():
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_sse_bytes("压缩到一半", stop_reason="length"),
                                  headers={"content-type": "text/event-stream"})

        with _tmp_settings(), _mock_upstream(handler):
            out = await compress_summary(original, _make_upstream())
        assert out == original, "截断时必须原样返回原文"

    import asyncio
    asyncio.run(case())


def test_compress_summary_happy_path():
    async def case():
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_sse_bytes("# 锚定\n压缩后\n# 近期\n近期\n# 远期\n暂无\n# 古老\n暂无"),
                                  headers={"content-type": "text/event-stream"})

        with _tmp_settings(), _mock_upstream(handler):
            out = await compress_summary("长摘要" * 500, _make_upstream())
        assert out.startswith("# 锚定")

    import asyncio
    asyncio.run(case())


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
