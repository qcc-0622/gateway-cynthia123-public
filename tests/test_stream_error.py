"""流式路径异常兜底测试（Issue #39，2026-08-28）。

背景：上游 ConnectError/超时/读断等异常此前直接穿透 StreamingResponse，
客户端只会看到 okhttp StreamResetException: INTERNAL_ERROR（200 已发出，
错误状态码来不及返回）。修复 = routers 的流式响应统一套 guarded_sse_stream，
任何异常转成一条带内 _error_sse 事件再正常收流。

⚠️ 铁律（同 test_unified_brain.py）：测试输入必须按真实调用方的形状构造。
真实调用链：routers/anthropic_proxy.py 把 stream_with_archive()（内部迭代
forward_stream_* 生成器）交给 guarded_sse_stream —— 所以此处构造的
async generator 也按"产出 SSE chunk 字符串、异常从迭代中抛出"的形状来。

设计要点：
- 不起服务、不连真实上游，直接构造 async generator 模拟异常源。
- 既可以 `python tests/test_stream_error.py` 直接跑（无需 pytest），
  也可以 `python -m pytest tests/test_stream_error.py -v` 跑（pytest 可用时）。
"""

import asyncio
import sys
from pathlib import Path

# 确保能 import gateway 包（tests/ 与 gateway/ 同级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import httpx  # noqa: E402

from gateway.services.proxy import guarded_sse_stream  # noqa: E402


async def _gen_ok():
    yield 'data: {"type":"ping"}\n\n'


async def _gen_connect_error():
    """按真实故障形状构造：先产出内容，再从迭代中抛 ConnectError（消息可为空，
    2026-08-28 线上就是空消息的 httpx.ConnectError）。"""
    yield 'data: {"type":"ping"}\n\n'
    raise httpx.ConnectError("")


async def _gen_generic_error():
    yield "chunk1"
    raise ValueError("boom")


def _status_error() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://upstream.test/v1/messages")
    response = httpx.Response(503, text="upstream busy", request=request)
    return httpx.HTTPStatusError("503", request=request, response=response)


async def _gen_status_error():
    raise _status_error()
    yield  # noqa: unreachable——仅为把函数变成 async generator（真实 forward_stream_* 都是生成器）


def _collect(agen) -> str:
    async def run():
        out = []
        async for chunk in agen:
            out.append(chunk)
        return "".join(out)
    return asyncio.run(run())


def test_normal_stream_passthrough():
    """无异常时逐字透传，一个字节都不能变。"""
    assert _collect(guarded_sse_stream(_gen_ok(), "anthropic", "55")) == 'data: {"type":"ping"}\n\n'


def test_archive_marker_passthrough():
    """兜底层不认识 __ARCHIVE__ 标记（拦截是 router 内层的职责），必须原样放行。"""
    async def gen():
        yield "\n__ARCHIVE__:{\"conversation_id\": \"x\"}\n"
    assert _collect(guarded_sse_stream(gen(), "anthropic", "55")) == '\n__ARCHIVE__:{"conversation_id": "x"}\n'


def test_connect_error_becomes_readable_sse_anthropic():
    """空消息 ConnectError（2026-08-28 线上真实形状）→ 带内错误 + 正常收流，不抛出。"""
    joined = _collect(guarded_sse_stream(_gen_connect_error(), "anthropic", "55"))
    assert "[网关错误]" in joined
    assert "上游 55 连接失败: ConnectError" in joined
    assert "message_stop" in joined  # anthropic 格式以 message_stop 收尾，客户端才能正常结束
    assert 'data: {"type":"ping"}' in joined  # 异常前已产出的内容不丢


def test_status_error_becomes_readable_sse():
    """上游 HTTP 错误（理论上内层已处理，兜底双保险）→ 带状态码和响应体摘要。"""
    joined = _collect(guarded_sse_stream(_gen_status_error(), "anthropic", "55"))
    assert "上游 55 返回 503" in joined
    assert "upstream busy" in joined


def test_generic_exception_also_guarded():
    """非 httpx 异常（如归档 db 写挂了）同样不能穿透。"""
    joined = _collect(guarded_sse_stream(_gen_generic_error(), "anthropic", "55"))
    assert "上游 55 连接失败: ValueError" in joined


def test_openai_format_shape():
    """openai 客户端格式：错误以 delta 文本 + data: [DONE] 收尾。"""
    joined = _collect(guarded_sse_stream(_gen_connect_error(), "openai", "小鸡"))
    assert "上游 小鸡 连接失败: ConnectError" in joined
    assert "data: [DONE]" in joined


if __name__ == "__main__":
    import traceback
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failed += 1
                print(f"FAIL {name}:")
                traceback.print_exc()
    print(f"{failed} failed" if failed else "ALL PASS")
    raise SystemExit(1 if failed else 0)
