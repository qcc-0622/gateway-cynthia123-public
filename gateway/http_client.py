"""进程级共享 httpx.AsyncClient（连接池复用）。

改动前 20+ 处调用都是 `async with httpx.AsyncClient(...)` 每请求新建——
每次上游请求都重新 TCP+TLS 握手，经中转站每请求多 100-300ms 延迟。
改为进程级共享实例后连接池复用；各调用方用 httpx 的 per-request
timeout 参数覆盖默认值，超时语义与原来完全一致。

测试注入 MockTransport 的方式不变：monkeypatch `httpx.AsyncClient`
（httpx 模块全局属性）后调 reset_client()，下一个请求就会用补丁类
重建共享实例（见 tests/test_openai_pipeline.py::_mock_upstream）。
"""

import httpx

_client: httpx.AsyncClient | None = None

# 保守默认值，只兜住忘传 timeout 的调用方；正式调用方一律 per-request 覆盖。
_DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)
    return _client


def reset_client() -> None:
    """丢弃共享实例（测试换 MockTransport 前后用）。不 await aclose——
    调用方要么在跑的事件循环即将结束（测试），要么应改用 close_client。"""
    global _client
    _client = None


async def close_client() -> None:
    """lifespan 关闭时优雅释放连接池。"""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception:
            pass
        _client = None
