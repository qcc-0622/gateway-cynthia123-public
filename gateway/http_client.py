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


# ── 代理白名单自检（2026-09-19）──────────────────────────────────────────
# 背景：httpx 默认 trust_env=True，会读 HTTPS_PROXY 把外呼送进代理；.env 的策略是
# "默认全走代理 + NO_PROXY 白名单放行"。于是**每新增一个上游都要记得补白名单**，
# 忘了就会去连代理→代理连不通目标站→抛 httpx.ConnectError，而
# `str(httpx.ConnectError)` 是**空字符串**、异常类型又长得像 TLS 问题，
# traceback 只落在 httpcore/_async/http_proxy.py（这个文件名是唯一线索）。
# 这个坑已经咬了三次（Issue #39 / #41 / 2026-09-19 relay-d），每次都耗掉几小时。
# 所以这里把"是不是被代理吃了"变成一句能直接读的提示，挂到失败日志/错误信息里。
_PROXY_ENV_KEYS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
                   "ALL_PROXY", "all_proxy")


def _no_proxy_covers(host: str, no_proxy: str | None = None) -> bool:
    """host 是否被 NO_PROXY 白名单覆盖（httpx/urllib 的后缀匹配语义）。"""
    if no_proxy is None:
        import os
        no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    host = (host or "").strip().lower().split(":")[0]
    if not host:
        return False
    for raw in (no_proxy or "").split(","):
        entry = raw.strip().lower()
        if not entry:
            continue
        if entry == "*":
            return True
        entry = entry.split("://")[-1].split("/")[0].split(":")[0].lstrip(".")
        if entry and (host == entry or host.endswith("." + entry)):
            return True
    return False


def proxy_hint(url: str) -> str:
    """若该请求很可能被代理拦截，返回一句提示；否则返回空串。

    判定：配了代理（任一 *_PROXY 环境变量）+ 目标域名不在 NO_PROXY 白名单里。
    这是"高度可疑"而不是"已证实"——所以措辞是"很可能"，且明确给出下一步动作。
    """
    import os
    from urllib.parse import urlparse
    proxy = next((os.environ.get(k) for k in _PROXY_ENV_KEYS if os.environ.get(k)), None)
    if not proxy:
        return ""
    host = urlparse(url if "://" in url else "https://" + url).hostname or ""
    if not host or _no_proxy_covers(host):
        return ""
    return (f"⚠️ 该域名 {host} 不在 .env 的 NO_PROXY/no_proxy 白名单里，"
            f"而代理 {proxy} 已配置 → 请求很可能被 httpx trust_env 送进了代理"
            f"（ConnectError 且 str 为空、traceback 落在 http_proxy.py 就是这个症状）。"
            f"修法：把 {host} 补进 .env 的 NO_PROXY 和 no_proxy **两行**后 systemctl restart chat-gateway")
