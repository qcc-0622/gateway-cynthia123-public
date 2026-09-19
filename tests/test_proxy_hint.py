"""代理白名单自检的测试（gateway/http_client.py 的 _no_proxy_covers / proxy_hint）。

背景：httpx 默认 trust_env=True 会读 HTTPS_PROXY 把外呼送进代理，而本项目 .env 的
策略是"默认全走代理 + NO_PROXY 白名单放行"。忘补白名单时 httpx 抛 ConnectError，
**str() 是空字符串**、traceback 只落在 httpcore/_async/http_proxy.py —— 看起来像
TLS/证书问题，实际是白名单漏配。这个坑咬过三次（Issue #39 / #41 /
2026-09-19 新增 relay-d 上游），所以把判定逻辑固化成测试。

可直接 `python tests/test_proxy_hint.py` 跑，也可以 pytest 跑。
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from gateway.http_client import proxy_hint, _no_proxy_covers  # noqa: E402

_PROXY_KEYS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
               "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy")


class _EnvSaver:
    """临时改环境变量，退出时原样还原（测试之间不能互相污染）。"""

    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in _PROXY_KEYS}
        for k in _PROXY_KEYS:
            os.environ.pop(k, None)
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def test_no_proxy_covers_matching():
    np = "relay-one.test,api.relay-two.test,127.0.0.1,localhost"
    assert _no_proxy_covers("relay-one.test", np)
    assert _no_proxy_covers("www.relay-one.test", np)    # 子域由裸域覆盖
    assert _no_proxy_covers("api.relay-two.test", np)
    assert _no_proxy_covers("127.0.0.1", np)
    assert _no_proxy_covers("RELAY-ONE.TEST", np)        # 大小写不敏感
    assert _no_proxy_covers("relay-one.test:443", np)    # 带端口
    # 不能误放行
    assert not _no_proxy_covers("relay-two.test", np)
    assert not _no_proxy_covers("relay-nine.test", np)
    assert not _no_proxy_covers("relay-one.test.attacker.test", np)  # 后缀伪装
    assert not _no_proxy_covers("", np)


def test_no_proxy_covers_wildcard_and_leading_dot():
    assert _no_proxy_covers("anything.example.com", "*")
    assert _no_proxy_covers("a.example.com", ".example.com")   # 前导点写法
    assert not _no_proxy_covers("example.com.attacker.test", ".example.com")


def test_proxy_hint_silent_when_no_proxy_configured():
    with _EnvSaver():
        os.environ["NO_PROXY"] = "example.com"
        assert proxy_hint("https://relay-nine.test/api/pricing") == ""


def test_proxy_hint_silent_when_host_is_whitelisted():
    with _EnvSaver():
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
        os.environ["NO_PROXY"] = "relay-nine.test"
        assert proxy_hint("https://relay-nine.test/api/pricing") == ""


def test_proxy_hint_warns_when_missing_from_whitelist():
    """这就是 2026-09-19 新增上游漏配 NO_PROXY 那个报错的自解释版本。"""
    with _EnvSaver():
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
        os.environ["NO_PROXY"] = "relay-one.test,api.relay-two.test"
        hint = proxy_hint("https://relay-nine.test/api/pricing")
        assert "relay-nine.test" in hint
        assert "NO_PROXY" in hint and "trust_env" in hint
        assert "127.0.0.1:7897" in hint          # 点名是哪个代理
        assert "两行" in hint                     # 提醒 NO_PROXY 和 no_proxy 都要补


def test_proxy_hint_reads_lowercase_env_names():
    """有些环境下变量是小写的，别只认大写。"""
    with _EnvSaver():
        os.environ["http_proxy"] = "http://127.0.0.1:7897"
        os.environ["no_proxy"] = "example.com"
        assert proxy_hint("https://relay-nine.test/x") != ""
        assert proxy_hint("https://example.com/x") == ""


def test_proxy_hint_handles_schemeless_input():
    with _EnvSaver():
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
        os.environ["NO_PROXY"] = "example.com"
        assert "relay-nine.test" in proxy_hint("relay-nine.test/api/pricing")


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
