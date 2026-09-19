"""上游价格自动同步测试（services/price_sync.py + costs.py 取价回退，2026-09-14）。

覆盖场景：
    1. usd_per_1m / convert_model：new-api 倍率 → $/1M 的换算口径
       （用 Claude Sonnet 官方价做锚点核对，比"跟实现对齐"可信）
    2. cache_ratio / create_cache_ratio **缺失 vs 显式 0** 的区别（坑 2）
    3. cache_ttl 1h/5m/off → 三档缓存写单价
    4. quota_type=1 的按次计费 + group_ratio 倍率
    5. parse_pricing_payload / pricing_urls 的形状兼容与错误处理
    6. **fetch_pricing 必须匿名裸调**（源码级回归：带 Authorization 会被
       new-api 判成坏凭据并直接 403，见 price_sync 模块开头坑 1）
    7. resolve_price 优先级：手工 > 自动 > 默认；自动价不跨上游串味；子串有歧义不猜
    8. sync_upstreams：落盘、失败保留旧价（stale）、已删上游的陈旧条目被清掉
    9. calculate_cost 端到端吃到自动价

⚠️ 铁律（同 test_models_autorefresh.py）：测试输入按真实调用方形状构造——
   真实链路是 routers/admin.py 的两个路由 + main.py 的 price_sync_loop 调
   sync_upstreams()，内部对 load_upstreams() 的每个 active 上游调
   fetch_pricing()（GET {base_url}/api/pricing），再经 costs.save_auto_prices 落盘。
   这里把网络层换成 canned 数据、把三个落盘目标全部重定向到临时目录。

既可 `python tests/test_auto_pricing.py` 直跑，也可 pytest 跑。
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── 在 import 被测代码之前，把落盘目标全部指向临时文件 ──
_TMP = Path(tempfile.mkdtemp(prefix="auto_pricing_test_"))
(_TMP / "settings.json").write_text("{}", encoding="utf-8")

from gateway import settings as settings_module  # noqa: E402
settings_module._SETTINGS_FILE = _TMP / "settings.json"

from gateway import upstream as upstream_module  # noqa: E402
upstream_module.UPSTREAMS_FILE = str(_TMP / "upstreams.json")

from gateway import costs  # noqa: E402
costs.AUTO_PRICES_FILE = str(_TMP / "auto_prices.json")

from gateway.services import price_sync  # noqa: E402
from gateway.upstream import Upstream, save_upstreams  # noqa: E402

import httpx  # noqa: E402


# ── 测试基建 ────────────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, payload, status=200, text=""):
        self._payload = payload
        self.status_code = status
        self.text = text or (json.dumps(payload) if isinstance(payload, (dict, list)) else "")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=self)

    def json(self):
        return self._payload


class _FakeClient:
    """url → payload（或 Exception）的假 httpx client，顺带记录调用参数。"""

    def __init__(self, routes: dict, calls: list):
        self.routes = routes
        self.calls = calls

    async def get(self, url, **kwargs):
        self.calls.append({"url": url, "kwargs": kwargs})
        result = self.routes.get(url)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, _FakeResp):
            return result            # 调用方自己造好了状态码/载荷
        if result is None:
            return _FakeResp({"success": False, "message": "no route"}, status=404)
        return _FakeResp(result)


def _install_fake_http(routes: dict) -> list:
    calls: list = []
    client = _FakeClient(routes, calls)
    price_sync.get_client = lambda: client
    return calls


def _make_upstream(**kw) -> Upstream:
    base = dict(name="55", base_url="https://api.test", api_key="sk-test",
                api_format="anthropic")
    base.update(kw)
    return Upstream(**base)


def _pricing_payload(models, group_ratio=None, success=True):
    return {
        "success": success,
        "data": models,
        "group_ratio": group_ratio if group_ratio is not None else {"default": 1, "vip": 1.5},
        "usable_group": {"default": "默认分组", "vip": "VIP"},
        "pricing_version": "test",
    }


def _sonnet():
    """真实 new-api 对 claude-sonnet-4-5 的常见配置：官方 $3/$15。"""
    return {
        "model_name": "claude-sonnet-4-5", "quota_type": 0, "model_ratio": 1.5,
        "model_price": 0, "completion_ratio": 5, "cache_ratio": 0.1,
        "create_cache_ratio": 1.25, "owner_by": "anthropic",
    }


def _reset_store():
    """清掉自动价文件 + mtime 缓存，保证用例互不污染。"""
    p = Path(costs.AUTO_PRICES_FILE)
    if p.exists():
        p.unlink()
    costs.purge_auto_cache()
    settings_module.update_settings(model_prices=[])


def _set_manual(prices: list):
    settings_module.update_settings(model_prices=prices)


# ── 1. 换算口径 ─────────────────────────────────────────────────────────

def test_usd_per_1m_anchor():
    # new-api: QuotaPerUnit = 500000 = $0.002/1K → 倍率 1 = $2/1M
    assert price_sync.usd_per_1m(1.0) == 2.0
    assert price_sync.usd_per_1m(1.5) == 3.0
    assert price_sync.usd_per_1m(1.5, 1.5) == 4.5   # group_ratio 是乘数
    assert price_sync.usd_per_1m(0.0) == 0.0


def test_convert_model_matches_official_claude_prices():
    """锚点核对：换算结果必须等于 Anthropic 官方价 / 中转站的常见倍率配置。
    官方 Sonnet 4.5：输入 $3 / 输出 $15 / 缓存读 0.1x / 缓存写 1h 2x / 5m 1.25x。"""
    entry = price_sync.convert_model(_sonnet(), group_ratio=1.0, cache_ttl="1h")
    assert entry["price_type"] == "token"
    assert entry["input"] == 3.0
    assert entry["output"] == 15.0
    assert entry["cache_read"] == 0.3          # 3 × 0.1
    assert entry["cache_write"] == 6.0         # 3 × 1.25 × 1.6
    assert entry["cache_write_5m"] == 3.75
    assert entry["cache_write_1h"] == 6.0


def test_convert_model_cache_ttl_three_levels():
    for ttl, expected in [("5m", 3.75), ("1h", 6.0), ("off", 0.0)]:
        entry = price_sync.convert_model(_sonnet(), 1.0, ttl)
        assert entry["cache_write"] == expected, ttl
    # 1h 相对 5m 的倍数就是源码里的 6/3.75
    assert abs(price_sync.CACHE_CREATION_1H_MULTIPLIER - 1.6) < 1e-9


def test_convert_model_group_ratio_scales_everything():
    entry = price_sync.convert_model(_sonnet(), group_ratio=2.0, cache_ttl="1h")
    assert entry["input"] == 6.0
    assert entry["output"] == 30.0
    assert entry["cache_read"] == 0.6
    assert entry["cache_write"] == 12.0


def test_convert_model_missing_vs_explicit_zero_cache_ratio():
    """坑 2：字段缺失 = 站点没配过 → 用 new-api 默认（1.0 / 1.25）；
    字段显式为 0 = 站点真的配了 0（如免费缓存读）→ 必须保留 0。
    new-api 用 *float64 + omitempty，指向 0 的指针照样会序列化，所以两者可区分。"""
    item = dict(_sonnet())
    item.pop("cache_ratio")
    item.pop("create_cache_ratio")
    entry = price_sync.convert_model(item, 1.0, "1h")
    assert entry["cache_read"] == 3.0          # 3 × 默认 1.0
    assert entry["cache_write_5m"] == 3.75     # 3 × 默认 1.25

    item2 = dict(_sonnet())
    item2["cache_ratio"] = 0
    item2["create_cache_ratio"] = 0
    entry2 = price_sync.convert_model(item2, 1.0, "1h")
    assert entry2["cache_read"] == 0.0, "显式 0 被 `or default` 吞掉了"
    assert entry2["cache_write"] == 0.0


def test_convert_model_per_call():
    item = {"model_name": "gpt-image", "quota_type": 1, "model_price": 0.05,
            "model_ratio": 0, "completion_ratio": 1}
    entry = price_sync.convert_model(item, group_ratio=2.0, cache_ttl="1h")
    assert entry["price_type"] == "per_call"
    assert abs(entry["price_per_call"] - 0.1) < 1e-9   # 0.05 × 2.0
    assert entry["input"] == 0.0


def test_convert_model_rejects_nameless():
    assert price_sync.convert_model({"model_ratio": 1.0}, 1.0, "1h") is None
    assert price_sync.convert_model({"model_name": "  "}, 1.0, "1h") is None


# ── 1.5 动态计费（billing_mode == "tiered_expr"）─────────────────────────
# 2026-09-14 用户对照上游定价页抓出的真 bug：这类模型的 model_ratio 是过时遗留值，
# 真实价格在 billing_expr 的系数里。小鸡 [Kiro3][手动标记] … 上游页面写 $1.875/$9.375，
# 而按 model_ratio=37.5 换算成 $75/$375 —— 错 40 倍。

_EXPR_SINGLE = ('tier("base", p * 1.875 + c * 9.375 + cr * 0.1875 '
                '+ cc * 2.34375 + cc1h * 3.75)')
_EXPR_TERNARY = ('len <= 272000 ?  tier("base", p * 5 + c * 30 + cr * 0.5 + cc * 6.25)'
                 ': tier("tier2", p * 10 + c * 45 + cr * 1 + cc * 12.5)')
_EXPR_NUM_FIRST = 'tier("base", 2 * p + 10 * c)'


def test_parse_billing_expr_real_samples():
    assert price_sync.parse_billing_expr(_EXPR_SINGLE) == {
        "p": 1.875, "c": 9.375, "cr": 0.1875, "cc": 2.34375, "cc1h": 3.75,
    }
    # 多档表达式只取第一档（base）—— 站点定价页也是把 base 列在最前
    assert price_sync.parse_billing_expr(_EXPR_TERNARY) == {
        "p": 5.0, "c": 30.0, "cr": 0.5, "cc": 6.25,
    }
    # 系数写在左边也能认
    assert price_sync.parse_billing_expr(_EXPR_NUM_FIRST) == {"p": 2.0, "c": 10.0}
    # cc1h 不能被 cc 吃掉（变量名长优先）
    assert "cc1h" in price_sync.parse_billing_expr(_EXPR_SINGLE)


def test_parse_billing_expr_rejects_garbage():
    for bad in ["", "乱七八糟", "model_ratio * 2", "tier("]:
        assert price_sync.parse_billing_expr(bad) == {}, bad
    # 档位名没加引号不影响取系数（解析器只关心系数，不解释档位语义）
    assert price_sync.parse_billing_expr("tier(base, p * 1)") == {"p": 1.0}


def test_convert_model_tiered_expr_matches_relay_pricing_page():
    """换算结果必须与上游定价页逐项一致，且**绝不能**回落到 model_ratio。"""
    item = {"model_name": "[Kiro3][手动标记] claude-opus-4-6-thinking [不补]",
            "quota_type": 0, "model_ratio": 37.5, "completion_ratio": 5,
            "billing_mode": "tiered_expr", "billing_expr": _EXPR_SINGLE}
    e = price_sync.convert_model(item, group_ratio=1.0, cache_ttl="1h")
    # 上游定价页（小鸡 站点）逐项：输入 1.875 / 输出 9.375 / 缓存读 0.1875
    #                            缓存写 2.3438 / 缓存写(1h) 3.75
    assert e["input"] == 1.875
    assert e["output"] == 9.375
    assert e["cache_read"] == 0.1875
    assert e["cache_write_5m"] == 2.34375
    assert e["cache_write_1h"] == 3.75
    assert e["cache_write"] == 3.75           # cache_ttl=1h
    assert e["billing"] == "tiered_expr"
    # 回归锁：绝不能是 model_ratio 那条路算出来的 37.5 × 2 = $75
    assert e["input"] != 75.0

    # 5m / off 两档
    assert price_sync.convert_model(item, 1.0, "5m")["cache_write"] == 2.34375
    assert price_sync.convert_model(item, 1.0, "off")["cache_write"] == 0.0


def test_tiered_expr_still_applies_group_ratio():
    """表达式系数是**分组前**价格（new-api 计费时再乘 groupRatio），所以要乘分组倍率。
    实测锚点：$1.875 × 0.8 = $1.5，正好等于用户手工表里那一行的值。"""
    item = {"model_name": "X", "quota_type": 0, "model_ratio": 37.5,
            "billing_mode": "tiered_expr", "billing_expr": _EXPR_SINGLE}
    e = price_sync.convert_model(item, group_ratio=0.8, cache_ttl="1h")
    assert abs(e["input"] - 1.5) < 1e-9
    assert abs(e["output"] - 7.5) < 1e-9


def test_convert_model_tiered_expr_unparseable_is_skipped_not_guessed():
    """表达式解析不出来时**必须跳过**（返回 None）。若回落 model_ratio，
    会给出错几十倍的价——这正是本次修掉的 bug。"""
    item = {"model_name": "Y", "quota_type": 0, "model_ratio": 37.5,
            "billing_mode": "tiered_expr", "billing_expr": "看不懂的表达式"}
    assert price_sync.convert_model(item, 1.0, "1h") is None


def test_convert_model_without_billing_mode_still_uses_ratio():
    """普通模型（无 billing_mode）必须继续走倍率换算，别被这次改动误伤。"""
    e = price_sync.convert_model(_sonnet(), 1.0, "1h")
    assert e["input"] == 3.0 and e["output"] == 15.0
    assert "billing" not in e


def test_fixed_expr_is_per_call():
    """`tier("base", fixed(N))` = 纯按次 N 美元/次。
    依据：55 站图片模型的**名字里就写着价**（[图-次-0.06￥] → fixed(0.06)、
    [图-次-0.26￥] → fixed(0.26)），用户手工表同一套命名也对得上。"""
    assert price_sync.parse_fixed_price('tier("base", fixed(0.06))') == 0.06
    assert price_sync.parse_fixed_price(_EXPR_SINGLE) is None       # 没有 fixed
    assert price_sync.parse_fixed_price("") is None

    item = {"model_name": "[图-次-0.06￥]gpt-image-2-med", "model_ratio": 37.5,
            "billing_mode": "tiered_expr", "billing_expr": 'tier("base", fixed(0.06))'}
    e = price_sync.convert_model(item, 1.0, "1h")
    assert e["price_type"] == "per_call"
    assert e["price_per_call"] == 0.06
    assert e["input"] == 0.0 and e["billing"] == "tiered_expr"
    # 分组倍率照样要乘
    assert abs(price_sync.convert_model(item, 0.8, "1h")["price_per_call"] - 0.048) < 1e-9


def test_fixed_yields_to_token_coefficients():
    """表达式里同时有 token 系数和 fixed() 时，走 token（别把有量计价误判成按次）。"""
    e = price_sync.convert_model(
        {"model_name": "mix", "billing_mode": "tiered_expr",
         "billing_expr": 'tier("base", p * 2 + fixed(0.06))'}, 1.0, "1h")
    assert e["price_type"] == "token" and e["input"] == 2.0


# ── 2. payload 解析与 URL ───────────────────────────────────────────────

def test_parse_pricing_payload_ok():
    models, groups, usable = price_sync.parse_pricing_payload(
        _pricing_payload([_sonnet()])
    )
    assert len(models) == 1 and models[0]["model_name"] == "claude-sonnet-4-5"
    assert groups["default"] == 1 and usable["vip"] == "VIP"


def test_parse_pricing_payload_rejects_bad_shapes():
    for bad in [
        {"success": False, "message": "no"},
        {"success": True},                 # 没有 data
        {"success": True, "data": "nope"},
        [],
        "nope",
    ]:
        try:
            price_sync.parse_pricing_payload(bad)
        except ValueError:
            continue
        raise AssertionError(f"该形状应当报 ValueError: {bad!r}")


def test_pricing_urls_handles_v1_suffix():
    assert price_sync.pricing_urls("https://a.example.com") == ["https://a.example.com/api/pricing"]
    assert price_sync.pricing_urls("https://a.example.com/") == ["https://a.example.com/api/pricing"]
    assert price_sync.pricing_urls("https://a.example.com/v1") == [
        "https://a.example.com/v1/api/pricing", "https://a.example.com/api/pricing",
    ]


# ── 3. 匿名裸调（源码级回归） ───────────────────────────────────────────

def test_fetch_pricing_sends_no_authorization():
    """坑 1 的回归锁：/api/pricing 走的是后台凭据中间件，把上游的转发
    key 塞进 Authorization 会被判成坏凭据并**直接 abort**（不是降级成匿名）。
    所以这里必须一个鉴权头都不带。"""
    calls = _install_fake_http({
        "https://api.test/api/pricing": _pricing_payload([_sonnet()]),
    })
    url, payload = asyncio.run(price_sync.fetch_pricing(_make_upstream()))
    assert url == "https://api.test/api/pricing"
    assert payload["success"] is True
    assert len(calls) == 1
    headers = calls[0]["kwargs"].get("headers") or {}
    lowered = {k.lower() for k in headers}
    assert not (lowered & {"authorization", "x-api-key"}), (
        f"fetch_pricing 带了鉴权头 {sorted(lowered)}——会被 new-api 直接 403"
    )


def test_fetch_pricing_falls_back_to_root_when_v1_404():
    _install_fake_http({
        "https://api.test/v1/api/pricing": _FakeResp({}, status=404),
        "https://api.test/api/pricing": _pricing_payload([_sonnet()]),
    })
    u = _make_upstream(base_url="https://api.test/v1")
    url, payload = asyncio.run(price_sync.fetch_pricing(u))
    assert url == "https://api.test/api/pricing"
    assert payload["success"] is True


def test_fetch_pricing_401_says_require_auth():
    """401 = 站点把 pricing 模块设成 requireAuth（需登录）。这是"永远拿不到"，
    必须说清楚，别让人当成网络抖动反复重试。"""
    _install_fake_http({
        "https://api.test/api/pricing": _FakeResp(
            {"code": "AUTH_UNAUTHORIZED", "message": "Unauthorized", "success": False},
            status=401),
    })
    try:
        asyncio.run(price_sync.fetch_pricing(_make_upstream()))
    except RuntimeError as e:
        msg = str(e)
        assert "401" in msg and "requireAuth" in msg and "手填" in msg, msg
        return
    raise AssertionError("401 应当抛出可读的 RuntimeError")


def test_fetch_pricing_403_cloudflare_is_not_blamed_on_newapi():
    """403 且响应体是 Cloudflare 的 error code 1010 → 是前置 CF 按 UA 拦截，
    跟 new-api 无关。2026-09-14 真实踩到：用 urllib 探测 relay-c 得到 1010，
    换成网关自己的 httpx UA 就是 200。错误信息必须点出 Cloudflare。"""
    _install_fake_http({
        "https://api.test/api/pricing": _FakeResp({}, status=403, text="error code: 1010 "),
    })
    try:
        asyncio.run(price_sync.fetch_pricing(_make_upstream()))
    except RuntimeError as e:
        msg = str(e)
        assert "Cloudflare" in msg and "1010" in msg, msg
        assert "不是 new-api 关了" in msg, msg
        return
    raise AssertionError("Cloudflare 403 应当抛出可读的 RuntimeError")


def test_fetch_pricing_403_newapi_module_disabled():
    """403 且响应体是 new-api 的错误体 → 站点后台关了价格页模块，只能手填。"""
    _install_fake_http({
        "https://api.test/api/pricing": _FakeResp(
            {"success": False, "message": "该模块已关闭"}, status=403),
    })
    try:
        asyncio.run(price_sync.fetch_pricing(_make_upstream()))
    except RuntimeError as e:
        msg = str(e)
        assert "403" in msg and "价格页模块" in msg, msg
        return
    raise AssertionError("403 应当抛出可读的 RuntimeError")


def test_fetch_pricing_all_404_says_station_has_no_pricing_api():
    """候选路径全 404 = 这个站压根没有 /api/pricing。
    2026-09-19 实测 relay-d(relay-d.test)：/api/status、/api/pricing、/api/about
    全是 Go 的 "404 page not found"，只有 /v1/* —— 不是 new-api 系，永远拿不到价。
    必须说清楚，否则下一个人会反复重试一个不可能成功的站。"""
    _install_fake_http({
        "https://api.test/api/pricing": _FakeResp({}, status=404, text="404 page not found"),
    })
    try:
        asyncio.run(price_sync.fetch_pricing(_make_upstream()))
    except RuntimeError as e:
        msg = str(e)
        assert "没有 /api/pricing" in msg and "请继续手填" in msg, msg
        return
    raise AssertionError("全 404 应当抛出可读的 RuntimeError")


def test_fetch_pricing_appends_proxy_hint_when_host_not_whitelisted():
    """忘补 NO_PROXY 时（relay-d 那次），失败信息里必须带上"很可能走了代理"的提示，
    否则 ConnectError 的 str 是空的、看起来像 TLS 问题，排查全靠猜。"""
    _install_fake_http({"https://api.test/api/pricing": RuntimeError("boom")})
    saved = {k: os.environ.get(k) for k in ("HTTPS_PROXY", "NO_PROXY", "no_proxy")}
    try:
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
        os.environ.pop("NO_PROXY", None)
        os.environ.pop("no_proxy", None)
        try:
            asyncio.run(price_sync.fetch_pricing(_make_upstream()))
        except RuntimeError as e:
            msg = str(e)
            assert "NO_PROXY" in msg and "trust_env" in msg, msg
        else:
            raise AssertionError("应当抛出 RuntimeError")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── 4. 取价优先级 ───────────────────────────────────────────────────────

def test_resolve_price_prefers_manual_over_auto():
    _reset_store()
    costs.save_auto_prices(
        [{"upstream": "55", "model": "claude-sonnet-4-5", "price_type": "token",
          "input": 3.0, "output": 15.0, "cache_write": 6.0, "cache_read": 0.3}],
        {"55": {"ok": True}}, "2026-09-14T00:00:00",
    )
    _set_manual([{"upstream": "55", "model": "claude-sonnet-4-5", "price_type": "token",
                  "input": 1.0, "output": 2.0, "cache_write": 3.0, "cache_read": 4.0}])
    entry, src = costs.resolve_price("55", "claude-sonnet-4-5")
    assert src == costs.SOURCE_MANUAL
    assert entry["input"] == 1.0

    _set_manual([])   # 撤掉手工 → 落到自动价
    entry, src = costs.resolve_price("55", "claude-sonnet-4-5")
    assert src == costs.SOURCE_AUTO
    assert entry["input"] == 3.0


def test_resolve_price_default_when_nothing_configured():
    _reset_store()
    entry, src = costs.resolve_price("55", "谁都不认识我")
    assert entry is None and src == costs.SOURCE_DEFAULT
    # 默认价的算钱结果 = 旧行为（Opus 4 官方价）
    cost, saved = costs.calculate_cost("55", "谁都不认识我", 1_000_000, 1_000_000, 0, 0)
    assert abs(cost - (15.0 + 75.0)) < 1e-9
    assert saved == 0.0


def test_auto_price_does_not_leak_across_upstreams():
    """A 站的价格绝不能算到 B 站头上（手工表有"不限上游"的兜底语义，
    自动表必须没有——每家中转站价格不同）。"""
    _reset_store()
    costs.save_auto_prices(
        [{"upstream": "A", "model": "claude-sonnet-4-5", "price_type": "token",
          "input": 3.0, "output": 15.0, "cache_write": 6.0, "cache_read": 0.3}],
        {"A": {"ok": True}}, "2026-09-14T00:00:00",
    )
    entry, src = costs.resolve_price("B", "claude-sonnet-4-5")
    assert entry is None and src == costs.SOURCE_DEFAULT
    entry, src = costs.resolve_price("A", "claude-sonnet-4-5")
    assert src == costs.SOURCE_AUTO


def test_auto_price_case_insensitive_then_unique_substring():
    _reset_store()
    costs.save_auto_prices(
        [
            {"upstream": "55", "model": "claude-opus-4-5", "price_type": "token",
             "input": 15.0, "output": 75.0, "cache_write": 30.0, "cache_read": 1.5},
            {"upstream": "55", "model": "claude-haiku-4-5", "price_type": "token",
             "input": 1.0, "output": 5.0, "cache_write": 2.0, "cache_read": 0.1},
        ],
        {"55": {"ok": True}}, "2026-09-14T00:00:00",
    )
    # 大小写不敏感精确
    entry, src = costs.resolve_price("55", "CLAUDE-OPUS-4-5")
    assert src == costs.SOURCE_AUTO and entry["input"] == 15.0
    # 唯一子串：客户端带了个下标后缀也能命中
    entry, src = costs.resolve_price("55", "claude-haiku-4-5-20251001")
    assert src == costs.SOURCE_AUTO and entry["input"] == 1.0


def test_auto_price_ambiguous_substring_falls_back():
    """有歧义时宁可退回默认价，也不猜一个可能错几倍的单价。"""
    _reset_store()
    costs.save_auto_prices(
        [
            {"upstream": "55", "model": "claude-opus-4-5", "price_type": "token",
             "input": 15.0, "output": 75.0, "cache_write": 30.0, "cache_read": 1.5},
            {"upstream": "55", "model": "claude-opus-4-5-thinking", "price_type": "token",
             "input": 15.0, "output": 75.0, "cache_write": 30.0, "cache_read": 1.5},
        ],
        {"55": {"ok": True}}, "2026-09-14T00:00:00",
    )
    entry, src = costs.resolve_price("55", "opus-4-5")
    assert entry is None and src == costs.SOURCE_DEFAULT


def test_load_auto_prices_survives_corrupt_file():
    _reset_store()
    Path(costs.AUTO_PRICES_FILE).write_text("{ 这不是 json", encoding="utf-8")
    costs.purge_auto_cache()
    snap = costs.load_auto_prices()
    assert snap["prices"] == [] and snap["upstreams"] == {}
    # 坏文件不能让归档崩：取价退回默认
    entry, src = costs.resolve_price("55", "x")
    assert entry is None and src == costs.SOURCE_DEFAULT


def test_calculate_cost_uses_auto_price():
    _reset_store()
    costs.save_auto_prices(
        [{"upstream": "55", "model": "claude-sonnet-4-5", "price_type": "token",
          "input": 3.0, "output": 15.0, "cache_write": 6.0, "cache_read": 0.3}],
        {"55": {"ok": True}}, "2026-09-14T00:00:00",
    )
    cost, saved = costs.calculate_cost("55", "claude-sonnet-4-5",
                                      1_000_000, 1_000_000, 1_000_000, 1_000_000)
    assert abs(cost - (3.0 + 15.0 + 6.0 + 0.3)) < 1e-9
    assert abs(saved - (3.0 - 0.3)) < 1e-9

    # 按次
    costs.save_auto_prices(
        [{"upstream": "55", "model": "gpt-image", "price_type": "per_call",
          "price_per_call": 0.02, "input": 0, "output": 0, "cache_write": 0, "cache_read": 0}],
        {"55": {"ok": True}}, "2026-09-14T00:00:00",
    )
    cost, saved = costs.calculate_cost("55", "gpt-image", 999, 999, 0, 0, msgs=3)
    assert abs(cost - 0.06) < 1e-9 and saved == 0.0


# ── 5. 同步流程 ─────────────────────────────────────────────────────────

def test_sync_upstreams_writes_snapshot():
    _reset_store()
    save_upstreams([_make_upstream(name="55", base_url="https://api.test"),
                    _make_upstream(name="B", base_url="https://b.test")])
    _install_fake_http({
        "https://api.test/api/pricing": _pricing_payload(
            [_sonnet(), {"model_name": "claude-opus-4-5", "quota_type": 0,
                         "model_ratio": 7.5, "completion_ratio": 5,
                         "cache_ratio": 0.1, "create_cache_ratio": 1.25}]),
        "https://b.test/api/pricing": _pricing_payload(
            [{"model_name": "gemini-3-pro", "quota_type": 1, "model_price": 0.02}],
            group_ratio={"default": 1}),
    })

    meta = asyncio.run(price_sync.sync_upstreams())
    assert meta["55"]["ok"] and meta["55"]["count"] == 2
    assert meta["B"]["ok"] and meta["B"]["count"] == 1
    assert meta["B"]["per_call_count"] == 1

    snap = costs.load_auto_prices()
    assert snap["synced_at"]
    assert len(snap["prices"]) == 3
    # 落盘条目带 upstream，且换算正确
    sonnet = next(p for p in snap["prices"]
                  if p["model"] == "claude-sonnet-4-5" and p["upstream"] == "55")
    assert sonnet["input"] == 3.0 and sonnet["output"] == 15.0
    # 元信息不该把上千条价格再抄一份（文件体积）
    assert "prices" not in meta["55"]
    # 马上就能取价
    entry, src = costs.resolve_price("55", "claude-sonnet-4-5")
    assert src == costs.SOURCE_AUTO and entry["input"] == 3.0


def test_sync_upstreams_keeps_old_prices_on_failure():
    """一次 500 不能让整站价格凭空消失（否则费用瞬间失真成默认价）。"""
    _reset_store()
    save_upstreams([_make_upstream(name="55", base_url="https://api.test")])
    _install_fake_http({"https://api.test/api/pricing": _pricing_payload([_sonnet()])})
    asyncio.run(price_sync.sync_upstreams())
    before = len(costs.load_auto_prices()["prices"])
    assert before == 1

    _install_fake_http({"https://api.test/api/pricing": RuntimeError("上游炸了")})
    meta = asyncio.run(price_sync.sync_upstreams())
    assert meta["55"]["ok"] is False
    assert meta["55"]["stale"] is True and meta["55"]["count"] == 1
    snap = costs.load_auto_prices()
    assert len(snap["prices"]) == 1, "失败时旧价被清掉了"
    entry, src = costs.resolve_price("55", "claude-sonnet-4-5")
    assert src == costs.SOURCE_AUTO


def test_sync_upstreams_drops_removed_upstream():
    _reset_store()
    save_upstreams([_make_upstream(name="55", base_url="https://api.test"),
                    _make_upstream(name="OLD", base_url="https://old.test")])
    _install_fake_http({
        "https://api.test/api/pricing": _pricing_payload([_sonnet()]),
        "https://old.test/api/pricing": _pricing_payload(
            [{"model_name": "gone-model", "quota_type": 0, "model_ratio": 1.0,
              "completion_ratio": 1}]),
    })
    asyncio.run(price_sync.sync_upstreams())
    assert any(p["upstream"] == "OLD" for p in costs.load_auto_prices()["prices"])

    # 把 OLD 从配置里删掉再同步 → 它的陈旧条目必须消失
    save_upstreams([_make_upstream(name="55", base_url="https://api.test")])
    asyncio.run(price_sync.sync_upstreams())
    snap = costs.load_auto_prices()
    assert all(p["upstream"] != "OLD" for p in snap["prices"])
    assert "OLD" not in snap["upstreams"]


def test_sync_one_upstream_leaves_others_untouched():
    """单站同步只替换被点名的上游，别家的价格和 meta 必须原样保留
    （否则点一次"同步此站"，其它站的价格会被静默清掉、费用掉回默认价）。"""
    _reset_store()
    save_upstreams([_make_upstream(name="55", base_url="https://api.test"),
                    _make_upstream(name="B", base_url="https://b.test")])
    _install_fake_http({
        "https://api.test/api/pricing": _pricing_payload([_sonnet()]),
        "https://b.test/api/pricing": _pricing_payload(
            [{"model_name": "gemini-3-pro", "quota_type": 0, "model_ratio": 0.5,
              "completion_ratio": 2}], group_ratio={"default": 1}),
    })
    asyncio.run(price_sync.sync_upstreams())          # 先全量，两家都有价
    assert {p["upstream"] for p in costs.load_auto_prices()["prices"]} == {"55", "B"}

    # 只同步 B：55 的条目和 meta 都得还在
    meta = asyncio.run(price_sync.sync_upstreams(["B"]))
    assert set(meta) == {"B"}
    snap = costs.load_auto_prices()
    assert {p["upstream"] for p in snap["prices"]} == {"55", "B"}
    assert "55" in snap["upstreams"]
    entry, src = costs.resolve_price("55", "claude-sonnet-4-5")
    assert src == costs.SOURCE_AUTO and entry["input"] == 3.0


def test_sync_unknown_upstream_writes_nothing():
    """点一个不存在的上游 → 一个字节都不写。曾经的实现会把快照清空，
    全站价格瞬间掉回 Opus 默认价。"""
    _reset_store()
    save_upstreams([_make_upstream(name="55", base_url="https://api.test")])
    _install_fake_http({"https://api.test/api/pricing": _pricing_payload([_sonnet()])})
    asyncio.run(price_sync.sync_upstreams())
    before = json.loads(Path(costs.AUTO_PRICES_FILE).read_text(encoding="utf-8"))

    meta = asyncio.run(price_sync.sync_upstreams(["不存在的中转站"]))
    assert meta == {}
    after = json.loads(Path(costs.AUTO_PRICES_FILE).read_text(encoding="utf-8"))
    assert before == after, "同步不存在的上游把快照改掉了"
    entry, src = costs.resolve_price("55", "claude-sonnet-4-5")
    assert src == costs.SOURCE_AUTO


def test_pricing_group_applies_and_warns_on_unknown():
    _reset_store()
    # 配了 vip 分组 → 用 vip 的倍率
    save_upstreams([_make_upstream(name="55", base_url="https://api.test",
                                   pricing_group="vip")])
    _install_fake_http({
        "https://api.test/api/pricing": _pricing_payload([_sonnet()]),
    })
    meta = asyncio.run(price_sync.sync_upstreams())
    assert meta["55"]["group"] == "vip" and meta["55"]["group_ratio"] == 1.5
    assert not meta["55"]["warning"]
    entry, _ = costs.resolve_price("55", "claude-sonnet-4-5")
    assert entry["input"] == 4.5            # 3.0 × 1.5

    # 配了个不存在的分组 → 回退 default 并给出警告（不静默算错）
    save_upstreams([_make_upstream(name="55", base_url="https://api.test",
                                   pricing_group="nope")])
    meta = asyncio.run(price_sync.sync_upstreams())
    assert meta["55"]["group"] == "default" and meta["55"]["group_ratio"] == 1.0
    assert "nope" in meta["55"]["warning"]


def test_should_auto_sync_respects_switch_and_interval():
    _reset_store()
    settings_module.update_settings(auto_price_enabled=False, auto_price_interval_hours=12)
    try:
        price_sync.reset_sync_state()
        assert price_sync.should_auto_sync() is False      # 总开关关着

        settings_module.update_settings(auto_price_enabled=True)
        assert price_sync.should_auto_sync() is True       # 冷启动首次即视为到期

        save_upstreams([_make_upstream(name="55", base_url="https://api.test")])
        _install_fake_http({"https://api.test/api/pricing": _pricing_payload([_sonnet()])})
        asyncio.run(price_sync.sync_upstreams())
        assert price_sync.should_auto_sync() is False      # 刚同步完，在冷却期内
        later = price_sync._last_sync + 12 * 3600 + 1
        assert price_sync.should_auto_sync(now=later) is True

        settings_module.update_settings(auto_price_interval_hours=1)
        assert price_sync.should_auto_sync(now=price_sync._last_sync + 3601) is True
    finally:
        settings_module.update_settings(auto_price_enabled=False,
                                        auto_price_interval_hours=12)


def test_sync_does_not_touch_upstreams_json_models():
    """价格同步只写 auto_prices.json，绝不能顺手改 upstreams.json——
    cached_models 是模型同步的职责，越界会让 /v1/models 行为变得难以解释。"""
    _reset_store()
    save_upstreams([_make_upstream(name="55", base_url="https://api.test")])
    _install_fake_http({"https://api.test/api/pricing": _pricing_payload([_sonnet()])})
    before = json.loads(Path(upstream_module.UPSTREAMS_FILE).read_text(encoding="utf-8"))
    asyncio.run(price_sync.sync_upstreams())
    after = json.loads(Path(upstream_module.UPSTREAMS_FILE).read_text(encoding="utf-8"))
    assert before[0]["cached_models"] == after[0]["cached_models"] == []


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
