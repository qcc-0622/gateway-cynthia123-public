"""管理后台页面渲染门禁（2026-09-14）。

为什么需要：/monitoring、/gateway-settings 这些 Jinja 页面此前**没有任何测试**——
模板里引用一个不存在的上下文键、或改路由时漏传一个变量，pytest 全绿而线上点开
就是 500。2026-09-14 加"上游价格自动同步"时就踩到了：路由忘记把 auto_price 传进
上下文，Python 侧一片绿。本测试把三个页面真渲染一遍（真路由 + 真模板 + 真 DB），
并断言关键片段，让模板/上下文漂移在门禁就红。

顺带锁住两条取价链路的行为：
    - 后台"同步价格"按钮在网络全挂时也要返回 JSON，且**保留旧价格**（不能清空）
    - 点一个不存在的上游同步 → 404 而不是 500，且不许改动快照

⚠️ DB 隔离铁律（CLAUDE.md 第九节）：pytest 按字母序收集，"谁先 import
   gateway.db"决定 DB_PATH 固化结果。所以 os.environ 之外还必须**直接改写
   db_module.DB_PATH** 兜底，否则会读写真实 data/gateway.db。

既可 `python tests/test_admin_render.py` 直跑，也可 pytest 跑。
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── 在 import gateway.* 之前把三个落盘目标 + DB 指向临时目录 ──
_TMP = Path(tempfile.mkdtemp(prefix="admin_render_test_"))
os.environ["DB_PATH"] = str(_TMP / "render.db")
os.environ["UPSTREAMS_FILE"] = str(_TMP / "upstreams.json")
os.environ["ADMIN_PASSWORD"] = "render-test"

from gateway import db as db_module  # noqa: E402
from gateway import upstream as upstream_module  # noqa: E402
from gateway import settings as settings_module  # noqa: E402
from gateway import costs  # noqa: E402

db_module.DB_PATH = str(_TMP / "render.db")          # 兜底：见文件头 DB 隔离铁律
upstream_module.UPSTREAMS_FILE = str(_TMP / "upstreams.json")
settings_module._SETTINGS_FILE = _TMP / "settings.json"
settings_module._SETTINGS_FILE.write_text("{}", encoding="utf-8")
costs.AUTO_PRICES_FILE = str(_TMP / "auto_prices.json")

from gateway.db import init_db, save_conversation, close_db  # noqa: E402
from gateway.upstream import Upstream, save_upstreams  # noqa: E402


def _seed_upstreams():
    save_upstreams([
        Upstream(name="55", base_url="https://api.test", api_key="sk-a",
                 api_format="anthropic",
                 cached_models=["claude-sonnet-4-5", "claude-opus-4-5"],
                 cache_ttl="1h"),
        Upstream(name="relay-b", base_url="https://b.test", api_key="sk-b",
                 api_format="openai", pricing_group="vip"),
    ])


def _seed_auto_prices():
    """一份含三种状态的快照：成功 / 失败但沿用旧价 / 分组回退警告。"""
    costs.save_auto_prices(
        [
            {"upstream": "55", "model": "claude-sonnet-4-5", "price_type": "token",
             "input": 3.0, "output": 15.0, "cache_write": 6.0, "cache_read": 0.3,
             "cache_write_5m": 3.75, "cache_write_1h": 6.0, "cache_ttl": "1h",
             "source": "new-api"},
            # 这条会被手工价压住（见 _seed_settings）—— 用来验证"上游价 vs 手工价"
            # 的偏差提示真的渲染出来（用户反馈"拉了价但价格没更新"的直接诉求）
            {"upstream": "55", "model": "claude-haiku-4-5", "price_type": "token",
             "input": 1.0, "output": 5.0, "cache_write": 1.25, "cache_read": 0.1,
             "cache_ttl": "1h", "source": "new-api"},
            {"upstream": "relay-b", "model": "gemini-3-pro", "price_type": "per_call",
             "price_per_call": 0.02, "input": 0, "output": 0,
             "cache_write": 0, "cache_read": 0},
        ],
        {
            "55": {"ok": True, "count": 2, "group": "default", "group_ratio": 1.0,
                   "group_options": {"default": 1.0, "vip": 1.5},
                   "synced_at": "2026-09-14T10:00:00", "warning": "", "error": "",
                   "stale": False},
            "relay-b": {"ok": False, "count": 1, "group": "default", "group_ratio": 1.0,
                        "group_options": {}, "synced_at": "2026-09-14T09:00:00",
                        "warning": "分组 'vip' 不在上游分组表里，已回退到 default（倍率 1.0）",
                        "error": "拉取价格表失败：https://b.test/api/pricing 返回 403",
                        "stale": True},
        },
        synced_at="2026-09-14T10:00:00",
    )


def _seed_settings():
    settings_module.update_settings(
        auto_price_enabled=True,
        auto_price_interval_hours=6,
        model_prices=[
            {"upstream": "55", "model": "claude-opus-4-5",
             "price_type": "token", "price_per_call": 0,
             "input": 15.0, "output": 75.0,
             "cache_write": 30.0, "cache_read": 1.5},
            # 故意只有上游价的 0.2 倍 —— 模拟"分组倍率没配对"造成的系统性偏差
            {"upstream": "55", "model": "claude-haiku-4-5",
             "price_type": "token", "price_per_call": 0,
             "input": 0.2, "output": 1.0,
             "cache_write": 0.25, "cache_read": 0.02},
        ],
    )


async def _seed_conversations():
    """每个用例都从**空库**开始：这些用例断言的是页面上渲染出来的具体金额，
    行一累积金额就翻倍（曾出现 4 个用例×5 行 → 断言 0.0423 而页面是 0.1692）。

    timestamp 必须显式传：db.save_conversation 默认是空串（真调用方都传 now），
    空串会被统计查询的 `timestamp >= date('now', ...)` 过滤掉。"""
    from datetime import datetime, timedelta
    await close_db()
    db_file = Path(db_module.DB_PATH)
    if db_file.exists():
        db_file.unlink()
    await init_db()
    now = datetime.now()
    for i in range(3):
        await save_conversation(
            conversation_id=f"c{i}", role="assistant", content="hi",
            model="claude-sonnet-4-5", client_model="claude-sonnet-4-5",
            tokens_in=1000, tokens_out=500, cache_write_tokens=200,
            cache_read_tokens=8000, upstream_name="55",
            cost_usd=0.01, saved_usd=0.02,
            timestamp=(now - timedelta(minutes=i)).isoformat(timespec="seconds"),
        )
    for cid, model, ups in [("c9", "gemini-3-pro", "relay-b"),
                            ("c10", "mystery-model", "55"),
                            ("c11", "claude-haiku-4-5", "55")]:
        await save_conversation(
            conversation_id=cid, role="assistant", content="hi",
            model=model, client_model=model, tokens_in=100, tokens_out=100,
            upstream_name=ups, timestamp=now.isoformat(timespec="seconds"),
        )
    await close_db()   # aiosqlite 连接绑 loop，交棒给 TestClient 前必须关（见 conftest）


def _install_offline_client():
    """把价格同步的网络层换成"立刻失败"，让后台按钮路径可测且不真发外呼。
    故意抛非连接类异常：fetch_pricing 对 SSL/ConnectError 会再试 no_verify
    （那是真的会出网），用 RuntimeError 走纯失败分支。"""
    from gateway.services import price_sync

    class _DeadClient:
        async def get(self, url, **kwargs):
            raise RuntimeError(f"offline test client: {url}")

    price_sync.get_client = lambda: _DeadClient()


def _client():
    from fastapi.testclient import TestClient
    from gateway.main import app
    from gateway.routers import admin as admin_mod
    admin_mod._check_auth = lambda request: True    # 免登录，只测渲染
    return TestClient(app)


def _prepare():
    _seed_upstreams()
    _seed_auto_prices()
    _seed_settings()
    asyncio.run(_seed_conversations())
    _install_offline_client()


def test_pages_render_with_auto_pricing_context():
    """三个后台页面必须 200，且把自动价格的状态真实渲染出来。
    回归锁：路由漏传 auto_price → 模板 Jinja StrictUndefined 报错 / 片段缺失。"""
    _prepare()
    with _client() as client:
        mon = client.get("/admin/monitoring")
        sets = client.get("/admin/gateway-settings")
        ups = client.get("/admin/upstreams")

    assert mon.status_code == 200, mon.text[:500]
    assert sets.status_code == 200, sets.text[:500]
    assert ups.status_code == 200, ups.text[:500]

    body = mon.text
    # 按 (中转站 × 模型) 统计真的渲染出来了（含无价模型）
    assert "model-stat-table" in body
    for model in ("claude-sonnet-4-5", "gemini-3-pro", "mystery-model"):
        assert model in body, f"监控页缺少模型行 {model}"
    # 取价来源标识：自动 / 手工 / 没价告警
    assert "自动" in body and "手工" in body
    assert "既没手工配价" in body and "快捷定价" in body

    # 上游拉回来的自动价必须摊在模型行里（用户反馈"拉了价但价格没更新"的诉求）：
    # ① 自动价就是生效价的那行 —— 直接显示上游价
    assert "上游价 入 $3 / 出 $15" in body
    # ② 自动价被手工价压住的那行 —— 连偏差倍数一起显示，否则看不出分组倍率配错
    assert "上游价 入 $1 / 出 $5" in body
    assert "手工价是它的 0.20×" in body

    # 自动同步状态面板：成功、失败沿用旧价、分组回退警告三种都要显示
    assert "上游价格自动同步" in body
    assert "不在上游分组表里" in body          # 引号被 Jinja 转义，只匹配无引号部分
    assert "仍用 1 条旧价格" in body
    assert "分组 default ×1.0" in body

    # 设置页：总开关 + 分组配置 + 两个 JS 入口
    sbody = sets.text
    for frag in ("上游价格自动同步", "开启自动同步", "各上游分组",
                 "syncAllPrices", "syncOnePrice", "savePricingGroup",
                 "/pricing-group"):
        assert frag in sbody, f"设置页缺少片段 {frag}"


def test_refresh_all_prices_endpoint_keeps_old_snapshot():
    """网络全挂时：接口返回 JSON（不是 500），且旧价格必须原样保留。
    否则一次上游抖动就让全站费用掉回 Opus 默认价。"""
    _prepare()
    before = json.loads(Path(costs.AUTO_PRICES_FILE).read_text(encoding="utf-8"))
    assert len(before["prices"]) == 3     # 55 两条 + relay-b 一条（见 _seed_auto_prices）

    with _client() as client:
        r = client.post("/admin/upstreams/refresh-all-prices")

    assert r.status_code == 200, r.text[:300]
    data = r.json()
    assert set(data) == {"55", "relay-b"}
    assert data["55"]["ok"] is False and data["55"]["stale"] is True
    assert data["55"]["count"] == 2            # 沿用了旧价（两条）
    assert data["55"]["error"]

    after = json.loads(Path(costs.AUTO_PRICES_FILE).read_text(encoding="utf-8"))
    assert {p["model"] for p in after["prices"]} == \
           {p["model"] for p in before["prices"]}, "同步失败把旧价格清掉了"


def test_unknown_upstream_endpoints_are_safe():
    """点一个不存在的上游：404 + 一个字节都不写（曾经会把快照清空）。"""
    _prepare()
    before = json.loads(Path(costs.AUTO_PRICES_FILE).read_text(encoding="utf-8"))

    with _client() as client:
        r = client.post("/admin/upstreams/不存在/refresh-prices")
        g = client.post("/admin/upstreams/55/pricing-group",
                        data={"pricing_group": "vip"})

    assert r.status_code == 404 and "不存在或未启用" in r.json()["error"]
    after = json.loads(Path(costs.AUTO_PRICES_FILE).read_text(encoding="utf-8"))
    assert before == after, "同步不存在的上游改动了快照"

    # 分组保存走 JSON（设置页是整页一个大 form，塞不进嵌套 form）
    assert g.status_code == 200 and g.json()["ok"] is True
    assert g.json()["pricing_group"] == "vip"
    assert upstream_module.load_upstreams()[0].pricing_group == "vip"


def test_monitoring_uses_same_price_source_as_archiver():
    """监控页的钱必须和归档同一套取价逻辑（手工 > 自动 > 默认）——
    旧版监控页自带了第二份匹配/算术实现，改一处忘一处。"""
    _prepare()
    with _client() as client:
        body = client.get("/admin/monitoring").text

    # claude-sonnet-4-5 只有自动价（$3/$15/$6/$0.3）→ 3 条消息：
    # (3000*3 + 1500*15 + 600*6 + 24000*0.3)/1e6 = (9000+22500+3600+7200)/1e6 = 0.0423
    entry, src = costs.resolve_price("55", "claude-sonnet-4-5")
    assert src == costs.SOURCE_AUTO
    assert abs(costs.calculate_cost("55", "claude-sonnet-4-5",
                                    3000, 1500, 600, 24000)[0] - 0.0423) < 1e-9
    assert "0.0423" in body, "监控页没有按自动价算出预期金额"
    # gemini-3-pro 走自动的按次价 $0.02/次
    assert "0.02" in body
    # 没价模型按默认 Opus 价：(100*15 + 100*75)/1e6 = 0.009
    assert "0.009" in body


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
