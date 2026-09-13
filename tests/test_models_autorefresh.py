"""上游模型列表自动同步测试（services/model_sync.py，2026-08-28）。

覆盖场景：
    1. refresh_one_upstream_models：主令牌 + 额外令牌分组拉取与缓存落盘
    2. auto_refresh_if_due：冷启动首次触发（/v1/models 钩子）
    3. 冷却期内不重复触发（防打爆上游）
    4. 冷却期过后再次触发
    5. settings 关闭总开关后不触发

⚠️ 铁律（同 test_unified_brain.py）：测试输入必须按真实调用方的形状构造——
    真实链路是 routers/models.py 的 list_models() 调 auto_refresh_if_due()，
    内部对 load_upstreams() 的每个 active 上游调 fetch_models_for_key()
    （models_url + api_key + api_format），再经 upstream.update_* 落盘。
    这里全部按该形状 mock，只是把网络层换成 canned 数据、把落盘重定向到临时文件。

设计要点：
- monkeypatch upstream.UPSTREAMS_FILE / settings._SETTINGS_FILE 到临时目录，
  绝不碰真实 data/（upstream.update_* 内部读的是模块全局，setattr 即可生效）。
- 既可以 `python tests/test_models_autorefresh.py` 直接跑（无需 pytest），
  也可以 `python -m pytest tests/test_models_autorefresh.py -v` 跑。
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# 确保能 import gateway 包（tests/ 与 gateway/ 同级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── 在 import 被测代码之前，把落盘目标全部指向临时文件 ──
_TMP = Path(tempfile.mkdtemp(prefix="model_sync_test_"))
(_TMP / "settings.json").write_text("{}", encoding="utf-8")

from gateway import settings as settings_module  # noqa: E402
settings_module._SETTINGS_FILE = _TMP / "settings.json"

from gateway import upstream as upstream_module  # noqa: E402
upstream_module.UPSTREAMS_FILE = str(_TMP / "upstreams.json")

from gateway.services import model_sync  # noqa: E402
from gateway.upstream import Upstream, load_upstreams, save_upstreams  # noqa: E402


def _make_upstream(**kw) -> Upstream:
    base = dict(
        name="55",
        base_url="https://api.test",
        api_key="sk-test",
        api_format="anthropic",
        extra_keys=[{"key": "sk-extra", "label": "备用", "purpose": "chat"}],
    )
    base.update(kw)
    return Upstream(**base)


def _install_fake_fetch(calls: list):
    """按真实形状 mock fetch_models_for_key：入参 (models_url, api_key, api_format)。"""
    canned = {"sk-test": {"model-a", "model-b"}, "sk-extra": {"model-c"}}

    async def fake_fetch(models_url, api_key, api_format):
        calls.append(api_key)
        assert models_url == "https://api.test/v1/models"
        assert api_format == "anthropic"
        return canned[api_key]

    model_sync.fetch_models_for_key = fake_fetch


def test_refresh_one_upstream_updates_grouped_caches():
    calls: list = []
    _install_fake_fetch(calls)
    u = _make_upstream()
    save_upstreams([u])

    result = asyncio.run(model_sync.refresh_one_upstream_models(u))

    assert result["ok"] is True and result["count"] == 3
    assert [g["key_index"] for g in result["groups"]] == ["primary", 0]
    reloaded = load_upstreams()[0]
    assert reloaded.primary_cached_models == ["model-a", "model-b"]
    assert reloaded.extra_keys[0]["cached_models"] == ["model-c"]
    assert reloaded.cached_models == ["model-a", "model-b", "model-c"]


def test_auto_refresh_triggers_when_cold_and_sets_cooldown():
    calls: list = []
    _install_fake_fetch(calls)
    save_upstreams([_make_upstream()])
    model_sync.reset_auto_refresh_state()

    import asyncio
    assert model_sync.should_auto_refresh() is True
    result = asyncio.run(model_sync.auto_refresh_if_due())
    assert result is not None and result["55"]["ok"] is True
    assert len(calls) == 2  # 主令牌 + 额外令牌各拉一次

    # 冷却期内：不再拉取
    assert model_sync.should_auto_refresh() is False
    assert asyncio.run(model_sync.auto_refresh_if_due()) is None
    assert len(calls) == 2


def test_auto_refresh_triggers_again_after_cooldown():
    calls: list = []
    _install_fake_fetch(calls)
    save_upstreams([_make_upstream()])
    model_sync.reset_auto_refresh_state()
    import asyncio
    asyncio.run(model_sync.auto_refresh_if_due())
    assert len(calls) == 2

    # 把冷却计时拨回去（模拟 30 分钟后）
    import time
    model_sync._last_auto_refresh -= (
        int(settings_module.get("models_auto_refresh_cooldown_min", 30)) * 60 + 1
    )
    assert model_sync.should_auto_refresh() is True
    asyncio.run(model_sync.auto_refresh_if_due())
    assert len(calls) == 4


def test_disabled_via_settings():
    calls: list = []
    _install_fake_fetch(calls)
    save_upstreams([_make_upstream()])
    model_sync.reset_auto_refresh_state()
    settings_module.update_settings(models_auto_refresh_on_client_pull=False)

    import asyncio
    try:
        assert model_sync.should_auto_refresh() is False
        assert asyncio.run(model_sync.auto_refresh_if_due()) is None
        assert calls == []
    finally:
        settings_module.update_settings(models_auto_refresh_on_client_pull=True)


def test_no_active_upstream_is_noop():
    calls: list = []
    _install_fake_fetch(calls)
    save_upstreams([_make_upstream(is_active=False)])
    model_sync.reset_auto_refresh_state()

    import asyncio
    result = asyncio.run(model_sync.refresh_all_upstream_models())
    assert result == {} and calls == []
    assert asyncio.run(model_sync.auto_refresh_if_due()) == {}  # 没有上游也安全：空结果、不炸、计冷却


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
