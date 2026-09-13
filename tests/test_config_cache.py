"""settings.json / upstreams.json mtime 缓存回归测试（2026-09-01）。

背景：热路径每请求多次同步读盘 + json.loads（settings._ensure 每请求 8+ 次、
load_upstreams 每请求 1-2 次）。加 (路径, mtime_ns, size) 签名缓存后要保证：
  1. 未变化时命中缓存（不读盘）
  2. 程序内写入（update_settings / save_upstreams）后立刻读到新值——
     不能依赖 mtime 变化（Windows 同尺寸快速重写粒度可能撞车）
  3. 外部手改文件（绕过程序）后 stat 变化 → 自动失效读到新值

既可以 `python tests/test_config_cache.py` 直接跑，也可以 pytest 跑。
⚠️ 本文件不 import gateway.db；文件路径 patch 只在测试函数内做并恢复，
不污染其它测试模块的模块级 patch（pytest 按字母序收集，本文件最先导入）。
"""

import json
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from gateway import settings as settings_module  # noqa: E402
from gateway import upstream as upstream_module  # noqa: E402
from gateway.upstream import Upstream  # noqa: E402


def test_settings_cache_hit_then_invalidate():
    tmp = Path(tempfile.mkdtemp(prefix="cfg_cache_test_")) / "settings.json"
    old_file = settings_module._SETTINGS_FILE
    old_cache = settings_module._cache
    settings_module._SETTINGS_FILE = tmp
    try:
        settings_module._cache = None
        tmp.write_text(json.dumps({"summary_compress_threshold": 111}), encoding="utf-8")

        # 首读 → 建缓存
        assert settings_module.get("summary_compress_threshold") == 111
        assert settings_module._cache is not None

        # 未变化：命中缓存（改 _DEFAULTS 验证不到缓存，改文件内容验证——见下）

        # 程序内写入：立刻读新值（不依赖 stat 变化）
        settings_module.update_settings(summary_compress_threshold=222)
        assert settings_module.get("summary_compress_threshold") == 222

        # 外部手改（绕过 update_settings）：stat 变化 → 自动失效
        data = json.loads(tmp.read_text(encoding="utf-8"))
        data["summary_compress_threshold"] = 333
        data["新增键"] = "x"   # 顺带改 size，防同尺寸 mtime 撞车
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        assert settings_module.get("summary_compress_threshold") == 333

        # get_settings 合并视图不受缓存污染
        merged = settings_module.get_settings()
        assert merged["summary_compress_threshold"] == 333
        assert merged["bp3_live_rounds"] == 24   # 代码默认值回落正常
    finally:
        settings_module._SETTINGS_FILE = old_file
        settings_module._cache = old_cache


def test_upstreams_cache_hit_then_invalidate():
    tmp_dir = Path(tempfile.mkdtemp(prefix="cfg_cache_test_"))
    old_file = upstream_module.UPSTREAMS_FILE
    old_cache = upstream_module._upstream_cache
    upstream_module.UPSTREAMS_FILE = str(tmp_dir / "upstreams.json")
    try:
        upstream_module._upstream_cache = None

        # 首读 → 建缓存，对象隔离（两次 load 不是同一个对象）
        ups = upstream_module.load_upstreams()
        assert isinstance(ups, list)
        ups.append(Upstream(name="ghost", base_url="http://x", api_key="k"))
        ups2 = upstream_module.load_upstreams()
        assert all(u.name != "ghost" for u in ups2), "调用方改动不得渗入缓存"

        # 程序内写入：立刻读到新值（save_upstreams → load_upstreams 零延迟）
        ups_all = upstream_module.load_upstreams()
        if not ups_all:
            ups_all = [Upstream(name="a", base_url="http://a", api_key="k")]
        ups_all[0].is_active = not ups_all[0].is_active
        flipped = ups_all[0].is_active
        upstream_module.save_upstreams(ups_all)
        fresh = upstream_module.load_upstreams()
        assert fresh[0].is_active == flipped, "save_upstreams 后必须立刻读到新值"

        # 外部手改：stat 变化 → 自动失效
        data = json.loads(Path(upstream_module.UPSTREAMS_FILE).read_text(encoding="utf-8"))
        data[0]["hidden"] = True
        data[0]["think_tag"] = "<thinking>\n"   # 改 size，防同尺寸 mtime 撞车
        Path(upstream_module.UPSTREAMS_FILE).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8")
        assert upstream_module.load_upstreams()[0].hidden is True
    finally:
        upstream_module.UPSTREAMS_FILE = old_file
        upstream_module._upstream_cache = old_cache


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
