"""监控页"每日明细按天×模型费用"聚合测试（2026-09-06 用户需求）。

需求：缓存监控区已有每天总费用，要求细化到每个模型用了多少钱、均价多少。
实现：_aggregate_day_model_rows（纯函数）把 day×model 统计行聚合成
每日明细 + 当天 models 列表（按费用降序，含 cost/avg_cost）。

既可以直接 `python tests/test_monitoring_breakdown.py` 跑，也可以 pytest 跑。
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from gateway.routers.admin import _aggregate_day_model_rows  # noqa: E402


def _row(day, upstream, model, inp, out, cw, cr, msgs):
    """形状对齐 db.get_cache_stats_by_day_and_model 的返回行。"""
    return {
        "day": day, "upstream": upstream, "model": model,
        "input_tokens": inp, "output_tokens": out,
        "cache_write": cw, "cache_read": cr, "messages": msgs,
        "stored_cost": 0, "stored_saved": 0,   # 路由不使用 stored，按价目重算
    }


def _fake_cost(upstream, model, inp, out, cw, cr, msgs):
    """确定性假单价：$0.01/次 + (输入+输出)/1M，便于手算核对。"""
    cost = 0.01 * msgs + (inp + out) / 1_000_000
    return cost, 0.0


def test_per_day_model_breakdown():
    rows = [
        _row("2026-09-05", "55", "claude-opus", 1_000_000, 500_000, 0, 0, 10),
        _row("2026-09-05", "55", "haiku", 100_000, 50_000, 0, 0, 4),
        _row("2026-09-04", "55", "claude-opus", 2_000_000, 0, 0, 0, 5),
    ]
    daily, totals = _aggregate_day_model_rows(rows, _fake_cost)

    # 两天的日期倒序
    assert [d["day"] for d in daily] == ["2026-09-05", "2026-09-04"]

    d5 = daily[0]
    # 每日合计：cost = 0.01*14 + 1.65/1e6*... 手算：opus 0.01*10+1.5=1.6；haiku 0.01*4+0.15=0.19 → 1.79
    assert d5["messages"] == 14
    assert abs(d5["cost"] - 1.79) < 1e-6
    # 均价/条
    assert abs(d5["avg_cost"] - round(1.79 / 14, 4)) < 1e-6

    # 当天模型明细：按费用降序（opus 1.6 > haiku 0.19），各带均价
    assert [m["model"] for m in d5["models"]] == ["claude-opus", "haiku"]
    opus = d5["models"][0]
    assert abs(opus["cost"] - 1.6) < 1e-6
    assert opus["avg_cost"] == round(1.6 / 10, 4)
    assert opus["messages"] == 10 and opus["upstream"] == "55"
    haiku = d5["models"][1]
    assert haiku["avg_cost"] == round(0.19 / 4, 4)

    # 总计跨天跨模型累加
    assert totals["messages"] == 19
    assert abs(totals["cost"] - (1.79 + 0.01 * 5 + 2.0)) < 1e-6


def test_breakdown_empty_rows():
    daily, totals = _aggregate_day_model_rows([], _fake_cost)
    assert daily == []
    assert totals["messages"] == 0 and totals["cost"] == 0.0


def test_breakdown_single_model_per_day():
    rows = [_row("2026-09-06", "relay-b", "gemini-x", 10, 20, 30, 40, 2)]
    daily, _ = _aggregate_day_model_rows(rows, _fake_cost)
    assert len(daily) == 1 and len(daily[0]["models"]) == 1
    m = daily[0]["models"][0]
    assert m["model"] == "gemini-x" and m["messages"] == 2
    # 命中率/复用率照常计算
    assert daily[0]["hit_rate"] == 50.0    # 40/(10+30+40)
    assert daily[0]["reuse_rate"] == 80.0  # 40/(10+40)


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
