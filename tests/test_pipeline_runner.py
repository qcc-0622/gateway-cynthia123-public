"""runner 单测（P2.2 第一步）：顺序执行 / async 步骤 / condition 跳过 /
trace 记录 / after 校验。纯新代码，不碰任何旧路径。

既可以直接 `python tests/test_pipeline_runner.py` 跑，也可以 pytest 跑。
"""

import asyncio
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from gateway.pipeline.runner import (  # noqa: E402
    PipelineContext,
    Step,
    run_pipeline,
    validate_pipeline,
)


def _ctx(**kw) -> PipelineContext:
    return PipelineContext(body={"messages": []}, upstream=None, **kw)


def test_steps_run_in_list_order_and_record_trace():
    calls = []

    def make(name):
        def fn(ctx):
            calls.append(name)
        return Step(name=name, fn=fn, after=[])

    ctx = _ctx()
    out = asyncio.run(run_pipeline([make("a"), make("b"), make("c")], ctx))
    assert calls == ["a", "b", "c"]
    assert out.trace == ["a", "b", "c"]
    assert out is ctx


def test_async_step_is_awaited():
    calls = []

    async def async_step(ctx):
        calls.append("async-ran")

    def sync_step(ctx):
        calls.append("sync-ran")

    ctx = _ctx()
    asyncio.run(run_pipeline(
        [Step("a", async_step), Step("b", sync_step)], ctx))
    assert calls == ["async-ran", "sync-ran"]
    assert ctx.trace == ["a", "b"]


def test_condition_skips_step():
    calls = []

    def fn(ctx):
        calls.append(ctx.tag)

    steps = [
        Step("only-daily", fn, after=[], condition=lambda c: c.tag == ""),
        Step("always", fn, after=[]),
    ]
    ctx = _ctx(tag="tech")
    asyncio.run(run_pipeline(steps, ctx))
    assert calls == ["tech"], "条件不满足的步骤应被跳过"
    assert ctx.trace == ["only-daily:SKIP", "always"]

    ctx2 = _ctx(tag="")
    calls.clear()
    asyncio.run(run_pipeline(steps, ctx2))
    assert calls == ["", ""] and ctx2.trace == ["only-daily", "always"]


def test_step_can_mutate_context_and_messages():
    def mutate(ctx):
        ctx.messages = ctx.messages + [{"role": "user", "content": "x"}]
        ctx.body["touched"] = True

    ctx = _ctx()
    asyncio.run(run_pipeline([Step("m", mutate)], ctx))
    assert ctx.body["touched"] is True
    assert len(ctx.messages) == 1


def test_validate_pipeline_accepts_valid_order():
    steps = [
        Step("a", lambda c: None, after=[]),
        Step("b", lambda c: None, after=["a"]),
        Step("c", lambda c: None, after=["a", "b"]),
    ]
    validate_pipeline(steps)  # 不抛即通过


def test_validate_pipeline_rejects_forward_reference():
    steps = [
        Step("a", lambda c: None, after=["b"]),   # b 还没出现
        Step("b", lambda c: None, after=[]),
    ]
    try:
        validate_pipeline(steps)
        raise SystemExit("应当抛 ValueError")
    except ValueError as e:
        assert "after='b'" in str(e) and "'a'" in str(e)


def test_validate_pipeline_rejects_unknown_dep():
    steps = [Step("a", lambda c: None, after=["ghost"])]
    try:
        validate_pipeline(steps)
        raise SystemExit("应当抛 ValueError")
    except ValueError:
        pass


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
