"""客户端断连漏归档修复的专项测试（2026-09-01）。

背景：流式响应在客户端中途断开时生成器被 cancel，流末尾的 __ARCHIVE__ 哨兵
永远到不了 → 该轮不归档（无记录、无成本统计）。修复 = forward_stream_* 边收
边把内容/token 写进 StreamArchiveState，routers 的
archiving_sse_stream 在 finally 里对"哨兵没到 + 有实际内容"的情况用后台任务
补归档（raw_response="[streamed-disconnect]"）。

测试形状（对齐真实调用方）：
    gen = forward_stream_xxx(body, upstream, archive_state=st)
    outer = archiving_sse_stream(gen, body=body, archive_state=st, archive_kwargs=...)
这里用手工 async generator 模拟 forward_stream_*（真实转发层的行为已由
test_openai_pipeline.py 的哨兵断言覆盖——state 字段进哨兵 JSON 的链路是同一条）。

既可以 `python tests/test_disconnect_archive.py` 直接跑，也可以 pytest 跑。
"""

import asyncio
import json
import sys
from contextlib import contextmanager
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from gateway.services.proxy import (  # noqa: E402
    archiving_sse_stream,
    StreamArchiveState,
)

# ⚠️ 不要在模块级 import gateway.services.archiver（它会连带 import gateway.db，
# 固化 DB_PATH，破坏 test_openai_pipeline/test_unified_brain 的临时库隔离——
# pytest 按文件名字母序收集，本文件排最前）。归档函数的 patch 走 _fake_archive()。


@contextmanager
def _fake_archive():
    """临时替换 archiver.archive，收集调用参数。函数内导入（原因见上）。"""
    from gateway.services import archiver as archiver_module
    calls = []
    real = archiver_module.archive

    async def fake_archive(**kw):
        calls.append(kw)

    archiver_module.archive = fake_archive
    try:
        yield calls
    finally:
        archiver_module.archive = real


def _make_state(**kw) -> StreamArchiveState:
    """模拟 forward_stream_* 收了一半上游流之后的 state。"""
    st = StreamArchiveState.for_body(
        {"model": "claude-x", "metadata": {"conversation_id": "conv-1"},
         "messages": [{"role": "user", "content": [{"type": "text", "text": "问题"}]}]},
        type("U", (), {"name": "fake-upstream"})(),
        "anthropic",
    )
    st.assistant_content = kw.get("assistant_content", "")
    st.tokens_in = kw.get("tokens_in", 0)
    st.tokens_out = kw.get("tokens_out", 0)
    st.failed = kw.get("failed", False)
    if kw.get("assistant_content") or kw.get("tokens_out") or kw.get("tokens_in"):
        # 有内容才算"上游真的吐过"，finalize 才有意义；空 state 保持 duration_ms=0
        st.finalize()
    return st


async def _fake_upstream_gen(st: StreamArchiveState, chunks: list[str]):
    """模拟 forward_stream_*：yield 客户端可见 chunk，收流时更新 state。"""
    for c in chunks:
        if c.startswith("TEXT:"):
            st.assistant_content += c[5:]
        yield c


async def _consume_then_disconnect(outer, stop_after: str):
    """模拟客户端断开：消费到 stop_after 就停止 + aclose（StreamingResponse
    在客户端断开时对生成器做的就是这件事）。"""
    got = []
    async for chunk in outer:
        got.append(chunk)
        if chunk == stop_after:
            break
    await outer.aclose()
    return got


async def _drain_background_tasks():
    """让 finally 里 create_task 派生的补归档任务跑完。"""
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def test_disconnect_mid_stream_archives_partial():
    """核心场景：断连后用已累积内容补归档。"""
    with _fake_archive() as calls:
        async def case():
            st = _make_state()
            outer = archiving_sse_stream(
                _fake_upstream_gen(st, ["TEXT:你", "TEXT:好", "TEXT:呀"]),
                body={"messages": []}, archive_state=st, archive_kwargs={"tag": "tech"},
            )
            got = await _consume_then_disconnect(outer, "TEXT:好")
            assert got == ["TEXT:你", "TEXT:好"]
            assert not st.archive_dispatched
            await _drain_background_tasks()

            assert len(calls) == 1, f"期望补归档 1 次，实际 {len(calls)}"
            info = calls[0]
            assert info["raw_response"] == "[streamed-disconnect]"
            assert info["assistant_content"] == "你好"      # 断连前已收到的部分
            assert info["conversation_id"].startswith("conv-") or info["conversation_id"]
            assert info["upstream_name"] == "fake-upstream"
            assert info["api_format"] == "anthropic"
            assert info["tag"] == "tech"
            assert info["duration_ms"] >= 0

        asyncio.run(case())


def test_normal_sentinel_archives_once_no_double():
    """正常结束：哨兵路径归档一次，结束后不再触发补归档（不双写）。"""
    with _fake_archive() as calls:
        async def case():
            st = _make_state()

            async def gen():
                # 对齐真实 forward_stream_*：先累积内容，再在结尾用最终 state 构建哨兵
                st.assistant_content += "完整回复"
                yield "TEXT:完整回复"
                st.finalize()
                yield "\n__ARCHIVE__:" + json.dumps(st.to_archive_info(), ensure_ascii=False)

            outer = archiving_sse_stream(
                gen(), body={"messages": []}, archive_state=st, archive_kwargs={},
            )
            got = [c async for c in outer]          # 全部消费（正常客户端）
            assert got == ["TEXT:完整回复"]          # 哨兵被拦截不透传
            assert st.archive_dispatched
            await _drain_background_tasks()
            assert len(calls) == 1
            assert calls[0]["raw_response"] == "[streamed]"
            assert calls[0]["assistant_content"] == "完整回复"

        asyncio.run(case())


def test_failed_stream_skips_disconnect_archive():
    """上游报错（failed=True）不补归档——与哨兵路径 stream_failed 早退一致。"""
    with _fake_archive() as calls:
        async def case():
            st = _make_state(assistant_content="部分内容", tokens_out=3, failed=True)
            outer = archiving_sse_stream(
                _fake_upstream_gen(st, ["TEXT:部分内容"]),
                body={"messages": []}, archive_state=st, archive_kwargs={},
            )
            await _consume_then_disconnect(outer, "TEXT:部分内容")
            await _drain_background_tasks()
            assert calls == []

        asyncio.run(case())


def test_empty_state_skips_disconnect_archive():
    """一个字没收到的断连（纯废请求）不归档，不污染账本。"""
    with _fake_archive() as calls:
        async def case():
            st = _make_state()   # 无内容无 token
            outer = archiving_sse_stream(
                _fake_upstream_gen(st, []),
                body={"messages": []}, archive_state=st, archive_kwargs={},
            )
            await _consume_then_disconnect(outer, "NEVER")
            await _drain_background_tasks()
            assert calls == []

        asyncio.run(case())


def test_worth_archiving_gate():
    """worth_archiving 三态：有内容/有 token → True；failed 或全空 → False。"""
    st = _make_state(assistant_content="x")
    assert st.worth_archiving()
    st = _make_state(tokens_in=10)
    assert st.worth_archiving()
    st = _make_state(assistant_content="x", failed=True)
    assert not st.worth_archiving()
    st = _make_state()
    assert not st.worth_archiving()


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
