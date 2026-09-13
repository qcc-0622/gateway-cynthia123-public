"""_ThinkingRewriter 状态机专项测试（REFACTOR_ROADMAP P1.1，2026-09-02）。

覆盖 gateway/services/proxy.py 里 ~190 行的 thinking 改写状态机——此前零测试，
却是 a2a 转发热路径上最复杂的改写逻辑（RikkaHub 的『深度思考』折叠全靠它）。

铁律（ROADMAP P1.1）：输入按真实调用方的形状构造——真实调用方
forward_stream_anthropic_to_anthropic 把上游 SSE data 行 json.loads 后的
event dict 逐个喂给 process()，输出是可直接 yield 给客户端的 SSE 帧字符串。

既可以 `python tests/test_thinking_rewriter.py` 直接跑，也可以 pytest 跑。
"""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from gateway.services.proxy import _ThinkingRewriter  # noqa: E402


def _frames_to_events(frames: list[str]) -> list[dict]:
    """把 _sse() 产出的 'event: X\\ndata: {...}\\n' 帧还原成 event dict 列表。"""
    events = []
    for f in frames:
        for line in f.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def _start(idx: int = 0, btype: str = "text") -> dict:
    return {"type": "content_block_start", "index": idx,
            "content_block": {"type": btype, "text": ""}}


def _delta(text: str, idx: int = 0) -> dict:
    return {"type": "content_block_delta", "index": idx,
            "delta": {"type": "text_delta", "text": text}}


def _stop(idx: int = 0) -> dict:
    return {"type": "content_block_stop", "index": idx}


def _new_rewriter() -> _ThinkingRewriter:
    rw = _ThinkingRewriter()
    rw.process(_start(0))   # 首块缓冲为 pending（真实流的第一步）
    return rw


def test_plain_text_passthrough():
    """无 thinking 标签的普通回复：缓冲确认后原样透传（pending start + delta）。"""
    rw = _ThinkingRewriter()
    assert rw.process(_start(0)) == []          # 缓冲，等首个 delta 判定
    evs = _frames_to_events(rw.process(_delta("Hello")))
    assert evs[0]["type"] == "content_block_start"
    assert evs[0]["content_block"]["type"] == "text"   # 原样补发 pending
    assert evs[1]["delta"]["type"] == "text_delta"
    assert evs[1]["delta"]["text"] == "Hello"
    assert evs[1]["index"] == 0
    # 确认后的后续 delta 直接透传
    evs2 = _frames_to_events(rw.process(_delta(" world")))
    assert len(evs2) == 1
    assert evs2[0]["delta"]["text"] == " world"
    assert evs2[0]["index"] == 0


def test_open_tag_confirmed_switches_to_thinking():
    rw = _new_rewriter()
    evs = _frames_to_events(rw.process(_delta("<think>")))
    assert evs[0]["type"] == "content_block_start"
    assert evs[0]["content_block"]["type"] == "thinking"   # text 块改写成 thinking
    evs2 = _frames_to_events(rw.process(_delta("推理内容")))
    assert evs2[0]["delta"]["type"] == "thinking_delta"
    assert evs2[0]["delta"]["thinking"] == "推理内容"


def test_partial_open_tag_buffered():
    """半个 open tag（可能是前缀）必须缓冲，不提前透传。"""
    rw = _new_rewriter()
    assert rw.process(_delta("<th")) == []
    evs = _frames_to_events(rw.process(_delta("ink>开始")))
    assert evs[0]["content_block"]["type"] == "thinking"
    assert evs[1]["delta"]["thinking"] == "开始"


def test_close_tag_emits_stop_and_shifted_text_block():
    rw = _new_rewriter()
    rw.process(_delta("<think>想法"))
    evs = _frames_to_events(rw.process(_delta("</think>答案")))
    assert [e["type"] for e in evs] == [
        "content_block_stop", "content_block_start", "content_block_delta",
    ]
    assert evs[0]["index"] == 0                                  # thinking 块收口
    assert evs[1]["index"] == 1                                  # 新 text 块偏移 +1
    assert evs[1]["content_block"]["type"] == "text"
    assert evs[2]["index"] == 1 and evs[2]["delta"]["text"] == "答案"


def test_close_tag_split_across_chunks():
    """close tag 被 chunk 边界切断时不能漏判。"""
    rw = _new_rewriter()
    rw.process(_delta("<think>想法"))
    evs = _frames_to_events(rw.process(_delta("更多</thi")))      # 半个 close tag 挂起
    deltas = [e for e in evs if e["type"] == "content_block_delta"]
    assert all(e["delta"]["type"] == "thinking_delta" for e in deltas)
    evs2 = _frames_to_events(rw.process(_delta("nk>收尾")))
    assert evs2[0]["type"] == "content_block_stop"
    assert evs2[0]["index"] == 0
    assert evs2[-1]["delta"]["text"] == "收尾"
    assert evs2[-1]["index"] == 1


def test_force_think_mode_first_block_is_thinking():
    """force_think：上游吞掉了 open tag（预填充被消费），首块直接当 thinking。"""
    rw = _ThinkingRewriter(force_think_mode=True)
    evs = _frames_to_events(rw.process(_start(0)))
    assert evs[0]["content_block"]["type"] == "thinking"
    evs2 = _frames_to_events(rw.process(_delta("直接开始思考")))
    assert evs2[0]["delta"]["type"] == "thinking_delta"


def test_stream_ends_inside_thinking_block():
    """流在 thinking 中结束：只补 thinking 块的 stop，不能让块悬空。"""
    rw = _ThinkingRewriter(force_think_mode=True)
    rw.process(_start(0))
    rw.process(_delta("思考到一半"))
    evs = _frames_to_events(rw.process(_stop(0)))
    assert evs == [{"type": "content_block_stop", "index": 0}]


def test_later_blocks_index_shifted_after_split():
    """thinking/text 拆分插入过一块之后，后续块的 index 必须 +1。"""
    rw = _new_rewriter()
    rw.process(_delta("<think>x"))
    rw.process(_delta("</think>y"))
    evs = _frames_to_events(rw.process(_start(1)))               # 第二个内容块
    assert evs[0]["index"] == 2
    evs2 = _frames_to_events(rw.process(_delta("B", idx=1)))
    assert evs2[0]["index"] == 2


def test_non_content_events_pass_through_unchanged():
    rw = _ThinkingRewriter()
    ev = {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 5}}}
    assert _frames_to_events(rw.process(ev)) == [ev]


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
