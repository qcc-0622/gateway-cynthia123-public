"""记忆新话题检测的省外呼短路测试（2026-09-06）。

is_new_topic 每请求固定 2 次 embedding 外呼是热路径最大外部延迟。
优化：完全相同消息对 → 结果缓存；字符 bigram 余弦极高（近似重复）→
跳过 embedding 直接判非新话题。两级都只朝"非新话题"方向短路。

既可以直接 `python tests/test_memory_shortcut.py` 跑，也可以 pytest 跑。
"""

import asyncio
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import gateway.memory as mem  # noqa: E402


def _msgs(*user_texts):
    """user 消息交替 assistant，末尾是最后一条 user（真实调用形状）。"""
    msgs = []
    for i, t in enumerate(user_texts):
        msgs.append({"role": "user", "content": t})
        msgs.append({"role": "assistant", "content": f"回复{i}"})
    return msgs


def _install_fake_embed(calls):
    async def fake_embed(text, http):
        calls.append(text)
        # 正交向量便于控制余弦：按文本长度给不同向量
        return [1.0, 0.0] if "量子" in text else [0.0, 1.0]
    real = mem._embed
    mem._embed = fake_embed
    mem._topic_check_cache.clear()
    return real


def test_near_identical_texts_shortcut_without_embedding():
    """近似重复消息（bigram 余弦 ≥ 0.9）→ 不打 embedding 直接判非新话题。"""
    calls = []
    real = _install_fake_embed(calls)
    try:
        # 末两条 user 只差一个标点 → bigram 余弦接近 1
        msgs = _msgs("今天晚上想吃火锅吗", "好呀", "今天晚上想吃火锅吗？")
        is_new, query = asyncio.run(mem.is_new_topic(msgs, http=None))
        assert is_new is False
        assert query == "今天晚上想吃火锅吗？"
        assert calls == [], "近似重复消息不应打 embedding"
    finally:
        mem._embed = real


def test_identical_pair_uses_result_cache():
    """完全相同的 (latest, prev) 对第二次判定走缓存，不再打 embedding。"""
    calls = []
    real = _install_fake_embed(calls)
    try:
        msgs = _msgs("我们聊聊量子物理吧", "好啊", "量子纠缠是什么原理")
        r1 = asyncio.run(mem.is_new_topic(msgs, http=None))
        assert calls == ["量子纠缠是什么原理", "我们聊聊量子物理吧"], "首次判定必须真实 embedding"
        r2 = asyncio.run(mem.is_new_topic(msgs, http=None))
        assert r1 == r2
        assert len(calls) == 2, "相同消息对第二次应命中缓存"
    finally:
        mem._embed = real


def test_dissimilar_texts_still_embed():
    """不相似的文本不短路，照常走 embedding 判定（正交向量 → 新话题）。"""
    calls = []
    real = _install_fake_embed(calls)
    try:
        msgs = _msgs("晚饭吃什么火锅", "随便", "我们讨论量子物理")
        is_new, query = asyncio.run(mem.is_new_topic(msgs, http=None))
        assert is_new is True          # 余弦 0 < 阈值
        assert query == "我们讨论量子物理"
        assert len(calls) == 2
    finally:
        mem._embed = real


def test_bigram_cosine_sanity():
    assert abs(mem._bigram_cosine("今天晚上吃火锅", "今天晚上吃火锅") - 1.0) < 1e-9
    assert mem._bigram_cosine("", "随便什么都行") == 0.0       # 空串不做短路
    assert mem._bigram_cosine("a", "a") == 0.0                 # 单字符无 bigram
    assert mem._bigram_cosine("量子物理很有趣", "晚饭吃火锅") < 0.2


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
