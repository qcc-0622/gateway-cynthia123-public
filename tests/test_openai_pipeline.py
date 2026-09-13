"""OpenAI 格式前端（mikeko）接入完整预处理管道测试（2026-08-31）。

背景：/v1/chat/completions 旧路径是 preprocess_openai 裸转发（只加时间戳），
Seamless 接续、BP2 滚动摘要、记忆召回、缓存断点全都没有 → OpenAI 格式前端
的助手拿不到任何摘要注入（2026-08-30 05:18-05:20 mikeko 实测，journalctl
该时间窗无一条 Seamless/BP2 日志，cache_read 全 0）。

修复 = routers/openai_proxy.py 对 anthropic 上游先把 openai body 转成
anthropic 格式，再走与 /v1/messages 完全相同的 preprocess_anthropic；
发上游前不再二次转换（body_format="anthropic"），响应侧仍转回 openai 格式。

测试形状（对齐真实调用方）：
    body = openai_to_anthropic(openai_body)
    body = await preprocess_anthropic(body, upstream=upstream)

设计要点：
- 临时 sqlite + 临时 settings.json，绝不碰真实 data/ 目录（同 test_unified_brain.py）。
- memory_recall_enabled=False 关掉召回，测试不出网。
- 流式转发用 httpx.MockTransport 模拟上游，不起服务。
- 既可以 `python tests/test_openai_pipeline.py` 直接跑，也可以 pytest 跑。
"""

import asyncio
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

# ── 在 import gateway.* 之前，把 DB_PATH 指向临时文件 ──
_TMP_DIR = Path(tempfile.mkdtemp(prefix="openai_pipeline_test_"))
_TMP_DB = _TMP_DIR / "test_gateway.db"
_TMP_SETTINGS = _TMP_DIR / "test_settings.json"
os.environ["DB_PATH"] = str(_TMP_DB)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import httpx  # noqa: E402

from gateway import db as db_module  # noqa: E402
from gateway import settings as settings_module  # noqa: E402

# ⚠️ 双保险（同 test_hooks_pipeline.py）：不依赖 import 顺序，直接改写路径常量
db_module.DB_PATH = str(_TMP_DB)

# 猴子补丁 settings 文件路径 + 关闭记忆召回（测试不出网）
settings_module._SETTINGS_FILE = _TMP_SETTINGS
_TMP_SETTINGS.write_text(json.dumps({"memory_recall_enabled": False}), encoding="utf-8")

from gateway.converter import openai_to_anthropic  # noqa: E402
from gateway.hooks import preprocess_anthropic  # noqa: E402
from gateway.services import proxy as proxy_module  # noqa: E402
from gateway.services.proxy import forward_nonstream, forward_stream_anthropic_body_to_openai  # noqa: E402
from gateway.summarizer import derive_conv_fingerprint  # noqa: E402
from gateway.upstream import Upstream  # noqa: E402


def run_async(coro):
    return asyncio.run(coro)


async def _fresh_db():
    """清空并重新初始化临时库（每个用例开头调用）。"""
    await db_module.close_db()
    cur = Path(db_module.DB_PATH)   # 用当前生效路径（可能被其他测试模块改写）
    if cur.exists():
        cur.unlink()
    for suffix in ("-wal", "-shm"):
        p = Path(str(cur) + suffix)
        if p.exists():
            p.unlink()
    await db_module.init_db()


async def _close_db():
    await db_module.close_db()


def _make_upstream(cache_ttl: str = "1h") -> Upstream:
    return Upstream(
        name="fake-upstream",
        base_url="http://upstream.test",
        api_key="sk-test",
        api_format="anthropic",
        cache_ttl=cache_ttl,
    )


def _msg_text(m: dict) -> str:
    c = m.get("content", "")
    if isinstance(c, str):
        return c
    return " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")


# ────────────────────────────────────────────────
# converter：内容块归一化
# ────────────────────────────────────────────────

def test_openai_to_anthropic_normalization():
    body = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "sys-a"},
            {"role": "system", "content": "sys-b"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": [{"type": "text", "text": "parts"}]},
        ],
    }
    out = openai_to_anthropic(body)
    assert out["system"] == "sys-a\n\nsys-b"
    assert out["messages"][0]["content"] == [{"type": "text", "text": "hello"}]
    assert out["messages"][1]["content"] == ""           # None → ""（后续被兜底丢弃）
    assert out["messages"][2]["content"] == [{"type": "text", "text": "parts"}]
    assert out["max_tokens"] == 4096


def test_openai_to_anthropic_preserves_thinking():
    body = {
        "model": "m",
        "max_tokens": 123,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "reasoning_effort": "high",
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = openai_to_anthropic(body)
    assert out["max_tokens"] == 123
    assert out["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert out["reasoning_effort"] == "high"


# ────────────────────────────────────────────────
# Seamless：openai 新会话注入最新摘要 + 归档尾巴
# ────────────────────────────────────────────────

async def _seed_archive():
    for i in range(3):
        await db_module.save_conversation(
            conversation_id=f"seed-{i}", role="user", content=f"历史用户消息{i}",
            model="m", timestamp=f"2026-08-30T0{i}:00:00+08:00", tag="",
            fingerprint=f"fp_seed{i}", history_hash=f"h{i}",
        )
        await db_module.save_conversation(
            conversation_id=f"seed-{i}", role="assistant", content=f"历史助手回复{i}",
            model="m", timestamp=f"2026-08-30T0{i}:00:30+08:00", tag="",
            fingerprint=f"fp_seed{i}", history_hash=f"h{i}",
        )


def test_seamless_injection_on_openai_path():
    run_async(_seamless_case())


async def _seamless_case():
    await _fresh_db()
    await db_module.save_summary_at("fp_seed0", messages_covered=100, summary="用户喜欢猫，养了只橘猫叫旺财。")
    await _seed_archive()
    try:
        openai_body = {
            "model": "claude-opus-4-6",
            "stream": True,
            "messages": [
                {"role": "system", "content": "你是用户的贴身助手"},
                {"role": "user", "content": "早上好呀"},
            ],
        }
        body = openai_to_anthropic(openai_body)
        out = await preprocess_anthropic(body, upstream=_make_upstream("1h"))

        msgs = out["messages"]
        # 1) Seamless 摘要头（user 摘要 + assistant ack）
        assert msgs[0]["role"] == "user"
        assert _msg_text(msgs[0]).startswith("[以下是我们之前对话的摘要，新窗口继续]")
        assert "旺财" in _msg_text(msgs[0])
        assert msgs[1]["role"] == "assistant"
        assert "好的" in _msg_text(msgs[1])
        # 2) 归档尾巴
        joined = "\n".join(_msg_text(m) for m in msgs)
        assert "历史用户消息0" in joined and "历史助手回复2" in joined
        # 3) 客户端原消息保留在末尾
        assert "早上好呀" in _msg_text(msgs[-1])
        # 4) system 变 block 并追加网关协议
        assert isinstance(out["system"], list) and len(out["system"]) == 2
        assert "Chat Gateway Metadata 协议" in out["system"][-1]["text"]
        # 5) 时间戳注入在最后一条 user
        assert "<time_reminder>" in _msg_text(msgs[-1])
        # 6) 缓存断点：system BP1（跟随上游 1h）+ BP4（固定 5m）
        def _cc(obj):
            n = 0
            if isinstance(obj, dict):
                n += 1 if "cache_control" in obj else 0
                n += sum(_cc(v) for v in obj.values())
            elif isinstance(obj, list):
                n += sum(_cc(x) for x in obj)
            return n
        assert _cc(out["system"]) == 1
        assert out["system"][-1]["cache_control"]["ttl"] == "1h"
        assert _cc(msgs) >= 1
        assert any(
            b.get("cache_control", {}).get("ttl") == "5m"
            for m in msgs if isinstance(m.get("content"), list)
            for b in m["content"] if isinstance(b, dict)
        )
    finally:
        await _close_db()


# ────────────────────────────────────────────────
# BP2：openai 长会话滚动摘要（完美命中路径）
# ────────────────────────────────────────────────

def test_bp2_perfect_match_on_openai_path():
    run_async(_bp2_case())


async def _bp2_case():
    await _fresh_db()
    try:
        # 34 个完整 pair + 当前 user → n_pairs=34，target_pos=(34-8)*2=52
        openai_msgs = [{"role": "system", "content": "你是助手"}]
        for i in range(34):
            openai_msgs.append({"role": "user", "content": f"问题{i}，请回答得详细一点"})
            openai_msgs.append({"role": "assistant", "content": f"回答{i}，这是一段足够长的回复内容"})
        openai_msgs.append({"role": "user", "content": "问题34，最新提问"})
        openai_body = {"model": "claude-opus-4-6", "stream": False, "messages": openai_msgs}

        body = openai_to_anthropic(openai_body)
        fp = derive_conv_fingerprint(body["messages"])
        await db_module.save_summary_at(fp, messages_covered=52, summary="这是滚动摘要：用户连问了 26 轮。")

        out = await preprocess_anthropic(body, upstream=_make_upstream("off"))
        msgs = out["messages"]
        # BP2 注入头（与 Seamless 的"新窗口继续"文案不同）
        assert _msg_text(msgs[0]).startswith("[以下是我们之前对话的摘要，请基于此继续]")
        assert "连问了 26 轮" in _msg_text(msgs[0])
        # 2 + (69-52) = 19 条：摘要 pair + frozen 8 轮 + live + 当前 user
        assert len(msgs) == 19
        # 切点 pos 52 → messages[52:] 开头是 pair26 的 user（问题26）
        assert "问题26" in _msg_text(msgs[2])
        assert "问题34" in _msg_text(msgs[-1])     # 当前提问保留
        # off TTL → 不注入任何 cache_control
        assert "cache_control" not in json.dumps(out, ensure_ascii=False)
    finally:
        await _close_db()


# ────────────────────────────────────────────────
# tag 频道：tech: 前缀跳过一切注入
# ────────────────────────────────────────────────

def test_tagged_openai_conversation_skips_injection():
    run_async(_tag_case())


async def _tag_case():
    await _fresh_db()
    await db_module.save_summary_at("fp_seed0", messages_covered=100, summary="不应出现的摘要")
    try:
        openai_body = {
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "system", "content": "你是助手"},
                {"role": "user", "content": "tech: 帮我写个正则"},
            ],
        }
        body = openai_to_anthropic(openai_body)
        out = await preprocess_anthropic(body, upstream=_make_upstream("1h"))
        joined = json.dumps(out["messages"], ensure_ascii=False)
        assert "不应出现的摘要" not in joined
        assert "[以下是我们之前对话的摘要" not in joined
        assert "帮我写个正则" in _msg_text(out["messages"][-1])
    finally:
        await _close_db()


# ────────────────────────────────────────────────
# 转发层：预转换 body → anthropic 上游 → openai SSE
# ────────────────────────────────────────────────

_UPSTREAM_SSE = (
    'event: message_start\n'
    'data: {"type":"message_start","message":{"id":"msg_x","type":"message","role":"assistant",'
    '"model":"claude-opus-4-6","usage":{"input_tokens":100,"output_tokens":0}}}\n\n'
    'event: content_block_start\n'
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"你好"}}\n\n'
    'event: content_block_stop\n'
    'data: {"type":"content_block_stop","index":0}\n\n'
    'event: message_delta\n'
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}\n\n'
    'event: message_stop\n'
    'data: {"type":"message_stop"}\n\n'
).encode("utf-8")


@contextmanager
def _mock_upstream(handler):
    real_client = proxy_module.httpx.AsyncClient

    class _PatchedClient(real_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    # 共享 client（gateway/http_client.py）已缓存实例时不会再用补丁类重建，
    # 先 reset 让下一个请求拿到 MockTransport 版本；测试间也要 reset 防串味。
    from gateway import http_client as _hc
    _hc.reset_client()
    proxy_module.httpx.AsyncClient = _PatchedClient
    try:
        yield
    finally:
        proxy_module.httpx.AsyncClient = real_client
        _hc.reset_client()


def _collect(agen):
    async def run():
        out = []
        async for chunk in agen:
            out.append(chunk)
        return out
    return asyncio.run(run())


def test_forward_stream_preconverted_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["beta"] = request.headers.get("anthropic-beta", "")
        return httpx.Response(200, content=_UPSTREAM_SSE, headers={"content-type": "text/event-stream"})

    body = {
        "model": "claude-opus-4-6",
        "stream": True,
        "max_tokens": 1000,
        "system": [{"type": "text", "text": "你是助手"}, {"type": "text", "text": "协议"}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "历史"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "回复", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]},
            {"role": "user", "content": [{"type": "text", "text": "新问题"}]},
        ],
        "_gateway_cache_ttl": "1h",
        "_gateway_fingerprint": "fp_test",
    }
    with _mock_upstream(handler):
        chunks = _collect(forward_stream_anthropic_body_to_openai(body, _make_upstream("1h")))

    # 上游收到的 body：预转换结果原样送达（缓存断点保留、私有字段剥掉）
    fwd = seen["body"]
    assert fwd["stream"] is True
    assert "_gateway_fingerprint" not in fwd and "_gateway_cache_ttl" not in fwd
    assert any("cache_control" in json.dumps(m) for m in fwd["messages"])
    assert seen["url"] == "http://upstream.test/v1/messages"
    assert seen["beta"] == "extended-cache-ttl-2025-04-11"

    # 客户端收到 openai chunk 流
    stream = "".join(chunks)
    archive_line = [c for c in chunks if c.startswith("\n__ARCHIVE__:")][-1]
    stream_visible = stream.replace(archive_line, "")
    data_chunks = []
    for line in stream_visible.splitlines():
        if line.startswith("data:"):
            raw = line[5:].strip()
            if raw and raw != "[DONE]":
                data_chunks.append(json.loads(raw))
    assert any(
        ch.get("choices", [{}])[0].get("delta", {}).get("content") == "你好"
        for ch in data_chunks
    )
    assert "data: [DONE]" in stream_visible
    assert any(ch.get("choices", [{}])[0].get("finish_reason") == "stop" for ch in data_chunks)
    info = json.loads(archive_line.split(":", 1)[1])
    assert info["tokens_in"] == 100 and info["tokens_out"] == 10
    assert info["api_format"] == "openai"
    assert info["assistant_content"] == "你好"
    assert "新问题" in info["user_content"]


def test_forward_nonstream_preconverted_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "id": "msg_y", "type": "message", "role": "assistant", "model": "claude-opus-4-6",
            "content": [{"type": "text", "text": "非流式回复"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 5},
        })

    body = {
        "model": "claude-opus-4-6",
        "stream": False,
        "max_tokens": 1000,
        "system": [{"type": "text", "text": "你是助手"}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "新问题"}]},
        ],
        "_gateway_cache_ttl": "1h",
    }
    with _mock_upstream(handler):
        response_data, archive_info = asyncio.run(
            forward_nonstream(body, _make_upstream("1h"), "openai", body_format="anthropic")
        )

    assert response_data["choices"][0]["message"]["content"] == "非流式回复"
    assert response_data["usage"]["prompt_tokens"] == 100
    assert archive_info["tokens_in"] == 100 and archive_info["tokens_out"] == 5
    assert archive_info["api_format"] == "openai"
    assert archive_info["upstream_name"] == "fake-upstream"
    assert "新问题" in archive_info["user_content"]


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
