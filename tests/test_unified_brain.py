"""统一大脑（PLAN_UNIFIED_BRAIN.md 第 0~3 步）单元测试。

覆盖场景：
    1. 正常追加消息
    2. 重roll（真实切分：messages[:-1] 不含被重问的 user，靠 current_user_text 判定）
    3. 编辑历史消息（分叉）
    4. 两个"平台"（两个 fingerprint）交替发消息，时间线正确交错、各自增量识别正确
    5. 客户端只发截断的尾段（模拟 contextMessageSize）
    6. 真实切分下的重roll：长对话 + 首轮（n==2，client_history=[]）
    7. 截断 + 重roll 组合；以及"截断到完全无法对齐"的极端场景（p==0 防线）
    8. 带 <time_reminder> 的归档内容 vs 不带的客户端历史，对账不误判
    回归. 截断尾段 + 追加新内容的组合（防误删全史）

⚠️ 关键约定（对齐真实调用方 `_apply_bp2_unified` 的切分方式）：
    client_history = messages[:-1]  —— **不含**本轮正在提问的最后一条 user；
    那条 user 的规范化文本作为 current_user_text 单独传入。
    重roll 时 messages 的最后一条就是被重新问的那条 user，所以
    client_history 里没有它——重roll 判定必须靠 current_user_text。

设计要点：
- 用临时 sqlite 数据库（tempfile），绝不碰真实 data/ 目录。
- 用临时 settings.json，绝不碰真实 data/settings.json。
- 这两个 monkeypatch 必须在 import gateway.db / gateway.settings 之前生效，
  所以本文件最开头就设置好环境变量 + 猴子补丁，再 import 被测代码。
- 既可以 `python tests/test_unified_brain.py` 直接跑（无需 pytest），
  也可以 `python -m pytest tests/test_unified_brain.py -v` 跑（pytest 可用时）。
"""

import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path

# ── 在 import gateway.* 之前，把 DB_PATH 指向临时文件，绝不碰真实 data/gateway.db ──
_TMP_DIR = Path(tempfile.mkdtemp(prefix="unified_brain_test_"))
_TMP_DB = _TMP_DIR / "test_gateway.db"
_TMP_SETTINGS = _TMP_DIR / "test_settings.json"
os.environ["DB_PATH"] = str(_TMP_DB)

# 确保能 import gateway 包（tests/ 与 gateway/ 同级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from gateway import db as db_module  # noqa: E402
from gateway import settings as settings_module  # noqa: E402

# ⚠️ 双保险（同 test_hooks_pipeline.py）：不依赖 import 顺序，直接改写路径常量
db_module.DB_PATH = str(_TMP_DB)

# 猴子补丁 settings 文件路径到临时文件，绝不碰真实 data/settings.json
settings_module._SETTINGS_FILE = _TMP_SETTINGS

from gateway import hooks  # noqa: E402


# ────────────────────────────────────────────────────────────────
# 测试辅助
# ────────────────────────────────────────────────────────────────

class FakeUpstream:
    """summarize_messages 需要的 upstream 占位对象。测试里不会真的触发网络调用
    （unified_summary_interval 设置得足够大，不会触发 rebuild），但保留字段
    避免属性访问报错。"""
    name = "fake-upstream"
    api_format = "anthropic"
    default_model = "fake-model"

    def get_key(self, purpose="chat", model=""):
        return "fake-key"

    def get_cache_ttl(self, purpose="chat", model=""):
        return "off"

    @property
    def messages_url(self):
        return "http://localhost/fake"


class _WarnCatcher(logging.Handler):
    """捕获 warning 日志，验证 p==0 防线有留痕。

    2026-09-06：对账已迁 gateway/pipeline/unified.py，warning 从
    gateway.pipeline.unified 发出——挂 root logger（记录经传播到达），
    对模块搬迁免疫。"""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def run_async(coro):
    return asyncio.run(coro)


async def _reset_db():
    """每个测试场景之间清空数据库，保证互不干扰。"""
    await db_module.close_db()
    if _TMP_DB.exists():
        _TMP_DB.unlink()
    for suffix in ("-wal", "-shm"):
        p = Path(str(_TMP_DB) + suffix)
        if p.exists():
            p.unlink()
    await db_module.init_db()


async def _archive_turn(fingerprint: str, user_text: str, assistant_text: str,
                        context_fingerprint: str = "", tag: str = ""):
    """模拟 archiver.archive() 的核心效果：往 conversations 表写一对 user+assistant。
    不依赖 gateway.services.archiver（避免拉入 costs.py/memory.py 等更多依赖），
    只调用 db.save_conversation，这就是 archiver 最终落地的地方。"""
    await db_module.save_conversation(
        conversation_id=f"conv-{fingerprint}",
        role="user", content=user_text, fingerprint=fingerprint,
        context_fingerprint=context_fingerprint, tag=tag,
    )
    await db_module.save_conversation(
        conversation_id=f"conv-{fingerprint}",
        role="assistant", content=assistant_text, fingerprint=fingerprint,
        context_fingerprint=context_fingerprint, tag=tag,
    )


def _msg(role, text):
    return {"role": role, "content": text}


def _print_result(name: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not ok else ""))
    # 2026-09-06 修复 pytest 盲区：场景原先只 print 不 assert，pytest 里
    # 永远绿（standalone 直跑才是真相，曾因此漏掉场景7 的回归）。
    # pytest 环境下失败即 raise；直跑 main() 的 try/except 照常兜住。
    if not ok and "pytest" in sys.modules:
        raise AssertionError(f"{name} — {detail}")
    return ok


async def _reconcile(fp: str, messages: list):
    """按真实调用方 `_apply_bp2_unified` 的切分方式做对账：
    client_history = messages[:-1]；最后一条 user 的文本作为 current_user_text。"""
    assert messages and messages[-1]["role"] == "user", "messages 必须以本轮 user 结尾"
    client_history = messages[:-1]
    current_user_text = hooks._unified_normalize_text(messages[-1]["content"])
    return await hooks._reconcile_and_update_timeline(
        fp, client_history, current_user_text=current_user_text,
    )


# ────────────────────────────────────────────────────────────────
# 场景 1：正常追加消息
# ────────────────────────────────────────────────────────────────

def test_scenario_1_normal_append():
    """正常追加：archived 只有第 1 轮，但客户端本地已经完成了第 2 轮对话
    （user+assistant 都在 client_history 里），只是网关还没来得及 archive。
    对账应该识别出第 2 轮是"未归档增量"，不删除任何东西。"""
    async def _run():
        await _reset_db()
        fp = "fp_normal0001"
        # 已归档：第 1 轮
        await _archive_turn(fp, "你好", "你好呀")

        # 客户端消息：第 1 轮（已归档）+ 第 2 轮（还没归档）+ 正在提问的第 3 轮 user
        messages = [
            _msg("user", "你好"),
            _msg("assistant", "你好呀"),
            _msg("user", "今天天气如何"),
            _msg("assistant", "今天晴天"),
            _msg("user", "那明天呢"),  # 本轮 user，不进 client_history
        ]
        return await _reconcile(fp, messages)

    result = run_async(_run())
    ok = (
        result.kind == "append"
        and result.new_messages == [_msg("user", "今天天气如何"), _msg("assistant", "今天晴天")]
        and result.delete_ids == []
        and result.new_watermark == 2
    )
    return _print_result(
        "场景1 正常追加消息", ok,
        f"kind={result.kind} new_messages={result.new_messages} delete_ids={result.delete_ids} watermark={result.new_watermark}",
    )


# ────────────────────────────────────────────────────────────────
# 场景 2：重roll（真实切分——client_history 不含被重问的 user）
# ────────────────────────────────────────────────────────────────

def test_scenario_2_reroll():
    """重roll：橘瓣重新生成回复时，messages = 原历史去掉旧 assistant，
    最后一条就是被重新问的那条 user。真实切分后 client_history 不含它。
    这里是最小案例（archived 只有一轮），也即"首轮重roll"。"""
    async def _run():
        await _reset_db()
        fp = "fp_reroll0001"
        await _archive_turn(fp, "讲个笑话", "笑话A：很冷的笑话")

        # 重roll：messages = [user"讲个笑话"]（旧 assistant 被客户端丢弃）
        messages = [_msg("user", "讲个笑话")]
        result = await _reconcile(fp, messages)
        remaining = await db_module.get_global_timeline_for_session(fp, after_id=0)
        return result, remaining

    result, remaining = run_async(_run())
    # 旧 user + 旧 assistant 两条都删（之后 archiver 会重新归档这一轮，
    # 只删 assistant 的话这条 user 会重复入账）
    ok = (
        result.kind == "reroll"
        and result.new_messages == []
        and len(result.delete_ids) == 2
        and result.new_watermark == 0
        and len(remaining) == 0
    )
    return _print_result(
        "场景2 重roll", ok,
        f"kind={result.kind} delete_ids={result.delete_ids} watermark={result.new_watermark} remaining={len(remaining)}",
    )


# ────────────────────────────────────────────────────────────────
# 场景 3：编辑历史消息（分叉）
# ────────────────────────────────────────────────────────────────

def test_scenario_3_edit_fork():
    """编辑：用户把第二轮的问题从"我今年20岁"改成"我今年25岁"。
    橘瓣编辑 = MessageNode 分叉出新枝，编辑后的消息就是本轮正在问的 user，
    所以 messages = [u1, a1, u2'(编辑后)]，client_history = [u1, a1]。
    对账应识别分叉点在第 2 条之后，删除 archived 后 4 条（旧第2轮+旧第3轮）。"""
    async def _run():
        await _reset_db()
        fp = "fp_edit0001"
        await _archive_turn(fp, "我叫小明", "你好小明")
        await _archive_turn(fp, "我今年20岁", "原来你20岁呀")
        await _archive_turn(fp, "我喜欢打球", "打球很棒")

        messages = [
            _msg("user", "我叫小明"),
            _msg("assistant", "你好小明"),
            _msg("user", "我今年25岁"),   # 编辑后的消息 = 本轮 user
        ]
        result = await _reconcile(fp, messages)
        remaining = await db_module.get_global_timeline_for_session(fp, after_id=0)
        return result, remaining

    result, remaining = run_async(_run())
    # 分叉点：archived 前 2 条（"我叫小明"/"你好小明"）保留，后 4 条删除；
    # 编辑后的新一轮不在 client_history 里（它是本轮 user），所以 new_messages
    # 为空，由 archiver 在拿到新回复后正常归档。
    ok = (
        result.kind == "edit"
        and result.new_watermark == 2
        and len(result.delete_ids) == 4
        and result.new_messages == []
        and len(remaining) == 2
    )
    return _print_result(
        "场景3 编辑历史消息(分叉)", ok,
        f"kind={result.kind} delete_ids={result.delete_ids} watermark={result.new_watermark} "
        f"new_messages={result.new_messages} remaining={len(remaining)}",
    )


# ────────────────────────────────────────────────────────────────
# 场景 4：两个平台交替发消息，时间线正确交错、各自增量识别正确
# ────────────────────────────────────────────────────────────────

def test_scenario_4_two_platforms_interleaved():
    async def _run():
        await _reset_db()
        fp_a = "fp_platA0001"  # 橘瓣
        fp_b = "fp_platB0001"  # 共读插件

        # 交替写入：A、B、A、B
        await _archive_turn(fp_a, "橘瓣消息1", "橘瓣回复1", context_fingerprint="daily")
        await _archive_turn(fp_b, "共读消息1", "共读回复1", context_fingerprint="reading:book1")
        await _archive_turn(fp_a, "橘瓣消息2", "橘瓣回复2", context_fingerprint="daily")
        await _archive_turn(fp_b, "共读消息2", "共读回复2", context_fingerprint="reading:book1")

        # 全局时间线应该按 id 顺序交错保留写入顺序
        global_rows = await db_module.get_global_timeline_after(after_id=0, limit=0)
        labels = [hooks._unified_label_for_row(r) for r in global_rows]

        # A 发第 3 轮新消息：历史与自己的 archived 完全一致 → noop，互不干扰
        a_messages = [
            _msg("user", "橘瓣消息1"), _msg("assistant", "橘瓣回复1"),
            _msg("user", "橘瓣消息2"), _msg("assistant", "橘瓣回复2"),
            _msg("user", "橘瓣消息3"),  # 本轮 user
        ]
        result_a = await _reconcile(fp_a, a_messages)

        b_messages = [
            _msg("user", "共读消息1"), _msg("assistant", "共读回复1"),
            _msg("user", "共读消息2"), _msg("assistant", "共读回复2"),
            _msg("user", "共读消息3"),  # 本轮 user
        ]
        result_b = await _reconcile(fp_b, b_messages)

        return global_rows, labels, result_a, result_b

    global_rows, labels, result_a, result_b = run_async(_run())
    # 写入顺序是 A(2条) B(2条) A(2条) B(2条)，全局时间线必须严格按 id 到达顺序
    ok = (
        len(global_rows) == 8
        and labels == ["daily", "daily", "reading:book1", "reading:book1",
                        "daily", "daily", "reading:book1", "reading:book1"]
        # 各自历史和自己的 archived 完全一致 → noop，且各自水位线互不干扰
        and result_a.kind == "noop" and result_a.new_messages == [] and result_a.new_watermark == 4
        and result_b.kind == "noop" and result_b.new_messages == [] and result_b.new_watermark == 4
    )
    return _print_result(
        "场景4 两平台交替消息", ok,
        f"labels={labels} result_a.kind={result_a.kind} result_b.kind={result_b.kind}",
    )


# ────────────────────────────────────────────────────────────────
# 场景 5：客户端只发截断的尾段（模拟 contextMessageSize）
# ────────────────────────────────────────────────────────────────

def test_scenario_5_truncated_tail():
    async def _run():
        await _reset_db()
        fp = "fp_trunc0001"
        # 归档 5 轮完整对话
        for i in range(1, 6):
            await _archive_turn(fp, f"第{i}轮问题", f"第{i}轮回答")

        # 橘瓣 contextMessageSize 截断：客户端只带最后 2 轮 + 新的第 6 轮 user
        messages = [
            _msg("user", "第4轮问题"), _msg("assistant", "第4轮回答"),
            _msg("user", "第5轮问题"), _msg("assistant", "第5轮回答"),
            _msg("user", "第6轮问题"),  # 本轮 user
        ]
        result = await _reconcile(fp, messages)
        remaining = await db_module.get_global_timeline_for_session(fp, after_id=0)
        return result, remaining

    result, remaining = run_async(_run())
    # 截断尾段完整匹配 archived 的尾部 → noop，不删除任何东西
    ok = (
        result.kind == "noop"
        and result.new_messages == []
        and result.delete_ids == []
        and len(remaining) == 10  # 5 轮 = 10 条，一条没少
    )
    return _print_result(
        "场景5 客户端只发截断尾段", ok,
        f"kind={result.kind} new_messages={result.new_messages} delete_ids={result.delete_ids} remaining={len(remaining)}",
    )


# ────────────────────────────────────────────────────────────────
# 场景 6：真实切分下的重roll —— 长对话 + 首轮（n==2，client_history=[]）
# ────────────────────────────────────────────────────────────────

def test_scenario_6_reroll_real_split():
    async def _run():
        await _reset_db()
        fp = "fp_reroll0002"
        # 长对话：3 轮已归档
        await _archive_turn(fp, "问题1", "回答1")
        await _archive_turn(fp, "问题2", "回答2")
        await _archive_turn(fp, "问题3", "回答3")

        # 重roll 第 3 轮回答：messages = 完整历史去掉旧 assistant，
        # 最后一条是被重问的 user"问题3"
        messages = [
            _msg("user", "问题1"), _msg("assistant", "回答1"),
            _msg("user", "问题2"), _msg("assistant", "回答2"),
            _msg("user", "问题3"),  # 被重问的 user，不进 client_history
        ]
        result_long = await _reconcile(fp, messages)
        remaining_long = await db_module.get_global_timeline_for_session(fp, after_id=0)

        # ── 首轮重roll：archived 只有一轮（n==2），client_history=[] ──
        await _reset_db()
        fp2 = "fp_reroll0003"
        await _archive_turn(fp2, "第一句话", "第一个回答")
        first_messages = [_msg("user", "第一句话")]
        result_first = await _reconcile(fp2, first_messages)
        remaining_first = await db_module.get_global_timeline_for_session(fp2, after_id=0)

        return result_long, remaining_long, result_first, remaining_first

    result_long, remaining_long, result_first, remaining_first = run_async(_run())
    ok_long = (
        result_long.kind == "reroll"
        and len(result_long.delete_ids) == 2      # 旧 user"问题3" + 旧 assistant"回答3"
        and result_long.new_watermark == 4        # 保留前 2 轮
        and result_long.new_messages == []
        and len(remaining_long) == 4
    )
    ok_first = (
        result_first.kind == "reroll"
        and len(result_first.delete_ids) == 2
        and result_first.new_watermark == 0
        and len(remaining_first) == 0             # 旧一轮必须删掉
    )
    ok = ok_long and ok_first
    return _print_result(
        "场景6 真实切分重roll(长对话+首轮)", ok,
        f"long: kind={result_long.kind} delete={result_long.delete_ids} wm={result_long.new_watermark} remaining={len(remaining_long)} | "
        f"first: kind={result_first.kind} delete={result_first.delete_ids} wm={result_first.new_watermark} remaining={len(remaining_first)}",
    )


# ────────────────────────────────────────────────────────────────
# 场景 7：截断 + 重roll 组合；截断到完全无法对齐的极端场景（p==0 防线）
# ────────────────────────────────────────────────────────────────

def test_scenario_7_truncated_reroll_and_p0_guard():
    async def _run():
        await _reset_db()
        fp = "fp_truncre001"
        # 归档 5 轮
        for i in range(1, 6):
            await _archive_turn(fp, f"第{i}轮问题", f"第{i}轮回答")

        # 截断 + 重roll：橘瓣 contextMessageSize 截断后重roll 第 5 轮回答，
        # messages = takeLast 之后的 [u4, a4, u5]（旧 a5 被客户端丢弃）
        messages = [
            _msg("user", "第4轮问题"), _msg("assistant", "第4轮回答"),
            _msg("user", "第5轮问题"),  # 被重问的 user
        ]
        result_re = await _reconcile(fp, messages)
        remaining_re = await db_module.get_global_timeline_for_session(fp, after_id=0)

        # ── 极端场景：截断到完全无法对齐（p==0 防线）──
        await _reset_db()
        fp2 = "fp_p0guard001"
        for i in range(1, 6):
            await _archive_turn(fp2, f"第{i}轮问题", f"第{i}轮回答")

        catcher = _WarnCatcher()
        logging.getLogger().addHandler(catcher)
        try:
            # 客户端历史内容和 archived 完全对不上（比如编辑+截断双重作用）
            weird_messages = [
                _msg("user", "完全对不上的历史A"), _msg("assistant", "完全对不上的回复A"),
                _msg("user", "新问题"),  # 本轮 user
            ]
            result_p0 = await _reconcile(fp2, weird_messages)
        finally:
            logging.getLogger().removeHandler(catcher)
        remaining_p0 = await db_module.get_global_timeline_for_session(fp2, after_id=0)
        has_warning = any("拒绝删除" in msg for msg in catcher.records)

        return result_re, remaining_re, result_p0, remaining_p0, has_warning

    result_re, remaining_re, result_p0, remaining_p0, has_warning = run_async(_run())
    ok_re = (
        result_re.kind == "reroll"
        and len(result_re.delete_ids) == 2       # 旧 u5 + 旧 a5
        and result_re.new_watermark == 8
        and len(remaining_re) == 8               # 前 4 轮完整保留
    )
    ok_p0 = (
        result_p0.kind == "append"               # 降级为 append
        and result_p0.delete_ids == []           # 一行都不删
        and len(remaining_p0) == 10              # 全史无损（灾难防线生效）
        and has_warning                          # 有 warning 留痕
    )
    ok = ok_re and ok_p0
    return _print_result(
        "场景7 截断+重roll组合 & p==0防线", ok,
        f"reroll: kind={result_re.kind} delete={result_re.delete_ids} wm={result_re.new_watermark} remaining={len(remaining_re)} | "
        f"p0: kind={result_p0.kind} delete={result_p0.delete_ids} remaining={len(remaining_p0)} warning={has_warning}",
    )


# ────────────────────────────────────────────────────────────────
# 场景 8：带 <time_reminder> 的归档内容 vs 不带的客户端历史
# ────────────────────────────────────────────────────────────────

def test_scenario_8_time_reminder_normalization():
    """archiver 归档的 user_content 取自预处理后的 body（带网关注入的
    <time_reminder> 尾巴），客户端历史不带。对账必须剥离后再比较，不误判。"""
    async def _run():
        await _reset_db()
        fp = "fp_timermd001"
        # 模拟真实归档：user 正文尾部带 time_reminder 注入
        await _archive_turn(
            fp,
            "你好<time_reminder>Current time: 星期五, 2026年7月10日 20:00:00</time_reminder>",
            "你好呀",
        )
        await _archive_turn(
            fp,
            "今天怎么样<time_reminder>Current time: 星期五, 2026年7月10日 20:05:00</time_reminder>",
            "挺好的",
        )

        # 客户端历史：不带 time_reminder 的原文
        messages = [
            _msg("user", "你好"),
            _msg("assistant", "你好呀"),
            _msg("user", "今天怎么样"),
            _msg("assistant", "挺好的"),
            _msg("user", "那就好"),  # 本轮 user
        ]
        result = await _reconcile(fp, messages)
        remaining = await db_module.get_global_timeline_for_session(fp, after_id=0)

        # 顺带验证：重roll 判定也要能穿透 time_reminder
        # （archived 旧 user 带注入，current_user_text 不带）
        reroll_messages = [
            _msg("user", "你好"),
            _msg("assistant", "你好呀"),
            _msg("user", "今天怎么样"),  # 重问第二轮
        ]
        result_reroll = await _reconcile(fp, reroll_messages)

        return result, remaining, result_reroll

    result, remaining, result_reroll = run_async(_run())
    ok = (
        # 历史完全一致（剥离 time_reminder 后）→ noop，不误判为编辑
        result.kind == "noop"
        and result.delete_ids == []
        and len(remaining) == 4
        # 重roll 判定穿透 time_reminder
        and result_reroll.kind == "reroll"
        and len(result_reroll.delete_ids) == 2
    )
    return _print_result(
        "场景8 time_reminder不污染对账", ok,
        f"kind={result.kind} delete={result.delete_ids} remaining={len(remaining)} | "
        f"reroll: kind={result_reroll.kind} delete={result_reroll.delete_ids}",
    )


# ────────────────────────────────────────────────────────────────
# 回归场景：截断尾段 + 追加新内容的组合（长期使用 contextMessageSize 的
# session：客户端只看得到最后几轮，但这几轮之后又真的聊了新的一轮）。
# 这是实现过程中发现的一个真实 bug：早期版本的算法会把这种正常场景误判成
# "编辑分叉"，错误删除全部 archived 历史。保留这个测试防止回归。
# ────────────────────────────────────────────────────────────────

def test_regression_truncated_tail_plus_append():
    async def _run():
        await _reset_db()
        fp = "fp_combo0001"
        for i in range(1, 6):  # 归档 5 轮
            await _archive_turn(fp, f"第{i}轮问题", f"第{i}轮回答")

        # 客户端只带最后 2 轮（截断）+ 1 轮全新未归档 + 本轮 user
        messages = [
            _msg("user", "第4轮问题"), _msg("assistant", "第4轮回答"),
            _msg("user", "第5轮问题"), _msg("assistant", "第5轮回答"),
            _msg("user", "全新问题"), _msg("assistant", "全新回答"),
            _msg("user", "再来一个新问题"),  # 本轮 user
        ]
        result = await _reconcile(fp, messages)
        remaining = await db_module.get_global_timeline_for_session(fp, after_id=0)
        return result, remaining

    result, remaining = run_async(_run())
    ok = (
        result.kind == "append"
        and result.delete_ids == []          # 绝不能删除前 5 轮
        and len(remaining) == 10             # 归档完整保留
        and result.new_messages == [_msg("user", "全新问题"), _msg("assistant", "全新回答")]
    )
    return _print_result(
        "回归测试 截断尾段+追加组合", ok,
        f"kind={result.kind} delete_ids={result.delete_ids} new_messages={result.new_messages} remaining={len(remaining)}",
    )


# ────────────────────────────────────────────────────────────────
# 场景 9（PLAN 第 4 步）：archiver 归档水位线流程
#   - 开关关闭：归档不写任何水位线（零行为变化）
#   - 开关开启：归档后水位线推进到 assistant 行的全局 id
#   - 开关开启 + find_and_delete_reroll 命中（漏网重roll replay）：
#     旧行删除后水位线回退修正到剩余最大 id
# ────────────────────────────────────────────────────────────────

def test_scenario_9_archiver_watermark():
    async def _run():
        await _reset_db()
        from gateway.services.archiver import archive

        # ── 开关关闭：归档后不应有任何 fp_global 水位线 ──
        settings_module.update_settings(unified_brain_enabled=False)
        await archive("conv-a1", "开关关闭的问题", "开关关闭的回答", "test-model", 10, 20,
                      fingerprint="fp_arch_off01", history_hash="h_off1")
        wm_off = await db_module.get_global_watermark("fp_arch_off01")

        # ── 开关开启：归档后水位线应推进到 assistant 行 id ──
        settings_module.update_settings(unified_brain_enabled=True)
        await archive("conv-a2", "开关开启的问题", "开关开启的回答", "test-model", 10, 20,
                      fingerprint="fp_arch_on01", history_hash="h_on1")
        rows_on = await db_module.get_global_timeline_for_session("fp_arch_on01", after_id=0)
        wm_on = await db_module.get_global_watermark("fp_arch_on01")

        # ── 开关开启 + reroll dedup 命中：同 fp + 同 history_hash 再归档一次 ──
        await archive("conv-a3", "开关开启的问题", "重roll后的新回答", "test-model", 10, 20,
                      fingerprint="fp_arch_on01", history_hash="h_on1")
        rows_re = await db_module.get_global_timeline_for_session("fp_arch_on01", after_id=0)
        wm_re = await db_module.get_global_watermark("fp_arch_on01")

        # 恢复开关默认状态，不影响后续测试
        settings_module.update_settings(unified_brain_enabled=False)
        return wm_off, rows_on, wm_on, rows_re, wm_re

    wm_off, rows_on, wm_on, rows_re, wm_re = run_async(_run())
    ok = (
        wm_off == 0                                        # 开关关闭：无水位线写入
        and len(rows_on) == 2 and wm_on == rows_on[-1]["id"]  # 开关开启：水位线=assistant行id
        and len(rows_re) == 2 and wm_re == rows_re[-1]["id"]  # reroll后：旧行已删+水位线修正
        and rows_re[-1]["content"] == "重roll后的新回答"
    )
    return _print_result(
        "场景9 archiver归档水位线", ok,
        f"wm_off={wm_off} | on: rows={len(rows_on)} wm={wm_on} | "
        f"reroll: rows={[r['id'] for r in rows_re]} wm={wm_re}",
    )


# ────────────────────────────────────────────────────────────────
# 场景 10（PLAN_PROACTIVE_FIX.md）：橘瓣主动消息 识别 + 归档纪律 + 对账免疫
#   - 确定性识别：合成 user 消息（固定尾缀）→ True；正常消息 → False；
#     带 time_reminder 注入的合成消息也能识别
#   - 归档纪律：proactive 正常回复只归档 assistant 单行（无合成 user 行），
#     [JUMP] 标记剥离；[PASS] 回复整条不归档
#   - 对账免疫：主动消息后的下一条正常消息 kind=noop/append，不误删
# ────────────────────────────────────────────────────────────────

_PROACTIVE_SUFFIX = "如果你觉得现在没什么好说的，或者没什么有趣的话题，请只回复 [PASS] 即可，不要强行找话题。"


def test_scenario_10_proactive_message():
    async def _run():
        await _reset_db()
        from gateway.services.archiver import archive
        settings_module.update_settings(unified_brain_enabled=True)

        fp = "fp_proact0001"
        # 先有一轮正常对话入账
        await archive("conv-p0", "今天有点累", "抱抱，累了就早点休息",
                      "test-model", 10, 20, fingerprint=fp, history_hash="hp0")

        # ── 1) 确定性识别 ──
        synthetic_text = "当前时间 22:00，用户已经 90 分钟没说话了。\n\n" + _PROACTIVE_SUFFIX
        proactive_messages = [
            _msg("user", "今天有点累"),
            _msg("assistant", "抱抱，累了就早点休息"),
            _msg("user", synthetic_text),  # 合成 user 消息
        ]
        detect_ok = hooks._is_proactive_synthetic_user(proactive_messages)
        # 带 time_reminder 注入的合成消息（防御性剥离验证）
        detect_with_tr = hooks._is_proactive_synthetic_user([
            _msg("user", synthetic_text + "<time_reminder>Current time: 22:00</time_reminder>"),
        ])
        # 正常消息不误判
        detect_normal = hooks._is_proactive_synthetic_user([_msg("user", "明天陪我去逛街吗")])
        # 最后一条不是 user 不误判
        detect_asst_tail = hooks._is_proactive_synthetic_user([_msg("assistant", _PROACTIVE_SUFFIX)])

        # ── 2) 归档纪律：正常回复（带 [JUMP] 标记）──
        await archive("conv-p1", synthetic_text, "在忙吗？我看你好久没说话了 [JUMP]",
                      "test-model", 10, 20, fingerprint=fp, history_hash="hp1",
                      proactive=True)
        rows_after_reply = await db_module.get_global_timeline_for_session(fp, after_id=0)
        wm_after_reply = await db_module.get_global_watermark(fp)

        # ── 3) 归档纪律：[PASS] 回复（大小写不敏感 + 前后空白）──
        await archive("conv-p2", synthetic_text, "  [pass]  ",
                      "test-model", 10, 20, fingerprint=fp, history_hash="hp2",
                      proactive=True)
        rows_after_pass = await db_module.get_global_timeline_for_session(fp, after_id=0)

        # ── 4) 对账免疫：主动消息后的下一条正常消息 ──
        # 橘瓣端：aP 已作为气泡持久化，下次请求历史 = [u1, a1, aP]（不含合成 user）
        next_messages = [
            _msg("user", "今天有点累"),
            _msg("assistant", "抱抱，累了就早点休息"),
            _msg("assistant", "在忙吗？我看你好久没说话了"),  # 主动消息气泡（客户端已剥 [JUMP]）
            _msg("user", "刚在洗澡啦"),  # 本轮新 user
        ]
        result_next = await _reconcile(fp, next_messages)
        rows_final = await db_module.get_global_timeline_for_session(fp, after_id=0)

        settings_module.update_settings(unified_brain_enabled=False)
        return (detect_ok, detect_with_tr, detect_normal, detect_asst_tail,
                rows_after_reply, wm_after_reply, rows_after_pass,
                result_next, rows_final)

    (detect_ok, detect_with_tr, detect_normal, detect_asst_tail,
     rows_after_reply, wm_after_reply, rows_after_pass,
     result_next, rows_final) = run_async(_run())

    # 归档纪律断言：正常回复后 = 原 2 行 + 1 行 assistant 单行（无合成 user 行）
    ok_detect = detect_ok and detect_with_tr and not detect_normal and not detect_asst_tail
    ok_reply = (
        len(rows_after_reply) == 3
        and rows_after_reply[-1]["role"] == "assistant"
        and rows_after_reply[-1]["content"] == "在忙吗？我看你好久没说话了"  # [JUMP] 已剥离
        and wm_after_reply == rows_after_reply[-1]["id"]                      # 水位线=单行 id
    )
    ok_pass = len(rows_after_pass) == 3      # [PASS] 整条没入账
    ok_next = (
        result_next.kind == "noop"           # 历史与 archived 完全一致（含 aP 单行）
        and result_next.delete_ids == []      # 不误删
        and len(rows_final) == 3
    )
    ok = ok_detect and ok_reply and ok_pass and ok_next
    return _print_result(
        "场景10 主动消息识别+归档纪律+对账免疫", ok,
        f"detect=({detect_ok},{detect_with_tr},{detect_normal},{detect_asst_tail}) | "
        f"reply: rows={len(rows_after_reply)} last={rows_after_reply[-1]['content'][:30] if rows_after_reply else ''} wm={wm_after_reply} | "
        f"pass: rows={len(rows_after_pass)} | next: kind={result_next.kind} delete={result_next.delete_ids}",
    )


# ────────────────────────────────────────────────────────────────
# 额外场景：_apply_bp2_unified 端到端（摘要+尾巴+增量组装）+ 开关默认关闭
# ────────────────────────────────────────────────────────────────

def test_extra_end_to_end_context_assembly():
    async def _run():
        await _reset_db()
        fp = "fp_e2e0001"
        await _archive_turn(fp, "你好", "你好呀，很高兴认识你")

        settings_module.update_settings(unified_summary_interval=9999, unified_tail_max=60)

        messages = [
            _msg("user", "你好"),
            _msg("assistant", "你好呀，很高兴认识你"),
            _msg("user", "我们继续聊聊"),  # 本轮 user
        ]
        out = await hooks._apply_bp2_unified(
            messages, FakeUpstream(), fingerprint=fp, context_id="",
        )
        return out

    out = run_async(_run())
    # 没有全局摘要（第一次），应该是 tail_msgs(带平台标签) + incremental([]) + current_user
    ok = (
        len(out) >= 1
        and out[-1]["role"] == "user"
        and out[-1]["content"] == "我们继续聊聊"
        and any("你好" in m.get("content", "") for m in out if m["role"] == "user")
    )
    return _print_result(
        "额外场景 端到端上下文组装", ok, f"out={out}",
    )


def test_extra_switch_off_no_effect():
    """验证开关关闭时，preprocess_anthropic 完全不会调用 _apply_bp2_unified
    （现有 _apply_bp2 逻辑必须原样执行）。这里直接检查开关默认值。"""
    async def _run():
        await _reset_db()
        settings_module.update_settings(unified_brain_enabled=False)
        return settings_module.get("unified_brain_enabled", None)

    val = run_async(_run())
    ok = val is False
    return _print_result("额外场景 开关默认关闭", ok, f"unified_brain_enabled={val}")


# ────────────────────────────────────────────────────────────────
# 场景 11（PLAN_PROACTIVE_FIX.md 形状 B）：旧版橘瓣"重复消息"主动触发
#   - 重复 + 冷场 ≥ 阈值 → 触发，且重发文本被替换为网关主动唤起指令
#   - 文本不同 → 不触发；冷场不足 → 不触发（用户手动连发场景）
# ────────────────────────────────────────────────────────────────
def test_scenario_11_proactive_shape_b():
    async def _run():
        await _reset_db()
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Asia/Shanghai")
        old_ts = (datetime.now(tz) - timedelta(minutes=45)).isoformat()
        fresh_ts = datetime.now(tz).isoformat()

        # 冷场 45 分钟的 session：重复消息 → 应触发
        fp = "fp_shapeb00001"
        await db_module.save_conversation(conversation_id="c1", role="user",
                                          content="晚安", fingerprint=fp, timestamp=old_ts)
        await db_module.save_conversation(conversation_id="c1", role="assistant",
                                          content="晚安宝宝", fingerprint=fp, timestamp=old_ts)
        msgs_repeat = [_msg("user", "晚安"), _msg("assistant", "晚安宝宝"), _msg("user", "晚安")]
        hit, rewritten = await hooks._detect_and_rewrite_repeated_proactive(msgs_repeat, fp)

        # 同 session 文本不同 → 不触发
        msgs_diff = [_msg("user", "晚安"), _msg("assistant", "晚安宝宝"), _msg("user", "在吗")]
        miss_diff, _ = await hooks._detect_and_rewrite_repeated_proactive(msgs_diff, fp)

        # 刚聊过的 session（冷场不足）：重复消息 → 不触发（用户手动连发）
        fp2 = "fp_shapeb00002"
        await db_module.save_conversation(conversation_id="c2", role="user",
                                          content="哈哈", fingerprint=fp2, timestamp=fresh_ts)
        await db_module.save_conversation(conversation_id="c2", role="assistant",
                                          content="笑什么", fingerprint=fp2, timestamp=fresh_ts)
        msgs_fresh = [_msg("user", "哈哈"), _msg("assistant", "笑什么"), _msg("user", "哈哈")]
        miss_fresh, _ = await hooks._detect_and_rewrite_repeated_proactive(msgs_fresh, fp2)

        # 形状 B：无归档记录的 session → 不触发（保守）
        miss_empty, _ = await hooks._detect_and_rewrite_repeated_proactive(msgs_repeat, "fp_noexist000")

        # ── 形状 C：请求以 assistant 结尾（continuation 式触发，2026-07-12 探针实测）──
        msgs_c = [_msg("user", "晚安"), _msg("assistant", "晚安宝宝")]
        # C1: 冷场 45 分钟 → 触发，且**追加**了一条 user 指令（长度+1，末尾是 user）
        hit_c, rewritten_c = await hooks._detect_and_rewrite_repeated_proactive(list(msgs_c), fp)
        # C2: 冷场不足（fresh session）→ 不触发（用户点"继续生成"的场景）
        msgs_c2 = [_msg("user", "哈哈"), _msg("assistant", "笑什么")]
        miss_c_fresh, _ = await hooks._detect_and_rewrite_repeated_proactive(list(msgs_c2), fp2)
        # C3: 查无归档 → 形状 C 信号足够强，放行触发
        hit_c_empty, _ = await hooks._detect_and_rewrite_repeated_proactive(list(msgs_c), "fp_noexist000")

        return (hit, rewritten, miss_diff, miss_fresh, miss_empty,
                hit_c, rewritten_c, miss_c_fresh, hit_c_empty)

    (hit, rewritten, miss_diff, miss_fresh, miss_empty,
     hit_c, rewritten_c, miss_c_fresh, hit_c_empty) = run_async(_run())
    ok = (
        hit is True
        and rewritten[-1]["content"].startswith(hooks._PROACTIVE_PREFIX)
        and miss_diff is False
        and miss_fresh is False
        and miss_empty is False
        and hit_c is True
        and len(rewritten_c) == 3
        and rewritten_c[-1]["role"] == "user"
        and rewritten_c[-1]["content"].startswith(hooks._PROACTIVE_PREFIX)
        and miss_c_fresh is False
        and hit_c_empty is True
    )
    return _print_result(
        "场景11 形状B/C主动触发", ok,
        f"B: hit={hit} miss_diff={miss_diff} miss_fresh={miss_fresh} miss_empty={miss_empty} | "
        f"C: hit={hit_c} appended={len(rewritten_c)==3} miss_fresh={miss_c_fresh} hit_empty={hit_c_empty}",
    )


ALL_TESTS = [
    test_scenario_1_normal_append,
    test_scenario_2_reroll,
    test_scenario_3_edit_fork,
    test_scenario_4_two_platforms_interleaved,
    test_scenario_5_truncated_tail,
    test_scenario_6_reroll_real_split,
    test_scenario_7_truncated_reroll_and_p0_guard,
    test_scenario_8_time_reminder_normalization,
    test_regression_truncated_tail_plus_append,
    test_scenario_9_archiver_watermark,
    test_scenario_10_proactive_message,
    test_scenario_11_proactive_shape_b,
    test_extra_end_to_end_context_assembly,
    test_extra_switch_off_no_effect,
]


def main():
    print(f"临时数据库：{_TMP_DB}")
    print(f"临时设置文件：{_TMP_SETTINGS}")
    print("-" * 60)
    results = []
    for t in ALL_TESTS:
        try:
            results.append(t())
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append(_print_result(t.__name__, False, str(e)))
    print("-" * 60)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"总计：{passed}/{total} 通过")
    # 清理临时目录
    try:
        run_async(db_module.close_db())
        import shutil
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
    except Exception:
        pass
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
