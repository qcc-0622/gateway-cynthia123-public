"""Persistent gateway settings stored in data/settings.json."""

import json
from pathlib import Path
from gateway.config import BASE_DIR

_SETTINGS_FILE = BASE_DIR / "data" / "settings.json"

_DEFAULTS = {
    "summary_model": "",        # empty = use haiku fallback
    "summary_upstream": "",     # empty = use active upstream; set to upstream name to override
    "summary_prompt": "",       # empty = use built-in Chinese prompt
    "bp3_cycle_size": 9,        # legacy: turns per freeze cycle (kept for compatibility)
    "bp3_frozen_rounds": 8,     # original turns retained as frozen context after BP2
    "bp3_live_rounds": 24,      # live turns accumulated before rewriting BP2 summary
    # Model pricing: list of {name, input, output, cache_write, cache_read} per 1M tokens
    "model_prices": [],

    # ── new-api 上游价格自动同步（2026-09-14） ────────────────────────
    # 上游都是 new-api 中转站，它的 GET /api/pricing 会返回每个模型的
    # model_ratio / completion_ratio / cache_ratio / create_cache_ratio，
    # 换算成 $/1M tokens 与 model_prices 结构完全一致。开启后由
    # services/price_sync.py 定时抓取写入 data/auto_prices.json，
    # costs.py 按「手工配置 > 自动同步 > 内置默认」取价。
    #
    # 为什么写单独文件而不是并进 settings.json：auto_prices 是机器生成的
    # 缓存（上千条 + 每次同步都变），塞进 settings.json 会让启动时的
    # get_overridden_defaults() 差异清单刷屏，也会让"用户显式配置"和
    # "机器同步结果"混成一份真相源。
    "auto_price_enabled": False,
    "auto_price_interval_hours": 12,   # 后台自动同步间隔（小时）

    # ── 统一大脑（全平台共享时间线 + 全局滚动摘要，PLAN_UNIFIED_BRAIN.md） ──
    # 总开关：默认 False。关闭时下面三个字段完全不影响任何现有行为——
    # 所有新逻辑必须包在 `if unified_brain_enabled:` 里，这是 Issue #36 的教训
    # （当年共享 BP2 出问题时，回退花了一整晚；这次回退只需要把这个开关改回 False）。
    "unified_brain_enabled":     False,  # 统一大脑总开关
    "unified_summary_interval": 30,      # 全局新增多少条消息触发一次全局摘要滚动压缩
    "unified_tail_max":         60,      # 全局时间线尾巴最多带多少条（超过靠摘要压缩兜底）

    # ── 摘要体重控制（2026-07-12，用户目标 8000-10000 字符） ──────────────
    # 滚动继承会让摘要一代代变肥（实测涨到 14000-23000），超过此阈值就调
    # compress_summary 压到 8000-10000。旧值 35000 形同虚设。
    "summary_compress_threshold": 12000,

    # ── 调试（REFACTOR_ROADMAP P0.4） ─────────────────────────────────
    # 管道追踪：开启后 preprocess 各关键节点打日志（消息数+尾部角色），排障用。
    # 默认关闭零开销；journalctl 里 grep PIPE 看轨迹。
    "debug_pipeline_trace": False,

    # ── 主动消息（PLAN_PROACTIVE_FIX.md） ──────────────────────────────
    # 形状 B 冷场判定阈值（分钟）：末尾 user 与上一条 user 完全相同 + 距上次
    # 归档闲置 ≥ N 分钟 → 判定为旧版橘瓣的主动触发（重发文本会被替换为
    # 网关主动唤起指令）。2026-07-12 实测触发器间隔仅 1~2 分钟，原默认 20
    # 挡住全部真实触发，降为 1。
    "proactive_repeat_idle_minutes": 1,

    # ── 多上游 failover（2026-09-06） ─────────────────────────────────
    # 首选上游在吐出第一个字节前失败（429/5xx/坏key/连接错误）时，自动换
    # 下一家能服务该模型的活跃上游。默认 False：备用上游余额不可控，未经
    # 用户在后台确认不自动消耗其他上游的钱。
    "upstream_failover_enabled": False,

    # ── 上游模型列表自动同步（2026-08-28） ─────────────────────────────
    # 客户端点"拉取模型"= GET /v1/models，网关借这次请求自动刷新所有上游的
    # 模型缓存（主令牌+额外令牌分组），本次响应即含新模型，不用再去后台手动
    # "一键拉取"。冷却期内直接用现有缓存，防止打爆上游。逻辑在 services/model_sync.py。
    "models_auto_refresh_on_client_pull": True,
    "models_auto_refresh_cooldown_min": 30,

    # ── Memory recall (ombre-brain integration) ───────────────────────
    "memory_recall_enabled":   True,    # 总开关
    "memory_autofeed_enabled": True,    # BP2 后自动喂 ombre
    "ombre_url":               "http://127.0.0.1:18001/mcp",
    "ombre_token":             "",      # 内网调用不用 token
    "ombre_buckets_dir":       "/opt/ombre-brain/buckets",
    # 新话题检测的 embedding 服务
    "embed_url":               "https://api.siliconflow.cn/v1/embeddings",
    "embed_model":             "BAAI/bge-m3",
    "embed_api_key":           "",      # 必填（同 ombre 用的那个 SiliconFlow key）
    "memory_new_topic_threshold": 0.55, # cosine 低于此值视为新话题
    "memory_max_recall":       5,       # 最多注入几条记忆
    "memory_random_k":         5,       # 随机注入几条
    "memory_breath_max_results": 8,
    "memory_breath_max_tokens":  2000,

    # ── 通知推送 ─────────────────────────────────────────────────────
    "notify_pushplus_token":        "",    # PushPlus 微信推送 token
    "notify_ntfy_url":              "",    # ntfy Android 推送地址
    "notify_balance_threshold":     5.0,   # 余额告警阈值（CNY/USD）
    "notify_balance_interval_hours": 6,    # 余额检查间隔（小时）
}


# 2026-09-01: mtime 缓存。热路径（hooks/memory/summarizer/archiver/notifier）
# 每请求会调 get()/get_settings() 多次，之前每次都同步读盘 + json.loads——
# 文件虽小，但都是事件循环上的磁盘 syscall。以 (路径, mtime_ns, size) 为签名，
# 后台改配置 / 外部手改文件 / 测试换路径都会让 stat 变化自动失效，无需手动清理。
_cache: tuple[str, int, int, dict] | None = None


def _ensure() -> dict:
    """读取 settings.json 里**显式设置过**的键（不含代码默认值）。

    ⚠️ 2026-07-12 根治"双真相源"暗坑（REFACTOR_ROADMAP P0.2）：
    旧版会把 _DEFAULTS 里 json 缺失的键物化写回 settings.json——从那一刻起
    代码里改默认值就永远无效（json 冻结副本优先），一个月内咬人三次
    （summary_prompt / proactive 阈值 / compress 阈值）。
    新版：json 只存显式设置过的键；缺失的键由 get()/get_settings() 运行时
    回落到 _DEFAULTS，不再写回。改代码默认值即刻生效，除非后台明确改过。"""
    global _cache
    try:
        st = _SETTINGS_FILE.stat()
        sig = (str(_SETTINGS_FILE), st.st_mtime_ns, st.st_size)
    except OSError:
        _cache = None
        _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SETTINGS_FILE.write_text("{}")
        return {}
    if _cache is not None and _cache[:3] == sig:
        return _cache[3]
    try:
        data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    _cache = (*sig, data)
    return data


def get_settings() -> dict:
    """合并视图：代码默认值 + json 显式覆盖。后台页面/调用方拿到全量键。"""
    return {**_DEFAULTS, **_ensure()}


def update_settings(**kwargs):
    global _cache
    data = dict(_ensure())  # 拷贝后再改，不污染 mtime 缓存里的同一份 dict
    for k, v in kwargs.items():
        if k in _DEFAULTS:
            data[k] = v
    _SETTINGS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    _cache = None  # 写后显式失效（Windows 同尺寸快速重写的 mtime 粒度可能撞车）
    return {**_DEFAULTS, **data}


def get(key: str, default=None):
    data = _ensure()
    if key in data:
        return data[key]
    if key in _DEFAULTS:
        return _DEFAULTS[key]
    return default


def get_overridden_defaults() -> list[tuple[str, object, object]]:
    """列出被 settings.json 覆盖且值与代码默认值不同的键：(key, 默认值, 实际值)。
    启动时打进日志 + 后台设置页顶部展示，让"配置盖住了代码"永远可见。"""
    data = _ensure()
    out = []
    for k, actual in data.items():
        if k in _DEFAULTS and actual != _DEFAULTS[k]:
            out.append((k, _DEFAULTS[k], actual))
    return out
