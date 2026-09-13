"""
gateway/costs.py — 费用计算公共模块
archiver.py 和 admin.py 都从这里 import，保证价格逻辑只有一份。

价格来源优先级（2026-09-14 起）：
    手工配置 settings.json:model_prices
      > 自动同步 data/auto_prices.json（上游 new-api 的 GET /api/pricing）
        > 内置默认价（_DEFAULT_PRICE，Claude Opus 4 官方价）

手工永远压过自动：自动同步只补齐"用户懒得手填"的那部分，不夺走覆盖能力——
想让某个模型走手填值，照旧填一行价格表即可，不用关总开关。

自动快照的**写入**在 services/price_sync.py（要联网），本模块只管读快照、
匹配、算钱，所以归档热路径上不会发任何网络请求。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from gateway.config import AUTO_PRICES_FILE

logger = logging.getLogger("gateway.costs")

_DEFAULT_PRICE = {
    "input": 15.0, "output": 75.0,
    "cache_write": 18.75, "cache_read": 1.5,
}

# 价格来源标识：监控页给每行标"这钱是按哪个价算的"，排障时一眼看出取价走没走对
SOURCE_MANUAL  = "manual"
SOURCE_AUTO    = "auto"
SOURCE_DEFAULT = "default"

SOURCE_LABEL = {
    SOURCE_MANUAL:  "手工配置",
    SOURCE_AUTO:    "上游自动同步",
    SOURCE_DEFAULT: "默认价（未定价）",
}


# ── 自动价格快照（data/auto_prices.json） ────────────────────────────────
# AUTO_PRICES_FILE 是模块级全局，测试 monkeypatch 它即可重定向（同 upstream.UPSTREAMS_FILE）。
# mtime 签名缓存与 settings.py / upstream.py 同一套思路：后台手动刷新写了新文件，
# stat 变化自动失效，无需显式清理。
_auto_cache: tuple[str, int, int, dict] | None = None


def _empty_auto() -> dict:
    """空快照。每次新建 dict——避免调用方原地改动污染共享常量。"""
    return {"synced_at": "", "prices": [], "upstreams": {}}


def purge_auto_cache() -> None:
    """测试/异常恢复用：丢弃 mtime 缓存。"""
    global _auto_cache
    _auto_cache = None


def load_auto_prices() -> dict:
    """读自动价格快照，返回 {"synced_at", "prices": [...], "upstreams": {...}}。

    文件缺失/损坏/半截 JSON 一律当空快照降级——价格同步是"锦上添花"，
    任何异常都不允许穿透到归档或监控渲染（取不到价就退回默认价）。
    """
    global _auto_cache
    path = Path(AUTO_PRICES_FILE)
    try:
        st = path.stat()
        sig = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        _auto_cache = None
        return _empty_auto()
    if _auto_cache is not None and _auto_cache[:3] == sig:
        return _auto_cache[3]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"顶层不是对象：{type(data).__name__}")
    except Exception:
        logger.warning("auto_prices.json 读取/解析失败，本次按空价目表处理：%s", path,
                       exc_info=True)
        _auto_cache = None
        return _empty_auto()
    data.setdefault("synced_at", "")
    if not isinstance(data.get("prices"), list):
        data["prices"] = []
    if not isinstance(data.get("upstreams"), dict):
        data["upstreams"] = {}
    _auto_cache = (*sig, data)
    return data


def save_auto_prices(prices: list[dict], upstreams: dict | None = None,
                     synced_at: str = "") -> Path:
    """写入自动价格快照。prices 为扁平列表（每条含 upstream/model/四个单价），
    与 model_prices 同构——这样匹配逻辑两边共用一份。"""
    global _auto_cache
    path = Path(AUTO_PRICES_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "synced_at": synced_at,
        "prices": prices,
        "upstreams": upstreams or {},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _auto_cache = None  # 写后显式失效（Windows 同尺寸重写的 mtime 粒度可能撞车）
    logger.info("auto_prices.json 已更新：%d 条价格 / %d 个上游 / synced_at=%s",
                len(prices), len(upstreams or {}), synced_at or "-")
    return path


# ── 取价 ────────────────────────────────────────────────────────────────

def _manual_prices() -> list[dict]:
    try:
        from gateway.settings import get as _get
        return _get("model_prices", []) or []
    except Exception:
        return []


def _match_manual(prices: list[dict], upstream: str, model: str) -> dict | None:
    """手工价目表匹配，保持 2026-09 起的原语义（不做"更聪明"的改动，
    免得老用户既有配置的取价结果悄悄变化）。
    优先级：(upstream, model) 精确 > model 精确 > model 子串。"""
    # 1. 精确匹配 (upstream, model)
    for p in prices:
        if (p.get("upstream") or "") == upstream and (p.get("model") or "") == model:
            return p
    # 2. 只匹配 model（不限上游）
    for p in prices:
        if not p.get("upstream") and (p.get("model") or "") == model:
            return p
    # 3. 子串匹配（兼容旧配置）
    for p in prices:
        if not p.get("upstream") and (p.get("model") or "").lower() in (model or "").lower():
            return p
    return None


def _match_auto(prices: list[dict], upstream: str, model: str) -> dict | None:
    """自动价目表匹配。

    与手工表的关键区别：auto 条目**永远带 upstream**（同一模型在不同中转站
    价格不同），所以绝不做"不限上游"的兜底——否则 A 站价格会算到 B 站头上。
    优先级：(upstream, model) 精确 > 大小写不敏感精确 > 同上游唯一子串。
    子串要求唯一命中：有歧义（如 opus-4-5 / opus-4-5-thinking 同时命中）时
    宁可退回默认价并在监控页标出来，也不猜一个可能错 5 倍的单价。"""
    if not upstream or not model:
        return None
    for p in prices:
        if (p.get("upstream") or "") == upstream and (p.get("model") or "") == model:
            return p
    low = model.lower()
    for p in prices:
        if (p.get("upstream") or "") == upstream and (p.get("model") or "").lower() == low:
            return p
    hits = []
    for p in prices:
        if (p.get("upstream") or "") != upstream:
            continue
        entry_model = (p.get("model") or "").lower()
        if entry_model and (entry_model in low or low in entry_model):
            hits.append(p)
    return hits[0] if len(hits) == 1 else None


def resolve_price(upstream: str, model: str) -> tuple[dict | None, str]:
    """按 手工 > 自动 > 默认 取价，返回 (条目, 来源)。来源见 SOURCE_*。"""
    manual = _manual_prices()
    if manual:
        entry = _match_manual(manual, upstream, model)
        if entry:
            return entry, SOURCE_MANUAL
    auto = load_auto_prices().get("prices") or []
    if auto:
        entry = _match_auto(auto, upstream, model)
        if entry:
            return entry, SOURCE_AUTO
    return None, SOURCE_DEFAULT


def get_price_entry(upstream: str, model: str) -> dict | None:
    """兼容旧调用点：只要价格条目，不要来源。"""
    return resolve_price(upstream, model)[0]


def price_from_entry(entry: dict | None) -> dict:
    """条目 → 四个单价（$/1M tokens）。缺字段按默认价回落。"""
    if not entry:
        return dict(_DEFAULT_PRICE)
    return {
        "input":       float(entry.get("input",       _DEFAULT_PRICE["input"])),
        "output":      float(entry.get("output",      _DEFAULT_PRICE["output"])),
        "cache_write": float(entry.get("cache_write", _DEFAULT_PRICE["cache_write"])),
        "cache_read":  float(entry.get("cache_read",  _DEFAULT_PRICE["cache_read"])),
    }


def compute_cost(entry: dict | None, inp: int, out: int, cw: int, cr: int,
                 msgs: int = 1) -> tuple[float, float]:
    """按条目算钱，返回 (cost_usd, saved_usd)。entry 为 None 时用默认价。
    - 按次计费（price_type == "per_call"）：cost = msgs × price_per_call，saved = 0
    - 按 token：cost = Σ(token 量 × 单价) / 1M
                 saved = cache_read × (input 单价 - cache_read 单价) / 1M

    监控页与归档共用这一份算术，避免两边漂移（旧版监控页自带了第二份实现）。"""
    if entry and entry.get("price_type") == "per_call":
        ppc = float(entry.get("price_per_call") or 0)
        return msgs * ppc, 0.0

    pr = price_from_entry(entry)
    cost = (
        inp * pr["input"] +
        out * pr["output"] +
        cw  * pr["cache_write"] +
        cr  * pr["cache_read"]
    ) / 1_000_000
    saved = cr * (pr["input"] - pr["cache_read"]) / 1_000_000
    return cost, saved


def calculate_cost(
    upstream: str,
    model: str,
    inp: int,
    out: int,
    cw: int,
    cr: int,
    msgs: int = 1,
) -> tuple[float, float]:
    """
    计算费用，返回 (cost_usd, saved_usd)。
    取价顺序见模块开头的"价格来源优先级"。条目里没写的维度按默认价回落。
    """
    entry, _source = resolve_price(upstream, model)
    return compute_cost(entry, inp, out, cw, cr, msgs)
