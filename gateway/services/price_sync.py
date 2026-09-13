"""new-api 上游价格自动同步（2026-09-14）。

背景：缓存监控的费用计算原本完全依赖用户手填价格表，而实际上游都是 new-api
中转站，它的 `GET /api/pricing` 已经把每个模型的计费倍率全吐出来了。本模块把
倍率换算成 $/1M tokens，落盘成 data/auto_prices.json，costs.py 按「手工配置 >
自动同步 > 内置默认」取价。

三个触发源共用同一套逻辑（唯一真相源，避免双份漂移）：
- 后台"立即同步价格"按钮（routers/admin.py /upstreams/refresh-all-prices）
- 后台定时循环（main.py lifespan → price_sync_loop）
- 每上游单独的"同步"按钮（同样走 sync_upstreams）

────────────────── new-api 计费口径（源码依据） ──────────────────
- 路由：`router/api-router.go` → `GET /api/pricing` → `controller.GetPricing`
- 响应：{success, data: [Pricing...], group_ratio, usable_group, vendors}
- `model/pricing.go` Pricing 结构：
    quota_type 0 = 按 token（用 model_ratio），1 = 按次（用 model_price）
    model_ratio / completion_ratio / cache_ratio / create_cache_ratio
- 换算（`common/constants.go` QuotaPerUnit = 500000，即 1 倍率 = $0.002/1K）：
    $/1M 输入 = model_ratio × group_ratio × 1_000_000 / 500_000 = ×2
    输出 = 输入 × completion_ratio
    缓存读 = 输入 × cache_ratio
    缓存写 = 输入 × create_cache_ratio（5m）；1h 再 × 1.6
            （relay/helper/price.go: claudeCacheCreation1hMultiplier = 6/3.75）
    按次 = model_price × group_ratio

⚠️ 两个必须记住的坑（都踩过）：
1. **不能带 Authorization 头**。`/api/pricing` 挂的是
   `middleware.HeaderNavModuleAuth("pricing")`，它只认后台凭据（会话/PAT）；
   把上游的转发 key（sk-...）塞进 Authorization 会被
   `classifyDashboardCredential` 判成「看起来像凭据但校验失败」并**直接
   abort 请求**（不是优雅降级成匿名）。所以只能匿名裸调。
   代价：匿名拿不到我们这把 key 所属分组的 group_ratio，只能按上游配的
   `pricing_group`（默认 default）取分组倍率，见 _resolve_group。
2. `cache_ratio` / `create_cache_ratio` 在 new-api 里是 `*float64` + omitempty。
   指针型 omitempty 只在 nil 时省略，**指向 0 的指针照样会序列化**。所以
   「字段缺失」= 站点没配过 → 用 new-api 的默认值（1.0 / 1.25）；
   「字段为 0」= 显式配了 0 → 用 0。绝不能写成 `x or default` 把 0 吞掉。
"""

import asyncio
import logging
import ssl
import time
import traceback
from datetime import datetime

import httpx

from gateway.costs import load_auto_prices, save_auto_prices
from gateway.http_client import get_client
from gateway.settings import get
from gateway.upstream import Upstream, _normalize_cache_ttl, load_upstreams

logger = logging.getLogger("gateway.price_sync")

# new-api 计费常量（源码：common/constants.go / relay/helper/price.go）
QUOTA_PER_UNIT = 500_000.0              # $0.002 / 1K tokens
CACHE_CREATION_1H_MULTIPLIER = 6 / 3.75  # 1h 缓存写相对 5m 的倍数 = 1.6
DEFAULT_CACHE_RATIO = 1.0               # setting/ratio_setting/cache_ratio.go
DEFAULT_CREATE_CACHE_RATIO = 1.25

DEFAULT_GROUP = "default"
_FETCH_TIMEOUT = 20

# 最近一次同步时间的单调钟（定时循环的冷却判据；重启后视为已冷却）
_last_sync = 0.0
_sync_lock = asyncio.Lock()


def reset_sync_state() -> None:
    """测试用：清零冷却计时。"""
    global _last_sync
    _last_sync = 0.0


# ── 换算 ────────────────────────────────────────────────────────────────

def usd_per_1m(ratio: float, group_ratio: float = 1.0) -> float:
    """new-api 倍率 → $/1M tokens。ratio=1 即 $2/1M（QuotaPerUnit 的定义）。"""
    return ratio * group_ratio * 1_000_000.0 / QUOTA_PER_UNIT


def _ratio_or_default(item: dict, key: str, default: float) -> float:
    """取倍率字段：**缺失**用默认值，显式为 0 就用 0（见模块开头坑 2）。"""
    value = item.get(key)
    return default if value is None else float(value)


def convert_model(item: dict, group_ratio: float, cache_ttl: str) -> dict | None:
    """单个 Pricing 条目 → 我方价格条目（结构与 model_prices 一致）。
    返回 None 表示这条没法用（没模型名）。"""
    model = str(item.get("model_name") or "").strip()
    if not model:
        return None

    entry = {
        "upstream": "",          # 由调用方按上游填
        "model": model,
        "source": "new-api",
        "cache_ttl": cache_ttl,
    }

    # quota_type == 1：站点给这个模型配了固定价（model_price），按次计费
    if int(item.get("quota_type") or 0) == 1:
        entry.update({
            "price_type": "per_call",
            "price_per_call": round(float(item.get("model_price") or 0) * group_ratio, 6),
            "input": 0.0, "output": 0.0, "cache_write": 0.0, "cache_read": 0.0,
            "cache_write_5m": 0.0, "cache_write_1h": 0.0,
        })
        return entry

    model_ratio = float(item.get("model_ratio") or 0)
    inp = usd_per_1m(model_ratio, group_ratio)
    completion = item.get("completion_ratio")
    cw_5m = inp * _ratio_or_default(item, "create_cache_ratio", DEFAULT_CREATE_CACHE_RATIO)
    cw_1h = cw_5m * CACHE_CREATION_1H_MULTIPLIER

    # 网关上这个上游实际用哪种 TTL（upstream.cache_ttl），就取哪个缓存写单价。
    # 注：归档时 calculate_cost 只拿得到上游名、拿不到令牌组，所以这里按
    # **上游级** cache_ttl 结算；额外令牌单独覆盖 TTL 的情况会略有偏差，
    # 改完 cache_ttl 记得点一次"同步价格"。
    if cache_ttl == "off":
        cw = 0.0
    elif cache_ttl == "1h":
        cw = cw_1h
    else:
        cw = cw_5m

    entry.update({
        "price_type": "token",
        "price_per_call": 0.0,
        "input": round(inp, 6),
        "output": round(inp * (1.0 if completion is None else float(completion)), 6),
        "cache_write": round(cw, 6),
        "cache_read": round(inp * _ratio_or_default(item, "cache_ratio", DEFAULT_CACHE_RATIO), 6),
        # 两个 TTL 档都留着：设置页展示用，也方便以后按令牌组细分
        "cache_write_5m": round(cw_5m, 6),
        "cache_write_1h": round(cw_1h, 6),
    })
    return entry


def parse_pricing_payload(payload) -> tuple[list[dict], dict, dict]:
    """解析 /api/pricing 响应 → (模型列表, group_ratio 表, usable_group 表)。
    结构不对直接抛 ValueError，由调用方转成该上游的失败结果。"""
    if not isinstance(payload, dict):
        raise ValueError(f"响应不是对象：{type(payload).__name__}")
    if payload.get("success") is False:
        raise ValueError(f"上游返回 success=false: {payload.get('message') or ''}")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("响应缺少 data 数组（可能不是 new-api 站点）")
    group_ratio = payload.get("group_ratio")
    usable_group = payload.get("usable_group")
    return (
        [x for x in data if isinstance(x, dict)],
        group_ratio if isinstance(group_ratio, dict) else {},
        usable_group if isinstance(usable_group, dict) else {},
    )


def pricing_urls(base_url: str) -> list[str]:
    """候选价格接口 URL。new-api 挂在站点根路径 /api/pricing；
    base_url 有的写成 https://host、有的写成 https://host/v1，两种都试。"""
    base = (base_url or "").rstrip("/")
    urls = [f"{base}/api/pricing"]
    if base.endswith("/v1"):
        urls.append(f"{base[:-3].rstrip('/')}/api/pricing")
    return urls


def _resolve_group(u: Upstream, group_ratio: dict) -> tuple[str, float, str]:
    """定分组倍率：上游配了 pricing_group 就用它，否则 default。
    返回 (实际用的分组名, 倍率, 警告文案)。"""
    wanted = (getattr(u, "pricing_group", "") or "").strip() or DEFAULT_GROUP
    if wanted in group_ratio:
        return wanted, float(group_ratio[wanted] or 0), ""
    fallback = float(group_ratio.get(DEFAULT_GROUP, 1.0) or 0)
    if not group_ratio:
        # 站点没返回分组表（老版本 new-api）。按 1.0 算并说清楚，别让用户
        # 以为价格一定准。
        return wanted, 1.0, f"上游未返回分组倍率表，{wanted} 按 1.0 计算"
    return DEFAULT_GROUP, fallback, (
        f"分组 {wanted!r} 不在上游分组表里，已回退到 {DEFAULT_GROUP}"
        f"（倍率 {fallback}）"
    )


# ── 抓取 ────────────────────────────────────────────────────────────────

async def fetch_pricing(u: Upstream) -> tuple[str, dict]:
    """匿名拉取上游价格表，返回 (实际生效的 URL, 解析后的 JSON)。

    匿名是硬要求，原因见模块开头坑 1——带 Authorization 反而会被判成
    坏凭据直接 403。失败会依次尝试：共享连接池 → 关掉证书校验（上游
    自签/链不全时；与 routers/admin.py fetch_models 同一套兜底理由）。"""
    errors: list[str] = []
    last_exc: Exception | None = None

    for url in pricing_urls(u.base_url):
        try:
            resp = await get_client().get(url, timeout=_FETCH_TIMEOUT)
            resp.raise_for_status()
            return url, resp.json()
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            # 403 是唯一"换了 URL 也没用"的情况：上游把价格页模块关了。
            # 其余状态码一律记下换下一个候选 URL 再试（反代对未知路径可能
            # 回 502/500 而不是 404）。
            if code == 403:
                raise RuntimeError(
                    f"{url} 返回 403：上游把价格页模块关掉了"
                    "（new-api 后台 HeaderNavModuleAuth），本上游无法自动取价"
                ) from exc
            last_exc = exc
            errors.append(f"{url} → HTTP {code}")
            continue
        except Exception as exc:  # noqa: BLE001 — 网络层异常统一降级
            last_exc = exc
            logger.warning("fetch_pricing %s 失败（%s）：%s", u.name, url, exc)
            # SSL/证书问题：换不校验重试一次（有些中转站证书链不全）
            if isinstance(exc, (ssl.SSLError, httpx.ConnectError, httpx.ConnectTimeout)):
                try:
                    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT, verify=False) as c:
                        resp = await c.get(url)
                        resp.raise_for_status()
                        logger.info("fetch_pricing %s: OK via no_verify (%s)", u.name, url)
                        return url, resp.json()
                except Exception as exc2:  # noqa: BLE001
                    last_exc = exc2
                    errors.append(f"{url} → {type(exc2).__name__}: {exc2}")
            else:
                errors.append(f"{url} → {type(exc).__name__}: {exc}")
            continue

    detail = "；".join(errors) if errors else (
        f"{type(last_exc).__name__}: {last_exc}" if last_exc else "未知错误"
    )
    raise RuntimeError(f"拉取价格表失败：{detail}")


# ── 同步 ────────────────────────────────────────────────────────────────

async def refresh_one_upstream_prices(u: Upstream) -> dict:
    """同步单个上游的价格。异常向上抛给调用方（由 sync_upstreams 兜住）。"""
    url, payload = await fetch_pricing(u)
    models, group_ratio, usable_group = parse_pricing_payload(payload)
    if not models:
        raise RuntimeError("上游价格表返回 0 个模型")

    group, rate, warning = _resolve_group(u, group_ratio)
    cache_ttl = _normalize_cache_ttl(getattr(u, "cache_ttl", ""), default="1h")

    prices: list[dict] = []
    per_call = 0
    for item in models:
        entry = convert_model(item, rate, cache_ttl)
        if entry is None:
            continue
        if entry.get("price_type") == "per_call":
            per_call += 1
        entry["upstream"] = u.name
        prices.append(entry)

    logger.info(
        "价格同步 %s：%d 个模型（按次 %d）· 分组=%s(×%s) · cache_ttl=%s · url=%s",
        u.name, len(prices), per_call, group, rate, cache_ttl, url,
    )
    return {
        "ok": True,
        "url": url,
        "count": len(prices),
        "per_call_count": per_call,
        "group": group,
        "group_ratio": rate,
        "warning": warning,
        "error": "",
        "stale": False,
        "cache_ttl": cache_ttl,
        # 全量分组表，给设置页做下拉候选
        "group_options": {k: float(v or 0) for k, v in group_ratio.items()},
        "usable_group": usable_group,
        "prices": prices,
    }


async def sync_upstreams(names: list[str] | None = None) -> dict:
    """同步指定（默认全部 active）上游的价格并落盘，返回 {上游名: meta}。

    names=None  = 全量同步：快照严格等于"当前配置里的 active 上游"，
                  所以配置里已删掉的上游，其陈旧条目会被清掉。
    names=[...] = 单站/部分同步：**只替换被点名的上游**，其它上游的价格与
                  meta 原样保留。点名单里没有匹配到任何上游时直接返回 {}，
                  一个字节都不写——"同步一个不存在的上游"绝不能把整份快照清空
                  （曾经就是这么把全站价格清成默认价的）。

    其它两条纪律：
    - 各上游并行抓取：彼此是不同站点，且最后统一写一次文件，不存在
      model_sync 那种"读-改-写互相覆盖丢更新"的问题。
    - **失败的上游保留上一轮价格**（meta.stale=True），否则一次 500 就会让
      整站价格回落到 Opus 默认价、账目瞬间失真。只有从没拉到过价的新上游才空着。
    """
    global _last_sync
    async with _sync_lock:
        configured = [u for u in load_upstreams() if u.is_active]
        if names:
            wanted = set(names)
            targets = [u for u in configured if u.name in wanted]
            if not targets:
                logger.warning("价格同步：指定上游 %s 无匹配（不存在或未启用），未改动快照",
                               sorted(wanted))
                return {}
            full_sync = False
        else:
            targets = configured
            full_sync = True

        results = await asyncio.gather(
            *[refresh_one_upstream_prices(u) for u in targets],
            return_exceptions=True,
        )

        old = load_auto_prices()
        old_by_upstream: dict[str, list[dict]] = {}
        for p in (old.get("prices") or []):
            old_by_upstream.setdefault(p.get("upstream") or "", []).append(p)
        old_meta: dict[str, dict] = dict(old.get("upstreams") or {})

        now_iso = datetime.now().isoformat(timespec="seconds")

        # 部分同步时，没被点名的上游原样带进新快照（价格 + meta 都不动）。
        # 注意区分两个 dict：file_meta 是"写进文件的全景"，result 是"本次真正
        # 处理的那些上游"——调用方要的是后者（后台按钮只关心自己点了谁）。
        all_prices: list[dict] = []
        file_meta: dict[str, dict] = {}
        result: dict[str, dict] = {}
        touched = {u.name for u in targets}
        if not full_sync:
            for name, entries in old_by_upstream.items():
                if name in touched:
                    continue
                all_prices.extend(entries)
                if name in old_meta:
                    file_meta[name] = old_meta[name]

        for u, res in zip(targets, results):
            if isinstance(res, BaseException):
                logger.warning("价格同步失败 %s：%s\n%s", u.name, res, traceback.format_exc())
                kept = old_by_upstream.get(u.name, [])
                prev = old_meta.get(u.name) or {}
                all_prices.extend(kept)
                info = {
                    "ok": False,
                    "count": len(kept),
                    "error": str(res)[:300],
                    "stale": bool(kept),
                    "synced_at": prev.get("synced_at", ""),
                    "group": prev.get("group", ""),
                    "group_ratio": prev.get("group_ratio", 0),
                    "warning": "",
                    "group_options": prev.get("group_options", {}),
                }
            else:
                all_prices.extend(res.pop("prices"))
                info = {**res, "synced_at": now_iso}
            file_meta[u.name] = info
            result[u.name] = info

        save_auto_prices(all_prices, upstreams=file_meta, synced_at=now_iso)
        _last_sync = time.monotonic()
        return result


async def refresh_all_upstream_prices() -> dict:
    """全部 active 上游。供后台"立即同步价格"按钮与定时循环调用。"""
    return await sync_upstreams()


def should_auto_sync(now: float | None = None) -> bool:
    """距上次同步是否已过间隔（且总开关开启）。"""
    if not get("auto_price_enabled", False):
        return False
    now = time.monotonic() if now is None else now
    interval = max(1, int(get("auto_price_interval_hours", 12))) * 3600
    return (now - _last_sync) >= interval


async def price_sync_loop():
    """后台无限循环。由 main.py lifespan 启动。

    价格同步没有"请求触发"这一路（模型同步有 /v1/models 那个钩子，价格纯粹是
    账目需要），所以只要一个定时循环：每 30 分钟醒一次看该不该刷，够用且
    开销可忽略。冷启动后先等 90 秒再首刷，给网关起完的时间（与
    balance_monitor 同一思路，错开一点避免同时外呼）。"""
    await asyncio.sleep(90)
    while True:
        try:
            if should_auto_sync():
                result = await refresh_all_upstream_prices()
                logger.info("自动价格同步完成：%s",
                            {k: v.get("count", 0) for k, v in (result or {}).items()})
        except Exception:
            logger.exception("price_sync_loop 未预期错误")
        await asyncio.sleep(1800)
