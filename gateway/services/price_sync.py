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
3. **排查"某站拉不到价"时必须用网关自己的客户端（httpx），不要用 curl / urllib。**
   2026-09-14 实测：relay-c 对 `python-urllib` 的 UA 回 Cloudflare `error code: 1010`，
   对 httpx 的 UA 回 200 完整数据。用 curl 试会得出"这站不支持"的错误结论
   （详见 _reject_reason 的三种情况表）。
"""
import asyncio
import logging
import re
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


# ── 动态计费表达式（billing_mode == "tiered_expr"）────────────────────────
# new-api 的 billing_expr 是一段类 JS 表达式，**系数直接就是 $/1M 价格**，例如：
#   tier("base", p * 1.875 + c * 9.375 + cr * 0.1875 + cc * 2.34375 + cc1h * 3.75)
#   len <= 272000 ? tier("base", p * 5 + c * 30 + cr * 0.5 + cc * 6.25)
#                 : tier("tier2", p * 10 + c * 45 + cr * 1 + cc * 12.5)
# 变量含义：p=输入token c=输出token cr=缓存读 cc=缓存写(5m) cc1h=缓存写(1h)
#          len=上下文长度（只影响选哪一档）
# 我们**只取第一档（base）的线性系数**：绝大多数请求落在 base 档，站点自己的定价页
# 也是把 base 列在最前。多档情况在 entry 里留 billing_expr 原文备查。
#
# 为什么不用完整 JS 解释器：为一个取价脚本引入 JS 求值器不值得，而且一旦求值出错
# 会静默给出错价。这里只认 `变量 * 数字` 的线性项，认不出来就返回 {} → 调用方跳过
# 该模型（不回落 model_ratio，见 convert_model 里的说明）。
_EXPR_VAR_RE = re.compile(r"\b(cc1h|cc|cr|p|c)\s*\*\s*(-?\d+(?:\.\d+)?)")
_EXPR_NUM_FIRST_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*\*\s*\b(cc1h|cc|cr|p|c)\b")
_EXPR_FIXED_RE = re.compile(r"fixed\(\s*(-?\d+(?:\.\d+)?)\s*\)", re.I)
_EXPR_FIRST_TIER_RE = re.compile(r"tier\(", re.I)


def _first_tier_body(expr: str) -> str | None:
    """取出第一个 tier(...) 的括号内容。括号配平扫，别用正则贪婪。

    注意 `tier(` 自己的那个左括号已经算一层，所以 depth 从 1 起。"""
    if not expr:
        return None
    m = _EXPR_FIRST_TIER_RE.search(expr)
    if not m:
        return None
    depth, start = 1, m.end()
    for j in range(m.end(), len(expr)):
        ch = expr[j]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return expr[start:j]
    return None


def parse_billing_expr(expr: str) -> dict[str, float]:
    """从 billing_expr 里取出**第一档**的 $/1M 系数，返回 {"p":…, "c":…, …}。

    只解析第一档（base）；解析失败返回 {}，让调用方跳过这个模型。
    变量名按长的优先匹配（cc1h 不能被 cc 吃掉）。"""
    body = _first_tier_body(expr)
    if body is None:
        return {}
    out: dict[str, float] = {}
    for var, num in _EXPR_VAR_RE.findall(body):
        out.setdefault(var, _to_float(num))
    for num, var in _EXPR_NUM_FIRST_RE.findall(body):
        out.setdefault(var, _to_float(num))
    return {k: v for k, v in out.items() if v is not None}


def parse_fixed_price(expr: str) -> float | None:
    """`tier("base", fixed(N))` → 每次 N（美元），纯按次计费。

    依据（2026-09-14 实测，不是猜的）：55 站 4 个图片模型的**模型名里就写着价**，
    与表达式一一对应 ——
        [图-次-0.06￥]gpt-image-2-med      → fixed(0.06)
        [图-次-0.09￥]gpt-image-2-max      → fixed(0.09)
        [图-次-0.02￥]gpt-image-2.5        → fixed(0.02)
        [图-次-0.26￥]gpt-image-2-med-124k → fixed(0.26)
    且用户手工表里同一套命名的条目（`逆[kiro3-次-0.05￥]` = $0.05/次、
    `逆[Ag1-次-0.25￥]` = $0.25/次）也逐一对上。所以 「次-N￥」的 N 就是每次的美元价。
    """
    body = _first_tier_body(expr)
    if body is None:
        return None
    m = _EXPR_FIXED_RE.search(body)
    if not m:
        return None
    return _to_float(m.group(1))


def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


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

    # ── 动态计费（billing_expr）────────────────────────────────────────
    # ⚠️ 这类模型**必须看 billing_expr，绝不能看 model_ratio**（2026-09-14 实测踩坑）：
    # 站点标了「动态计费」时，billing_expr 的系数本身就是 $/1M 价格，而
    # model_ratio 是过时的遗留值。反例：小鸡 [Kiro3][手动标记] … 上游定价页写明
    # 输入 $1.875/输出 $9.375，但 model_ratio=37.5 → 按倍率换算成 $75/$375，**错 40 倍**。
    # 解析不出来就**跳过这个模型**（宁可不给价，也绝不给一个错 40 倍的价）。
    if (item.get("billing_mode") or "") == "tiered_expr":
        expr = item.get("billing_expr") or ""
        coeff = parse_billing_expr(expr)
        if not coeff or "p" not in coeff:
            # 没有 token 系数 → 可能是纯按次（fixed(N)）
            fixed = parse_fixed_price(expr)
            if fixed is not None:
                entry.update({
                    "price_type": "per_call",
                    "price_per_call": round(fixed * group_ratio, 6),
                    "input": 0.0, "output": 0.0,
                    "cache_write": 0.0, "cache_read": 0.0,
                    "cache_write_5m": 0.0, "cache_write_1h": 0.0,
                    "billing": "tiered_expr",
                    "billing_expr": expr[:300],
                })
                return entry
            logger.warning(
                "动态计费模型 %s 的 billing_expr 解析不出输入价，跳过（不用 model_ratio 兜底，"
                "那会错几十倍）：%r", model, expr[:120])
            return None
        inp = coeff["p"] * group_ratio
        cw_5m = coeff.get("cc", 0.0) * group_ratio
        cw_1h = coeff.get("cc1h", 0.0) * group_ratio or cw_5m * CACHE_CREATION_1H_MULTIPLIER
        entry.update({
            "price_type": "token",
            "price_per_call": 0.0,
            "input": round(inp, 6),
            "output": round(coeff.get("c", inp) * group_ratio, 6),
            "cache_read": round(coeff.get("cr", 0.0) * group_ratio, 6),
            "cache_write": round(cw_1h if cache_ttl == "1h" else (0.0 if cache_ttl == "off" else cw_5m), 6),
            "cache_write_5m": round(cw_5m, 6),
            "cache_write_1h": round(cw_1h, 6),
            "billing": "tiered_expr",
            "billing_expr": (item.get("billing_expr") or "")[:300],
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

def _reject_reason(url: str, code: int, resp) -> str:
    """把 401/403 翻译成"该站为什么拿不到价、还能不能拿到"。

    2026-09-14 在本机真实上游上实测到的三种情况（都是只靠自己猜绝对猜不到的）：

    | 现象 | 真实原因 | 能不能自动取价 |
    |---|---|---|
    | 403 + `error code: 1010` | **前置 Cloudflare 按 UA 指纹拦截**，与 new-api 无关 | 能——httpx 的 UA 能过，`urllib`/`curl` 的会被拦（**排查时务必用网关自己的客户端试**，别用 curl，否则会误判成站点不支持） |
    | 403 + new-api 的 JSON 错误体 | 站点后台关了 pricing 模块（`HeaderNavModules.pricing.enabled=false`） | 不能，只能手填 |
    | 401 | 站点把 pricing 模块设成了 `requireAuth=true`（`GET /api/status` 里能看到） | 不能（匿名拿不到后台凭据），只能手填 |

    所以这里把响应体片段一起带上——让下一个人一眼看出是哪一种，而不是照着
    "拉取失败"去重试一个永远不会成功的站点。"""
    body = ""
    try:
        body = " ".join((resp.text or "").split())[:160]
    except Exception:
        pass
    if code == 401:
        return (f"{url} 返回 401：该站价格页要求登录（new-api 的 pricing 模块设了 "
                f"requireAuth=true），网关是匿名调用拿不到，这一站只能继续手填价格")
    if "error code: 1" in body or "cloudflare" in body.lower():
        return (f"{url} 返回 403：被前置 Cloudflare 拦截（{body}），不是 new-api 关了"
                f"价格页。httpx 自带 UA 通常能通过，改用 curl/urllib 复现会被拦")
    return (f"{url} 返回 403：站点拒绝匿名访问（new-api 后台可能关闭了价格页模块）。"
            f"响应片段：{body or '(空)'}")


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
            # 401/403 是"这一站换 URL 也没用"，必须当场给出精确定位——它们的原因
            # 完全不同、处置也不同，糊成一句"拉取失败"会把人带偏（2026-09-14 实测
            # 本机 5 个上游里两种都真实出现：ksir 401 / relay-c 403，见下）。
            if code in (401, 403):
                raise RuntimeError(_reject_reason(url, code, exc.response)) from exc
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
