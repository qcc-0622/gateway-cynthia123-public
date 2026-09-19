"""Admin GUI routes: dashboard, conversation logs, upstream management."""

import hashlib
import logging
import time
from collections import defaultdict
from pathlib import Path

import httpx
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from gateway.config import ADMIN_PASSWORD, ADMIN_SECRET_KEY, UPSTREAM_TIMEOUT
from gateway.db import get_stats, get_conversations, get_conversation_messages, get_cache_stats, get_cache_stats_by_model, get_cache_stats_by_day_and_model, list_context_summaries
from gateway import costs
from gateway.upstream import (
    load_upstreams, save_upstreams, add_upstream, remove_upstream, toggle_upstream, toggle_hidden,
    set_default_model, set_cache_ttl, set_pricing_group, update_cached_models, update_primary_cached_models, update_extra_key_cached_models,
)
from gateway.settings import get_settings, update_settings

logger = logging.getLogger("gateway.admin")
router = APIRouter(prefix="/admin")
templates = Jinja2Templates(directory="gateway/templates")

_TOKEN_KEY = "admin_token"
_BASE = "/chat-gateway/admin"

# 登录失败限速：每 IP 最多连续失败 5 次，锁定 10 分钟
_MAX_FAILURES = 5
_LOCKOUT_SECONDS = 600
_login_failures: dict[str, list[float]] = defaultdict(list)  # ip -> [timestamp, ...]


def _get_client_ip(request: Request) -> str:
    return request.headers.get("x-real-ip") or request.headers.get("x-forwarded-for", "").split(",")[0].strip() or "unknown"


def _is_locked(ip: str) -> tuple[bool, int]:
    """Returns (locked, seconds_remaining)."""
    now = time.time()
    failures = [t for t in _login_failures[ip] if now - t < _LOCKOUT_SECONDS]
    _login_failures[ip] = failures
    if len(failures) >= _MAX_FAILURES:
        remaining = int(_LOCKOUT_SECONDS - (now - failures[-_MAX_FAILURES]))
        return True, max(0, remaining)
    return False, 0


def _record_failure(ip: str):
    _login_failures[ip].append(time.time())
    logger.warning("Admin login failure from %s (total recent: %d)", ip, len(_login_failures[ip]))


def _clear_failures(ip: str):
    _login_failures[ip] = []


def _make_token() -> str:
    return hashlib.sha256((ADMIN_SECRET_KEY + "admin").encode()).hexdigest()[:32]


def _check_auth(request: Request):
    if not ADMIN_PASSWORD:
        return True
    token = request.cookies.get(_TOKEN_KEY, "")
    return token == _make_token()


def _t(request: Request, name: str, ctx: dict):
    ctx["admin_base"] = _BASE
    return templates.TemplateResponse(request, name, ctx)


# 模型拉取逻辑已抽到 services/model_sync.py（/v1/models 自动刷新共用，勿在此再实现一份）
from gateway.services.model_sync import (  # noqa: E402
    fetch_models_for_key as _fetch_models_for_key,
    model_headers as _model_headers,
    parse_models_payload as _parse_models_payload,
    refresh_all_upstream_models,
)
# 价格同步同理抽到 services/price_sync.py（后台定时循环共用）
from gateway.services.price_sync import (  # noqa: E402
    refresh_all_upstream_prices,
    sync_upstreams as _sync_upstreams,
)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if not ADMIN_PASSWORD or _check_auth(request):
        return RedirectResponse(f"{_BASE}/", status_code=302)
    return _t(request, "login.html", {"error": ""})


@router.post("/login")
async def login_submit(request: Request, password: str = Form(...)):
    ip = _get_client_ip(request)
    locked, remaining = _is_locked(ip)
    if locked:
        mins = remaining // 60
        secs = remaining % 60
        logger.warning("Blocked login attempt from locked IP %s", ip)
        return _t(request, "login.html", {"error": f"登录失败次数过多，请 {mins}分{secs}秒 后再试"})

    if password != ADMIN_PASSWORD:
        _record_failure(ip)
        _, remaining2 = _is_locked(ip)
        locked2 = remaining2 > 0
        if locked2:
            return _t(request, "login.html", {"error": f"密码错误，IP 已被锁定 10 分钟"})
        attempts_left = _MAX_FAILURES - len(_login_failures[ip])
        return _t(request, "login.html", {"error": f"密码错误（还剩 {attempts_left} 次机会）"})

    _clear_failures(ip)
    logger.info("Admin login success from %s", ip)
    response = RedirectResponse(f"{_BASE}/", status_code=302)
    response.set_cookie(_TOKEN_KEY, _make_token(), max_age=86400 * 7, httponly=True)
    return response


@router.get("/logout")
async def logout():
    response = RedirectResponse(f"{_BASE}/login", status_code=302)
    response.delete_cookie(_TOKEN_KEY)
    return response


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    stats = await get_stats()
    upstreams = load_upstreams()
    active = next((u for u in upstreams if u.is_active), None)
    return _t(request, "dashboard.html", {
        "stats": stats,
        "active_upstream": active,
        "upstream_count": len(upstreams),
    })


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    page = int(request.query_params.get("page", 1))
    search = request.query_params.get("search", "")
    rows, total = await get_conversations(page=page, per_page=20, search=search)
    total_pages = max(1, (total + 19) // 20)
    return _t(request, "logs.html", {
        "conversations": rows,
        "page": page,
        "total_pages": total_pages,
        "total": total,
        "search": search,
    })


@router.get("/timeline", response_class=HTMLResponse)
async def timeline_page(request: Request):
    """全局时间线视图（统一大脑，PLAN 第 5 步）：按到达顺序（id）混排所有
    fingerprint 的归档行，带来源标签。只读查询视图，开关关闭时也能看
    （它只是 conversations 表的另一种展示，不影响任何行为）。"""
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    try:
        limit = int(request.query_params.get("limit", 100) or 100)
    except ValueError:
        limit = 100
    limit = min(500, max(20, limit))

    from gateway.db import get_global_timeline_after, get_global_summary
    from gateway.hooks import _unified_label_for_row
    from gateway.settings import get as _sget

    rows = await get_global_timeline_after(after_id=0, limit=limit)
    items = []
    for r in rows:
        items.append({
            "id": r["id"],
            "role": r["role"],
            "content": r["content"] or "",
            "timestamp": (r["timestamp"] or "")[:19].replace("T", " "),
            "label": _unified_label_for_row(r),
            "tag": r["tag"] or "",
        })
    _, summary_pos = await get_global_summary()
    return _t(request, "timeline.html", {
        "items": items,
        "limit": limit,
        "summary_pos": summary_pos,
        "unified_enabled": bool(_sget("unified_brain_enabled", False)),
    })


@router.get("/logs/{conversation_id}", response_class=HTMLResponse)
async def conversation_detail(request: Request, conversation_id: str):
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    messages = await get_conversation_messages(conversation_id)
    return _t(request, "conversation.html", {
        "conversation_id": conversation_id,
        "messages": messages,
    })


@router.post("/logs/{conversation_id}/delete")
async def delete_conversation(request: Request, conversation_id: str):
    """Delete all rows belonging to a single conversation_id."""
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    from gateway.db import get_db
    db = get_db()
    cur = await db.execute(
        "DELETE FROM conversations WHERE conversation_id=?", (conversation_id,)
    )
    await db.commit()
    logger.info("Deleted conversation %s (%d rows)", conversation_id, cur.rowcount)
    return RedirectResponse(f"{_BASE}/logs?deleted={cur.rowcount}", status_code=302)


@router.post("/logs/delete-range")
async def delete_conversations_range(request: Request):
    """Delete conversations within a time range. Form params: start, end (ISO8601 or YYYY-MM-DDTHH:MM)."""
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    form = await request.form()
    start = form.get("start", "").strip()
    end = form.get("end", "").strip()
    if not start and not end:
        return RedirectResponse(f"{_BASE}/logs?err=missing_range", status_code=302)
    from gateway.db import get_db
    db = get_db()
    where = []
    params: list = []
    if start:
        where.append("timestamp >= ?")
        params.append(start)
    if end:
        where.append("timestamp < ?")
        params.append(end)
    sql = f"DELETE FROM conversations WHERE {' AND '.join(where)}"
    cur = await db.execute(sql, params)
    await db.commit()
    logger.info("Deleted %d rows in range [%s, %s)", cur.rowcount, start, end)
    return RedirectResponse(f"{_BASE}/logs?deleted={cur.rowcount}", status_code=302)


@router.get("/upstreams", response_class=HTMLResponse)
async def upstreams_page(request: Request):
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    upstreams = load_upstreams()
    return _t(request, "upstreams.html", {
        "upstreams": upstreams,
        "error": "",
        "success": "",
    })


@router.post("/upstreams/add")
async def add_upstream_handler(
    request: Request,
    name: str = Form(...),
    base_url: str = Form(...),
    api_key: str = Form(...),
    api_format: str = Form("anthropic"),
):
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    try:
        add_upstream(name, base_url, api_key, api_format)
        logger.info("Added upstream: %s", name)
    except ValueError as e:
        upstreams = load_upstreams()
        return _t(request, "upstreams.html", {
            "upstreams": upstreams,
            "error": str(e),
            "success": "",
        })
    return RedirectResponse(f"{_BASE}/upstreams", status_code=302)


@router.post("/upstreams/{name}/delete")
async def delete_upstream_handler(request: Request, name: str):
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    remove_upstream(name)
    logger.info("Removed upstream: %s", name)
    return RedirectResponse(f"{_BASE}/upstreams", status_code=302)


@router.post("/upstreams/{name}/toggle")
async def toggle_upstream_handler(request: Request, name: str):
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    toggle_upstream(name)
    logger.info("Toggled upstream: %s", name)
    return RedirectResponse(f"{_BASE}/upstreams", status_code=302)


@router.post("/upstreams/refresh-all-models")
async def refresh_all_models(request: Request):
    """一次性从所有 active 上游拉取模型列表，更新缓存。
    返回 {upstream_name: {groups: [...]}} JSON 摘要。
    实现在 services/model_sync.py（与 /v1/models 自动刷新共用）。"""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return JSONResponse(await refresh_all_upstream_models())


@router.post("/upstreams/refresh-all-prices")
async def refresh_all_prices(request: Request):
    """从所有 active 上游的 new-api /api/pricing 拉最新价格并落盘。
    返回 {upstream_name: {ok, count, group, group_ratio, error, ...}} JSON 摘要。
    实现在 services/price_sync.py（与后台定时循环共用）。
    失败的上游会保留上一轮价格（meta.stale=True），不会让整站价格凭空消失。"""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return JSONResponse(await refresh_all_upstream_prices())


@router.post("/upstreams/{name}/refresh-prices")
async def refresh_one_price(request: Request, name: str):
    """只同步某一个上游的价格（其它上游的条目原样保留）。"""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    result = await _sync_upstreams([name])
    if name not in result:
        return JSONResponse({"error": f"上游 {name} 不存在或未启用"}, status_code=404)
    return JSONResponse(result[name])


@router.post("/upstreams/{name}/pricing-group")
async def set_pricing_group_handler(
    request: Request,
    name: str,
    pricing_group: str = Form(""),
):
    """设置该上游的分组名（决定自动价格的 group_ratio 倍率）。
    设置页是整页一个大 form，塞不进嵌套 form，所以这里返回 JSON 由 JS 调。"""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    set_pricing_group(name, pricing_group)
    logger.info("Upstream %s pricing_group = %r", name, pricing_group)
    return JSONResponse({"ok": True, "name": name, "pricing_group": pricing_group.strip()})


@router.get("/upstreams/{name}/models")
async def fetch_models(request: Request, name: str):
    """从上游拉取模型列表，返回 JSON。"""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    upstreams = load_upstreams()
    upstream = next((u for u in upstreams if u.name == name), None)
    if not upstream:
        return JSONResponse({"error": "Upstream not found"}, status_code=404)

    headers = _model_headers(upstream.api_key, upstream.api_format)

    # 2026-09-08: use shared client + detailed diagnostics (fixes SSL issue with relay-c)
    import traceback as _tb
    from gateway.http_client import get_client, proxy_hint

    async def _try_shared():
        client = get_client()
        resp = await client.get(upstream.models_url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    async def _try_system_ca():
        async with httpx.AsyncClient(timeout=15, verify="/etc/ssl/certs/ca-certificates.crt") as c:
            resp = await c.get(upstream.models_url, headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def _try_no_verify():
        async with httpx.AsyncClient(timeout=15, verify=False) as c:
            resp = await c.get(upstream.models_url, headers=headers)
            resp.raise_for_status()
            return resp.json()

    data = None
    for label, fn in [("shared_client", _try_shared), ("system_ca", _try_system_ca), ("no_verify", _try_no_verify)]:
        try:
            data = await fn()
            logger.info("fetch_models %s: OK via %s", name, label)
            break
        except Exception as e:
            # proxy_hint：三级 fallback 全失败最常见的原因不是证书，而是该域名
            # 没进 .env 的 NO_PROXY 白名单 → httpx trust_env 把请求送进代理
            # （ConnectError + str 为空 + traceback 落在 http_proxy.py 就是这个症状）
            hint = proxy_hint(upstream.models_url)
            logger.warning(
                "fetch_models %s: %s failed: %s(%s)%s\n%s",
                name, label, type(e).__name__, e,
                ("\n" + hint) if hint else "",
                _tb.format_exc()
            )
    if data is None:
        hint = proxy_hint(upstream.models_url)
        return JSONResponse({"error": f"All fetch attempts failed for {name}"
                                      + (f"｜{hint}" if hint else "")})

    primary_models = sorted(_parse_models_payload(data) - {""})
    groups = [{
        "key_index": "primary",
        "label": "主令牌",
        "purpose": "chat",
        "models": primary_models,
        "count": len(primary_models),
    }]
    all_model_ids = set(primary_models)
    update_primary_cached_models(name, primary_models)

    # 额外令牌：分别拉取，分别缓存，让前端能看见 key 分组
    for idx, ek in enumerate(upstream.extra_keys or []):
        if not (isinstance(ek, dict) and ek.get("key")):
            continue
        ek_models = sorted(await _fetch_models_for_key(upstream.models_url, ek["key"], upstream.api_format))
        update_extra_key_cached_models(name, idx, ek_models)
        all_model_ids |= set(ek_models)
        groups.append({
            "key_index": idx,
            "label": ek.get("label") or f"额外令牌 {idx + 1}",
            "purpose": ek.get("purpose", "chat"),
            "models": ek_models,
            "count": len(ek_models),
        })

    models = sorted(all_model_ids - {""})

    # 缓存到本地
    update_cached_models(name, models)
    return JSONResponse({"models": models, "groups": groups})


def _aggregate_day_model_rows(rows, cost_fn) -> tuple[list[dict], dict]:
    """day×model 统计行 → （每日明细列表, 总计 dict）。

    每日条目附带 models 列表（当天各 中转站×模型 的 messages/cost/avg_cost，
    按费用降序），供监控页"每日明细"行展开看当天各模型费用（2026-09-06 用户需求：
    "已经有每天用了多少钱，想细化到每个模型用了多少钱，均价多少"）。
    cost_fn(upstream, model, inp, out, cw, cr, msgs) -> (cost, saved)。
    纯函数，不碰 db/settings，方便单测（tests/test_monitoring_breakdown.py）。
    """
    daily_map: dict[str, dict] = {}
    totals = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0, "messages": 0,
              "cost": 0.0, "saved": 0.0}
    for r in rows:
        day = r["day"]
        upstream_name = r["upstream"]
        model_name = r["model"]
        inp = r["input_tokens"] or 0
        out = r["output_tokens"] or 0
        cw  = r["cache_write"]   or 0
        cr  = r["cache_read"]    or 0
        msgs = r["messages"]    or 0
        cost, saved = cost_fn(upstream_name, model_name, inp, out, cw, cr, msgs)

        d = daily_map.setdefault(day, {
            "day": day, "input": 0, "output": 0,
            "cache_write": 0, "cache_read": 0,
            "messages": 0, "cost": 0.0, "saved": 0.0,
            "models": [],
        })
        d["models"].append({
            "upstream": upstream_name,
            "model": model_name,
            "messages": msgs,
            "input": inp, "output": out,
            "cache_write": cw, "cache_read": cr,
            "cost": cost, "saved": saved,
            "avg_cost": round(cost / msgs, 4) if msgs else 0,
        })
        d["input"]       += inp
        d["output"]      += out
        d["cache_write"] += cw
        d["cache_read"]  += cr
        d["messages"]    += msgs
        d["cost"]        += cost
        d["saved"]       += saved

        totals["input"]       += inp
        totals["output"]      += out
        totals["cache_write"] += cw
        totals["cache_read"]  += cr
        totals["messages"]    += msgs
        totals["cost"]        += cost
        totals["saved"]       += saved

    # 整理 daily 列表（按日期倒序）
    daily = []
    for day in sorted(daily_map.keys(), reverse=True):
        d = daily_map[day]
        tc = d["input"] + d["cache_write"] + d["cache_read"]
        d["hit_rate"] = round(d["cache_read"] / tc * 100, 1) if tc else 0
        tr = d["input"] + d["cache_read"]
        d["reuse_rate"] = round(d["cache_read"] / tr * 100, 1) if tr else 0
        d["avg_cost"] = round(d["cost"] / d["messages"], 4) if d["messages"] else 0
        d["models"].sort(key=lambda m: m["cost"], reverse=True)
        for m in d["models"]:
            m["cost"] = round(m["cost"], 4)
            m["saved"] = round(m["saved"], 4)
        d["cost"] = round(d["cost"], 4)
        d["saved"] = round(d["saved"], 4)
        daily.append(d)
    return daily, totals


@router.get("/monitoring", response_class=HTMLResponse)
async def monitoring(request: Request):
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    days = int(request.query_params.get("days", 7))
    settings = get_settings()

    # 取价唯一真相源在 gateway/costs.py（手工配置 > 上游自动同步 > 内置默认）。
    # 这里只在它上面包一层"给页面看的形状"：_price_entries 是手工价目表（用于
    # 下方"已配置的所有价格"表格），实际算钱一律走 costs.resolve_price，
    # 免得监控页再维护第二份匹配/算术（旧版就是两份，改一处忘一处）。
    _price_entries: list[dict] = []
    for mp in (settings.get("model_prices") or []):
        # 新 schema: upstream + model；旧 schema: name 仅按 substring 匹配
        ups = (mp.get("upstream") or "").strip()
        model = (mp.get("model") or mp.get("name") or "").strip()
        if not model:
            continue
        _price_entries.append({
            "upstream":      ups,
            "model":         model,
            "price_type":    mp.get("price_type", "token"),   # "token" | "per_call"
            "price_per_call": float(mp.get("price_per_call", 0) or 0),
            "prices": costs.price_from_entry(mp),
        })

    _auto = costs.load_auto_prices()
    _auto_count = len(_auto.get("prices") or [])

    def _price_for(upstream_name: str, model_name: str) -> dict:
        entry, _src = costs.resolve_price(upstream_name, model_name)
        return costs.price_from_entry(entry)

    def _cost_for(upstream_name: str, model_name: str,
                  inp: int, out: int, cw: int, cr: int, msgs: int) -> tuple[float, float]:
        """返回 (cost, saved)，自动区分按次/按Token。"""
        entry, _src = costs.resolve_price(upstream_name, model_name)
        return costs.compute_cost(entry, inp, out, cw, cr, msgs)

    def _source_for(upstream_name: str, model_name: str) -> tuple[str, str]:
        """返回 (来源标识, 来源中文标签)。"""
        _entry, src = costs.resolve_price(upstream_name, model_name)
        return src, costs.SOURCE_LABEL.get(src, src)

    def _auto_price_for(upstream_name: str, model_name: str) -> dict | None:
        """该模型在上游自动价目表里的条目（不看优先级，纯粹"拉到了没有"）。

        为什么监控页要单独把自动价摊出来（2026-09-14 用户反馈"拉了价但价格没更新"）：
        实际用到的模型大多已经被手工价覆盖，而取价优先级是手工 > 自动，所以页面上
        数字一个都不变——这本身是对的（手工价往往更懂上游的按次促销变体），但用户
        看不到"拉回来的价长什么样、跟手工价差多少"，就会以为同步没生效。
        摊出来后，分组倍率配错导致的系统性偏差（实测有差 5 倍的）也能一眼看出来。"""
        return costs.find_auto_entry(upstream_name, model_name)

    # ── 用 day × model 数据精确算每日和总费用 ──
    # 旧实现是用单一 PRICE 算所有数据，多模型混用时不准
    day_model_rows = await get_cache_stats_by_day_and_model(days=days)

    # 按日聚合（费用 = 各模型实际单价相加）；每天附带各模型费用明细（每日行可展开）
    daily, totals = _aggregate_day_model_rows(day_model_rows, _cost_for)

    total_cost = round(totals["cost"], 4)
    total_saved = round(totals["saved"], 4)
    tc = totals["input"] + totals["cache_write"] + totals["cache_read"]
    total_hit_rate = round(totals["cache_read"] / tc * 100, 1) if tc else 0
    tr = totals["input"] + totals["cache_read"]
    total_reuse_rate = round(totals["cache_read"] / tr * 100, 1) if tr else 0
    avg_cost_per_msg = round(total_cost / totals["messages"], 4) if totals["messages"] else 0
    has_cache_data = totals["cache_write"] > 0 or totals["cache_read"] > 0

    # 按模型聚合统计（用户切多个模型，想看每个模型的缓存健康度）
    model_rows = await get_cache_stats_by_model(days=days)
    models = []
    for row in model_rows:
        inp = row["input_tokens"] or 0
        out = row["output_tokens"] or 0
        cw  = row["cache_write"]   or 0
        cr  = row["cache_read"]    or 0
        msgs = row["messages"]     or 0
        if msgs == 0:
            continue
        display_name = row["display_model"]
        upstream_model_name = row["upstream_model"]
        upstream_name = row["upstream"]
        cost, saved = _cost_for(upstream_name, display_name, inp, out, cw, cr, msgs)
        price_source, price_source_label = _source_for(upstream_name, display_name)
        tc = inp + cw + cr
        hit_rate = round(cr / tc * 100, 1) if tc else 0
        tr = inp + cr
        reuse_rate = round(cr / tr * 100, 1) if tr else 0
        avg_cost = round(cost / msgs, 4) if msgs else 0
        # 缓存"健康度"评分：综合命中率 + 是否有缓存写入
        if cw == 0 and cr == 0:
            health = "no-cache"  # 完全没缓存
        elif hit_rate >= 70:
            health = "excellent"
        elif hit_rate >= 40:
            health = "good"
        elif hit_rate >= 15:
            health = "weak"
        else:
            health = "poor"

        # 上游自动价（如果有）：摊出来给用户看"拉回来的价长什么样"，以及跟当前
        # 生效价差多少——分组倍率配错会让整站自动价系统性偏移，这里能一眼看出。
        _auto_entry = _auto_price_for(upstream_name, display_name)
        _auto_text, _auto_gap = "", ""
        if _auto_entry:
            _ap = costs.price_from_entry(_auto_entry)
            if _auto_entry.get("price_type") == "per_call":
                _auto_text = "$%g/次" % float(_auto_entry.get("price_per_call") or 0)
            else:
                _auto_text = "入 $%g / 出 $%g" % (_ap["input"], _ap["output"])
            if price_source == costs.SOURCE_MANUAL:
                _used_entry, _ = costs.resolve_price(upstream_name, display_name)
                if _used_entry and (_used_entry.get("price_type") or "token") == "token" \
                        and (_auto_entry.get("price_type") or "token") == "token":
                    _up_in = float(_used_entry.get("input") or 0)
                    if _ap["input"] > 0 and _up_in > 0:
                        _r = _up_in / _ap["input"]
                        if 0.9 <= _r <= 1.1:
                            _auto_gap = "与手工价一致"
                        else:
                            _auto_gap = "手工价是它的 %.2f×" % _r

        models.append({
            "upstream": upstream_name,                                  # 中转站名
            "model": display_name,                                      # 客户端模型名
            "upstream_model": upstream_model_name,                      # 上游返回的真实模型名
            "model_differs": display_name != upstream_model_name,
            "messages": msgs,
            "input": inp, "output": out,
            "cache_write": cw, "cache_read": cr,
            "hit_rate": hit_rate,
            "reuse_rate": reuse_rate,
            "cost": round(cost, 4),
            "saved": round(saved, 4),
            "avg_cost": avg_cost,
            "health": health,
            "last_used": (row["last_used"] or "")[:19].replace("T", " "),
            "price": _price_for(upstream_name, display_name),  # 当前生效价格
            # 取价来源：manual / auto / default。模板据此决定显示"已定价"
            # 还是"⚠️ 没价"，以及这一行的钱是按哪个价算的。
            "price_source": price_source,
            "price_source_label": price_source_label,
            "has_price": price_source != costs.SOURCE_DEFAULT,
            # 上游拉回来的自动价（None = 该站价格表里没这个模型名）。
            # 手工价压着它时，模板会把两者一起显示出来，便于发现偏差。
            "auto_price_text": _auto_text,
            "auto_gap": _auto_gap,
        })

    # Token 分布面板的估算单价：取"token 量最大"的那个模型的实际生效价，
    # 这样面板里的分项金额能和按模型汇总的总费用大致对上。旧版固定用
    # _price_entries[0]（第一条手工规则），一旦开了自动取价、手工表是空的，
    # 面板会退回 Opus 默认价，和下面的真实费用对不上。
    _dominant_key, _dominant_tokens = None, -1
    for row in model_rows:
        _tok = sum(row[k] or 0 for k in
                   ("input_tokens", "output_tokens", "cache_write", "cache_read"))
        if _tok > _dominant_tokens:
            _dominant_key = (row["upstream"], row["display_model"])
            _dominant_tokens = _tok
    if _dominant_key:
        PRICE = _price_for(*_dominant_key)
        PRICE_LABEL = f"{_dominant_key[0]} / {_dominant_key[1]}"
    else:
        PRICE = _price_entries[0]["prices"] if _price_entries else costs.price_from_entry(None)
        PRICE_LABEL = "默认价" if not _price_entries else "手工配置"

    # has_custom_price 保留旧键名（模板多处引用），语义放宽为"有价可依"：
    # 手工配置或上游自动同步到价都算，不再把自动同步的模型误报成"未配置"。
    has_custom_price = bool(_price_entries) or _auto_count > 0

    return _t(request, "monitoring.html", {
        "daily": daily,
        "totals": totals,
        "total_cost": round(total_cost, 4),
        "total_saved": round(total_saved, 4),
        "total_hit_rate": total_hit_rate,
        "total_reuse_rate": total_reuse_rate,
        "avg_cost_per_msg": avg_cost_per_msg,
        "days": days,
        "PRICE": PRICE,
        "PRICE_LABEL": PRICE_LABEL,
        "has_cache_data": has_cache_data,
        "has_custom_price": has_custom_price,
        "models": models,
        "price_entries": _price_entries,
        "auto_price": {
            "enabled": bool(settings.get("auto_price_enabled", False)),
            "count": _auto_count,
            "synced_at": _auto.get("synced_at", ""),
            "upstreams": _auto.get("upstreams", {}),
            "file": str(costs.AUTO_PRICES_FILE),
        },
    })


@router.post("/clear-zero-cache", response_class=HTMLResponse)
async def clear_zero_cache(request: Request):
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    from gateway.db import get_db
    db = get_db()
    cur = await db.execute("DELETE FROM conversations")
    deleted = cur.rowcount
    await db.commit()
    logger.info("Cleared all %d conversation records", deleted)
    return RedirectResponse(f"{_BASE}/monitoring?cleared={deleted}", status_code=302)


@router.get("/gateway-settings", response_class=HTMLResponse)
async def gateway_settings_page(request: Request):
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    settings = get_settings()
    upstreams = load_upstreams()
    # 扁平去重列表（向后兼容）
    all_models: list[str] = sorted(set(
        m for u in upstreams for m in (u.cached_models or [])
    ))
    # 带上游前缀的分组列表：[{name, models: ["upstream::model", ...]}, ...]
    grouped_models = []
    for u in upstreams:
        if u.primary_cached_models:
            grouped_models.append({
                "name": f"{u.name} / 主令牌",
                "models": sorted(f"{u.name}::{m}" for m in (u.primary_cached_models or [])),
            })
        for idx, key_info in enumerate(u.extra_keys or []):
            if not isinstance(key_info, dict):
                continue
            models = key_info.get("cached_models") or key_info.get("models") or []
            if models:
                label = key_info.get("label") or f"令牌 {idx + 1}"
                grouped_models.append({
                    "name": f"{u.name} / {label}",
                    "models": sorted(f"{u.name}::{m}" for m in models),
                })
        if not u.primary_cached_models and u.cached_models:
            grouped_models.append({
                "name": u.name,
                "models": sorted(f"{u.name}::{m}" for m in (u.cached_models or [])),
            })
    _auto = costs.load_auto_prices()
    return _t(request, "gateway_settings.html", {
        "settings": settings,
        "upstreams": upstreams,
        "all_models": all_models,
        "grouped_models": grouped_models,
        "saved": request.query_params.get("saved") == "1",
        "auto_price": {
            "count": len(_auto.get("prices") or []),
            "synced_at": _auto.get("synced_at", ""),
            "upstreams": _auto.get("upstreams", {}),
        },
    })


@router.post("/gateway-settings")
async def gateway_settings_save(request: Request):
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    form = await request.form()

    summary_model    = str(form.get("summary_model", "")).strip()
    summary_upstream = str(form.get("summary_upstream", "")).strip()
    summary_prompt   = str(form.get("summary_prompt", "")).strip()
    bp3_cycle_size   = max(3, int(form.get("bp3_cycle_size", 9) or 9))
    bp3_frozen_rounds = max(1, int(form.get("bp3_frozen_rounds", 8) or 8))
    bp3_live_rounds   = max(1, int(form.get("bp3_live_rounds", 24) or 24))

    # Parse model price rows (multi-value fields).
    # New schema: upstream + model + 4 prices; both upstream and model are
    # exact-match strings. upstream can be empty (matches any upstream).
    upstreams_in = form.getlist("price_upstream[]")
    models_in    = form.getlist("price_model[]")
    inputs       = form.getlist("price_input[]")
    outputs      = form.getlist("price_output[]")
    cws          = form.getlist("price_cw[]")
    crs          = form.getlist("price_cr[]")
    model_prices = []
    for i, model in enumerate(models_in):
        model = (model or "").strip()
        if not model:
            continue
        ups = (upstreams_in[i] if i < len(upstreams_in) else "").strip()
        try:
            _ptypes = form.getlist("price_type[]")
            _ppc    = form.getlist("price_per_call[]")
            model_prices.append({
                "upstream":      ups,
                "model":         model,
                "price_type":    (_ptypes[i] if i < len(_ptypes) else "token") or "token",
                "price_per_call": float(_ppc[i] if i < len(_ppc) else 0 or 0),
                "input":       float(inputs[i]  or 0),
                "output":      float(outputs[i] or 0),
                "cache_write": float(cws[i]     or 0),
                "cache_read":  float(crs[i]     or 0),
            })
        except (IndexError, ValueError):
            pass

    # Memory recall (ombre integration) fields
    def _float_or(default, key):
        try:
            return float(form.get(key, default))
        except (ValueError, TypeError):
            return default

    def _int_or(default, key):
        try:
            return int(form.get(key, default))
        except (ValueError, TypeError):
            return default

    update_settings(
        summary_model=summary_model,
        summary_upstream=summary_upstream,
        summary_prompt=summary_prompt,
        bp3_cycle_size=bp3_cycle_size,
        bp3_frozen_rounds=bp3_frozen_rounds,
        bp3_live_rounds=bp3_live_rounds,
        model_prices=model_prices,
        memory_recall_enabled=    bool(form.get("memory_recall_enabled")),
        memory_autofeed_enabled=  bool(form.get("memory_autofeed_enabled")),
        ombre_url=                str(form.get("ombre_url", "")).strip() or "http://127.0.0.1:18001/mcp",
        ombre_token=              str(form.get("ombre_token", "")).strip(),
        ombre_buckets_dir=        str(form.get("ombre_buckets_dir", "")).strip() or "/opt/ombre-brain/buckets",
        embed_url=                str(form.get("embed_url", "")).strip() or "https://api.siliconflow.cn/v1/embeddings",
        embed_model=              str(form.get("embed_model", "")).strip() or "BAAI/bge-m3",
        embed_api_key=            str(form.get("embed_api_key", "")).strip(),
        memory_new_topic_threshold=_float_or(0.55, "memory_new_topic_threshold"),
        memory_max_recall=        _int_or(4, "memory_max_recall"),
        memory_random_k=          _int_or(2, "memory_random_k"),
        # 通知推送
        notify_pushplus_token=        str(form.get("notify_pushplus_token", "")).strip(),
        notify_ntfy_url=              str(form.get("notify_ntfy_url", "")).strip(),
        notify_balance_threshold=     _float_or(5.0, "notify_balance_threshold"),
        notify_balance_interval_hours=_int_or(6,   "notify_balance_interval_hours"),
        # 上游价格自动同步（new-api /api/pricing）—— 注意：这里只存"开关/间隔"，
        # 同步结果本身在 data/auto_prices.json（机器生成，不做成配置项）
        auto_price_enabled=       bool(form.get("auto_price_enabled")),
        auto_price_interval_hours=max(1, _int_or(12, "auto_price_interval_hours")),
        # 统一大脑（PLAN_UNIFIED_BRAIN.md）：checkbox 不勾时浏览器不提交该字段，
        # bool(form.get(...)) 自然为 False —— 和 memory_recall_enabled 同一模式
        upstream_failover_enabled=bool(form.get("upstream_failover_enabled")),
        unified_brain_enabled=    bool(form.get("unified_brain_enabled")),
        unified_summary_interval= max(5, _int_or(30, "unified_summary_interval")),
        unified_tail_max=         max(10, _int_or(60, "unified_tail_max")),
    )
    logger.info("Gateway settings updated: model_prices=%d entries", len(model_prices))
    return RedirectResponse(f"{_BASE}/gateway-settings?saved=1", status_code=302)


@router.get("/summaries", response_class=HTMLResponse)
async def summaries_page(request: Request):
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    rows = await list_context_summaries()
    # 把 created_at 从 UTC 换算到本地时区显示
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from gateway.config import TIMEZONE
    tz = ZoneInfo(TIMEZONE)
    formatted = []
    for r in rows:
        d = dict(r)
        ca = r["created_at"] or ""
        if ca:
            try:
                # SQLite 存的是 datetime.utcnow().isoformat()，无时区信息
                dt_utc = datetime.fromisoformat(ca).replace(tzinfo=ZoneInfo("UTC"))
                dt_local = dt_utc.astimezone(tz)
                d["created_at_local"] = dt_local.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                d["created_at_local"] = ca[:19].replace("T", " ")
        else:
            d["created_at_local"] = ""
        formatted.append(d)
    # 按 conv_fingerprint 分组，组内按 messages_covered 升序（旧→新）
    from collections import defaultdict
    _groups: dict = defaultdict(list)
    for d in formatted:
        _groups[d["conv_fingerprint"]].append(d)

    # 查询每个 fp 的首末 user 消息预览
    from gateway.db import get_db as _get_db
    _db = _get_db()
    _SKIP_PREFIX = "[以下是我们之前对话的摘要"

    async def _get_user_previews(fp: str):
        """返回 (first_preview, last_preview)，各截 50 字符"""
        # 首条 user 消息（跳过网关注入的摘要前缀）
        first_preview = ""
        cursor = await _db.execute(
            "SELECT content FROM conversations WHERE fingerprint=? AND role='user' ORDER BY id ASC LIMIT 5",
            (fp,),
        )
        rows = await cursor.fetchall()
        for row in rows:
            text = (row[0] or "").strip()
            if not text.startswith(_SKIP_PREFIX):
                first_preview = text[:50]
                break

        # 末条 user 消息
        last_preview = ""
        cursor = await _db.execute(
            "SELECT content FROM conversations WHERE fingerprint=? AND role='user' ORDER BY id DESC LIMIT 1",
            (fp,),
        )
        row = await cursor.fetchone()
        if row:
            last_preview = (row[0] or "").strip()[:50]

        return first_preview, last_preview

    grouped_summaries = []
    for fp, items in _groups.items():
        # 按 messages_covered 升序排，计算每个 cycle 的轮次范围
        items_asc = sorted(items, key=lambda x: x.get("messages_covered") or 0)
        prev_covered = 0
        for item in items_asc:
            cur_covered = item.get("messages_covered") or 0
            item["round_start"] = int(prev_covered // 2) + 1
            item["round_end"]   = int(cur_covered // 2)
            prev_covered = cur_covered

        # 展示时仍按 messages_covered 降序（新 cycle 在前）
        items_sorted = sorted(items_asc, key=lambda x: x.get("messages_covered") or 0, reverse=True)

        latest_raw   = max((i.get("created_at") or "") for i in items)
        latest_local = max((i.get("created_at_local") or "") for i in items)

        # 查询首末 user 消息预览
        first_preview, last_preview = await _get_user_previews(fp)

        grouped_summaries.append({
            "fingerprint":         fp,
            "latest_raw":          latest_raw,
            "latest_local":        latest_local,
            "count":               len(items),
            "total_chars":         sum(len(i.get("summary") or "") for i in items),
            "summaries":           items_sorted,
            "first_user_preview":  first_preview,
            "last_user_preview":   last_preview,
        })
    # 最新 session 在最前
    grouped_summaries.sort(key=lambda x: x["latest_raw"], reverse=True)

    return _t(request, "summaries.html", {
        "summaries": formatted,           # 保留兼容
        "grouped_summaries": grouped_summaries,
    })


@router.post("/summaries/{conv_fingerprint}/{cycle}/delete")
async def delete_summary(request: Request, conv_fingerprint: str, cycle: int):
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    from gateway.db import get_db
    db = get_db()
    await db.execute(
        "DELETE FROM context_summaries WHERE conv_fingerprint=? AND freeze_cycle=?",
        (conv_fingerprint, cycle),
    )
    await db.commit()
    logger.info("Deleted BP2 summary: %s cycle %d", conv_fingerprint, cycle)
    return RedirectResponse(f"{_BASE}/summaries", status_code=302)


@router.post("/summaries/{conv_fingerprint}/{cycle}/edit")
async def edit_summary(request: Request, conv_fingerprint: str, cycle: int):
    """Save manually edited summary text."""
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    form = await request.form()
    new_text = form.get("summary", "").strip()
    if not new_text:
        return RedirectResponse(f"{_BASE}/summaries?err=empty", status_code=302)
    from gateway.db import get_db
    db = get_db()
    cur = await db.execute(
        "UPDATE context_summaries SET summary=?, model=? WHERE conv_fingerprint=? AND freeze_cycle=?",
        (new_text, "manual-edit", conv_fingerprint, cycle),
    )
    await db.commit()
    logger.info("Edited BP2 summary: %s cycle %d (%d chars, %d rows updated)",
                conv_fingerprint, cycle, len(new_text), cur.rowcount)
    return RedirectResponse(f"{_BASE}/summaries?edited=1", status_code=302)



@router.get("/notifications-api")
async def notifications_api(request: Request):
    """从日志文件提取重要通知，返回 JSON。"""
    import re as _re
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    log_file = Path(__file__).parent.parent.parent / "data" / "gateway.log"
    events = []

    if not log_file.exists():
        return JSONResponse([])

    try:
        text = log_file.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()[-3000:]
    except Exception:
        return JSONResponse([])

    # 解析日志行
    # 格式: 2026-05-30 01:13:25,621 LEVEL module: message
    LINE_RE = _re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ (\w+) ([\w\.]+): (.+)$"
    )

    for raw in lines:
        m = LINE_RE.match(raw.strip())
        if not m:
            # 可能是多行错误续行，附到上一条
            if events and raw.strip():
                events[-1]["detail"] = (events[-1].get("detail","") + " " + raw.strip())[:400]
            continue

        ts, level, module, msg = m.group(1), m.group(2), m.group(3), m.group(4)

        # 跳过太频繁的无用 WARNING
        if "ombre.breath timed out" in msg:
            continue
        if level == "DEBUG":
            continue
        # 跳过纯 http 请求日志（200 OK）
        if "HTTP/1.1 200" in msg or "HTTP/1.1 304" in msg or "HTTP/1.1 302" in msg:
            continue

        category = None
        icon = ""
        importance = 0   # 越大越重要

        # 429 限流
        if "429" in msg:
            category = "rate_limit"
            icon = "🚫"
            importance = 3
            short = "上游限流 429"
        # 503 / 不可用
        elif "503" in msg or "No available channel" in msg or "model_not_found" in msg:
            category = "unavailable"
            icon = "⛔"
            importance = 3
            short = "上游不可用"
        # 余额
        elif "balance_monitor" in module and "balance" in msg:
            category = "balance"
            icon = "💰"
            importance = 2
            # 提取金额
            bm = _re.search(r"balance (.+?): ([\d.]+) (\w+).*?threshold=([\d.]+)", msg)
            if bm:
                name, val, unit, thr = bm.group(1), float(bm.group(2)), bm.group(3), float(bm.group(4))
                short = f"余额 {name}: {val} {unit}"
                importance = 4 if val < thr else 1
                category = "balance_low" if val < thr else "balance_ok"
                icon = "⚠️" if val < thr else "💰"
            else:
                short = "余额检查"
        # BP2 摘要开始
        elif "summarizer" in module and "Summarizing (stream)" in msg:
            category = 'bp2_start'
            icon = '⏳'
            importance = 1
            bm = _re.search(r"upstream='([^']+)'.*model='([^']+)'.*new_turns=(\d+)", msg)
            if bm:
                short = f"BP2 摘要生成中 via {bm.group(1)} · {bm.group(2)} · {bm.group(3)}轮"
            else:
                short = 'BP2 摘要生成中'
        # BP2 摘要完成
        elif "summarizer" in module and "Generated BP2 summary" in msg:
            category = "bp2_done"
            icon = "✅"
            importance = 1
            bm = _re.search(r"Generated BP2 summary \((\d+) chars.*?\) from (\d+) turns", msg)
            short = f"BP2 摘要完成 {bm.group(2)}轮→{int(bm.group(1))//1000}k字" if bm else "BP2 摘要完成"
        # 服务重启
        elif module == "gateway" and ("Gateway stopped" in msg or "Gateway started" in msg or "Starting chat gateway" in msg):
            category = "restart"
            icon = "🔄"
            importance = 2
            short = "服务重启" if "stopped" in msg or "started" in msg else "网关启动"
        # 摘要失败
        elif "summarizer" in module and "error" in msg.lower() and level == "ERROR":
            category = "summary_fail"
            icon = "❌"
            importance = 3
            short = "摘要请求失败"
        # 其他 ERROR
        elif level == "ERROR":
            category = "error"
            icon = "🔴"
            importance = 3
            short = msg[:60]
        # 其他 WARNING（去掉内存类）
        elif level == "WARNING" and "memory" not in module:
            category = "warning"
            icon = "⚠️"
            importance = 2
            short = msg[:60]
        else:
            continue

        events.append({
            "ts": ts,
            "level": level,
            "module": module,
            "category": category,
            "icon": icon,
            "short": short,
            "detail": msg[:300],
            "importance": importance,
        })

    # 最新的排前面，最多 60 条
    events = events[-60:]
    events.reverse()
    return JSONResponse(events)


@router.get("/system-logs", response_class=HTMLResponse)
async def system_logs(request: Request):
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    log_file = Path(__file__).parent.parent.parent / "data" / "gateway.log"
    # 支持 ?n=行数 参数
    try:
        n = min(int(request.query_params.get("n", 500)), 5000)
    except (ValueError, TypeError):
        n = 500
    lines = []
    total_lines = 0
    if log_file.exists():
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
            all_lines = text.splitlines()
            total_lines = len(all_lines)
            lines = all_lines[-n:]
        except Exception as e:
            lines = [f"读取日志失败: {e}"]
    return _t(request, "system_logs.html", {
        "lines": lines,
        "total_lines": total_lines,
        "n": n,
    })


@router.post("/upstreams/{name}/set-model")
async def set_model_handler(request: Request, name: str, model: str = Form(...)):
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    set_default_model(name, model)
    logger.info("Set default model for %s: %s", name, model)
    return RedirectResponse(f"{_BASE}/upstreams", status_code=302)


@router.get("/doc", response_class=HTMLResponse)
async def gateway_doc(request: Request):
    """渲染 docs/architecture.md 架构文档。"""
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    doc_path = Path(__file__).parent.parent.parent / "docs" / "architecture.md"
    if not doc_path.exists():
        return _t(request, "doc.html", {
            "html": "<p style='color:var(--text-3)'>文档文件不存在</p>",
            "raw_path": str(doc_path),
            "mtime": "",
        })
    try:
        import markdown
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from gateway.config import TIMEZONE
        text = doc_path.read_text(encoding="utf-8")
        md = markdown.Markdown(extensions=[
            "tables", "fenced_code", "toc", "sane_lists",
            "nl2br", "attr_list",
        ])
        html = md.convert(text)
        toc_html = getattr(md, "toc", "") or ""
        # mtime 转本地时区
        mtime_ts = doc_path.stat().st_mtime
        mtime_dt = datetime.fromtimestamp(mtime_ts, tz=ZoneInfo(TIMEZONE))
        mtime_str = mtime_dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.exception("Failed to render doc")
        html = f"<pre style='color:#e07060'>渲染失败: {e}</pre>"
        toc_html = ""
        mtime_str = ""
    return _t(request, "doc.html", {
        "html": html,
        "toc": toc_html,
        "raw_path": str(doc_path),
        "mtime": mtime_str,
    })




@router.get("/push-vapid-key")
async def push_vapid_key(request: Request):
    """返回 VAPID 公钥（前端订阅用）。"""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        from gateway.webpush_manager import get_vapid_keys
        keys = get_vapid_keys()
        return JSONResponse({"publicKey": keys["public_key_b64"]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/push-subscribe")
async def push_subscribe(request: Request):
    """保存浏览器推送订阅。"""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        body = await request.json()
        from gateway.webpush_manager import save_subscription
        save_subscription(body)
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@router.delete("/push-subscribe")
async def push_unsubscribe(request: Request):
    """删除推送订阅。"""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        body = await request.json()
        from gateway.webpush_manager import remove_subscription
        remove_subscription(body.get("endpoint", ""))
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)



@router.post("/quick-price")
async def quick_price(request: Request):
    """快捷设置单个模型的价格（不影响其他设置）。"""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        body = await request.json()
        upstream = (body.get("upstream") or "").strip()
        model    = (body.get("model")    or "").strip()
        if not model:
            return JSONResponse({"error": "model required"}, status_code=400)
        price_type    = body.get("price_type", "token")
        price_per_call = float(body.get("price_per_call", 0) or 0)
        inp  = float(body.get("input",       0) or 0)
        out  = float(body.get("output",      0) or 0)
        cw   = float(body.get("cache_write", 0) or 0)
        cr   = float(body.get("cache_read",  0) or 0)

        from gateway.settings import get as _sget, update_settings as _upd
        prices = list(_sget("model_prices", []) or [])
        # 删除已有条目（同上游+模型）
        prices = [p for p in prices
                  if not ((p.get("upstream") or "") == upstream
                          and (p.get("model")    or "") == model)]
        new_entry = {"upstream": upstream, "model": model,
                     "price_type": price_type,
                     "price_per_call": price_per_call,
                     "input": inp, "output": out,
                     "cache_write": cw, "cache_read": cr}
        prices.append(new_entry)
        _upd(model_prices=prices)
        logger.info("quick-price: %s / %s set", upstream, model)
        return JSONResponse({"ok": True, "total": len(prices)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@router.post("/test-notify")
async def test_notify(request: Request):
    """发送测试推送通知（PushPlus + ntfy）。"""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    import asyncio as _aio
    from gateway.notifier import notify as _notify, _send_ntfy
    # ntfy 单独调用（不经过 PushPlus 冷却逻辑）
    ntfy_ok = await _send_ntfy(
        "🔔 Chat Gateway 测试通知",
        "ntfy 推送配置成功！",
        dedup_key="test_ntfy_0",
    )
    # PushPlus
    pushplus_ok = await _notify(
        "🔔 Chat Gateway 测试通知",
        "推送配置成功！网关正在正常运行。",
        dedup_key="test_notify",
        cooldown=0,
    )
    if pushplus_ok or ntfy_ok:
        channels = []
        if pushplus_ok: channels.append("微信(PushPlus)")
        if ntfy_ok: channels.append("Android(ntfy)")
        return JSONResponse({"ok": True, "channels": channels})
    else:
        return JSONResponse({"ok": False, "error": "未配置推送渠道或发送失败（请检查 PushPlus Token / ntfy 地址）"})



@router.post("/upstreams/{name}/toggle-think")
async def toggle_upstream_think(request: Request, name: str):
    """切换上游「强制思考」开关。"""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        body = await request.json()
        from gateway.upstream import load_upstreams, save_upstreams
        upstreams = load_upstreams()
        for u in upstreams:
            if u.name == name:
                u.force_think = body.get("force_think", not u.force_think)
                if "think_tag" in body:
                    u.think_tag = body["think_tag"] or "<think>\n"
                save_upstreams(upstreams)
                return JSONResponse({"ok": True, "force_think": u.force_think, "think_tag": u.think_tag})
        return JSONResponse({"error": "upstream not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@router.post("/upstreams/{name}/cache-ttl")
async def update_upstream_cache_ttl(request: Request, name: str):
    """更新上游缓存 TTL：1h / 5m / off。"""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        body = await request.json()
        cache_ttl = (body.get("cache_ttl") or "1h").strip()
        set_cache_ttl(name, cache_ttl)
        upstream = next((u for u in load_upstreams() if u.name == name), None)
        if not upstream:
            return JSONResponse({"error": "upstream not found"}, status_code=404)
        return JSONResponse({"ok": True, "cache_ttl": upstream.cache_ttl})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@router.post("/upstreams/{name}/add-key")
async def add_extra_key(request: Request, name: str):
    """给上游添加一个额外令牌。"""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    form = await request.form()
    key = (form.get("key") or "").strip()
    label = (form.get("label") or "").strip()
    purpose = (form.get("purpose") or "chat").strip()
    cache_ttl = (form.get("cache_ttl") or "").strip()
    model_prefixes = [
        p.strip() for p in (form.get("model_prefixes") or "").replace("，", ",").split(",")
        if p.strip()
    ]
    models = [
        m.strip() for m in (form.get("models") or "").replace("，", ",").split(",")
        if m.strip()
    ]
    if not key:
        return RedirectResponse(f"{_BASE}/upstreams?error=key_empty", status_code=302)
    upstreams = load_upstreams()
    for u in upstreams:
        if u.name == name:
            u.extra_keys = u.extra_keys or []
            u.extra_keys.append({
                "key": key,
                "label": label,
                "purpose": purpose,
                "model_prefixes": model_prefixes,
                "models": models,
                "cached_models": [],
                "cache_ttl": cache_ttl,
            })
            break
    save_upstreams(upstreams)
    return RedirectResponse(f"{_BASE}/upstreams", status_code=302)


@router.post("/upstreams/{name}/update-key/{idx}")
async def update_extra_key(request: Request, name: str, idx: int):
    """更新额外令牌的前端分组/模型匹配规则。"""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    form = await request.form()
    label = (form.get("label") or "").strip()
    purpose = (form.get("purpose") or "chat").strip()
    cache_ttl = (form.get("cache_ttl") or "").strip()
    model_prefixes = [
        p.strip() for p in (form.get("model_prefixes") or "").replace("，", ",").split(",")
        if p.strip()
    ]
    models = [
        m.strip() for m in (form.get("models") or "").replace("，", ",").split(",")
        if m.strip()
    ]
    upstreams = load_upstreams()
    for u in upstreams:
        if u.name != name:
            continue
        keys = u.extra_keys or []
        if 0 <= idx < len(keys) and isinstance(keys[idx], dict):
            keys[idx]["label"] = label
            keys[idx]["purpose"] = purpose
            keys[idx]["model_prefixes"] = model_prefixes
            keys[idx]["models"] = models
            keys[idx]["cache_ttl"] = cache_ttl
        u.extra_keys = keys
        break
    save_upstreams(upstreams)
    return RedirectResponse(f"{_BASE}/upstreams", status_code=302)


@router.post("/upstreams/{name}/delete-key/{idx}")
async def delete_extra_key(request: Request, name: str, idx: int):
    """删除上游的某个额外令牌（按索引）。"""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    upstreams = load_upstreams()
    for u in upstreams:
        if u.name == name:
            keys = u.extra_keys or []
            if 0 <= idx < len(keys):
                keys.pop(idx)
            u.extra_keys = keys
            break
    save_upstreams(upstreams)
    return RedirectResponse(f"{_BASE}/upstreams", status_code=302)



@router.get("/api/bp2-boundary")
async def bp2_boundary(request: Request, fingerprint: str, covered: int):
    """Get BP2 summary boundary: first message, last covered, next uncovered."""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    from gateway.db import get_db
    
    try:
        db = get_db()
        
        # First message
        cur = await db.execute(
            """SELECT id, timestamp, role, substr(content, 1, 180) AS preview
               FROM conversations
               WHERE fingerprint=? AND role IN ('user','assistant')
                 AND content IS NOT NULL AND content!=''
               ORDER BY id ASC LIMIT 1""",
            (fingerprint,)
        )
        first = await cur.fetchone()
        
        # Last covered message (Nth message, 1-indexed)
        cur = await db.execute(
            """SELECT id, timestamp, role, substr(content, 1, 180) AS preview
               FROM conversations
               WHERE fingerprint=? AND role IN ('user','assistant')
                 AND content IS NOT NULL AND content!=''
               ORDER BY id ASC LIMIT 1 OFFSET ?""",
            (fingerprint, covered - 1)
        )
        covered_last = await cur.fetchone()
        
        # Next uncovered message
        cur = await db.execute(
            """SELECT id, timestamp, role, substr(content, 1, 180) AS preview
               FROM conversations
               WHERE fingerprint=? AND role IN ('user','assistant')
                 AND content IS NOT NULL AND content!=''
               ORDER BY id ASC LIMIT 1 OFFSET ?""",
            (fingerprint, covered)
        )
        next_uncovered = await cur.fetchone()
        
        return JSONResponse({
            "fingerprint": fingerprint,
            "messages_covered": covered,
            "first": dict(first) if first else None,
            "covered_last": dict(covered_last) if covered_last else None,
            "next_uncovered": dict(next_uncovered) if next_uncovered else None,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/api/bp2-around")
async def bp2_around(request: Request, fingerprint: str, rn: int, radius: int = 5):
    """Get messages around Nth message (1-indexed row number)."""
    if not _check_auth(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    from gateway.db import get_db
    
    try:
        db = get_db()
        start_rn = max(1, rn - radius)
        end_rn = rn + radius
        
        cur = await db.execute(
            """SELECT rn, id, timestamp, role, substr(content, 1, 180) AS preview
               FROM (
                 SELECT
                   ROW_NUMBER() OVER (ORDER BY id ASC) AS rn,
                   id, timestamp, role, content
                 FROM conversations
                 WHERE fingerprint=? AND role IN ('user','assistant')
                   AND content IS NOT NULL AND content!=''
               )
               WHERE rn BETWEEN ? AND ?
               ORDER BY rn""",
            (fingerprint, start_rn, end_rn)
        )
        rows = await cur.fetchall()
        
        return JSONResponse({
            "fingerprint": fingerprint,
            "center_rn": rn,
            "radius": radius,
            "messages": [dict(r) for r in rows],
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



@router.get("/bp2-test", response_class=HTMLResponse)
async def bp2_test_page(request: Request):
    """BP2 boundary API test page."""
    if not _check_auth(request):
        return RedirectResponse(f"{_BASE}/login", status_code=302)
    return '\n<!DOCTYPE html>\n<html>\n<head>\n  <meta charset="utf-8">\n  <title>BP2 边界测试</title>\n  <style>\n    body { font-family: system-ui, sans-serif; margin: 2rem; }\n    input, button { padding: 0.5rem; margin: 0.5rem 0; }\n    pre { background: #f5f5f5; padding: 1rem; overflow-x: auto; }\n    .section { margin: 2rem 0; border: 1px solid #ddd; padding: 1rem; }\n  </style>\n</head>\n<body>\n  <h1>BP2 边界查询测试</h1>\n  \n  <div class="section">\n    <h2>1. 查询摘要边界</h2>\n    <p>Fingerprint: <input id="fp1" value="fp_1d5733c3b07133de" style="width:300px"></p>\n    <p>Covered: <input id="covered1" value="106" style="width:100px"></p>\n    <button onclick="queryBoundary()">查询边界</button>\n    <pre id="result1"></pre>\n  </div>\n  \n  <div class="section">\n    <h2>2. 查询第N条附近</h2>\n    <p>Fingerprint: <input id="fp2" value="fp_1d5733c3b07133de" style="width:300px"></p>\n    <p>Row Number: <input id="rn" value="106" style="width:100px"></p>\n    <p>Radius: <input id="radius" value="5" style="width:100px"></p>\n    <button onclick="queryAround()">查询附近</button>\n    <pre id="result2"></pre>\n  </div>\n  \n  <script>\n    async function queryBoundary() {\n      const fp = document.getElementById(\'fp1\').value;\n      const covered = document.getElementById(\'covered1\').value;\n      const url = `/chat-gateway/admin/api/bp2-boundary?fingerprint=${fp}&covered=${covered}`;\n      try {\n        const res = await fetch(url);\n        const data = await res.json();\n        document.getElementById(\'result1\').textContent = JSON.stringify(data, null, 2);\n      } catch (e) {\n        document.getElementById(\'result1\').textContent = \'Error: \' + e.message;\n      }\n    }\n    \n    async function queryAround() {\n      const fp = document.getElementById(\'fp2\').value;\n      const rn = document.getElementById(\'rn\').value;\n      const radius = document.getElementById(\'radius\').value;\n      const url = `/chat-gateway/admin/api/bp2-around?fingerprint=${fp}&rn=${rn}&radius=${radius}`;\n      try {\n        const res = await fetch(url);\n        const data = await res.json();\n        document.getElementById(\'result2\').textContent = JSON.stringify(data, null, 2);\n      } catch (e) {\n        document.getElementById(\'result2\').textContent = \'Error: \' + e.message;\n      }\n    }\n  </script>\n</body>\n</html>\n'
