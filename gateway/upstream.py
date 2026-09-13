import copy
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from gateway.config import UPSTREAMS_FILE


def _as_list(value) -> list[str]:
    """Normalize JSON/list/comma-separated model filters."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        parts = value.replace("，", ",").replace("；", ",").replace(";", ",").split(",")
        return [p.strip() for p in parts if p.strip()]
    return []


def _normalize_cache_ttl(value, default: str = "1h") -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "": default,
        "auto": default,
        "inherit": default,
        "1h": "1h",
        "60m": "1h",
        "60min": "1h",
        "1hour": "1h",
        "5m": "5m",
        "5min": "5m",
        "5minutes": "5m",
        "off": "off",
        "none": "off",
        "disable": "off",
        "disabled": "off",
        "false": "off",
        "0": "off",
    }
    return aliases.get(text, default)


def _purpose_matches(configured: str, requested: str) -> bool:
    configured = (configured or "chat").strip()
    requested = (requested or "chat").strip()
    return configured == requested or configured == "all"


def _key_filter_lists(key_info: dict) -> tuple[list[str], list[str]]:
    exact = _as_list(key_info.get("models")) + _as_list(key_info.get("cached_models"))
    prefixes = _as_list(key_info.get("model_prefixes"))
    return list(dict.fromkeys(exact)), list(dict.fromkeys(prefixes))


def key_has_model_filter(key_info: dict) -> bool:
    exact, prefixes = _key_filter_lists(key_info)
    return bool(exact or prefixes)


def key_matches_model(key_info: dict, model: str) -> bool:
    """Return True when an extra key is allowed to serve `model`."""
    model = (model or "").strip()
    if not model:
        return False
    exact, prefixes = _key_filter_lists(key_info)
    if model in exact:
        return True
    return any(model.startswith(prefix) for prefix in prefixes if prefix)


@dataclass
class Upstream:
    name: str
    base_url: str
    api_key: str
    api_format: str = "anthropic"
    is_active: bool = True
    default_model: str = ""
    cached_models: list = field(default_factory=list)
    primary_cached_models: list = field(default_factory=list)
    extra_keys: list = field(default_factory=list)  # [{"key": str, "label": str, "purpose": "chat"|"summary"|"all", "models": [], "model_prefixes": [], "cached_models": []}]
    hidden: bool = False   # 隐藏上游：不出现在 /v1/models，但仍可用于摘要等内部用途
    force_think: bool = False  # 强制思考：在消息末尾注入 <think> 助手预填充，引导推理
    think_tag: str = "<think>\n"  # 预填充内容，默认 <think>，也可改成 <thinking> 等
    cache_ttl: str = "1h"  # "1h" | "5m" | "off"; extra_keys can override per token group
    # new-api 价格自动同步用的分组名（决定 group_ratio 倍率）。空 = default。
    # 为什么不能自动探测：/api/pricing 只认后台凭据，匿名调用拿不到"我们这把
    # key 属于哪个分组"，只能由用户指定（见 services/price_sync.py 坑 1）。
    pricing_group: str = ""

    @property
    def messages_url(self) -> str:
        base = self.base_url.rstrip("/")
        if self.api_format == "openai":
            return f"{base}/v1/chat/completions"
        return f"{base}/v1/messages"

    def get_key(self, purpose: str = "chat", model: str = "") -> str:
        """
        按用途和模型取 API key。
        先找 purpose 匹配且 model 精确/前缀匹配的 extra key；
        再找 purpose 匹配且未限制模型的通用 extra key；最后回退默认 api_key。
        purpose: "chat" | "summary" | "all"
        """
        candidates = [
            k for k in (self.extra_keys or [])
            if isinstance(k, dict)
            and k.get("key")
            and _purpose_matches(k.get("purpose", "chat"), purpose)
        ]
        if model:
            for k in candidates:
                if key_has_model_filter(k) and key_matches_model(k, model):
                    return k["key"]
        for k in candidates:
            if not key_has_model_filter(k):
                return k["key"]
        return self.api_key

    def get_cache_ttl(self, purpose: str = "chat", model: str = "") -> str:
        """Return cache TTL for the selected token group."""
        default = _normalize_cache_ttl(self.cache_ttl, default="1h")
        candidates = [
            k for k in (self.extra_keys or [])
            if isinstance(k, dict)
            and k.get("key")
            and _purpose_matches(k.get("purpose", "chat"), purpose)
        ]
        if model:
            for k in candidates:
                if key_has_model_filter(k) and key_matches_model(k, model):
                    return _normalize_cache_ttl(k.get("cache_ttl"), default=default)
        for k in candidates:
            if not key_has_model_filter(k):
                return _normalize_cache_ttl(k.get("cache_ttl"), default=default)
        return default

    @property
    def models_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/models"


def _ensure_file():
    path = Path(UPSTREAMS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("[]", encoding="utf-8")


def _normalize_item(item: dict) -> None:
    """兼容旧格式（没有 default_model / cached_models 字段）。幂等，只影响缓存填充。"""
    item.setdefault("default_model", "")
    item.setdefault("cached_models", [])
    item.setdefault("primary_cached_models", [])
    item.setdefault("extra_keys", [])
    for k in item["extra_keys"]:
        if isinstance(k, dict):
            k.setdefault("label", "")
            k.setdefault("purpose", "chat")
            k.setdefault("models", [])
            k.setdefault("model_prefixes", [])
            k.setdefault("cached_models", [])
    item.setdefault("hidden", False)
    item.setdefault("force_think", False)
    item.setdefault("think_tag", "<think>\n")
    item.setdefault("pricing_group", "")
    item["cache_ttl"] = _normalize_cache_ttl(item.get("cache_ttl"), default="1h")
    for k in item["extra_keys"]:
        if isinstance(k, dict):
            k.setdefault("cache_ttl", "")


# 2026-09-01: mtime 缓存（与 settings.py 同一套思路）。之前每个代理请求都同步
# 读盘 + json.loads + 归一化（resolve_model_and_upstream → load_upstreams）。
# 以 (路径, mtime_ns, size) 为签名，后台改 upstreams.json / 测试换路径都会因
# stat 变化自动失效；save_upstreams 写完显式清缓存（Windows 同尺寸快速重写的
# mtime 粒度可能撞车，不能只靠 stat）。命中后按次 deepcopy 归一化数据，调用方
# 拿到的仍是彼此隔离的 Upstream 对象（与每次 json.loads 的新鲜语义一致）。
_upstream_cache: tuple[str, int, int, list[dict]] | None = None


def load_upstreams() -> list[Upstream]:
    global _upstream_cache
    _ensure_file()
    path = Path(UPSTREAMS_FILE)
    st = path.stat()
    sig = (str(path), st.st_mtime_ns, st.st_size)
    if _upstream_cache is None or _upstream_cache[:3] != sig:
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            _normalize_item(item)
        _upstream_cache = (*sig, data)
    result = []
    for item in _upstream_cache[3]:
        result.append(Upstream(**copy.deepcopy(item)))
    return result


def save_upstreams(upstreams: list[Upstream]):
    global _upstream_cache
    _ensure_file()
    data = [asdict(u) for u in upstreams]
    Path(UPSTREAMS_FILE).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _upstream_cache = None  # 写后显式失效（见 load_upstreams 注释）


def get_active_upstream() -> Upstream | None:
    for u in load_upstreams():
        if u.is_active:
            return u
    return None


def get_upstream_for_model(model: str) -> Upstream | None:
    """Return the active upstream whose cached_models contains the requested model.
    Falls back to the first active upstream if no match found."""
    upstreams = [u for u in load_upstreams() if u.is_active]
    if not upstreams:
        return None
    if model:
        for u in upstreams:
            if model in (u.cached_models or []) or model in (u.primary_cached_models or []):
                return u
            for key_info in (u.extra_keys or []):
                if (
                    isinstance(key_info, dict)
                    and _purpose_matches(key_info.get("purpose", "chat"), "chat")
                    and key_matches_model(key_info, model)
                ):
                    return u
    return upstreams[0]


def get_all_active_upstreams() -> list[Upstream]:
    return [u for u in load_upstreams() if u.is_active]


def add_upstream(name: str, base_url: str, api_key: str, api_format: str = "anthropic"):
    upstreams = load_upstreams()
    for u in upstreams:
        if u.name == name:
            raise ValueError(f"Upstream '{name}' already exists")
    upstreams.append(Upstream(
        name=name,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        api_format=api_format,
    ))
    save_upstreams(upstreams)


def remove_upstream(name: str):
    upstreams = load_upstreams()
    upstreams = [u for u in upstreams if u.name != name]
    save_upstreams(upstreams)


def toggle_upstream(name: str):
    upstreams = load_upstreams()
    for u in upstreams:
        if u.name == name:
            u.is_active = not u.is_active
            break
    save_upstreams(upstreams)


def set_default_model(name: str, model: str):
    upstreams = load_upstreams()
    for u in upstreams:
        if u.name == name:
            u.default_model = model
            break
    save_upstreams(upstreams)


def set_cache_ttl(name: str, cache_ttl: str):
    upstreams = load_upstreams()
    for u in upstreams:
        if u.name == name:
            u.cache_ttl = _normalize_cache_ttl(cache_ttl, default="1h")
            break
    save_upstreams(upstreams)


def set_pricing_group(name: str, pricing_group: str):
    """设置 new-api 价格同步用的分组名（空 = default）。"""
    upstreams = load_upstreams()
    for u in upstreams:
        if u.name == name:
            u.pricing_group = (pricing_group or "").strip()[:64]
            break
    save_upstreams(upstreams)


def update_cached_models(name: str, models: list[str]):
    upstreams = load_upstreams()
    for u in upstreams:
        if u.name == name:
            u.cached_models = models
            break
    save_upstreams(upstreams)


def update_primary_cached_models(name: str, models: list[str]):
    upstreams = load_upstreams()
    for u in upstreams:
        if u.name == name:
            u.primary_cached_models = models
            merged = list(dict.fromkeys(list(models or []) + list(u.cached_models or [])))
            u.cached_models = merged
            break
    save_upstreams(upstreams)


def update_extra_key_cached_models(name: str, idx: int, models: list[str]):
    upstreams = load_upstreams()
    for u in upstreams:
        if u.name != name:
            continue
        keys = u.extra_keys or []
        if 0 <= idx < len(keys) and isinstance(keys[idx], dict):
            keys[idx]["cached_models"] = models
            merged = list(u.primary_cached_models or [])
            for key_info in keys:
                if isinstance(key_info, dict):
                    merged += _as_list(key_info.get("cached_models"))
                    merged += _as_list(key_info.get("models"))
            u.cached_models = list(dict.fromkeys(merged or list(u.cached_models or [])))
        break
    save_upstreams(upstreams)


def resolve_model_and_upstream(model: str) -> tuple[str, "Upstream | None"]:
    """
    解析可选的上游前缀，返回 (clean_model, upstream)。

    支持两种格式：
      - "upstream_name::model_name"  → 指定上游（精确路由）
      - "model_name"                 → 按 cached_models 匹配（原有逻辑）

    示例：
      "55al::claude-sonnet-4-5"  → 路由到 55al，发给上游的 model = "claude-sonnet-4-5"
      "claude-sonnet-4-5"         → 匹配 cached_models 含该名的第一个上游
    """
    if "::" in model:
        upstream_name, clean_model = model.split("::", 1)
        upstreams = [u for u in load_upstreams() if u.is_active and u.name == upstream_name]
        if upstreams:
            return clean_model.strip(), upstreams[0]
        # 指定的上游不存在或已禁用，fallback 到普通匹配
    return model, get_upstream_for_model(model)


def _serves_model(u: "Upstream", model: str) -> bool:
    """该上游（主令牌或 chat 用途 extra key）能否服务该模型。"""
    if not model:
        return False
    if model in (u.cached_models or []) or model in (u.primary_cached_models or []):
        return True
    for key_info in (u.extra_keys or []):
        if (
            isinstance(key_info, dict)
            and _purpose_matches(key_info.get("purpose", "chat"), "chat")
            and key_matches_model(key_info, model)
        ):
            return True
    return False


def resolve_model_candidates(model: str, limit: int = 3) -> list["Upstream"]:
    """能服务该模型的活跃上游候选列表（failover 用，2026-09-06）。

    ⚙️ 受 settings.upstream_failover_enabled 开关控制（默认 False，备用上游
    余额不可控时关掉即可只走首选）：关闭时永远只返回首选一个候选。
    开启后首个与 resolve_model_and_upstream 的选择一致；其后按 upstreams.json
    顺序追加同样能服务该模型的其他活跃上游（limit 截断，防全站轮询放大故障）。
    显式 "upstream::model" 点名时不自动换站——用户点名了就尊重，想换站改前缀。
    """
    _, primary = resolve_model_and_upstream(model)
    candidates: list[Upstream] = [primary] if primary else []
    from gateway.settings import get as _get
    if not bool(_get("upstream_failover_enabled", False)):
        return candidates
    if primary and "::" in model:
        return candidates
    clean_model = model.split("::", 1)[1].strip() if "::" in model else model
    for u in load_upstreams():
        if len(candidates) >= max(1, limit):
            break
        if not u.is_active or any(u.name == c.name for c in candidates):
            continue
        if _serves_model(u, clean_model):
            candidates.append(u)
    return candidates

def toggle_hidden(name: str):
    upstreams = load_upstreams()
    for u in upstreams:
        if u.name == name:
            u.hidden = not u.hidden
            break
    save_upstreams(upstreams)
