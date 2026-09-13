import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", secrets.token_hex(32))
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "data" / "gateway.db"))
UPSTREAMS_FILE = os.getenv("UPSTREAMS_FILE", str(BASE_DIR / "data" / "upstreams.json"))
# new-api 自动价格快照（services/price_sync.py 写，costs.py 读）
AUTO_PRICES_FILE = os.getenv("AUTO_PRICES_FILE", str(BASE_DIR / "data" / "auto_prices.json"))
TIMEZONE = os.getenv("TIMEZONE", "Asia/Shanghai")
LISTEN_HOST = os.getenv("LISTEN_HOST", "127.0.0.1")
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "8899"))
UPSTREAM_TIMEOUT = int(os.getenv("UPSTREAM_TIMEOUT", "120"))

# Keep-alive: 活跃时段内每 55 分钟无对话则静默刷新缓存
KEEPALIVE_ENABLED    = os.getenv("KEEPALIVE_ENABLED", "true").lower() == "true"
KEEPALIVE_START_HOUR = int(os.getenv("KEEPALIVE_START_HOUR", "8"))   # 北京时间
KEEPALIVE_END_HOUR   = int(os.getenv("KEEPALIVE_END_HOUR", "23"))
KEEPALIVE_INTERVAL   = int(os.getenv("KEEPALIVE_INTERVAL", "55"))    # 分钟

# BP2 摘要模型（留空则用对话里的同款模型）
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "")
