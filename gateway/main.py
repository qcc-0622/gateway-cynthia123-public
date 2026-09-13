import asyncio
import logging
import logging.handlers
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from gateway.db import init_db, close_db
from gateway.http_client import close_client
from gateway.routers.anthropic_proxy import router as anthropic_router
from gateway.routers.openai_proxy import router as openai_router
from gateway.routers.models import router as models_router
from gateway.routers.admin import router as admin_router
from gateway.keepalive import keepalive_loop
from gateway.balance_monitor import balance_monitor_loop
from gateway.services.price_sync import price_sync_loop

LOG_FILE = Path(__file__).parent.parent / "data" / "gateway.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
_fh = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_fh.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger().addHandler(_fh)
logger = logging.getLogger("gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting chat gateway...")
    # 配置覆盖清单（REFACTOR_ROADMAP P0.2）：settings.json 里与代码默认值不同的键，
    # 启动时醒目打印一次——"配置盖住了代码"从此不再是暗坑
    try:
        from gateway.settings import get_overridden_defaults
        _overrides = get_overridden_defaults()
        if _overrides:
            logger.warning("⚙️ settings.json 覆盖了 %d 个代码默认值：", len(_overrides))
            for _k, _d, _a in _overrides:
                logger.warning("   %s: 默认=%r → 实际=%r", _k, _d, _a)
        else:
            logger.info("⚙️ settings.json 无差异覆盖（与代码默认值一致或仅显式设置）")
    except Exception:
        logger.exception("配置覆盖清单生成失败（非致命）")
    await init_db()
    logger.info("Database initialized")
    # P2.2：启动时校验声明式管道的 after 约束（顺序错直接拒绝启动）
    from gateway.pipeline.runner import validate_pipeline
    from gateway.pipeline.steps import PIPELINE_ANTHROPIC
    validate_pipeline(PIPELINE_ANTHROPIC)
    logger.info("Pipeline validated: %d steps", len(PIPELINE_ANTHROPIC))
    # 一次性历史修复任务（早期几次生产事故的补丁）已从 lifespan 迁出，不再每次启动空跑
    task_keepalive = asyncio.create_task(keepalive_loop())
    task_balance = asyncio.create_task(balance_monitor_loop())
    task_price = asyncio.create_task(price_sync_loop())
    yield
    task_keepalive.cancel()
    task_balance.cancel()
    task_price.cancel()
    for t in (task_keepalive, task_balance, task_price):
        try:
            await t
        except asyncio.CancelledError:
            pass
    await close_client()
    await close_db()
    logger.info("Gateway stopped")


app = FastAPI(title="Chat Gateway", lifespan=lifespan, docs_url=None, redoc_url=None)

app.include_router(anthropic_router)
app.include_router(openai_router)
app.include_router(models_router)
app.include_router(admin_router)
app.mount("/static", StaticFiles(directory="gateway/static"), name="static")


@app.get("/health")
async def health():
    return {"status": "ok"}
