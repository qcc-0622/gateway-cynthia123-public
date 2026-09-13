"""pytest 全局收尾：每个测试模块跑完关掉 gateway 的 aiosqlite 连接。

为什么需要：aiosqlite 的 worker 线程是非 daemon 线程，连接不关就永远不退出，
进程会卡在 threading._shutdown。多个测试文件都只在 `if __name__ == "__main__"`
里关库（直跑模式够用），pytest 模式下那段不执行 → 最后一个模块的连接没人关
→ 整包 `pytest tests/` 打印 "N passed" 后永久挂住（不会返回 shell）。

注意 import 必须放在 fixture 里延迟执行：conftest 在测试模块之前被导入，
若在模块导入期 `import gateway.db`，会先于测试文件把 gateway.config.DB_PATH
固化到真实库，破坏 CLAUDE.md 里的「测试 DB 隔离铁律」。
"""

import asyncio

import pytest


@pytest.fixture(autouse=True, scope="module")
def _close_gateway_db_after_module():
    yield
    try:
        from gateway import db as db_module

        asyncio.run(db_module.close_db())
    except Exception:
        # 收尾失败不该让测试结果变红；这里的唯一目的是让进程能正常退出
        pass
