"""声明式管道 runner（P2.2，PLAN_PIPELINE_DECLARATIVE.md 第五节/第七节）。

PipelineContext：步骤间数据传递的唯一载体（不再往 body 塞隐式键；存量
body["_gateway_*"] 由步骤/入口按原位置写入，下游 archiver/proxy 不受影响）。
Step：name + fn(ctx)（同步或 async）+ after 依赖声明（只做校验不做排序）+
可选 condition（跳过条件，如统一大脑分支）。
run_pipeline：按列表顺序执行；每步前后自动 _pipe_trace，trace 记录在 ctx。

设计决策（文档第五节）：不用拓扑排序——after 只用于启动校验（顺序错直接
报错而不是自动排），因为管道步骤有微妙顺序依赖，自动排序自由度太大。
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable

from gateway.pipeline.common import _pipe_trace

logger = logging.getLogger("gateway.pipeline.runner")


@dataclass
class PipelineContext:
    """管道上下文对象，步骤之间的数据传递全走这里。"""

    # ── 输入 ──
    body: dict                  # 请求 body（浅拷贝，步骤逐步修改——与旧实现一致）
    upstream: object | None     # 上游配置对象

    # ── 身份（阶段 A 产出） ──
    tag: str = ""
    fingerprint: str = ""
    context_id: str = ""
    context_fingerprint: str = ""
    history_hash: str = ""
    cache_ttl: str = "1h"
    cache_control: dict | None = None   # _cache_control_for_ttl(cache_ttl)，BP 断点注入用
    proactive_trigger: bool = False
    unified_enabled: bool = False   # 统一大脑总开关（文档第九节-3 的分支条件）

    # ── 消息列表（步骤共享的可变状态） ──
    messages: list = field(default_factory=list)

    # ── 调试 ──
    trace: list[str] = field(default_factory=list)  # 每步自动追加


@dataclass
class Step:
    name: str
    fn: Callable[[PipelineContext], object]   # fn(ctx) -> None 或 coroutine
    after: list = field(default_factory=list)  # 声明式依赖（校验用，不排序）
    condition: Callable[[PipelineContext], bool] | None = None  # 返回 False 跳过


def validate_pipeline(steps: list[Step]) -> None:
    """校验 after 约束是否被列表顺序满足（启动时调用，见文档第七节）。"""
    seen: set[str] = set()
    for step in steps:
        for dep in step.after:
            if dep not in seen:
                raise ValueError(
                    f"Pipeline step '{step.name}' declares after='{dep}', "
                    f"but '{dep}' has not appeared before it in the list."
                )
        seen.add(step.name)


async def run_pipeline(steps: list[Step], ctx: PipelineContext) -> PipelineContext:
    """按列表顺序执行步骤表。每步前后自动 _pipe_trace（debug_pipeline_trace
    开启时可见），trace 记录进 ctx.trace。"""
    for step in steps:
        if step.condition is not None and not step.condition(ctx):
            _pipe_trace(f"{step.name}:SKIP", ctx.messages)
            ctx.trace.append(f"{step.name}:SKIP")
            continue
        _pipe_trace(f"{step.name}:BEFORE", ctx.messages)
        result = step.fn(ctx)
        if asyncio.iscoroutine(result):
            await result
        _pipe_trace(f"{step.name}:AFTER", ctx.messages)
        ctx.trace.append(step.name)
    return ctx
