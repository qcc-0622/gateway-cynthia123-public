"""导入卫生检查：全 gateway 模块的 Load 名字必须可解析（2026-09-06）。

背景：P2.1/P2.2 绞杀搬迁时模块被拆来拆去，函数体里引用的 hashlib /
asyncio 等模块级导入留在了原文件——只有生产数据路径才会触发 NameError
（archive 报废、 Seamless 注入 500），pytest 的临时库路径盖不住。
本测试静态扫描全部模块，任何"名字在模块里被引用但既未导入也未定义"
的情况直接失败，让缺导入永远无法再上线。

白名单：except 处理器名（e/exc/c 等，AST 的 ExceptHandler.name 不是
Name-Store 节点，这里按名字模式排除）、dunder、类体字符串标注等。
"""

import ast
import builtins
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
_GATEWAY_DIR = _PROJECT_ROOT / "gateway"
_EXC_NAME_OK = {"e", "e2", "e3", "exc", "err", "c", "x", "ex"}


def _unresolved_names(tree: ast.Module) -> set[str]:
    imported: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            imported |= {a.asname or a.name for a in n.names}
        elif isinstance(n, ast.Import):
            imported |= {(a.asname or a.name).split(".")[0] for a in n.names}
    defined = {n.name for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    assigned: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            assigned.add(n.id)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = n.args
            assigned |= {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
            if a.vararg:
                assigned.add(a.vararg.arg)
            if a.kwarg:
                assigned.add(a.kwarg.arg)
        if isinstance(n, ast.ExceptHandler) and n.name:
            assigned.add(n.name)          # except XxxError as e
    bi = set(dir(builtins)) | {"__name__", "__file__", "__doc__"}
    missing = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if (n.id not in imported and n.id not in defined
                    and n.id not in assigned and n.id not in bi):
                missing.add(n.id)
    return missing


def test_all_gateway_modules_resolve_names():
    problems = []
    for f in sorted(_GATEWAY_DIR.rglob("*.py")):
        if "_archive" in str(f):
            continue   # 留档模块不参与运行
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            problems.append(f"{f}: SyntaxError: {e}")
            continue
        missing = {m for m in _unresolved_names(tree) if m not in _EXC_NAME_OK}
        if missing:
            problems.append(f"{f.relative_to(_PROJECT_ROOT)}: {sorted(missing)}")
    assert not problems, "存在未解析名字（缺导入/缺定义）：\n" + "\n".join(problems)


def test_pipeline_table_valid():
    """PIPELINE_ANTHROPIC 的 after 约束必须始终满足（等价于启动校验）。"""
    from gateway.pipeline.runner import validate_pipeline
    from gateway.pipeline.steps import PIPELINE_ANTHROPIC
    validate_pipeline(PIPELINE_ANTHROPIC)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
