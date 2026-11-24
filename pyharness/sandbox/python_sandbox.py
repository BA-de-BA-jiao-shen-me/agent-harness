from __future__ import annotations

import ast
import io
import json
import math
import statistics
from contextlib import redirect_stdout
from typing import Any

from pyharness.runtime.engine import ToolRegistry
from pyharness.runtime.schemas import RiskLevel, ToolSpec

_ALLOWED_IMPORTS = {"math", "statistics", "json", "datetime", "collections", "re"}
_FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "breakpoint",
    "input",
}
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


class SandboxSecurityError(ValueError):
    """Raised when Python code fails static AST security audit."""


class PythonSandbox:
    """
    AST-Audited Restricted Python Computation Sandbox.

    Defense-in-depth protections (great interview talking points):
    1. Parses code via `ast.parse` BEFORE execution.
    2. Blocks all dunder attribute access (e.g., `__class__`, `__subclasses__`, `__globals__`)
       preventing classic Python sandbox breakout chains.
    3. Restricts `import` strictly to a pure-math/data whitelist (`math`, `statistics`, `json`...).
    4. Strips `__builtins__` down to a minimal safe subset and captures stdout cleanly.
    """

    @staticmethod
    def audit_ast(code: str) -> ast.Module:
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            raise SyntaxError(f"Python syntax error at line {exc.lineno}: {exc.msg}") from exc

        for node in ast.walk(tree):
            # 1. Block dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in _ALLOWED_IMPORTS:
                        raise SandboxSecurityError(
                            f"Import of '{alias.name}' is forbidden in sandbox. Allowed: {sorted(_ALLOWED_IMPORTS)}"
                        )
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root not in _ALLOWED_IMPORTS:
                    raise SandboxSecurityError(
                        f"Import from '{node.module}' is forbidden in sandbox."
                    )
            # 2. Block dunder attribute access (prevents object.__subclasses__() escape!)
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__"):
                    raise SandboxSecurityError(
                        f"Access to dunder attribute '{node.attr}' is forbidden for sandbox safety."
                    )
            # 3. Block forbidden builtin calls & dunder names
            elif isinstance(node, ast.Name):
                if node.id.startswith("__") or node.id in _FORBIDDEN_CALLS:
                    raise SandboxSecurityError(f"Use of '{node.id}' is forbidden in sandbox.")
        return tree

    def run(self, code: str) -> str:
        tree = self.audit_ast(code)
        safe_globals: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            "math": math,
            "statistics": statistics,
            "json": json,
        }
        # Support whitelisted `import math` statements safely
        def _safe_import(name: str, *_args: Any, **_kwargs: Any) -> Any:
            root = name.split(".")[0]
            if root not in _ALLOWED_IMPORTS:
                raise SandboxSecurityError(f"Forbidden import: {name}")
            return __import__(name)

        safe_globals["__builtins__"] = {**_SAFE_BUILTINS, "__import__": _safe_import}
        local_ns: dict[str, Any] = {}
        stdout_buf = io.StringIO()

        with redirect_stdout(stdout_buf):
            exec(compile(tree, filename="<pyharness_sandbox>", mode="exec"), safe_globals, local_ns)

        output = stdout_buf.getvalue().strip()
        result_val = local_ns.get("result")
        payload = {
            "stdout": output,
            "result": result_val if isinstance(result_val, (int, float, str, list, dict, bool, type(None))) else repr(result_val),
        }
        return json.dumps(payload, ensure_ascii=False)

    def register_tools(self, registry: ToolRegistry) -> None:
        registry.register(
            ToolSpec(
                name="run_python_sandbox",
                description="在 AST 安全审计受限沙箱中执行纯数值/统计 Python 代码（可赋值给 result 或 print 输出）",
                parameters={
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "待执行的 Python 代码"}},
                    "required": ["code"],
                },
                risk_level=RiskLevel.MEDIUM,
                timeout_seconds=10.0,
            ),
            self.run,
        )
