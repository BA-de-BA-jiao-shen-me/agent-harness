from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import json

from pyharness.runtime.engine import ToolRegistry
from pyharness.runtime.schemas import RiskLevel, ToolSpec
from pyharness.sandbox.providers import (
    SandboxProvider,
    SubprocessSandboxProvider,
)

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
    "dir",
    "type",
}
_REFLECTIVE_ATTRS = {"format", "format_map"}


class SandboxSecurityError(ValueError):
    """Raised when Python code fails static AST security pre-flight audit."""


class PythonSandbox:
    """
    Governed Python Computation Sandbox backed by a pluggable `SandboxProvider` seam
    (`DockerSandboxProvider` | `E2BSandboxProvider` | `SubprocessSandboxProvider`).

    Defense-in-depth protections:
    1. Pre-flight `audit_ast` rejects non-whitelisted imports, ALL private/dunder attributes
       (`attr.startswith("_")` — closing `collections._sys` & `__subclasses__` chains),
       and C-level string formatting reflection (`str.format` / `str.format_map`).
    2. Delegates execution out-of-process to `SandboxProvider` (ephemeral `--network=none`
       Docker container, Firecracker E2B MicroVM, or scrubbed-env isolated subprocess)
       eliminating host GIL starvation, host secret leaks, and cross-thread stdout bleed.
    """

    def __init__(
        self,
        provider: SandboxProvider | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.provider: SandboxProvider = provider or SubprocessSandboxProvider()
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def audit_ast(code: str) -> ast.Module:
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            raise SyntaxError(f"Python syntax error at line {exc.lineno}: {exc.msg}") from exc

        for node in ast.walk(tree):
            # 1. Block non-whitelisted imports
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
            # 2. Block ALL single/double underscore private & dunder attributes + str.format reflection
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("_") or node.attr in _REFLECTIVE_ATTRS:
                    raise SandboxSecurityError(
                        f"Access to private/dunder or reflective attribute '{node.attr}' is forbidden for sandbox safety."
                    )
            # 3. Block forbidden builtin calls & underscore identifiers
            elif isinstance(node, ast.Name):
                if node.id.startswith("_") or node.id in _FORBIDDEN_CALLS:
                    raise SandboxSecurityError(f"Use of '{node.id}' is forbidden in sandbox.")
        return tree

    async def arun(self, code: str) -> str:
        # Layer 1: zero-latency host AST pre-check before touching container/VM runtime
        self.audit_ast(code)
        # Layer 2: isolated process / Docker container / E2B MicroVM execution
        exec_res = await self.provider.execute(code=code, timeout_seconds=self.timeout_seconds)
        payload = {
            "stdout": exec_res.stdout,
            "result": exec_res.result,
            "sandbox_provider": exec_res.provider,
        }
        return json.dumps(payload, ensure_ascii=False)

    def run(self, code: str) -> str:
        """Synchronous convenience wrapper preserving backward compatibility for callers/tests."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(code))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(self.arun(code))).result()

    async def aclose(self) -> None:
        await self.provider.aclose()

    def register_tools(self, registry: ToolRegistry) -> None:
        registry.register(
            ToolSpec(
                name="run_python_sandbox",
                description="在进程/容器隔离且经 AST 安全审计的受限沙箱中执行纯数值/统计 Python 代码（可赋值给 result 或 print 输出）",
                parameters={
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "待执行的 Python 代码"}},
                    "required": ["code"],
                },
                risk_level=RiskLevel.MEDIUM,
                timeout_seconds=self.timeout_seconds + 2.0,
            ),
            self.arun,
        )
